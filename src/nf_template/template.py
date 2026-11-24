"""``NFBinnedTemplate``: binned background template of a version 3 NF model.

The template is the NF density integrated over every Compton-data-space bin,

    P_i = integral over bin i of p(Em, phi, psi, chi) dEm dphi dOmega,

with p in keV^-1 rad^-1 sr^-1, so P_i is dimensionless. The nodes and weights
of the integral come from ``quadrature``; the density comes from the NF model
through cosipy (``BackgroundDensityApproximation``).
"""

import logging
import time
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
import torch
import healpy as hp
import astropy.units as u
from astropy.time import Time
from histpy import Histogram, Axes, Axis, HealpixAxis
from scoords import SpacecraftFrame
from cosipy.spacecraftfile import SpacecraftHistory                                      # cosipy's spacecraft_file.py
from cosipy.response.ml.NFModelIO import load_nf_model                                   # Pascal's NFModelIO.py
from cosipy.background_estimation.ml.NFBackground import BackgroundDensityApproximation  # cosipy's NFBackground.py (version 3: Pascal's branch)
from cosipy.background_estimation.ml.NFBackgroundModels import ConstantBackgroundRate    # Pascal's NFBackgroundModels.py (version 3)
from cosipy.background_estimation.free_norm_threeml_binned_bkg import FreeNormBinnedBackground   # cosipy's free_norm_threeml_binned_bkg.py

from .quadrature import (as_breaks, binned_gl_nodes, node_to_bin_matrix,
                         near_pole, healpix_subpixel_nodes, polar_pixel_nodes)
from .galactic import galactic_conversion_matrix

logger = logging.getLogger(__name__)


class NFBinnedTemplate:
    """Binned background template of an NF model, by integrating its density.

        P_i = integral over bin i of p(Em, phi, psi, chi) dEm dphi dOmega

    p is in keV^-1 rad^-1 sr^-1, so P_i is dimensionless (probability per bin).
    The binning is given later, to ``compute`` or ``get_binned_background``.

    Parameters
    ----------
    model_path : str or Path
        NF model file (.fits), major version 3.
    batch_size : int, default 4096
        Points per flow call (speed and memory only; keep >= 2048).
    device : str, default "cpu"
        Torch device.
    seam_delta_deg, polar_cap_deg, polar_subpixel_level, pole_colat_deg,
    pole_subpixel_level, near_pole_cap_deg, near_pole_subpixel_level
        Sky-node corrections at the chi = 0 seam and near the poles, for the
        version 3.00 models. Defaults validated in notebook 09; normally leave
        them unchanged.

    Attributes
    ----------
    probability : np.ndarray (n_em_bins, n_phi_bins, npix)
        P_i; NaN where not computed.
    subpixel_level_used : np.ndarray of int (npix,)
        Sub-pixel level used per pixel; -1 where not computed.
    model_version : str
        Version of the model file.
    model_counts, model_livetime : float
        Event count and livetime [s] from the model file.
    rate : float
        Model rate R = model_counts / model_livetime [1/s].
    """

    def __init__(self,
                 model_path: str | Path,
                 batch_size: int = 4096,
                 device: str = "cpu",
                 seam_delta_deg: float = 0.2,
                 polar_cap_deg: float = 4.0,
                 polar_subpixel_level: int = 5,
                 pole_colat_deg: float = 1.0,
                 pole_subpixel_level: int = 7,
                 near_pole_cap_deg: float = 10.0,
                 near_pole_subpixel_level: int = 3) -> None:

        self.model_path = model_path
        self.batch_size = _check_int(batch_size, "batch_size")
        self.device = device

        # chi = 0 seam: half-width [rad] of the strip where sky nodes are not placed
        self.seam_delta_rad = np.deg2rad(_check_seam_delta(seam_delta_deg))

        # Polar pixels: finer sky nodes, and finer still close to the pole
        self.polar_cap_rad = np.deg2rad(_check_angle(polar_cap_deg, "polar_cap_deg"))
        self.polar_subpixel_level = _check_int(polar_subpixel_level, "polar_subpixel_level", minimum=0)
        self.pole_colat_rad = np.deg2rad(_check_angle(pole_colat_deg, "pole_colat_deg"))
        self.pole_subpixel_level = _check_int(pole_subpixel_level, "pole_subpixel_level", minimum=0)

        # Pixels near (not touching) a pole: a minimum sub-pixel level
        self.near_pole_cap_rad = np.deg2rad(_check_angle(near_pole_cap_deg, "near_pole_cap_deg"))
        self.near_pole_subpixel_level = _check_int(near_pole_subpixel_level, "near_pole_subpixel_level",
                                                   minimum=0)

        self._load_model(model_path)

        # No binning yet: set by the first compute / get_binned_background call
        self._settings = None
        self.probability = None
        self.subpixel_level_used = None

    def _load_model(self, model_path: str | Path) -> None:
        """
        Read the model file: the density (shape) and the rate (normalization).

        Sets ``model_version``, ``model_counts``, ``model_livetime``, ``rate``
        and the density. The tests replace this method with an analytic density.
        """
        # A dict with "version", "density_input" (weights and settings of the flow)
        # and "rate_input" (counts and livetime)
        model_file = load_nf_model(model_path)                          # Pascal's NFModelIO.py
        self.model_version = model_file["version"]
        _check_model_version(self.model_version)
        major_version = int(str(self.model_version).split(".")[0])

        # Normalization: constant rate R = counts / livetime
        rate_input = model_file["rate_input"]
        self.model_counts = float(rate_input["counts"])
        self.model_livetime = float(rate_input["livetime"])
        self.rate = ConstantBackgroundRate(rate_input).rate             # Pascal's NFBackgroundModels.py (version 3)

        # Density: picks the flow class from the version and checks the input dimensions.
        # Arguments: (major_version, density_input, device, batch_size, compile_mode); None = no torch.compile
        self._density = BackgroundDensityApproximation(major_version, model_file["density_input"],   # version 3: Pascal's branch
                                                       self.device, self.batch_size, None)
        self._density_model = self._density._model     # the flow itself (used directly in notebooks 08b, 09)

    # -----------------------------------------------------------------------
    # 1. Binning: Em and Phi nodes (set once per binning)
    # -----------------------------------------------------------------------

    def _set_binning(self,
                     axes: Axes,
                     n_em_per_bin: int,
                     n_phi_per_bin: int,
                     em_breaks_kev: ArrayLike | None,
                     phi_breaks_deg: ArrayLike | None) -> None:
        """
        Build the Em and Phi nodes for this binning; reset results if it changed.

        The results (``probability``, ``subpixel_level_used``) belong to one
        binning and one set of template settings. The same binning as the last
        call keeps them; a different one discards them (with a warning).

        """
        em_edges_kev, phi_edges_deg, nside, scheme = _check_axes(axes)
        n_em_per_bin = _check_int(n_em_per_bin, "n_em_per_bin")
        n_phi_per_bin = _check_int(n_phi_per_bin, "n_phi_per_bin")
        em_breaks_kev = as_breaks(em_breaks_kev)
        phi_breaks_deg = as_breaks(phi_breaks_deg)

        # Same binning and settings as the last call: reuse nodes and results
        settings = (tuple(em_edges_kev), 
                    tuple(phi_edges_deg), 
                    nside, 
                    scheme,
                    n_em_per_bin, 
                    n_phi_per_bin, 
                    tuple(em_breaks_kev), 
                    tuple(phi_breaks_deg),
                    self.seam_delta_rad,                                        # chi = 0 strip
                    self.polar_cap_rad, self.polar_subpixel_level,              # polar pixels
                    self.pole_colat_rad, self.pole_subpixel_level,              # ... and their refinement at the pole
                    self.near_pole_cap_rad, self.near_pole_subpixel_level)      # pixels near a pole
        if settings == self._settings:
            return

        # Something changed: stored results no longer apply
        if self.subpixel_level_used is not None:
            n_lost = int((self.subpixel_level_used >= 0).sum())
            if n_lost:
                logger.warning(f"Binning or settings changed: results of {n_lost} computed pixel(s) discarded.")

        # New binning: store it ...
        self.em_edges_kev, self.phi_edges_deg = em_edges_kev, phi_edges_deg
        self.nside, self.scheme, self.nest = nside, scheme, scheme == "nested"
        self.npix = hp.nside2npix(nside)
        self.n_em_bins, self.n_phi_bins = len(em_edges_kev) - 1, len(phi_edges_deg) - 1

        # ... build the Em and Phi nodes (the same for every pixel) ...
        self.em_nodes_kev, em_node_weights, em_bin_index = binned_gl_nodes(      # Em nodes, weights [keV],
            em_edges_kev, n_em_per_bin, em_breaks_kev)                           # and bin of each node

        self.phi_nodes_rad, phi_node_weights, phi_bin_index = binned_gl_nodes(   # same for Phi, in rad
            np.deg2rad(phi_edges_deg), n_phi_per_bin, np.deg2rad(phi_breaks_deg))

        # ... and the weights per bin: dEm and dphi of the integral (0 for nodes of other bins)
        self._em_weights = node_to_bin_matrix(em_node_weights, em_bin_index, self.n_em_bins)       # dEm [keV], (Em bins, Em nodes)
        self._phi_weights = node_to_bin_matrix(phi_node_weights, phi_bin_index, self.n_phi_bins)   # dphi [rad], (Phi bins, Phi nodes)

        # Start over: nothing computed for this binning yet
        self.probability = np.full((self.n_em_bins, self.n_phi_bins, self.npix), np.nan)
        self.subpixel_level_used = np.full(self.npix, -1, dtype=np.int64)
        self._settings = settings

    # -----------------------------------------------------------------------
    # 2. The NF density
    # -----------------------------------------------------------------------

    def _evaluate_density(
            self, 
            em: np.ndarray, 
            phi: np.ndarray,
            chi: np.ndarray, 
            psi: np.ndarray
            ) -> np.ndarray:
        """
        NF density at paired points.

        Parameters
        ----------
        em, phi, chi, psi : np.ndarray (n,)
            Em [keV], Phi, chi, psi [rad], spacecraft frame.

        Returns
        -------
        np.ndarray (n,)
            Density [keV^-1 rad^-1 sr^-1], float64.
        """
        source = torch.from_numpy(np.stack([em, phi, chi, psi], axis=1))   # (n, 4), the column order the model expects
        context = torch.zeros((len(em), 1), dtype=torch.float32)       
            # time: ignored by version 3 models
        return self._density.evaluate_density(context, source).numpy().astype(np.float64)   # cosipy's NFBase.py

    def _density_on_grid(
            self, 
            psi: np.ndarray, 
            chi: np.ndarray
            ) -> np.ndarray:
        """
        NF density at every (Em node, Phi node, sky point).

        Parameters
        ----------
        psi, chi : np.ndarray (n_sky,)
            Paired sky points [rad], spacecraft frame (psi = colatitude, chi = longitude).

        Returns
        -------
        np.ndarray (n_em_nodes, n_phi_nodes, n_sky)
            Density [keV^-1 rad^-1 sr^-1], float64.
        """
        shape = (len(self.em_nodes_kev), len(self.phi_nodes_rad), len(psi))

        # chi = 0 seam: the flow's density is wrong in a thin strip there, so nodes are moved just outside it
        chi = np.clip(chi, self.seam_delta_rad, 2 * np.pi - self.seam_delta_rad)

        # Every Em node x every Phi node x every sky point
        em_grid = np.broadcast_to(self.em_nodes_kev[:, None, None], shape)     # Em varies along axis 0
        phi_grid = np.broadcast_to(self.phi_nodes_rad[None, :, None], shape)   # Phi varies along axis 1
        chi_grid = np.broadcast_to(chi[None, None, :], shape)                  # chi and psi vary together along
        psi_grid = np.broadcast_to(psi[None, None, :], shape)                  # axis 2: each sky point is one pair

        density = self._evaluate_density(em_grid.ravel(), phi_grid.ravel(), chi_grid.ravel(), psi_grid.ravel())
        return density.reshape(shape)

    # -----------------------------------------------------------------------
    # 3. Plan: sky level of each pixel, normal or polar
    # -----------------------------------------------------------------------

    def subpixel_levels(
            self, 
            nside: int, 
            pixels: ArrayLike, 
            nest: bool, 
            subpixel_level: int
            ) -> np.ndarray:
        """
        Sub-pixel level each pixel gets when ``subpixel_level`` is requested.

        ``subpixel_level`` in general, at least ``near_pole_subpixel_level``
        near a pole, and at least ``polar_subpixel_level`` in polar pixels.

        Returns
        -------
        np.ndarray of int (n_pixels,)
        """
        pixels = np.atleast_1d(np.asarray(pixels, dtype=np.int64))
        near = near_pole(nside, pixels, nest, self.near_pole_cap_rad)       # within near_pole_cap_deg of a pole
        polar = near_pole(nside, pixels, nest, self.polar_cap_rad)          # the polar pixels

        levels = np.where(near, max(subpixel_level, self.near_pole_subpixel_level), subpixel_level)   # near-pole (<= 10 deg): at least 3; others: subpixel_level
        levels = np.where(polar, max(subpixel_level, self.polar_subpixel_level), levels)             # polar (<= 4 deg): at least 5; others: unchanged
        return levels


    def _split_pixels(
            self, 
            nside: int, 
            pixels: np.ndarray, nest: bool,
            subpixel_level: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normal and polar pixels, each with its level (shared by ``compute`` and ``n_evaluations``).

        Polar pixels use refined sky nodes (``polar_pixel_nodes``), so they are
        integrated separately.

        Returns
        -------
        normal, normal_levels, polar, polar_levels : np.ndarray of int
        """
        levels = self.subpixel_levels(nside, pixels, nest, subpixel_level)
        polar = near_pole(nside, pixels, nest, self.polar_cap_rad)        # polar pixels go to the polar routine
        return pixels[~polar], levels[~polar], pixels[polar], levels[polar]

    # -----------------------------------------------------------------------
    # 4. The integral
    # -----------------------------------------------------------------------

    def compute(
            self,
            axes: Axes,
            pixels: ArrayLike | None = None,
            subpixel_level: int = 2,
            n_em_per_bin: int = 4,
            n_phi_per_bin: int = 4,
            em_breaks_kev: ArrayLike | None = None,
            phi_breaks_deg: ArrayLike | None = None,
            points_per_chunk: int = 2_000_000,
            verbose: bool = False
            ) -> np.ndarray:
        """
        Integrate the NF density over every (Em, Phi, PsiChi) bin of the given pixels.

        Results are stored in ``probability`` chunk by chunk. The given pixels
        are always recomputed.

        Parameters
        ----------
        axes : histpy.Axes
            Binning with "Em", "Phi" and "PsiChi"; other axes are ignored.
        pixels : array_like of int, optional
            PsiChi pixel indices; None = all pixels.
        subpixel_level : int, default 2
            Quadrature (this and the next four set accuracy and cost of the integral):
            each PsiChi pixel is split into 4**subpixel_level sub-pixels (sky nodes).
        n_em_per_bin : int, default 4
            Gauss-Legendre nodes per Em bin.
        n_phi_per_bin : int, default 4
            Gauss-Legendre nodes per Phi bin.
        em_breaks_kev : array_like of float, optional
            Em values [keV] where a bin is split into pieces, each with its own nodes; None = no split.
        phi_breaks_deg : array_like of float, optional
            Same for Phi [deg].
        points_per_chunk : int, default 2_000_000
            Density evaluations per chunk (bounds memory).
        verbose : bool, default False
            Print progress.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins, n_pixels)
            P_i (dimensionless) of the given pixels.
        """
        # 1. Prepare: Em and Phi nodes for this binning, and check the inputs

        self._set_binning(axes, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg)
        assert self.probability is not None and self.subpixel_level_used is not None   # set by _set_binning
        pixels = self._check_pixels(pixels)
        subpixel_level = _check_int(subpixel_level, "subpixel_level", minimum=0)
        points_per_chunk = _check_int(points_per_chunk, "points_per_chunk")

        # 2. Split the pixels: normal (and near-pole) or polar, each with its level
        normal, normal_levels, polar, polar_levels = self._split_pixels(self.nside, pixels, self.nest,subpixel_level)

        n_em_phi_nodes = len(self.em_nodes_kev) * len(self.phi_nodes_rad)

        # 3. Normal pixels: grouped by level, in chunks of about points_per_chunk evaluations
        for level in np.unique(normal_levels):
            group = normal[normal_levels == level]
            points_per_pixel = n_em_phi_nodes * 4**int(level)                  # evaluations for one pixel
            pixels_per_chunk = max(1, points_per_chunk // points_per_pixel)    # pixels that fit in one chunk
            n_chunks = -(-len(group) // pixels_per_chunk)                      # ceiling division

            if verbose:
                print(f"{len(group)} pixels, level {level}: {points_per_pixel * len(group):,} evaluations "
                      f"in {n_chunks} chunks of up to {pixels_per_chunk} pixels")
            start_time = time.perf_counter()

            for i, start in enumerate(range(0, len(group), pixels_per_chunk)):
                # Integrate one chunk and store it, so finished chunks survive an interruption
                chunk = group[start:start + pixels_per_chunk]
                self.probability[:, :, chunk] = self._integrate_pixels(chunk, int(level))
                self.subpixel_level_used[chunk] = level

                if verbose:
                    elapsed = time.perf_counter() - start_time
                    done = start + len(chunk)
                    print(f"  chunk {i + 1}/{n_chunks}: {done}/{len(group)} pixels, {elapsed:.0f} s, "
                          f"{done * points_per_pixel / elapsed:,.0f} evaluations/s")

        # 4. Polar pixels: one at a time (each has many sky nodes)
        for i, (pixel, level) in enumerate(zip(polar, polar_levels)):
            start_time = time.perf_counter()
            self.probability[:, :, pixel], n_sky = self._integrate_polar_pixel(pixel, int(level), points_per_chunk)
            self.subpixel_level_used[pixel] = level

            if verbose:
                elapsed = time.perf_counter() - start_time
                print(f"  polar pixel {pixel} ({i + 1}/{len(polar)}): {n_sky:,} sky nodes, {elapsed:.0f} s, "
                      f"{n_sky * n_em_phi_nodes / elapsed:,.0f} evaluations/s")

        # 5. P_i of the requested pixels
        return self.probability[:, :, pixels].copy()

    def _integrate_pixels(
            self, 
            pixels: np.ndarray, 
            level: int
            ) -> np.ndarray:
        """
        P_i of normal (non-polar) pixels at one sub-pixel level.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins, n_pixels)
        """
        # 1. Sky nodes: 4**level sub-pixel centres per pixel, weight = sub-pixel area [sr]
        psi, chi, sky_weights = healpix_subpixel_nodes(self.nside, level, pixels, nest=self.nest)   # (n_pixels, 4**level)

        # 2. NF density at every (Em node, Phi node, sky node)
        density = self._density_on_grid(psi.ravel(), chi.ravel())                  # all sky points as one list
        density = density.reshape(len(self.em_nodes_kev), len(self.phi_nodes_rad), len(pixels), 4**level)                           # sky -> (pixel, sub-pixel)

        # 3. The integral: weighted sum over the nodes of each bin
        # P[a, b, c] = sum_e sum_p sum_s  dEm[a, e] * dphi[b, p] * density[e, p, c, s] * dOmega[c, s]
        # a: Em bin, b: Phi bin, c: pixel; e, p, s: Em, Phi and sky nodes
        #   sum over e with _em_weights   (dEm)    -> integral over Em
        #   sum over p with _phi_weights  (dphi)   -> integral over Phi
        #   sum over s with sky_weights   (dOmega) -> integral over the pixel
        return np.einsum("ae,bp,epcs,cs->abc", self._em_weights, self._phi_weights,
                         density, sky_weights, optimize=True)

    def _integrate_polar_pixel(
            self, 
            pixel: int, 
            level: int, 
            points_per_chunk: int
            ) -> tuple[np.ndarray, int]:
        """
        P_i of one polar pixel, with refined sky nodes near the pole.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins)
            P_i of the pixel.
        int
            Number of sky nodes used.
        """
        # 1. Sky nodes: the pixel's level; those close to the pole at the pole level
        psi, chi, sky_weights = polar_pixel_nodes(self.nside, pixel, self.nest, level,
                                                  max(level, self.pole_subpixel_level), self.pole_colat_rad)

        # 2.-3. Density and the integral as in _integrate_pixels, in pieces of about points_per_chunk evaluations
        n_sky = max(1, points_per_chunk // (len(self.em_nodes_kev) * len(self.phi_nodes_rad)))   # sky nodes per piece
        total = np.zeros((self.n_em_bins, self.n_phi_bins))
        
        for a in range(0, len(sky_weights), n_sky):
            density = self._density_on_grid(psi[a:a + n_sky], chi[a:a + n_sky])   # (Em, Phi, sky)
            # Integral over Em (dEm), Phi (dphi) and this piece of the pixel (dOmega)
            total += np.einsum("ae,bp,eps,s->ab", self._em_weights, self._phi_weights,
                               density, sky_weights[a:a + n_sky], optimize=True)
        return total, len(sky_weights)

    # -----------------------------------------------------------------------
    # 5. Outputs
    # -----------------------------------------------------------------------

    def get_binned_background(
            self,
            axes: Axes,
            frame: str,
            sc_history: SpacecraftHistory | str | Path,
            tmin: Time | float | None = None,
            tmax: Time | float | None = None,
            pixels: ArrayLike | None = None,
            subpixel_level: int = 2,
            n_em_per_bin: int = 4,
            n_phi_per_bin: int = 4,
            em_breaks_kev: ArrayLike | None = None,
            phi_breaks_deg: ArrayLike | None = None,
            scatt_nside: int = 32,
            points_per_chunk: int = 2_000_000
            ) -> Histogram:
        
        """
        Expected background counts, binned in Em x Phi x PsiChi.

        N_i = R * T * P_i, with R = ``rate`` [1/s] and T the live time [s] of
        the history. Missing pixels are computed first.

        Parameters
        ----------
        axes, pixels, subpixel_level, n_em_per_bin, n_phi_per_bin,
        em_breaks_kev, phi_breaks_deg, points_per_chunk
            As in ``compute``. ``pixels`` only for "local" (others are NaN).
        frame : str
            "local" (spacecraft frame) or "galactic".
        sc_history : SpacecraftHistory, str or Path
            Attitude and live time: a SpacecraftHistory, or an orientation file.
        tmin, tmax : astropy Time or float (unix time [s]), optional
            Time window; None = the whole history.
        scatt_nside : int, default 32
            Attitude grouping for the Galactic rotation.

        Returns
        -------
        histpy.Histogram (n_em_bins, n_phi_bins, npix)
            Expected counts.
        """
        _check_frame(frame, pixels)
        history = _open_history(sc_history, tmin, tmax)

        # 1. Integrate the requested pixels not yet computed at their level
        self._set_binning(axes, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg)
        assert self.probability is not None and self.subpixel_level_used is not None   # set by _set_binning

        level = _check_int(subpixel_level, "subpixel_level", minimum=0)
        missing = self._missing_pixels(pixels, level)

        if missing.size:
            self.compute(axes, missing, level, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg,
                         points_per_chunk)

        # 2a. Local frame: counts = R * T * P
        if frame == "local":
            livetime = _total_livetime(history)

            # A pixel stored at another sub-pixel level (e.g. from a call with a
            # different level) is NaN in the result, like one never computed
            level_now = self.subpixel_levels(self.nside, np.arange(self.npix), self.nest, level)
            probability = np.where(self.subpixel_level_used == level_now, self.probability, np.nan)
            return Histogram(self._axes(SpacecraftFrame()), contents=self.rate * livetime * probability)

        # 2b. Galactic frame: rotate with W (live time each Galactic pixel spends in each local pixel)
        W = galactic_conversion_matrix(history, self.nside, self.nest, scatt_nside, sparse=True)   # (Galactic, local) [s]
        P = self.probability.reshape(-1, self.npix)                     # (Em x Phi bins, local pixels)
        counts = self.rate * (W @ P.T).T                                # (Em x Phi bins, Galactic pixels)
        return Histogram(self._axes("galactic"), contents=counts.reshape(self.probability.shape))

    def get_free_norm_background(
            self,
            axes: Axes,
            frame: str,
            sc_history: SpacecraftHistory | str | Path,
            tmin: Time | float | None = None,
            tmax: Time | float | None = None,
            subpixel_level: int = 2,
            n_em_per_bin: int = 4,
            n_phi_per_bin: int = 4,
            em_breaks_kev: ArrayLike | None = None,
            phi_breaks_deg: ArrayLike | None = None,
            scatt_nside: int = 32
            ) -> FreeNormBinnedBackground:
        """
        The template as cosipy's binned background model, for the fit.

        cosipy's ``FreeNormBinnedBackground`` normalizes the template to sum 1
        and predicts norm [Hz] * live time [s] * template. The norm starts at
        the model rate inside this binning (sum of N_i / T), so the starting
        prediction equals ``get_binned_background``. All pixels are needed.

        Parameters
        ----------
        All as in ``get_binned_background``.

        Returns
        -------
        cosipy.background_estimation.FreeNormBinnedBackground
            Background with label "bkg_norm" and the starting norm set.
        """
        history = _open_history(sc_history, tmin, tmax)
        counts = self.get_binned_background(axes, frame, history, pixels=None, subpixel_level=subpixel_level,
                                            n_em_per_bin=n_em_per_bin, n_phi_per_bin=n_phi_per_bin,
                                            em_breaks_kev=em_breaks_kev, phi_breaks_deg=phi_breaks_deg,
                                            scatt_nside=scatt_nside)
        
        contents = cast(np.ndarray, counts.contents)                     # dense, no units (cast: type checker only)
        if np.isnan(contents).any():
            raise RuntimeError("The template has NaN bins; cosipy's normalization would fail.")

        background = FreeNormBinnedBackground(counts, history)          # cosipy's free_norm_threeml_binned_bkg.py
        livetime = history.cumulative_livetime().to_value(u.s)          # the live time cosipy uses
        background.set_norm(u.Quantity(np.sum(contents) / livetime, u.Hz))
        return background

    def expected_counts(self) -> np.ndarray:
        """
        Expected counts over the model livetime: model_counts * P_i.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins, npix)
            Counts. NaN = not computed.
        """
        self._check_binning_set()
        assert self.probability is not None                             # guaranteed by _check_binning_set
        return self.model_counts * self.probability

    def to_histogram(self) -> Histogram:
        """
        Template P_i as a histpy Histogram, spacecraft frame (all pixels needed).

        Returns
        -------
        histpy.Histogram (n_em_bins, n_phi_bins, npix)
            P_i (dimensionless). Axes "Em" [keV], "Phi" [deg], "PsiChi".
        """
        self._check_binning_set()
        self._check_all_computed()
        return Histogram(self._axes(SpacecraftFrame()), contents=self.probability)

    def _missing_pixels(
            self, 
            pixels: ArrayLike | None, 
            subpixel_level: int
            ) -> np.ndarray:
        """
        Requested pixels (None = all) not yet computed at their level.
        """
        assert self.subpixel_level_used is not None                     # set by _set_binning
        pixels = self._check_pixels(pixels)
        level_now = self.subpixel_levels(self.nside, pixels, self.nest, subpixel_level)
        return pixels[self.subpixel_level_used[pixels] != level_now]

    def _axes(
            self, 
            coordsys: SpacecraftFrame | str
            ) -> Axes:
        """
        Em, Phi, PsiChi axes of the template, PsiChi in the given frame.
        """
        return Axes([Axis(self.em_edges_kev, unit=u.keV, label="Em"),
                     Axis(self.phi_edges_deg, unit=u.deg, label="Phi"),
                     HealpixAxis(nside=self.nside, scheme=self.scheme,
                                 coordsys=coordsys, label="PsiChi")])

    # -----------------------------------------------------------------------
    # 6. Cost estimate (counts evaluations; not part of the integral)
    # -----------------------------------------------------------------------

    def n_evaluations(self,
                      axes: Axes,
                      pixels: ArrayLike | None = None,
                      subpixel_level: int = 2,
                      n_em_per_bin: int = 4,
                      n_phi_per_bin: int = 4,
                      em_breaks_kev: ArrayLike | None = None,
                      phi_breaks_deg: ArrayLike | None = None) -> int:
        """
        Number of density evaluations a ``compute`` call would need.

        Arguments as in ``compute``. Divide by the throughput (~7,000
        evaluations/s on CPU) to estimate the run time. Nothing is computed
        and the stored results are not touched.

        Returns
        -------
        int
            Number of density evaluations.
        """
        em_edges_kev, phi_edges_deg, nside, scheme = _check_axes(axes)

        n_em_nodes = len(binned_gl_nodes(em_edges_kev, _check_int(n_em_per_bin, "n_em_per_bin"),
                                         as_breaks(em_breaks_kev))[0])
        
        n_phi_nodes = len(binned_gl_nodes(phi_edges_deg, _check_int(n_phi_per_bin, "n_phi_per_bin"),
                                          as_breaks(phi_breaks_deg))[0])
        
        level = _check_int(subpixel_level, "subpixel_level", minimum=0)
        nest = scheme == "nested"
        pixels = np.arange(hp.nside2npix(nside)) if pixels is None else np.atleast_1d(np.asarray(pixels, dtype=np.int64))

        # Sky nodes: 4**level per normal pixel, plus the refined nodes of each polar pixel
        normal, normal_levels, polar, polar_levels = self._split_pixels(nside, pixels, nest, level)
        n_sky = int((4**normal_levels).sum())
        for pixel, pixel_level in zip(polar, polar_levels):
            weights = polar_pixel_nodes(nside, pixel, nest, int(pixel_level),
                                        max(int(pixel_level), self.pole_subpixel_level), self.pole_colat_rad)[2]
            n_sky += len(weights)
        return n_em_nodes * n_phi_nodes * n_sky

    # -----------------------------------------------------------------------
    # Checks that need the current binning
    # -----------------------------------------------------------------------

    def _check_pixels(self, pixels: ArrayLike | None) -> np.ndarray:
        """Pixel indices as a 1D array of unique, valid ints (None = all)."""
        if pixels is None:
            return np.arange(self.npix)
        pixels = np.atleast_1d(np.asarray(pixels, dtype=np.int64))
        if pixels.ndim != 1:
            raise ValueError("pixels must be a 1D list of pixel indices.")
        if np.any(pixels < 0) or np.any(pixels >= self.npix):
            raise ValueError(f"pixel indices must be in [0, {self.npix - 1}].")
        if len(np.unique(pixels)) != len(pixels):
            raise ValueError("pixel indices must be unique.")
        return pixels

    def _check_binning_set(self) -> None:
        """A binning must have been set by compute / get_binned_background."""
        if self._settings is None:
            raise RuntimeError("No binning yet: call compute or get_binned_background first.")

    def _check_all_computed(self) -> None:
        """Every pixel must be computed (no NaN left)."""
        assert self.probability is not None                             # called after _check_binning_set
        missing = np.isnan(np.asarray(self.probability)).any(axis=(0, 1))
        if missing.any():
            raise RuntimeError(f"{missing.sum()} of {self.npix} pixels are not computed yet.")


# ---------------------------------------------------------------------------
# Input checks and small helpers (no template state needed)
# ---------------------------------------------------------------------------

def _check_axes(axes: Axes) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Em edges [keV], Phi edges [deg], PsiChi NSIDE and scheme from histpy Axes."""
    labels = list(axes.labels)
    for label in ("Em", "Phi", "PsiChi"):
        if label not in labels:
            raise ValueError(f"axes must contain 'Em', 'Phi' and 'PsiChi', got {labels}.")

    # cast only informs the type checker: axes[label] is one axis, and its edges carry units
    em_edges_q = cast(u.Quantity, cast(Axis, axes["Em"]).edges)
    phi_edges_q = cast(u.Quantity, cast(Axis, axes["Phi"]).edges)
    em_edges = _check_edges(em_edges_q.to_value(u.keV), "Em edges")
    phi_edges = _check_edges(phi_edges_q.to_value(u.deg), "Phi edges")
    if phi_edges[0] < 0 or phi_edges[-1] > 180:
        raise ValueError("Phi edges must lie within [0, 180] deg.")

    psichi = cast(HealpixAxis, axes["PsiChi"])
    return em_edges, phi_edges, _check_nside(psichi.nside), _check_scheme(psichi.scheme)


def _check_edges(edges: ArrayLike, name: str) -> np.ndarray:
    """Bin edges: 1D, strictly increasing, at least one bin."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be a 1D, strictly increasing array with at least 2 edges.")
    return edges


def _check_nside(nside: int) -> int:
    """NSIDE must be a power of 2 (sub-pixels are NESTED children)."""
    if not hp.isnsideok(nside, nest=True):
        raise ValueError(f"PsiChi nside must be a power of 2, got {nside}.")
    return int(nside)


def _check_scheme(scheme: str) -> str:
    """Pixel ordering: "nested" or "ring" (case-insensitive)."""
    scheme = str(scheme).lower()
    if scheme not in ("nested", "ring"):
        raise ValueError(f"PsiChi scheme must be 'nested' or 'ring', got {scheme!r}.")
    return scheme


def _check_int(value: int, name: str, minimum: int = 1) -> int:
    """Integer >= minimum."""
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}.")
    return int(value)


def _check_seam_delta(value: float) -> float:
    """chi = 0 strip half-width [deg], in [0, 1); 0 = off."""
    value = float(value)
    if not 0 <= value < 1:
        raise ValueError(f"seam_delta_deg must be in [0, 1) deg, got {value}.")
    return value


def _check_angle(value: float, name: str) -> float:
    """Angle from a pole [deg], in [0, 90); 0 = off."""
    value = float(value)
    if not 0 <= value < 90:
        raise ValueError(f"{name} must be in [0, 90) deg, got {value}.")
    return value


def _check_model_version(version: str) -> None:
    """Only major version 3 (time-independent) models are supported."""
    if int(str(version).split(".")[0]) != 3:
        raise NotImplementedError(
            f"Only major version 3 (time-independent) models are supported, got version {version}.")


def _check_frame(frame: str, pixels: ArrayLike | None) -> None:
    """Frame "local" or "galactic"; a pixel subset only in "local"."""
    if frame not in ("local", "galactic"):
        raise ValueError(f"frame must be 'local' or 'galactic', got {frame!r}.")
    if frame == "galactic" and pixels is not None:
        raise ValueError("The Galactic frame mixes all spacecraft-frame pixels; use pixels=None.")


def _open_history(sc_history: SpacecraftHistory | str | Path,
                  tmin: Time | float | None, tmax: Time | float | None) -> SpacecraftHistory:
    """
    SpacecraftHistory from an orientation file or an existing history, cut to [tmin, tmax] if given.
    """
    history = sc_history if isinstance(sc_history, SpacecraftHistory) else SpacecraftHistory.open(sc_history)   # cosipy's spacecraft_file.py
    if tmin is None and tmax is None:
        return history
    if tmin is None or tmax is None:
        raise ValueError("Give both tmin and tmax, or neither.")
    tmin = tmin if isinstance(tmin, Time) else Time(tmin, format="unix")
    tmax = tmax if isinstance(tmax, Time) else Time(tmax, format="unix")
    return history.select_interval(tmin, tmax)


def _total_livetime(history: SpacecraftHistory) -> float:
    """
    Total live time of the history [s].
    """
    return float(np.sum(cast(np.ndarray, history.livetime.to_value(u.s))))
