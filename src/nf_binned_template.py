"""Binned COSI background templates from an NF background model.

Integrates the density of a time-independent (version 3) normalizing-flow
background model over each Compton-data-space bin, instead of sampling events.

Entry points
------------
NFBinnedTemplate
    Computes the template; ``get_binned_background`` returns expected counts.
galactic_conversion_matrix
    Rotates a spacecraft-frame template into the Galactic frame.

Example (DC4 511 keV imaging binning)
-------------------------------------
    from src.nf_binned_template import NFBinnedTemplate

    template = NFBinnedTemplate("model/nfbackground_part-01_v3-00.fits")
    axes = binned_data.axes        # any histpy Axes with "Em", "Phi", "PsiChi"

    counts = template.get_binned_background(
        axes, "galactic", "data/DC4_orientation.ori",
        tmin=1835487300.036206, tmax=1843467254.887687, em_breaks_kev=[511.0])

    print(template.n_evaluations(axes, [255, 444]))   # cost before running
    rows = template.compute(axes, [255, 444])         # shape (1, 60, 2)
"""

import time
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import torch
import healpy as hp
import astropy.units as u
from astropy.time import Time
from histpy import Histogram, Axes, Axis, HealpixAxis
from scoords import SpacecraftFrame
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.ml.NFModelIO import load_nf_model
from cosipy.background_estimation.ml.NFBackgroundModels import BackgroundDensityECDFSplineDGaussianARQSFlow


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
        Sky-node corrections at the chi = 0 seam and near the poles.
        Defaults validated in notebook 09; normally leave them unchanged.

    Attributes
    ----------
    probability : np.ndarray (n_em_bins, n_phi_bins, npix)
        P_i; NaN where not computed.
    subpixel_level_used : np.ndarray of int (npix,)
        Sub-pixel level used per pixel; -1 where not computed.
    model_counts, model_livetime : float
        Event count and livetime [s] from the model file.
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
        self.batch_size = self._check_int(batch_size, "batch_size")
        self.device = device
        self.seam_delta_rad = np.deg2rad(self._check_seam_delta(seam_delta_deg))   # chi = 0 strip half-width [rad]

        # Polar pixels: finer sky nodes, and finer still close to the pole
        self.polar_cap_rad = np.deg2rad(self._check_angle(polar_cap_deg, "polar_cap_deg"))
        self.pole_colat_rad = np.deg2rad(self._check_angle(pole_colat_deg, "pole_colat_deg"))
        self.polar_subpixel_level = self._check_int(polar_subpixel_level, "polar_subpixel_level", minimum=0)
        self.pole_subpixel_level = self._check_int(pole_subpixel_level, "pole_subpixel_level", minimum=0)

        # Pixels near (not touching) a pole: a minimum sub-pixel level
        self.near_pole_cap_rad = np.deg2rad(self._check_angle(near_pole_cap_deg, "near_pole_cap_deg"))
        self.near_pole_subpixel_level = self._check_int(near_pole_subpixel_level, "near_pole_subpixel_level",
                                                        minimum=0)

        # --- Load the model ---
        # cosipy returns a dict with "version", "density_input" (weights and
        # settings of the flow) and "rate_input" (counts and livetime)
        model_file = load_nf_model(model_path)
        self.model_version = model_file["version"]
        self._check_model_version(self.model_version)

        # Normalization of the model: R = model_counts / model_livetime
        rate_input = model_file["rate_input"]
        self.model_counts = float(rate_input["counts"])
        self.model_livetime = float(rate_input["livetime"])

        # The network that returns p(Em, phi, psi, chi).
        # Arguments: (density_input, device, batch_size, compile_mode)
        self._density_model = BackgroundDensityECDFSplineDGaussianARQSFlow(
            model_file["density_input"], device, self.batch_size, None)

        # --- No binning yet: set by the first compute / get_binned_background call ---
        self._settings = None
        self.probability = None
        self.subpixel_level_used = None

    # -----------------------------------------------------------------------
    # The integral
    # -----------------------------------------------------------------------

    def compute(self,
                axes: Axes,
                pixels: ArrayLike | None = None,
                subpixel_level: int = 2,
                n_em_per_bin: int = 4,
                n_phi_per_bin: int = 4,
                em_breaks_kev: ArrayLike | None = None,
                phi_breaks_deg: ArrayLike | None = None,
                points_per_chunk: int = 2_000_000,
                verbose: bool = False) -> np.ndarray:
        """Integrate the NF density over every (Em, Phi) bin of the given pixels.

        Results are stored in ``probability`` chunk by chunk.

        Parameters
        ----------
        axes : histpy.Axes
            Binning with "Em", "Phi" and "PsiChi"; other axes are ignored.
        pixels : array_like of int, optional
            PsiChi pixel indices; None = all pixels.
        subpixel_level, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg
            Quadrature: 4**subpixel_level sky nodes per pixel, Gauss-Legendre
            nodes per Em / Phi bin, break points [keV] / [deg] where bins split.
        points_per_chunk : int, default 2_000_000
            Density evaluations per chunk (bounds memory).
        verbose : bool, default False
            Print progress.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins, n_pixels)
            P_i (dimensionless) of the given pixels.
        """

        self._set_binning(                  # Em and phi nodes for this binning (reused if unchanged)
            axes,
            n_em_per_bin,
            n_phi_per_bin,
            em_breaks_kev,
            phi_breaks_deg,
            verbose
            )
        pixels = self._check_pixels(pixels)

        k = self._check_int(
            subpixel_level,
            "subpixel_level",
            minimum=0)

        # Level each pixel gets: k, or more near the poles (rule in _levels_used).
        # Polar pixels have their own sky rule, so they are done separately
        levels = self._levels_used(self.nside, pixels, self.nest, k)
        is_polar = self._is_polar(self.nside, pixels, self.nest, self.polar_cap_rad)
        normal, polar = pixels[~is_polar], pixels[is_polar]
        normal_levels, polar_levels = levels[~is_polar], levels[is_polar]

        for level in np.unique(normal_levels):
            group = normal[normal_levels == level]

            # Group these pixels into chunks of about points_per_chunk evaluations
            points_per_pixel = self._evaluations_per_pixel(level)             # n_Em x n_phi x 4**level
            pixels_per_chunk = max(1, points_per_chunk // points_per_pixel)   # pixels per chunk
            n_chunks = -(-len(group) // pixels_per_chunk)                     # ceiling division

            if verbose:
                print(f"{len(group)} pixels, level {level}: {points_per_pixel * len(group):,} evaluations "
                      f"in {n_chunks} chunks of up to {pixels_per_chunk} pixels")

            start_time = time.perf_counter()

            for i, start in enumerate(range(0, len(group), pixels_per_chunk)):
                # Integrate one chunk and store it, so finished chunks survive an interruption
                chunk = group[start:start + pixels_per_chunk]
                self.probability[:, :, chunk] = self._integrate_pixels(chunk, level)   # the integral of this chunk
                self.subpixel_level_used[chunk] = level                                # level actually used

                if verbose:
                    elapsed = time.perf_counter() - start_time
                    done = start + len(chunk)
                    print(f"  chunk {i + 1}/{n_chunks}: {done}/{len(group)} pixels, {elapsed:.0f} s, "
                          f"{done * points_per_pixel / elapsed:,.0f} evaluations/s")

        # Polar pixels, one at a time
        for i, (pixel, level) in enumerate(zip(polar, polar_levels)):
            start_time = time.perf_counter()
            self.probability[:, :, pixel], n_sky = self._integrate_polar_pixel(pixel, int(level), points_per_chunk)
            self.subpixel_level_used[pixel] = level          # at least polar_subpixel_level

            if verbose:
                elapsed = time.perf_counter() - start_time
                n_points = n_sky * len(self.em_nodes_kev) * len(self.phi_nodes_rad)
                print(f"  polar pixel {pixel} ({i + 1}/{len(polar)}): {n_sky:,} sky nodes, {elapsed:.0f} s, "
                      f"{n_points / elapsed:,.0f} evaluations/s")

        return self.probability[:, :, pixels].copy()

    def _evaluations_per_pixel(self, k: int) -> int:
        """Evaluations per non-polar pixel: n_Em nodes x n_phi nodes x 4**k."""
        return len(self.em_nodes_kev) * len(self.phi_nodes_rad) * 4**k

    def _integrate_pixels(self, pixels: np.ndarray, k: int) -> np.ndarray:
        """P_i of non-polar pixels at sub-pixel level k.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins, n_pixels)
        """
        # 1. Sky nodes: 4**k sub-pixel centres per pixel, weight = sub-pixel area [sr]
        colatitude, longitude, sky_weights = self._healpix_subpixel_nodes(
            self.nside, k, pixels, nest=self.nest)                       # each (n_pixels, 4**k)

        # 2. NF density at every (Em node, phi node, sky node)
        density = self._density_on_grid(colatitude.ravel(), longitude.ravel())   # all sky points as one list
        density = density.reshape(len(self.em_nodes_kev), len(self.phi_nodes_rad), len(pixels), 4**k)  # sky -> (pixel, sub-pixel)

        # 3. Weighted sum over the nodes of each bin:
        # P[a, b, c] = sum_e sum_p sum_s  M_E[a, e] * M_phi[b, p] * density[e, p, c, s] * w_sky[c, s]
        # a: Em bin, b: phi bin, c: pixel; e, p, s: Em, phi and sky nodes
        return np.einsum("ae,bp,epcs,cs->abc", self._em_matrix, self._phi_matrix,
                         density, sky_weights, optimize=True)

    def _integrate_polar_pixel(self, pixel: int, level: int, points_per_chunk: int) -> tuple[np.ndarray, int]:
        """P_i of one polar pixel, with refined sky nodes near the pole.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins)
            P_i of the pixel.
        int
            Number of sky nodes used.
        """
        # 1. Sky nodes: the pixel's level; those close to the pole at the pole level
        colatitude, longitude, sky_weights = self._polar_pixel_nodes(self.nside, pixel, self.nest, level)

        # 2.-3. Density and weighted sum as in _integrate_pixels, in pieces of about points_per_chunk evaluations
        n_sky = max(1, points_per_chunk // (len(self.em_nodes_kev) * len(self.phi_nodes_rad)))   # sky nodes per piece
        total = np.zeros((self.n_em_bins, self.n_phi_bins))
        for a in range(0, len(sky_weights), n_sky):
            density = self._density_on_grid(colatitude[a:a + n_sky], longitude[a:a + n_sky])   # (Em, phi, sky)
            total += np.einsum("ae,bp,eps,s->ab", self._em_matrix, self._phi_matrix,
                               density, sky_weights[a:a + n_sky], optimize=True)
        return total, len(sky_weights)

    def _density_on_grid(self, colatitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """NF density on (Em nodes) x (phi nodes) x (sky points).

        Parameters
        ----------
        colatitude, longitude : np.ndarray (n_sky,)
            Paired sky points psi, chi [rad], spacecraft frame.

        Returns
        -------
        np.ndarray (n_em_nodes, n_phi_nodes, n_sky)
            Density [keV^-1 rad^-1 sr^-1], float64.
        """
        shape = (len(self.em_nodes_kev), len(self.phi_nodes_rad), len(colatitude))

        # chi = 0 seam: the flow's density is wrong in a thin strip there, so nodes are moved just outside it
        longitude = np.clip(longitude, self.seam_delta_rad, 2 * np.pi - self.seam_delta_rad)

        # Every Em node x every phi node x every sky point
        grid = np.empty((4,) + shape)                 # 4 layers: Em, phi, chi, psi
        grid[0] = self.em_nodes_kev[:, None, None]    # Em varies along axis 0
        grid[1] = self.phi_nodes_rad[None, :, None]   # phi varies along axis 1
        grid[2] = longitude[None, None, :]            # chi and psi vary together along axis 2:
        grid[3] = colatitude[None, None, :]           # each sky point is one (psi, chi) pair
        em, phi, chi, psi = torch.from_numpy(grid.reshape(4, -1))   # 4 flat lists

        # Version 3 models ignore the context (time); it only sets the number of points
        context = torch.zeros(em.shape[0], dtype=torch.float32)
        density = self._density_model.evaluate_density(context, em, phi, chi, psi)   # float32
        return density.numpy().astype(np.float64).reshape(shape)                     # back to (n_Em, n_phi, n_sky)

    # -----------------------------------------------------------------------
    # Em and phi nodes
    # -----------------------------------------------------------------------

    def _set_binning(self, axes: Axes, n_em_per_bin: int, n_phi_per_bin: int,
                     em_breaks_kev: ArrayLike | None, phi_breaks_deg: ArrayLike | None,
                     verbose: bool = False) -> None:
        """Build the Em and phi nodes for this binning; reset results if it changed."""
        em_edges_kev, phi_edges_deg, nside, scheme = self._check_axes(axes)
        n_em_per_bin = self._check_int(n_em_per_bin, "n_em_per_bin")
        n_phi_per_bin = self._check_int(n_phi_per_bin, "n_phi_per_bin")
        em_breaks_kev = self._as_breaks(em_breaks_kev)
        phi_breaks_deg = self._as_breaks(phi_breaks_deg)

        # Same binning and settings as the last call: reuse nodes and results.
        # Template settings are included, so changing one also starts over
        settings = (tuple(em_edges_kev), tuple(phi_edges_deg), nside, scheme,
                    n_em_per_bin, n_phi_per_bin, tuple(em_breaks_kev), tuple(phi_breaks_deg),
                    self.seam_delta_rad,                                        # chi = 0 strip
                    self.polar_cap_rad, self.polar_subpixel_level,              # polar pixels
                    self.pole_colat_rad, self.pole_subpixel_level,              # ... and their refinement at the pole
                    self.near_pole_cap_rad, self.near_pole_subpixel_level)      # pixels near a pole
        if settings == self._settings:
            return

        # Something changed: stored results no longer apply. Say so if computed pixels are lost
        if verbose and self.subpixel_level_used is not None:
            n_lost = int((self.subpixel_level_used >= 0).sum())
            if n_lost:
                print(f"Settings changed: results of {n_lost} computed pixel(s) discarded.")

        # New binning: store it ...
        self.em_edges_kev, self.phi_edges_deg = em_edges_kev, phi_edges_deg
        self.nside, self.scheme, self.nest = nside, scheme, scheme == "nested"
        self.npix = hp.nside2npix(nside)
        self.n_em_bins, self.n_phi_bins = len(em_edges_kev) - 1, len(phi_edges_deg) - 1

        # ... build the Em and phi nodes (the same for every pixel) ...
        self.em_nodes_kev, em_weights, em_bin_index = self._binned_gl_nodes(         # Em nodes, weights [keV],
            em_edges_kev, n_em_per_bin, em_breaks_kev)                               # and bin of each node
        self.phi_nodes_rad, phi_weights, phi_bin_index = self._binned_gl_nodes(      # same for phi, in rad
            np.deg2rad(phi_edges_deg), n_phi_per_bin, np.deg2rad(phi_breaks_deg))

        # ... and the matrices that sum weighted node values into bins
        self._em_matrix = self._node_to_bin_matrix(em_weights, em_bin_index, self.n_em_bins)       # (Em bins, Em nodes)
        self._phi_matrix = self._node_to_bin_matrix(phi_weights, phi_bin_index, self.n_phi_bins)   # (phi bins, phi nodes)

        # Start over: nothing computed for this binning yet
        self.probability = np.full((self.n_em_bins, self.n_phi_bins, self.npix), np.nan)
        self.subpixel_level_used = np.full(self.npix, -1, dtype=np.int64)
        self._settings = settings

    @staticmethod
    def _binned_gl_nodes(edges: ArrayLike, n_per_bin: int,
                         breaks: ArrayLike = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gauss-Legendre nodes for every bin; bins are split at break points.

        Returns
        -------
        nodes, weights : np.ndarray (n_nodes,)
            Node positions and weights, in the unit of ``edges``.
        bin_index : np.ndarray of int (n_nodes,)
            Bin of each node.
        """
        edges = np.asarray(edges, dtype=np.float64)
        breaks = np.unique(np.asarray(breaks, dtype=np.float64))

        nodes, weights, bin_index = [], [], []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            # Split the bin at the break points inside it, then GL nodes in each piece
            inside = breaks[(breaks > lo) & (breaks < hi)]
            pieces = np.concatenate([[lo], inside, [hi]])
            for a, b in zip(pieces[:-1], pieces[1:]):
                x, w = NFBinnedTemplate._gl_nodes(a, b, n_per_bin)
                nodes.append(x)
                weights.append(w)
                bin_index.append(np.full(n_per_bin, i))

        return np.concatenate(nodes), np.concatenate(weights), np.concatenate(bin_index)

    @staticmethod
    def _gl_nodes(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Legendre nodes and weights on [a, b]; weights sum to b - a."""
        x, w = np.polynomial.legendre.leggauss(n)     # standard nodes and weights on [-1, 1]
        # Map [-1, 1] linearly onto [a, b]
        return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

    @staticmethod
    def _node_to_bin_matrix(weights: np.ndarray, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
        """Matrix (n_bins, n_nodes) that sums weighted node values into bins."""
        matrix = np.zeros((n_bins, len(weights)))
        matrix[bin_index, np.arange(len(weights))] = weights    # row = bin, column = node, value = its weight
        return matrix

    @staticmethod
    def _as_breaks(breaks: ArrayLike | None) -> np.ndarray:
        """Break points as a 1D float array (None = empty)."""
        return np.atleast_1d(np.asarray([] if breaks is None else breaks, dtype=np.float64))

    # -----------------------------------------------------------------------
    # Sky nodes
    # -----------------------------------------------------------------------

    def _levels_used(self, nside: int, pixels: np.ndarray, nest: bool, k: int) -> np.ndarray:
        """Sub-pixel level each pixel gets when level k is requested.

        k in general, at least ``near_pole_subpixel_level`` near a pole, and at
        least ``polar_subpixel_level`` in polar pixels.
        """
        near = self._is_polar(nside, pixels, nest, self.near_pole_cap_rad)     # within near_pole_cap_deg of a pole
        polar = self._is_polar(nside, pixels, nest, self.polar_cap_rad)        # the polar pixels
        levels = np.where(near, max(k, self.near_pole_subpixel_level), k)
        return np.where(polar, max(k, self.polar_subpixel_level), levels)

    @staticmethod
    def _is_polar(nside: int, pixels: ArrayLike, nest: bool, polar_cap_rad: float) -> np.ndarray:
        """Boolean mask: pixel centre within ``polar_cap_rad`` of a pole."""
        colatitude, _ = hp.pix2ang(nside, pixels, nest=nest)
        return np.minimum(colatitude, np.pi - colatitude) < polar_cap_rad   # distance to the nearer pole

    @staticmethod
    def _healpix_subpixel_nodes(nside: int, k: int, pixels: ArrayLike,
                                nest: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sky nodes: centres of the 4**k sub-pixels of each pixel.

        Returns
        -------
        colatitude, longitude : np.ndarray (n_pixels, 4**k)
            Sub-pixel centres [rad].
        weights : np.ndarray (n_pixels, 4**k)
            Sub-pixel areas [sr]; they sum to the pixel area.
        """
        pixels = np.atleast_1d(np.asarray(pixels, dtype=np.int64))
        parents_nest = pixels if nest else hp.ring2nest(nside, pixels)         # sub-pixels are defined in NESTED numbering

        n_sub = 4**k                                                           # sub-pixels per pixel
        nside_sub = nside * 2**k                                               # finer grid of the sub-pixels
        children = parents_nest[:, None] * n_sub + np.arange(n_sub)[None, :]   # sub-pixel indices, (n_pixels, n_sub)

        colatitude, longitude = hp.pix2ang(nside_sub, children.ravel(), nest=True)   # centre of each sub-pixel
        weights = np.full(children.shape, hp.nside2pixarea(nside_sub))         # all equal in HEALPix
        return colatitude.reshape(children.shape), longitude.reshape(children.shape), weights

    def _polar_pixel_nodes(self, nside: int, pixel: int, nest: bool,
                           level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sky nodes of one polar pixel, with this template's pole settings."""
        return self._polar_refined_nodes(nside, pixel, nest, level,
                                         max(level, self.pole_subpixel_level), self.pole_colat_rad)

    @staticmethod
    def _polar_refined_nodes(nside: int, pixel: int, nest: bool, k: int, k_pole: int,
                             pole_colat_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sky nodes of one pixel at level k; sub-pixels near the pole at k_pole.

        Returns
        -------
        colatitude, longitude, weights : np.ndarray (n_sky,)
            Node positions [rad] and weights [sr]; weights sum to the pixel area.
        """
        parent_nest = pixel if nest else hp.ring2nest(nside, pixel)
        children = parent_nest * 4**k + np.arange(4**k)                         # NESTED children at nside * 2**k
        colat_c, _ = hp.pix2ang(nside * 2**k, children, nest=True)
        near = np.minimum(colat_c, np.pi - colat_c) < pole_colat_rad            # sub-pixels close to the pole

        m = 4**(k_pole - k)                                                     # children of each of them at k_pole
        fine = (children[near][:, None] * m + np.arange(m)).ravel()
        colat_a, lon_a = hp.pix2ang(nside * 2**k, children[~near], nest=True)
        colat_b, lon_b = hp.pix2ang(nside * 2**k_pole, fine, nest=True)
        weights = np.concatenate([np.full(len(colat_a), hp.nside2pixarea(nside * 2**k)),
                                  np.full(len(colat_b), hp.nside2pixarea(nside * 2**k_pole))])
        return np.concatenate([colat_a, colat_b]), np.concatenate([lon_a, lon_b]), weights

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    def get_binned_background(
            self,
            axes: Axes,
            frame: str,
            ori_file: str | Path,
            tmin: float,
            tmax: float,
            pixels: ArrayLike | None = None,
            subpixel_level: int = 2,
            n_em_per_bin: int = 4,
            n_phi_per_bin: int = 4,
            em_breaks_kev: ArrayLike | None = None,
            phi_breaks_deg: ArrayLike | None = None,
            scatt_nside: int = 32
            ) -> Histogram:
        """Expected background counts in [tmin, tmax], binned in Em x Phi x PsiChi.

        N_i = R * T * P_i, with R = model_counts / model_livetime [1/s] and T the
        live time [s] in the window. Missing pixels are computed first.

        Parameters
        ----------
        axes, pixels, subpixel_level, n_em_per_bin, n_phi_per_bin,
        em_breaks_kev, phi_breaks_deg
            As in ``compute``. ``pixels`` only for "local" (others are NaN).
        frame : str
            "local" (spacecraft frame) or "galactic".
        ori_file : str or Path
            Orientation file with attitude and live time.
        tmin, tmax : float
            Time window, unix time [s].
        scatt_nside : int, default 32
            Attitude grouping for the Galactic rotation.

        Returns
        -------
        histpy.Histogram (n_em_bins, n_phi_bins, npix)
            Expected counts.
        """
        self._check_frame(frame, pixels)
        self._set_binning(axes, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg)
        k = self._check_int(subpixel_level, "subpixel_level", minimum=0)

        # 1. Integrate the requested pixels not yet computed at level k
        missing = self._missing_pixels(pixels, k)
        if missing.size:
            self.compute(axes, missing, k, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg)

        # 2. Model rate, and attitude / live time in the window
        history = SpacecraftHistory.open(ori_file).select_interval(
            Time(tmin, format="unix"), Time(tmax, format="unix"))
        rate = self.model_counts / self.model_livetime          # events / s

        # 3a. Local frame: counts = rate * live time * P
        if frame == "local":
            livetime = float(history.livetime.to_value(u.s).sum())          # total live time in [tmin, tmax] [s]

            # A pixel stored at another sub-pixel level (e.g. from a call with a
            # different k) is NaN in the result, like one never computed
            level_now = self._levels_used(self.nside, np.arange(self.npix), self.nest, k)   # level each pixel gets now
            probability = np.where(self.subpixel_level_used == level_now, self.probability, np.nan)
            return Histogram(self._axes(SpacecraftFrame()), contents=rate * livetime * probability)

        # 3b. Galactic frame: rotate with W (live time each Galactic pixel spends in each local pixel)
        W = galactic_conversion_matrix(history, self.nside, self.nest, scatt_nside)   # (Galactic pix, local pix) [s]
        counts = rate * (self.probability @ W.T)
        return Histogram(self._axes("galactic"), contents=counts)

    def _missing_pixels(self, pixels: ArrayLike | None, k: int) -> np.ndarray:
        """Requested pixels (None = all) not yet computed at their level."""
        pixels = self._check_pixels(pixels)
        return pixels[self.subpixel_level_used[pixels] != self._levels_used(self.nside, pixels, self.nest, k)]

    def _axes(self, coordsys: SpacecraftFrame | str) -> Axes:
        """Em, Phi, PsiChi axes of the template, PsiChi in the given frame."""
        return Axes([Axis(self.em_edges_kev, unit=u.keV, label="Em"),
                     Axis(self.phi_edges_deg, unit=u.deg, label="Phi"),
                     HealpixAxis(nside=self.nside, scheme=self.scheme,
                                 coordsys=coordsys, label="PsiChi")])

    def expected_counts(self) -> np.ndarray:
        """Expected counts over the model livetime: model_counts * P_i.

        Returns
        -------
        np.ndarray (n_em_bins, n_phi_bins, npix)
            Counts. NaN = not computed.
        """
        self._check_binning_set()
        return self.model_counts * self.probability

    def to_histogram(self) -> Histogram:
        """Template P_i as a histpy Histogram, spacecraft frame (all pixels needed).

        Returns
        -------
        histpy.Histogram (n_em_bins, n_phi_bins, npix)
            P_i (dimensionless). Axes "Em" [keV], "Phi" [deg], "PsiChi".
        """
        self._check_binning_set()
        self._check_all_computed()
        return Histogram(self._axes(SpacecraftFrame()), contents=self.probability)

    def n_evaluations(self,
                      axes: Axes,
                      pixels: ArrayLike | None = None,
                      subpixel_level: int = 2,
                      n_em_per_bin: int = 4,
                      n_phi_per_bin: int = 4,
                      em_breaks_kev: ArrayLike | None = None,
                      phi_breaks_deg: ArrayLike | None = None) -> int:
        """Number of density evaluations a ``compute`` call would need.

        Arguments as in ``compute``. Divide by the throughput (~7,000
        evaluations/s on CPU) to estimate the run time.

        Returns
        -------
        int
            Number of density evaluations.
        """
        em_edges_kev, phi_edges_deg, nside, _ = self._check_axes(axes)
        n_em_nodes = len(self._binned_gl_nodes(em_edges_kev, self._check_int(n_em_per_bin, "n_em_per_bin"),
                                               self._as_breaks(em_breaks_kev))[0])
        n_phi_nodes = len(self._binned_gl_nodes(phi_edges_deg, self._check_int(n_phi_per_bin, "n_phi_per_bin"),
                                                self._as_breaks(phi_breaks_deg))[0])
        k = self._check_int(subpixel_level, "subpixel_level", minimum=0)
        nest = self._check_scheme(axes["PsiChi"].scheme) == "nested"
        pixels = np.arange(hp.nside2npix(nside)) if pixels is None else np.atleast_1d(np.asarray(pixels, dtype=np.int64))

        # Sky nodes: 4**level per normal pixel, plus the refined nodes of each polar pixel
        levels = self._levels_used(nside, pixels, nest, k)
        is_polar = self._is_polar(nside, pixels, nest, self.polar_cap_rad)
        n_sky = int((4**levels[~is_polar]).sum()) + sum(
            len(self._polar_pixel_nodes(nside, p, nest, int(level))[2])
            for p, level in zip(pixels[is_polar], levels[is_polar]))
        return n_em_nodes * n_phi_nodes * n_sky

    # -----------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_axes(axes: Axes) -> tuple[np.ndarray, np.ndarray, int, str]:
        """Em edges [keV], Phi edges [deg], PsiChi NSIDE and scheme from histpy Axes."""
        labels = list(axes.labels)
        for label in ("Em", "Phi", "PsiChi"):
            if label not in labels:
                raise ValueError(f"axes must contain 'Em', 'Phi' and 'PsiChi', got {labels}.")

        em_edges = NFBinnedTemplate._check_edges(axes["Em"].edges.to_value(u.keV), "Em edges")
        phi_edges = NFBinnedTemplate._check_edges(axes["Phi"].edges.to_value(u.deg), "Phi edges")
        if phi_edges[0] < 0 or phi_edges[-1] > 180:
            raise ValueError("Phi edges must lie within [0, 180] deg.")

        psichi = axes["PsiChi"]
        return em_edges, phi_edges, NFBinnedTemplate._check_nside(psichi.nside), \
            NFBinnedTemplate._check_scheme(psichi.scheme)

    @staticmethod
    def _check_edges(edges: ArrayLike, name: str) -> np.ndarray:
        """Bin edges: 1D, strictly increasing, at least one bin."""
        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError(f"{name} must be a 1D, strictly increasing array with at least 2 edges.")
        return edges

    @staticmethod
    def _check_nside(nside: int) -> int:
        """NSIDE must be a power of 2 (sub-pixels are NESTED children)."""
        if not hp.isnsideok(nside, nest=True):
            raise ValueError(f"PsiChi nside must be a power of 2, got {nside}.")
        return int(nside)

    @staticmethod
    def _check_scheme(scheme: str) -> str:
        """Pixel ordering: "nested" or "ring" (case-insensitive)."""
        scheme = str(scheme).lower()
        if scheme not in ("nested", "ring"):
            raise ValueError(f"PsiChi scheme must be 'nested' or 'ring', got {scheme!r}.")
        return scheme

    @staticmethod
    def _check_int(value: int, name: str, minimum: int = 1) -> int:
        """Integer >= minimum."""
        if int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value}.")
        return int(value)

    @staticmethod
    def _check_seam_delta(value: float) -> float:
        """chi = 0 strip half-width [deg], in [0, 1); 0 = off."""
        value = float(value)
        if not 0 <= value < 1:
            raise ValueError(f"seam_delta_deg must be in [0, 1) deg, got {value}.")
        return value

    @staticmethod
    def _check_angle(value: float, name: str) -> float:
        """Angle from a pole [deg], in [0, 90); 0 = off."""
        value = float(value)
        if not 0 <= value < 90:
            raise ValueError(f"{name} must be in [0, 90) deg, got {value}.")
        return value

    @staticmethod
    def _check_model_version(version: str) -> None:
        """Only major version 3 (time-independent) models are supported."""
        if int(str(version).split(".")[0]) != 3:
            raise NotImplementedError(
                f"Only major version 3 (time-independent) models are supported, "
                f"got version {version}.")

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
        missing = np.isnan(self.probability).any(axis=(0, 1))
        if missing.any():
            raise RuntimeError(f"{missing.sum()} of {self.npix} pixels are not computed yet.")

    @staticmethod
    def _check_frame(frame: str, pixels: ArrayLike | None) -> None:
        """Frame "local" or "galactic"; a pixel subset only in "local"."""
        if frame not in ("local", "galactic"):
            raise ValueError(f"frame must be 'local' or 'galactic', got {frame!r}.")
        if frame == "galactic" and pixels is not None:
            raise ValueError("The Galactic frame mixes all spacecraft-frame pixels; use pixels=None.")


# ---------------------------------------------------------------------------
# Spacecraft frame -> Galactic frame (separate tool, also usable without a template)
# ---------------------------------------------------------------------------

def galactic_conversion_matrix(
        history: SpacecraftHistory,
        nside: int,
        nest: bool,
        scatt_nside: int = 32,
        attitudes_per_chunk: int = 512
        ) -> np.ndarray:
    """Live time each Galactic pixel spends in each spacecraft-frame pixel.

    For a spacecraft-frame rate r [1/s], the Galactic counts are ``W @ r``.
    Same nearest-pixel method as cosipy's Galactic responses.

    Parameters
    ----------
    history : SpacecraftHistory
        Attitude and live time, already cut to the time window.
    nside, nest : int, bool
        HEALPix NSIDE and ordering (True = NESTED) of both pixelizations.
    scatt_nside : int, default 32
        Resolution of the attitude grouping.
    attitudes_per_chunk : int, default 512
        Attitudes processed at once (bounds memory).

    Returns
    -------
    np.ndarray (npix, npix)
        W [s]; each row sums to the total live time.
    """
    # Attitudes grouped on a HEALPix grid, each with its live time
    scatt = history.get_scatt_map(nside=scatt_nside, target_coord=None, earth_occ=False)
    livetime = np.asarray(scatt.weights.to_value(u.s), dtype=np.float64)
    rotation = scatt.attitudes.rot.inv().as_matrix()            # (n_att, 3, 3): Galactic -> spacecraft frame

    npix = hp.nside2npix(nside)
    galactic_pix = np.arange(npix)
    galactic_dirs = np.array(hp.pix2vec(nside, galactic_pix, nest=nest))   # (3, npix), Galactic Cartesian

    # For each attitude: rotate every Galactic pixel centre into the spacecraft frame,
    # find the local pixel it lands in, and add the attitude's live time to W[g, l]
    W = np.zeros(npix * npix)
    for a in range(0, len(rotation), attitudes_per_chunk):
        local_dirs = np.einsum("cij,jp->cip", rotation[a:a + attitudes_per_chunk], galactic_dirs)
        local_pix = hp.vec2pix(nside, local_dirs[:, 0], local_dirs[:, 1], local_dirs[:, 2], nest=nest)
        W += np.bincount((galactic_pix * npix + local_pix).ravel(),
                         weights=np.repeat(livetime[a:a + attitudes_per_chunk], npix),
                         minlength=npix * npix)
    return W.reshape(npix, npix)
