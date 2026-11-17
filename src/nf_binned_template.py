"""Binned background templates from a normalizing-flow (NF) background model.

Method
------
Pascal Janowski's NF background models give a probability density
p(Em, phi, psi, chi) over the Compton data space (CDS). Instead of sampling
events from the flow and histogramming them, this module integrates the
density over every CDS bin B_i:

    P_i = integral over B_i of p(Em, phi, psi, chi) dEm dphi dOmega

The integral is approximated by a weighted sum over quadrature nodes:

    P_i ~ sum_e sum_p sum_s  wE_e * wphi_p * wOmega_s * p(Em_e, phi_p, psi_s, chi_s)

where e runs over the Em nodes inside the energy bin, p over the phi nodes
inside the phi bin, and s over the sky nodes inside the HEALPix pixel. The
expected counts per bin are

    N_i = R * T * P_i,

with R * T the number of events behind the model (``model_counts``).

Quadrature rules
----------------
- Em and phi: Gauss-Legendre (GL) nodes inside each bin. A bin can be split at
  break points (e.g. at 511 keV, where the density peaks), with the same number
  of GL nodes in each piece.
- Sky: each HEALPix pixel at NSIDE is split into its 4**k NESTED children at
  NSIDE * 2**k. The nodes are the child centers and each weight is the child
  area (a midpoint rule on the sphere). All sky nodes come from
  ``NFBinnedTemplate._healpix_subpixel_nodes``; a different sky rule can replace
  it as long as it returns the same (colatitude, longitude, weight) arrays.
- Sky, polar pixels: the flow's density has a tall, narrow peak at the poles
  (psi = 0 and 180 deg), so pixels whose centre is within ``polar_cap_deg`` of a
  pole use level ``polar_subpixel_level`` (at least k), and their sub-pixels with
  centre within ``pole_colat_deg`` of the pole are split further to
  ``pole_subpixel_level`` (``_polar_refined_nodes``). Defaults (4 deg, 5, 1 deg,
  7) select the 8 pixels touching the poles at NSIDE 16 and agree with uniform
  level 7 within 0.3% (notebook 09, Section 10). The other pixels with centre
  within ``near_pole_cap_deg`` (10 deg: the next two rings at NSIDE 16) use at
  least ``near_pole_subpixel_level`` (3); at level 2 they are up to 6% low
  (notebook 09, Section 11.1).

Units and conventions
---------------------
- The density is in keV^-1 rad^-1 sr^-1 (the Jacobian of the flow's internal
  coordinates is applied inside cosipy), so P_i is dimensionless.
- Em in keV, phi in rad internally (bin edges are given in degrees).
- Sky coordinates are in the spacecraft frame, with COSIpy names:
  psi = colatitude ("Psi local"), chi = longitude ("Chi local"), both in rad.
  The NF source code uses swapped names; this module always uses COSIpy names.
- Solid angle: dOmega = dcos(psi) dchi.
- chi = 0 seam: the flow works in u = chi / 2pi in [0, 1], so chi = 0 is the
  edge of its domain, and its density is wrong in a thin strip there
  (~0.05 deg on the chi < 0 side, ~0.1-0.15 deg on the chi > 0 side).
  HEALPix sub-pixel centres lie exactly on chi = 0, so sky nodes with
  chi < delta are evaluated at chi = delta, and those with chi > 2pi - delta
  at 2pi - delta (``seam_delta_deg``, default 0.2 deg; 0 turns it off).
  Results do not depend on delta for 0.1-0.3 deg (notebook 09, Section 9).
- The density is evaluated without truncation to the flow's sampling domain,
  so the P_i are not renormalized by the domain mass Z <= 1. The density is
  returned by cosipy as float32; all sums here are accumulated in float64.

Only major version 3 models (unconditional, time-independent) are supported.

Binned background
-----------------
``get_binned_background`` returns the expected counts in a time window,
N_i = R * (live time in the window) * P_i with R = model_counts / model_livetime,
as a histpy Histogram with axes Em, Phi, PsiChi:
- frame "local": PsiChi in the spacecraft frame (the frame of the NF density);
- frame "galactic": the spacecraft-frame template is rotated into Galactic
  coordinates with the attitude history, weighted by live time (same
  nearest-pixel method cosipy uses for its Galactic responses, see
  ``galactic_conversion_matrix``).
The binning is given as histpy Axes with labels "Em", "Phi" and "PsiChi",
e.g. the axes of cosipy's BinnedData or of a response.

Layout of this file
-------------------
- ``NFBinnedTemplate``: created from the model file only. The binning and the
  quadrature settings are arguments of ``compute`` / ``get_binned_background``.
  Order in the class: the integral (``compute`` -> ``_integrate_pixels``), its
  quadrature nodes, the outputs, and at the end the input checks.
- ``galactic_conversion_matrix``: spacecraft frame -> Galactic frame; a separate
  tool, also usable without a template (e.g. to rotate binned samples).

Usage (DC4 511 keV imaging binning)
-----------------------------------
    from src.nf_binned_template import NFBinnedTemplate

    template = NFBinnedTemplate("model/nfbackground_part-01_v3-00.fits")
    axes = binned_data.axes        # any histpy Axes with "Em", "Phi", "PsiChi" (other axes are ignored)

    background = template.get_binned_background(
        axes, "galactic", "data/DC4_orientation.ori",
        tmin=1835487300.036206, tmax=1843467254.887687,
        subpixel_level=2, n_em_per_bin=4, n_phi_per_bin=4, em_breaks_kev=[511.0])

    # Local frame on a few pixels only (the others are NaN):
    local = template.get_binned_background(axes, "local", "data/DC4_orientation.ori",
                                           1835487300.036206, 1843467254.887687,
                                           pixels=[255, 444], em_breaks_kev=[511.0])

    # Step by step (pixel indices in the scheme of the PsiChi axis):
    print(template.n_evaluations(axes, [255, 444]))   # cost before running
    rows = template.compute(axes, [255, 444])         # shape (1, 60, 2)
"""

import time
import numpy as np
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
    """Binned template P_i of an NF background model, by integration of its density.

    Created from the model file only. The binning (histpy Axes with "Em",
    "Phi", "PsiChi") and the quadrature settings are given to ``compute`` and
    ``get_binned_background``. The last results are kept in ``probability``
    (NaN for pixels not computed yet) while the binning, the quadrature settings
    and the template settings below (chi = 0 strip, pole corrections) stay the
    same; a call with different ones starts over and says how many computed
    pixels it discards. ``get_binned_background`` computes only the pixels still
    missing; ``compute`` always computes the pixels it is given.

    Parameters
    ----------
    model_path : str or Path
        NF background model file (.fits), major version 3.
    batch_size : int
        Number of points per forward pass of the flow. It changes the speed and the
        memory, not the nodes or the weights. Measured on the MacBook CPU (4 threads,
        model part 01): 4096 gives ~7,500 evaluations/s with ~1.6 GB of RAM;
        100,000 gives 3,300-5,000 evaluations/s with ~11 GB. From 2048 to 100,000
        the densities are bit-identical; below 2048 they change slightly (float32
        rounding: ~1e-5 on the integral of a pixel), so keep it >= 2048.
    device : str
        Torch device for the flow, e.g. "cpu" or "cuda".
    seam_delta_deg : float
        Half-width [deg] of the strip at chi = 0 where the flow's density is
        wrong: sky nodes with chi < delta (or > 360 deg - delta) are evaluated
        at chi = delta (or 360 deg - delta). 0 turns the correction off.
    polar_cap_deg : float
        Pixels whose centre is within this angle [deg] of a pole are polar
        pixels and get the refined sky rule below. 0 turns the refinement off.
    polar_subpixel_level : int
        Sub-pixel level of polar pixels (the larger of this and ``subpixel_level``).
    pole_colat_deg : float
        Sub-pixels of a polar pixel with centre within this angle [deg] of the
        pole are split further.
    pole_subpixel_level : int
        Level of those sub-pixels (at least the level of the polar pixel).
    near_pole_cap_deg : float
        Other pixels whose centre is within this angle [deg] of a pole use at
        least ``near_pole_subpixel_level``. 0 turns this off.
    near_pole_subpixel_level : int
        Minimum sub-pixel level of those pixels.

    Attributes
    ----------
    model_path, batch_size, device
        As the parameters. Fixed when the template is created (the flow is built
        with them); changing them later has no effect.
    model_version : str
        Version of the model file.
    model_counts : float
        R * T, the number of events behind the model.
    model_livetime : float
        Livetime of the model [s].
    seam_delta_rad : float
        Half-width of the chi = 0 strip [rad] (from ``seam_delta_deg``).
    polar_cap_rad, pole_colat_rad : float
        ``polar_cap_deg`` and ``pole_colat_deg`` in rad.
    near_pole_cap_rad : float
        ``near_pole_cap_deg`` in rad.
    polar_subpixel_level, pole_subpixel_level, near_pole_subpixel_level : int
        As the parameters.
    probability : np.ndarray, shape (n_em_bins, n_phi_bins, npix), or None
        P_i of the current binning, float64; NaN where not computed; None before
        the first call.
    subpixel_level_used : np.ndarray of int, shape (npix,), or None
        Sub-pixel level really used for each pixel (above the requested one near the
        poles; for polar pixels, the level before the extra refinement at the pole);
        -1 where not computed.
    """

    def __init__(self, 
                 model_path, 
                 batch_size=4096,
                 device="cpu",
                 seam_delta_deg=0.2,
                 polar_cap_deg=4.0,
                 polar_subpixel_level=5,
                 pole_colat_deg=1.0,
                 pole_subpixel_level=7,
                 near_pole_cap_deg=10.0,
                 near_pole_subpixel_level=3):

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

        # --- Load the model once, in this process ---
        # cosipy's reader returns the file as a dict with "version", "density_input" (the weights and
        # settings of the flow) and "rate_input" (counts and livetime)
        model_file = load_nf_model(model_path)
        self.model_version = model_file["version"]
        self._check_model_version(self.model_version)

        # Normalization of the model: R = model_counts / model_livetime
        rate_input = model_file["rate_input"]
        self.model_counts = float(rate_input["counts"])
        self.model_livetime = float(rate_input["livetime"])

        # The density network. Arguments: (density_input, device, batch_size, compile_mode).
        self._density_model = BackgroundDensityECDFSplineDGaussianARQSFlow(       # the network that returns
            model_file["density_input"], device, self.batch_size, None)           # p(Em, phi, psi, chi)

        # --- No binning yet: set by the first compute / get_binned_background call ---
        self._settings = None
        self.probability = None
        self.subpixel_level_used = None

    # -----------------------------------------------------------------------
    # The integral
    # -----------------------------------------------------------------------

    def compute(self, 
                axes, 
                pixels=None, 
                subpixel_level=2, 
                n_em_per_bin=4, 
                n_phi_per_bin=4,
                em_breaks_kev=None, 
                phi_breaks_deg=None, 
                points_per_chunk=2_000_000, 
                verbose=True):
        """Integrate the NF density over every (Em, phi) bin of the given pixels.

        Pixels are processed in chunks of about ``points_per_chunk`` density
        evaluations to bound memory (about 50 bytes per evaluation, i.e.
        ~100 MB for the default). Results are written into ``probability``
        chunk by chunk, so an interrupted run keeps the chunks already done.
        The given pixels are always computed, also if they were computed before
        (``get_binned_background`` computes only the missing ones).

        Parameters
        ----------
        axes : histpy.Axes
            Binning, with axes "Em" [energy], "Phi" [angle] and "PsiChi" (HealpixAxis).
            Other axes (e.g. "Time") are ignored.
        pixels : array_like of int or None
            Pixel indices (in the scheme of PsiChi); None means all pixels.
        subpixel_level : int
            Sub-pixel level k: 4**k sky nodes per pixel. Polar pixels use the
            polar rule instead (see ``polar_cap_deg``), and pixels near a pole
            at least ``near_pole_subpixel_level``.
        n_em_per_bin, n_phi_per_bin : int
            GL nodes per Em / phi bin (per piece if the bin is split at a break).
        em_breaks_kev, phi_breaks_deg : array_like or None
            Points at which Em [keV] / phi [deg] bins are split (e.g. em_breaks_kev=[511.0]).
        points_per_chunk : int
            Target number of density evaluations per chunk (at least one pixel).
        verbose : bool
            Print one line per chunk with progress and throughput, and a notice
            if a change of settings discards computed pixels.

        Returns
        -------
        np.ndarray, shape (n_em_bins, n_phi_bins, n_pixels)
            P_i for the given pixels, in the order given.
        """

        self._set_binning(                  # Em and phi points for this binning (reused if unchanged)
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

        # Level each pixel really gets: k, or more near the poles (the rule is in _levels_used).
        # Polar pixels also get their own sky rule (see the module notes), so they are done separately
        levels = self._levels_used(self.nside, pixels, self.nest, k)
        is_polar = self._is_polar(self.nside, pixels, self.nest, self.polar_cap_rad)
        normal, polar = pixels[~is_polar], pixels[is_polar]
        normal_levels, polar_levels = levels[~is_polar], levels[is_polar]

        for level in np.unique(normal_levels):
            group = normal[normal_levels == level]

            # Group these pixels into chunks of about points_per_chunk density evaluations
            points_per_pixel = self._evaluations_per_pixel(level)             # n_Em points x n_phi points x 4**level
            pixels_per_chunk = max(1, points_per_chunk // points_per_pixel)   # how many pixels fit in one chunk
            n_chunks = -(-len(group) // pixels_per_chunk)    # ceiling division

            if verbose:
                print(f"{len(group)} pixels, level {level}: {points_per_pixel * len(group):,} evaluations "
                      f"in {n_chunks} chunks of up to {pixels_per_chunk} pixels")

            start_time = time.perf_counter()

            for i, start in enumerate(range(0, len(group), pixels_per_chunk)):
                # Integrate one chunk and store it, so finished chunks survive an interruption
                chunk = group[start:start + pixels_per_chunk]
                self.probability[:, :, chunk] = self._integrate_pixels(chunk, level)   # THE INTEGRAL of this chunk, stored
                self.subpixel_level_used[chunk] = level          # the level really used (above k near the poles)

                if verbose:
                    elapsed = time.perf_counter() - start_time
                    done = start + len(chunk)
                    print(f"  chunk {i + 1}/{n_chunks}: {done}/{len(group)} pixels, {elapsed:.0f} s, "
                          f"{done * points_per_pixel / elapsed:,.0f} evaluations/s")

        # Polar pixels, one at a time
        for i, (pixel, level) in enumerate(zip(polar, polar_levels)):
            start_time = time.perf_counter()
            self.probability[:, :, pixel], n_sky = self._integrate_polar_pixel(pixel, int(level), points_per_chunk)
            self.subpixel_level_used[pixel] = level          # the level really used (at least polar_subpixel_level)

            if verbose:
                elapsed = time.perf_counter() - start_time
                n_points = n_sky * len(self.em_nodes_kev) * len(self.phi_nodes_rad)
                print(f"  polar pixel {pixel} ({i + 1}/{len(polar)}): {n_sky:,} sky nodes, {elapsed:.0f} s, "
                      f"{n_points / elapsed:,.0f} evaluations/s")

        return self.probability[:, :, pixels].copy()


    def _integrate_pixels(self, pixels, k):
        """P_i for a group of pixels: the weighted sum over all quadrature nodes.

        Parameters
        ----------
        pixels : np.ndarray of int, shape (n_pixels,)
            Pixel indices (in ``scheme``).
        k : int
            Sub-pixel level.

        Returns
        -------
        np.ndarray, shape (n_em_bins, n_phi_bins, n_pixels)
        """
        # 1. Sky nodes: 4**k sub-pixel centres per pixel, weight = sub-pixel area [sr]
        colatitude, longitude, sky_weights = self._healpix_subpixel_nodes(
            self.nside, k, pixels, nest=self.nest)                       # each (n_pixels, 4**k)

        # 2. NF density at every (Em node, phi node, sky node)
        density = self._density_on_grid(colatitude.ravel(), longitude.ravel())   # all sky points as one list
        density = density.reshape(len(self.em_nodes_kev), len(self.phi_nodes_rad), len(pixels), 4**k)  # split sky axis into (pixel, sub-pixel)

        # 3. Weighted sum over the nodes of each bin:
        # P[a, b, c] = sum_e sum_p sum_s  M_E[a, e] * M_phi[b, p] * density[e, p, c, s] * w_sky[c, s]
        # a: energy bin, b: phi bin, c: pixel; e, p, s: Em, phi and sky nodes.
        return np.einsum("ae,bp,epcs,cs->abc", self._em_matrix, self._phi_matrix,   # weights x density, added
                         density, sky_weights, optimize=True)                         # up per bin -> P_i

    def _integrate_polar_pixel(self, pixel, level, points_per_chunk):
        """P_i of one polar pixel at sub-pixel level ``level``, with the refined sky nodes of ``_polar_pixel_nodes``.

        Returns
        -------
        np.ndarray, shape (n_em_bins, n_phi_bins)
            P_i of the pixel.
        int
            Number of sky nodes used.
        """
        # 1. Sky nodes: the pixel's level (from _levels_used); those close to the pole at the pole level
        colatitude, longitude, sky_weights = self._polar_pixel_nodes(self.nside, pixel, self.nest, level)

        # 2.-3. Density and weighted sum as in _integrate_pixels, in pieces of about points_per_chunk evaluations
        n_sky = max(1, points_per_chunk // (len(self.em_nodes_kev) * len(self.phi_nodes_rad)))   # sky nodes per piece
        total = np.zeros((self.n_em_bins, self.n_phi_bins))
        for a in range(0, len(sky_weights), n_sky):
            density = self._density_on_grid(colatitude[a:a + n_sky], longitude[a:a + n_sky])   # (Em, phi, sky)
            total += np.einsum("ae,bp,eps,s->ab", self._em_matrix, self._phi_matrix,
                               density, sky_weights[a:a + n_sky], optimize=True)
        return total, len(sky_weights)

    def _density_on_grid(self, colatitude, longitude):
        """NF density on (Em nodes) x (phi nodes) x (list of sky points).

        Parameters
        ----------
        colatitude, longitude : np.ndarray, shape (n_sky,)
            Paired sky points [rad] in the spacecraft frame (not a grid).

        Returns
        -------
        np.ndarray, shape (n_em_nodes, n_phi_nodes, n_sky), float64
            Density [keV^-1 rad^-1 sr^-1].
        """
        shape = (len(self.em_nodes_kev), len(self.phi_nodes_rad), len(colatitude))

        # chi = 0 seam: nodes inside the strip where the flow's density is wrong are moved just outside it
        longitude = np.clip(longitude, self.seam_delta_rad, 2 * np.pi - self.seam_delta_rad)

        # List of points to evaluate: every Em point x every phi point x every sky point
        grid = np.empty((4,) + shape)                 # 4 layers (Em, phi, chi, psi), one cell per combination
        grid[0] = self.em_nodes_kev[:, None, None]    # Em varies along axis 0, repeated along the others
        grid[1] = self.phi_nodes_rad[None, :, None]   # phi varies along axis 1
        grid[2] = longitude[None, None, :]            # chi varies along axis 2 (sky point) ...
        grid[3] = colatitude[None, None, :]           # ... together with psi: each sky point is one (psi, chi) pair
        em, phi, chi, psi = torch.from_numpy(grid.reshape(4, -1))   # flatten: 4 lists of n_Em*n_phi*n_sky values

        # Evaluate the flow. The context (time) is ignored by version 3 models; it only sets the number of points.
        context = torch.zeros(em.shape[0], dtype=torch.float32)
        density = self._density_model.evaluate_density(context, em, phi, chi, psi)   # p at every point
        return density.numpy().astype(np.float64).reshape(shape)                     # back to (n_Em, n_phi, n_sky)

    def _set_binning(self, axes, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg, verbose=True):
        """Prepare the quadrature nodes for this binning; keep the results if nothing changed.

        If something changed, the stored results no longer apply and are discarded
        (with a one-line notice if ``verbose`` and computed pixels are lost).
        """
        em_edges_kev, phi_edges_deg, nside, scheme = self._check_axes(axes)
        n_em_per_bin = self._check_int(n_em_per_bin, "n_em_per_bin")
        n_phi_per_bin = self._check_int(n_phi_per_bin, "n_phi_per_bin")
        em_breaks_kev = self._as_breaks(em_breaks_kev)
        phi_breaks_deg = self._as_breaks(phi_breaks_deg)

        # Same binning and settings as the last call: reuse the stored nodes and results.
        # The template settings are included, so changing one on an existing template also starts over
        settings = (tuple(em_edges_kev), tuple(phi_edges_deg), nside, scheme,
                    n_em_per_bin, n_phi_per_bin, tuple(em_breaks_kev), tuple(phi_breaks_deg),
                    self.seam_delta_rad,                                        # chi = 0 strip
                    self.polar_cap_rad, self.polar_subpixel_level,              # polar pixels
                    self.pole_colat_rad, self.pole_subpixel_level,              # ... and their refinement at the pole
                    self.near_pole_cap_rad, self.near_pole_subpixel_level)      # pixels near a pole
        if settings == self._settings:
            return

        # Something changed: the stored results no longer apply. Say so if computed pixels are lost
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
        self.em_nodes_kev, em_weights, em_bin_index = self._binned_gl_nodes(         # Em points, weights [keV],
            em_edges_kev, n_em_per_bin, em_breaks_kev)                               # and the bin of each point
        self.phi_nodes_rad, phi_weights, phi_bin_index = self._binned_gl_nodes(      # same for phi, in rad
            np.deg2rad(phi_edges_deg), n_phi_per_bin, np.deg2rad(phi_breaks_deg))

        # ... and the matrices that sum weighted node values into bins
        self._em_matrix = self._node_to_bin_matrix(em_weights, em_bin_index, self.n_em_bins)       # (Em bins, Em points)
        self._phi_matrix = self._node_to_bin_matrix(phi_weights, phi_bin_index, self.n_phi_bins)   # (phi bins, phi points)

        # Start over: nothing computed for this binning yet
        self.probability = np.full((self.n_em_bins, self.n_phi_bins, self.npix), np.nan)
        self.subpixel_level_used = np.full(self.npix, -1, dtype=np.int64)
        self._settings = settings

    # -----------------------------------------------------------------------
    # Quadrature nodes
    # -----------------------------------------------------------------------

    @staticmethod
    def _gl_nodes(a, b, n):
        """Gauss-Legendre nodes and weights on [a, b]; the weights sum to b - a."""
        x, w = np.polynomial.legendre.leggauss(n)     # standard GL points and weights on [-1, 1]
        # Map the standard nodes on [-1, 1] linearly onto [a, b]
        return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

    @staticmethod
    def _binned_gl_nodes(edges, n_per_bin, breaks=()):
        """Gauss-Legendre nodes for every bin of a 1D binning.

        A bin that contains break points strictly inside it is split at those
        points, and each piece gets ``n_per_bin`` nodes.

        Returns
        -------
        nodes, weights : np.ndarray, shape (n_nodes,)
            Node positions and weights, in the unit of ``edges``.
        bin_index : np.ndarray of int, shape (n_nodes,)
            Index of the bin that contains each node.
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
    def _healpix_subpixel_nodes(nside, k, pixels, nest=True):
        """Sky nodes: centres of the 4**k sub-pixels of each pixel, weight = sub-pixel area.

        In NESTED ordering the children of parent p at ``nside * 2**k`` are
        p * 4**k, ..., p * 4**k + 4**k - 1; the weights of one parent sum
        exactly to its area.

        Returns
        -------
        colatitude, longitude : np.ndarray, shape (n_pixels, 4**k)
            Sub-pixel centres [rad].
        weights : np.ndarray, shape (n_pixels, 4**k)
            Sub-pixel areas [sr].
        """
        pixels = np.atleast_1d(np.asarray(pixels, dtype=np.int64))
        parents_nest = pixels if nest else hp.ring2nest(nside, pixels)         # sub-pixels are defined in NESTED numbering

        n_sub = 4**k                                                           # sub-pixels per pixel
        nside_sub = nside * 2**k                                               # finer HEALPix grid of the sub-pixels
        children = parents_nest[:, None] * n_sub + np.arange(n_sub)[None, :]   # indices of the sub-pixels, (n_pixels, n_sub)

        colatitude, longitude = hp.pix2ang(nside_sub, children.ravel(), nest=True)   # centre of each sub-pixel
        weights = np.full(children.shape, hp.nside2pixarea(nside_sub))         # weight = sub-pixel area (all equal in HEALPix)
        return colatitude.reshape(children.shape), longitude.reshape(children.shape), weights

    @staticmethod
    def _is_polar(nside, pixels, nest, polar_cap_rad):
        """True for pixels whose centre is within polar_cap_rad of a pole."""
        colatitude, _ = hp.pix2ang(nside, pixels, nest=nest)
        return np.minimum(colatitude, np.pi - colatitude) < polar_cap_rad   # distance to the nearer pole

    def _levels_used(self, nside, pixels, nest, k):
        """Sub-pixel level each pixel really gets when level k is requested (what ``subpixel_level_used`` stores).

        The one place where this rule is written: k for most pixels, at least
        near_pole_subpixel_level near a pole, and at least polar_subpixel_level in
        polar pixels (before their extra refinement at the pole, see ``_polar_pixel_nodes``).
        The pixel grid (nside, nest) is an argument so that ``n_evaluations`` can use it
        before any binning is set.
        """
        near = self._is_polar(nside, pixels, nest, self.near_pole_cap_rad)     # within near_pole_cap_deg of a pole
        polar = self._is_polar(nside, pixels, nest, self.polar_cap_rad)        # the polar pixels
        levels = np.where(near, max(k, self.near_pole_subpixel_level), k)
        return np.where(polar, max(k, self.polar_subpixel_level), levels)

    def _polar_pixel_nodes(self, nside, pixel, nest, level):
        """Sky nodes of one polar pixel at its level (from ``_levels_used``), with this template's pole settings.

        The one place where the pole refinement is chosen: sub-pixels within pole_colat_deg
        of the pole are split to pole_subpixel_level (at least ``level``).
        Returns colatitude, longitude [rad] and weights [sr], as ``_polar_refined_nodes``.
        """
        return self._polar_refined_nodes(nside, pixel, nest, level,
                                         max(level, self.pole_subpixel_level), self.pole_colat_rad)

    @staticmethod
    def _polar_refined_nodes(nside, pixel, nest, k, k_pole, pole_colat_rad):
        """Sky nodes of one polar pixel: its 4**k sub-pixels, with those close to the pole split further.

        Sub-pixels with centre within ``pole_colat_rad`` of a pole are replaced by
        their 4**(k_pole - k) children at ``nside * 2**k_pole``. Each weight is the
        area of its (sub-)sub-pixel, so the weights still sum to the pixel area.

        Returns
        -------
        colatitude, longitude, weights : np.ndarray, shape (n_sky,)
            Node positions [rad] and weights [sr].
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

    @staticmethod
    def _node_to_bin_matrix(weights, bin_index, n_bins):
        """M[b, n] = weights[n] if node n lies in bin b, else 0; M @ f sums f over each bin."""
        matrix = np.zeros((n_bins, len(weights)))
        matrix[bin_index, np.arange(len(weights))] = weights    # row = bin, column = point, value = its weight
        return matrix

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    def get_binned_background(
            self, 
            axes, 
            frame, 
            ori_file, 
            tmin, 
            tmax, 
            pixels=None,
            subpixel_level=2, 
            n_em_per_bin=4, 
            n_phi_per_bin=4,
            em_breaks_kev=None, 
            phi_breaks_deg=None, 
            scatt_nside=32
            ):
        """Expected background counts in [tmin, tmax], binned in Em x Phi x PsiChi.

        N_i = R * P_i * (live time), with R = model_counts / model_livetime and
        the live time taken from the orientation file. Requested pixels not yet
        computed at ``subpixel_level`` (with this binning and these settings) are
        computed first; all pixels take hours.

        Parameters
        ----------
        axes : histpy.Axes
            Binning, with axes "Em", "Phi" and "PsiChi" (others are ignored).
            Only the NSIDE and scheme of PsiChi are used, not its frame.
        frame : str
            "local" (spacecraft frame) or "galactic".
        ori_file : str or Path
            Orientation file (.ori or .fits) with attitude and live time.
        tmin, tmax : float
            Time window [s] (unix seconds, as the event TimeTags).
        pixels : array_like of int or None
            Local frame only: pixels to compute (in the scheme of PsiChi). Pixels
            computed in earlier calls with the same setup are kept in the result;
            pixels never computed, or stored at another sub-pixel level than this
            call gives them, are NaN. None means all pixels. The Galactic frame
            needs every pixel, so pixels must be None.
        subpixel_level, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg
            Quadrature settings, as in ``compute``.
        scatt_nside : int
            Attitude grouping for the Galactic rotation (see ``galactic_conversion_matrix``).

        Returns
        -------
        histpy.Histogram, shape (n_em_bins, n_phi_bins, npix)
            Axes "Em" [keV], "Phi" [deg], "PsiChi" (NSIDE and scheme of ``axes``,
            spacecraft or Galactic frame).
        """
        self._check_frame(frame, pixels)
        self._set_binning(axes, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg)
        k = self._check_int(subpixel_level, "subpixel_level", minimum=0)

        # 1. Integrate the requested pixels not yet computed at level k
        missing = self._missing_pixels(pixels, k)
        if missing.size:
            self.compute(axes, missing, k, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg)

        # 2. Rate of the model and attitude / live time in the window
        history = SpacecraftHistory.open(ori_file).select_interval(
            Time(tmin, format="unix"), Time(tmax, format="unix"))
        rate = self.model_counts / self.model_livetime          # events / s

        # 3a. Local frame: counts = rate * live time * P
        if frame == "local":
            livetime = float(history.livetime.to_value(u.s).sum())          # total live time in [tmin, tmax] [s]

            # Only pixels computed with the rule of this call: a pixel stored at another sub-pixel level
            # (e.g. from a test with a different k on this template) is NaN in the result, like one never computed
            level_now = self._levels_used(self.nside, np.arange(self.npix), self.nest, k)   # level each pixel gets in this call
            probability = np.where(self.subpixel_level_used == level_now, self.probability, np.nan)
            return Histogram(self._axes(SpacecraftFrame()), contents=rate * livetime * probability)

        # 3b. Galactic frame: rotate with W (live time each Galactic pixel spends in each local pixel)
        W = galactic_conversion_matrix(history, self.nside, self.nest, scatt_nside)   # (Galactic pix, local pix) [s]
        counts = rate * (self.probability @ W.T)                # each Galactic pixel collects local P weighted by live time
        return Histogram(self._axes("galactic"), contents=counts)

    def expected_counts(self):
        """Expected counts N_i = R * T * P_i of the current binning, shape (n_em_bins, n_phi_bins, npix); NaN if not computed."""
        self._check_binning_set()
        return self.model_counts * self.probability

    def to_histogram(self):
        """Template P_i as a histpy Histogram (axes Em [keV], Phi [deg], PsiChi in the spacecraft frame).

        Only allowed once every pixel has been computed.
        """
        self._check_binning_set()
        self._check_all_computed()
        return Histogram(self._axes(SpacecraftFrame()), contents=self.probability)

    def n_evaluations(self, axes, pixels=None, subpixel_level=2, n_em_per_bin=4, n_phi_per_bin=4,
                      em_breaks_kev=None, phi_breaks_deg=None):
        """Number of density evaluations a ``compute`` call with these arguments would need.

        Does not change the stored results. Divide by the measured throughput
        (about 7,000 evaluations/s on CPU) to estimate the run time.
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

    def _evaluations_per_pixel(self, k):
        # Grid points per pixel: Em nodes x phi nodes x sky nodes
        return len(self.em_nodes_kev) * len(self.phi_nodes_rad) * 4**k

    def _axes(self, coordsys):
        # Em, Phi, PsiChi axes of the template, with PsiChi in the given frame
        return Axes([Axis(self.em_edges_kev, unit=u.keV, label="Em"),
                     Axis(self.phi_edges_deg, unit=u.deg, label="Phi"),
                     HealpixAxis(nside=self.nside, scheme=self.scheme,
                                 coordsys=coordsys, label="PsiChi")])

    def _missing_pixels(self, pixels, k):
        # Requested pixels (None -> all) not computed yet at the level they get for a requested level k
        pixels = self._check_pixels(pixels)
        return pixels[self.subpixel_level_used[pixels] != self._levels_used(self.nside, pixels, self.nest, k)]

    @staticmethod
    def _as_breaks(breaks):
        # Break points as a 1D float array (None -> empty)
        return np.atleast_1d(np.asarray([] if breaks is None else breaks, dtype=np.float64))

    # -----------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_axes(axes):
        # Binning from histpy Axes: Em edges [keV], Phi edges [deg] within [0, 180], PsiChi NSIDE and scheme
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
    def _check_edges(edges, name):
        # Bin edges: 1D, strictly increasing, at least one bin
        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError(f"{name} must be a 1D, strictly increasing array with at least 2 edges.")
        return edges

    @staticmethod
    def _check_nside(nside):
        # Sub-pixels are NESTED children, so NSIDE must be a power of 2 for both schemes
        if not hp.isnsideok(nside, nest=True):
            raise ValueError(f"PsiChi nside must be a power of 2, got {nside}.")
        return int(nside)

    @staticmethod
    def _check_scheme(scheme):
        # Pixel ordering: "nested" or "ring" (case-insensitive)
        scheme = str(scheme).lower()
        if scheme not in ("nested", "ring"):
            raise ValueError(f"PsiChi scheme must be 'nested' or 'ring', got {scheme!r}.")
        return scheme

    @staticmethod
    def _check_int(value, name, minimum=1):
        # Integer >= minimum
        if int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value}.")
        return int(value)

    @staticmethod
    def _check_seam_delta(value):
        # Half-width of the chi = 0 strip [deg]: a number in [0, 1); 0 turns the correction off
        value = float(value)
        if not 0 <= value < 1:
            raise ValueError(f"seam_delta_deg must be in [0, 1) deg, got {value}.")
        return value

    @staticmethod
    def _check_angle(value, name):
        # Angle from a pole [deg]: a number in [0, 90); 0 turns the refinement off
        value = float(value)
        if not 0 <= value < 90:
            raise ValueError(f"{name} must be in [0, 90) deg, got {value}.")
        return value

    @staticmethod
    def _check_model_version(version):
        # Only major version 3 (time-independent) models are supported; version is a string like "3.00"
        if int(str(version).split(".")[0]) != 3:
            raise NotImplementedError(
                f"Only major version 3 (time-independent) models are supported, "
                f"got version {version}.")

    def _check_pixels(self, pixels):
        # Pixel indices as a 1D array of unique, valid ints; None means all pixels
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

    def _check_binning_set(self):
        # A binning must have been given by a compute / get_binned_background call
        if self._settings is None:
            raise RuntimeError("No binning yet: call compute or get_binned_background first.")

    def _check_all_computed(self):
        # Every pixel must be computed (no NaN left in probability)
        missing = np.isnan(self.probability).any(axis=(0, 1))
        if missing.any():
            raise RuntimeError(f"{missing.sum()} of {self.npix} pixels are not computed yet.")

    @staticmethod
    def _check_frame(frame, pixels):
        # Frame "local" or "galactic"; a pixel subset only makes sense in the local frame
        if frame not in ("local", "galactic"):
            raise ValueError(f"frame must be 'local' or 'galactic', got {frame!r}.")
        if frame == "galactic" and pixels is not None:
            raise ValueError("The Galactic frame mixes all spacecraft-frame pixels; use pixels=None.")


# ---------------------------------------------------------------------------
# Spacecraft frame -> Galactic frame (separate tool, also used without a template)
# ---------------------------------------------------------------------------

def galactic_conversion_matrix(
        history, 
        nside, 
        nest, 
        scatt_nside=32, 
        attitudes_per_chunk=512
        ):
    """Live time that each Galactic pixel spends in each spacecraft-frame pixel.

    W[g, l] is the live time [s] during which the center of Galactic pixel g
    lies in spacecraft-frame pixel l. For a spacecraft-frame rate r[l] [1/s],
    the expected counts per Galactic pixel are W @ r.

    This is the nearest-pixel method cosipy uses for its Galactic responses
    (``FullDetectorResponse.get_point_source_response``): the attitudes are
    grouped with ``SpacecraftHistory.get_scatt_map`` (live-time weights, no
    Earth occultation), and for each attitude the Galactic pixel centers are
    rotated into the spacecraft frame.

    Parameters
    ----------
    history : cosipy.spacecraftfile.SpacecraftHistory
        Attitude and live time, already cut to the time window.
    nside : int
        HEALPix NSIDE of both pixelizations.
    nest : bool
        Ordering of both pixelizations (True: NESTED, False: RING).
    scatt_nside : int
        Resolution of the attitude grouping (``get_scatt_map``).
    attitudes_per_chunk : int
        Attitudes processed at once (bounds memory).

    Returns
    -------
    np.ndarray, shape (npix, npix)
        W [s]; every row sums to the total live time of ``history``.
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
