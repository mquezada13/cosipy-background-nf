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

Units and conventions
---------------------
- The density is in keV^-1 rad^-1 sr^-1 (the Jacobian of the flow's internal
  coordinates is applied inside cosipy), so P_i is dimensionless.
- Em in keV, phi in rad internally (bin edges are given in degrees).
- Sky coordinates are in the spacecraft frame, with COSIpy names:
  psi = colatitude ("Psi local"), chi = longitude ("Chi local"), both in rad.
  The NF source code uses swapped names; this module always uses COSIpy names.
- Solid angle: dOmega = dcos(psi) dchi.
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
from cosipy.background_estimation.ml import NFBackground
from cosipy.background_estimation.ml.NFBackgroundModels import BackgroundDensityECDFSplineDGaussianARQSFlow


class NFBinnedTemplate:
    """Binned template P_i of an NF background model, by integration of its density.

    Created from the model file only. The binning (histpy Axes with "Em",
    "Phi", "PsiChi") and the quadrature settings are given to ``compute`` and
    ``get_binned_background``. The last results are kept in ``probability``
    (NaN for pixels not computed yet) and reused while the binning and the
    quadrature settings stay the same; a call with different ones starts over.

    Parameters
    ----------
    model_path : str or Path
        NF background model file (.fits), major version 3.
    batch_size : int
        Number of points per forward pass of the flow (bounds the network's memory).
    device : str
        Torch device for the flow, e.g. "cpu" or "cuda".

    Attributes
    ----------
    model_version : str
        Version of the model file.
    model_counts : float
        R * T, the number of events behind the model.
    model_livetime : float
        Livetime of the model [s].
    probability : np.ndarray, shape (n_em_bins, n_phi_bins, npix), or None
        P_i of the current binning, float64; NaN where not computed; None before
        the first call.
    subpixel_level_used : np.ndarray of int, shape (npix,), or None
        Sub-pixel level used for each pixel; -1 where not computed.
    """

    def __init__(self, 
                 model_path, 
                 batch_size=100_000, 
                 device="cpu"):
        
        self.model_path = model_path
        self.batch_size = self._check_int(batch_size, "batch_size")
        self.device = device

        # --- Load the model once, in this process (no `devices` argument -> no worker pool) ---
        nf_background = NFBackground(model_path, density_batch_size=self.batch_size,      # reads the .fits:
                                     density_compile_mode=None, show_progress=False)      # weights, version, counts, livetime
        self._check_model_version(nf_background)
        self.model_version = nf_background._version

        # Normalization of the model: R = model_counts / model_livetime
        rate_input = nf_background._ckpt["rate_input"]
        self.model_counts = float(rate_input["counts"])
        self.model_livetime = float(rate_input["livetime"])

        # The density network. Arguments: (density_input, device, batch_size, compile_mode).
        self._density_model = BackgroundDensityECDFSplineDGaussianARQSFlow(       # the network that returns
            nf_background._density_input, device, self.batch_size, None)          # p(Em, phi, psi, chi)

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

        Parameters
        ----------
        axes : histpy.Axes
            Binning, with axes "Em" [energy], "Phi" [angle] and "PsiChi" (HealpixAxis).
            Other axes (e.g. "Time") are ignored.
        pixels : array_like of int or None
            Pixel indices (in the scheme of PsiChi); None means all pixels.
        subpixel_level : int
            Sub-pixel level k: 4**k sky nodes per pixel.
        n_em_per_bin, n_phi_per_bin : int
            GL nodes per Em / phi bin (per piece if the bin is split at a break).
        em_breaks_kev, phi_breaks_deg : array_like or None
            Points at which Em [keV] / phi [deg] bins are split (e.g. em_breaks_kev=[511.0]).
        points_per_chunk : int
            Target number of density evaluations per chunk (at least one pixel).
        verbose : bool
            Print one line per chunk with progress and throughput.

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
            phi_breaks_deg
            )
        pixels = self._check_pixels(pixels)

        k = self._check_int(
            subpixel_level, 
            "subpixel_level", 
            minimum=0)

        # Group pixels into chunks of about points_per_chunk density evaluations
        points_per_pixel = self._evaluations_per_pixel(k)                 # n_Em points x n_phi points x 4**k
        pixels_per_chunk = max(1, points_per_chunk // points_per_pixel)   # how many pixels fit in one chunk
        n_chunks = -(-len(pixels) // pixels_per_chunk)    # ceiling division

        if verbose:
            print(f"{len(pixels)} pixels, k = {k}: {points_per_pixel * len(pixels):,} evaluations "
                  f"in {n_chunks} chunks of up to {pixels_per_chunk} pixels")

        start_time = time.perf_counter()

        for i, start in enumerate(range(0, len(pixels), pixels_per_chunk)):
            # Integrate one chunk and store it, so finished chunks survive an interruption
            chunk = pixels[start:start + pixels_per_chunk]
            self.probability[:, :, chunk] = self._integrate_pixels(chunk, k)   # THE INTEGRAL of this chunk, stored
            self.subpixel_level_used[chunk] = k                                # remember the k of each pixel

            if verbose:
                elapsed = time.perf_counter() - start_time
                done = start + len(chunk)
                print(f"  chunk {i + 1}/{n_chunks}: {done}/{len(pixels)} pixels, {elapsed:.0f} s, "
                      f"{done * points_per_pixel / elapsed:,.0f} evaluations/s")

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

    def _set_binning(self, axes, n_em_per_bin, n_phi_per_bin, em_breaks_kev, phi_breaks_deg):
        """Prepare the quadrature nodes for this binning; keep the results if nothing changed."""
        em_edges_kev, phi_edges_deg, nside, scheme = self._check_axes(axes)
        n_em_per_bin = self._check_int(n_em_per_bin, "n_em_per_bin")
        n_phi_per_bin = self._check_int(n_phi_per_bin, "n_phi_per_bin")
        em_breaks_kev = self._as_breaks(em_breaks_kev)
        phi_breaks_deg = self._as_breaks(phi_breaks_deg)

        # Same binning and settings as the last call: reuse the stored nodes and results
        settings = (tuple(em_edges_kev), tuple(phi_edges_deg), nside, scheme,
                    n_em_per_bin, n_phi_per_bin, tuple(em_breaks_kev), tuple(phi_breaks_deg))
        if settings == self._settings:
            return

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
            Local frame only: pixels to compute (in the scheme of PsiChi); pixels
            never computed are NaN in the result. None means all pixels. The
            Galactic frame needs every pixel, so pixels must be None.
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
            return Histogram(self._axes(SpacecraftFrame()), contents=rate * livetime * self.probability)

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
        n_pixels = hp.nside2npix(nside) if pixels is None else len(np.atleast_1d(pixels))
        return n_em_nodes * n_phi_nodes * 4**self._check_int(subpixel_level, "subpixel_level", minimum=0) * n_pixels

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
        # Requested pixels (None -> all) not computed yet at sub-pixel level k
        pixels = self._check_pixels(pixels)
        return pixels[self.subpixel_level_used[pixels] != k]

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
    def _check_model_version(nf_background):
        # Only major version 3 (time-independent) models are supported
        if nf_background._major_version != 3:
            raise NotImplementedError(
                f"Only major version 3 (time-independent) models are supported, "
                f"got version {nf_background._version}.")

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
