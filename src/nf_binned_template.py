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
  ``healpix_subpixel_nodes``; a different sky rule can replace that function
  as long as it returns the same (colatitude, longitude, weight) arrays.

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
The axes are those of cosipy's BinnedData after ``.project("Em", "Phi", "PsiChi")``.

Usage (DC4 511 keV imaging binning)
-----------------------------------
    import numpy as np
    from src.nf_binned_template import NFBinnedTemplate

    template = NFBinnedTemplate(
        "model/nfbackground_part-01_v3-00.fits",
        em_edges_kev=[509.0, 513.0],
        phi_edges_deg=np.arange(0, 181, 3),
        nside=16, scheme="ring",
        n_em_per_bin=4, n_phi_per_bin=4, subpixel_level=2,
        em_breaks_kev=[511.0])

    background = template.get_binned_background(
        "galactic", ori_file="data/DC4_orientation.ori",
        tmin=1835487300.036206, tmax=1843467254.887687)

    # Step by step instead (pixel indices in the scheme of the template):
    print(template.n_evaluations([255, 444]))      # cost before running
    rows = template.compute([255, 444])            # shape (1, 60, 2)
    template.compute()                             # all pixels (hours)
    hist = template.to_histogram()                 # probabilities P_i, axes Em, Phi, PsiChi
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

# ---------------------------------------------------------------------------
# Quadrature nodes
# ---------------------------------------------------------------------------

def gl_nodes(a, b, n):
    """Gauss-Legendre nodes and weights on the interval [a, b].

    Parameters
    ----------
    a, b : float
        Interval limits (any unit; the weights carry the same unit).
    n : int
        Number of nodes.

    Returns
    -------
    nodes, weights : np.ndarray, shape (n,)
        The weights sum to b - a.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    # Map the standard nodes on [-1, 1] linearly onto [a, b].
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def binned_gl_nodes(edges, n_per_bin, breaks=None):
    """Gauss-Legendre nodes for every bin of a 1D binning.

    A bin that contains break points strictly inside it is split at those
    points, and each piece gets ``n_per_bin`` nodes. Break points on a bin edge
    or outside the binning are ignored.

    Parameters
    ----------
    edges : array_like, shape (n_bins + 1,)
        Increasing bin edges.
    n_per_bin : int
        Number of GL nodes per bin (per piece for a split bin).
    breaks : array_like or None
        Break points, in the same unit as ``edges``.

    Returns
    -------
    nodes, weights : np.ndarray, shape (n_nodes,)
        Node positions and weights, in the unit of ``edges``.
    bin_index : np.ndarray of int, shape (n_nodes,)
        Index of the bin that contains each node.
    """
    edges = np.asarray(edges, dtype=np.float64)
    breaks = np.unique(np.asarray([] if breaks is None else breaks, dtype=np.float64))

    nodes, weights, bin_index = [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        inside = breaks[(breaks > lo) & (breaks < hi)]
        pieces = np.concatenate([[lo], inside, [hi]])
        for a, b in zip(pieces[:-1], pieces[1:]):
            x, w = gl_nodes(a, b, n_per_bin)
            nodes.append(x)
            weights.append(w)
            bin_index.append(np.full(n_per_bin, i))

    return np.concatenate(nodes), np.concatenate(weights), np.concatenate(bin_index)


def healpix_subpixel_nodes(nside, k, pixels, nest=True):
    """Sky quadrature nodes inside HEALPix pixels.

    Each parent pixel at ``nside`` is split into its 4**k children at
    ``nside * 2**k``. In NESTED ordering the children of parent p are
    p * 4**k, ..., p * 4**k + 4**k - 1. The nodes are the child centers and
    each weight is the child area, so the weights of one parent sum exactly to
    its area. The integral of f over a parent is approximated by
    sum_children weight * f(center).

    Parameters
    ----------
    nside : int
        NSIDE of the parent pixels (a power of 2).
    k : int
        Sub-pixel level (k = 0 uses the parent center only).
    pixels : array_like of int, shape (n_pixels,)
        Parent pixel indices, in NESTED ordering if ``nest`` is True,
        otherwise in RING ordering.
    nest : bool
        Ordering of ``pixels``.

    Returns
    -------
    colatitude, longitude : np.ndarray, shape (n_pixels, 4**k)
        Child centers [rad]; colatitude in [0, pi], longitude in [0, 2 pi).
    weights : np.ndarray, shape (n_pixels, 4**k)
        Child areas [sr].
    """
    pixels = np.atleast_1d(np.asarray(pixels, dtype=np.int64))
    parents_nest = pixels if nest else hp.ring2nest(nside, pixels)

    n_sub = 4**k
    nside_sub = nside * 2**k
    children = parents_nest[:, None] * n_sub + np.arange(n_sub)[None, :]   # (n_pixels, n_sub)

    colatitude, longitude = hp.pix2ang(nside_sub, children.ravel(), nest=True)
    weights = np.full(children.shape, hp.nside2pixarea(nside_sub))         # all HEALPix pixels have equal area
    return colatitude.reshape(children.shape), longitude.reshape(children.shape), weights


def node_to_bin_matrix(weights, bin_index, n_bins):
    """Matrix that sums weighted node values into bins.

    M[b, n] = weights[n] if node n lies in bin b, else 0. For values f at the
    nodes, M @ f gives the quadrature sum of f over each bin.

    Parameters
    ----------
    weights : np.ndarray, shape (n_nodes,)
        Quadrature weights.
    bin_index : np.ndarray of int, shape (n_nodes,)
        Bin of each node.
    n_bins : int
        Number of bins.

    Returns
    -------
    np.ndarray, shape (n_bins, n_nodes)
    """
    matrix = np.zeros((n_bins, len(weights)))
    matrix[bin_index, np.arange(len(weights))] = weights
    return matrix


# ---------------------------------------------------------------------------
# Spacecraft frame -> Galactic frame
# ---------------------------------------------------------------------------

def galactic_conversion_matrix(history, nside, nest, scatt_nside=32, attitudes_per_chunk=512):
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
    scatt = history.get_scatt_map(nside=scatt_nside, target_coord=None, earth_occ=False)
    livetime = np.asarray(scatt.weights.to_value(u.s), dtype=np.float64)
    rotation = scatt.attitudes.rot.inv().as_matrix()            # (n_att, 3, 3): Galactic -> spacecraft frame

    npix = hp.nside2npix(nside)
    galactic_pix = np.arange(npix)
    galactic_dirs = np.array(hp.pix2vec(nside, galactic_pix, nest=nest))   # (3, npix), Galactic Cartesian

    W = np.zeros(npix * npix)
    for a in range(0, len(rotation), attitudes_per_chunk):
        local_dirs = np.einsum("cij,jp->cip", rotation[a:a + attitudes_per_chunk], galactic_dirs)
        local_pix = hp.vec2pix(nside, local_dirs[:, 0], local_dirs[:, 1], local_dirs[:, 2], nest=nest)
        W += np.bincount((galactic_pix * npix + local_pix).ravel(),
                         weights=np.repeat(livetime[a:a + attitudes_per_chunk], npix),
                         minlength=npix * npix)
    return W.reshape(npix, npix)


def _check_edges(edges, name):
    # Bin edges must be a 1D, strictly increasing array with at least one bin.
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be a 1D, strictly increasing array with at least 2 edges.")
    return edges


def _check_positive_int(value, name, minimum=1):
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}.")
    return int(value)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

class NFBinnedTemplate:
    """Binned template P_i of an NF background model, by integration of its density.

    The binning is Em x Phi x PsiChi (HEALPix, spacecraft frame). Results are
    kept in ``probability`` (shape (n_em_bins, n_phi_bins, npix), NaN for pixels
    not computed yet), so the sky can be computed in several ``compute`` calls,
    for example with a different sub-pixel level in some regions.

    Parameters
    ----------
    model_path : str or Path
        NF background model file (.fits), major version 3.
    em_edges_kev : array_like
        Measured-energy bin edges [keV].
    phi_edges_deg : array_like
        Compton scattering angle bin edges [deg], within [0, 180].
    nside : int
        HEALPix NSIDE of the PsiChi axis (a power of 2).
    scheme : str
        Pixel ordering of the PsiChi axis, "nested" or "ring".
    n_em_per_bin : int
        GL nodes per energy bin (per piece if the bin is split at a break).
    n_phi_per_bin : int
        GL nodes per phi bin (per piece if the bin is split at a break).
    subpixel_level : int
        Default sub-pixel level k: 4**k sky nodes per pixel.
    em_breaks_kev : array_like or None
        Energies [keV] at which energy bins are split (e.g. [511.0]).
    phi_breaks_deg : array_like or None
        Angles [deg] at which phi bins are split.
    batch_size : int
        Number of points per forward pass of the flow (bounds the network's memory).
    device : str
        Torch device for the flow, e.g. "cpu" or "cuda".

    Attributes
    ----------
    model_counts : float
        R * T, the number of events behind the model.
    model_livetime : float
        Livetime of the model [s].
    probability : np.ndarray, shape (n_em_bins, n_phi_bins, npix)
        P_i, float64; NaN where not computed.
    subpixel_level_used : np.ndarray of int, shape (npix,)
        Sub-pixel level used for each pixel; -1 where not computed.
    """

    def __init__(self, model_path, em_edges_kev, phi_edges_deg, nside, scheme,
                 n_em_per_bin, n_phi_per_bin, subpixel_level,
                 em_breaks_kev=None, phi_breaks_deg=None, batch_size=100_000, device="cpu"):

        # --- Binning and quadrature settings ---
        self.model_path = model_path
        self.em_edges_kev = _check_edges(em_edges_kev, "em_edges_kev")
        self.phi_edges_deg = _check_edges(phi_edges_deg, "phi_edges_deg")
        if self.phi_edges_deg[0] < 0 or self.phi_edges_deg[-1] > 180:
            raise ValueError("phi_edges_deg must lie within [0, 180].")

        # Sub-pixels are built from NESTED children, so NSIDE must be a power of 2
        # for both schemes.
        if not hp.isnsideok(nside, nest=True):
            raise ValueError(f"nside must be a power of 2, got {nside}.")
        self.nside = int(nside)
        self.npix = hp.nside2npix(self.nside)

        self.scheme = scheme.lower()
        if self.scheme not in ("nested", "ring"):
            raise ValueError(f"scheme must be 'nested' or 'ring', got {scheme!r}.")
        self.nest = self.scheme == "nested"

        self.n_em_per_bin = _check_positive_int(n_em_per_bin, "n_em_per_bin")
        self.n_phi_per_bin = _check_positive_int(n_phi_per_bin, "n_phi_per_bin")
        self.subpixel_level = _check_positive_int(subpixel_level, "subpixel_level", minimum=0)
        self.em_breaks_kev = em_breaks_kev
        self.phi_breaks_deg = phi_breaks_deg
        self.batch_size = _check_positive_int(batch_size, "batch_size")
        self.device = device

        self.n_em_bins = len(self.em_edges_kev) - 1
        self.n_phi_bins = len(self.phi_edges_deg) - 1

        # --- Em and phi nodes: the same for every pixel, so built once ---
        self.em_nodes_kev, self.em_weights_kev, self.em_bin_index = binned_gl_nodes(
            self.em_edges_kev, self.n_em_per_bin, em_breaks_kev)

        phi_breaks_rad = None if phi_breaks_deg is None else np.deg2rad(phi_breaks_deg)
        self.phi_nodes_rad, self.phi_weights_rad, self.phi_bin_index = binned_gl_nodes(
            np.deg2rad(self.phi_edges_deg), self.n_phi_per_bin, phi_breaks_rad)

        # Weighted node-to-bin matrices: (n_em_bins, n_em_nodes) and (n_phi_bins, n_phi_nodes).
        self._em_matrix = node_to_bin_matrix(self.em_weights_kev, self.em_bin_index, self.n_em_bins)
        self._phi_matrix = node_to_bin_matrix(self.phi_weights_rad, self.phi_bin_index, self.n_phi_bins)

        # --- Load the model once, in this process ---
        # Without the `devices` argument NFBackground starts no worker pool.
        nf_background = NFBackground(model_path, density_batch_size=self.batch_size,
                                     density_compile_mode=None, show_progress=False)
        if nf_background._major_version != 3:
            raise NotImplementedError(
                f"Only major version 3 (time-independent) models are supported, "
                f"got version {nf_background._version}.")
        self.model_version = nf_background._version

        rate_input = nf_background._ckpt["rate_input"]
        self.model_counts = float(rate_input["counts"])
        self.model_livetime = float(rate_input["livetime"])

        # Arguments: (density_input, device, batch_size, compile_mode).
        self._density_model = BackgroundDensityECDFSplineDGaussianARQSFlow(
            nf_background._density_input, device, self.batch_size, None)

        # --- Results ---
        self.probability = np.full((self.n_em_bins, self.n_phi_bins, self.npix), np.nan)
        self.subpixel_level_used = np.full(self.npix, -1, dtype=np.int64)

    # -----------------------------------------------------------------------

    def _check_pixels(self, pixels):
        # Pixel indices (in self.scheme) as a 1D int array; None means all pixels.
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

    def _evaluations_per_pixel(self, k):
        # Grid points per pixel: Em nodes x phi nodes x sky nodes.
        return len(self.em_nodes_kev) * len(self.phi_nodes_rad) * 4**k

    def n_evaluations(self, pixels=None, subpixel_level=None):
        """Number of density evaluations a ``compute`` call would need.

        Divide by the measured throughput (about 7,000 evaluations/s on CPU) to
        estimate the run time.

        Parameters
        ----------
        pixels : array_like of int or None
            Pixel indices (in ``scheme``); None means all pixels.
        subpixel_level : int or None
            Sub-pixel level k; None uses the default of the instance.

        Returns
        -------
        int
        """
        k = self.subpixel_level if subpixel_level is None else subpixel_level
        return self._evaluations_per_pixel(k) * len(self._check_pixels(pixels))

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

        # One array holds the four coordinates of every grid point; broadcasting
        # fills it without building index grids.
        grid = np.empty((4,) + shape)
        grid[0] = self.em_nodes_kev[:, None, None]
        grid[1] = self.phi_nodes_rad[None, :, None]
        grid[2] = longitude[None, None, :]        # chi
        grid[3] = colatitude[None, None, :]       # psi
        em, phi, chi, psi = torch.from_numpy(grid.reshape(4, -1))

        # The context is ignored by version 3 models; it only sets the number of points.
        context = torch.zeros(em.shape[0], dtype=torch.float32)
        density = self._density_model.evaluate_density(context, em, phi, chi, psi)
        return density.numpy().astype(np.float64).reshape(shape)

    def _integrate_pixels(self, pixels, k):
        """P_i for a group of pixels.

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
        colatitude, longitude, sky_weights = healpix_subpixel_nodes(
            self.nside, k, pixels, nest=self.nest)                       # each (n_pixels, 4**k)

        density = self._density_on_grid(colatitude.ravel(), longitude.ravel())
        density = density.reshape(len(self.em_nodes_kev), len(self.phi_nodes_rad), len(pixels), 4**k)

        # P[a, b, c] = sum_e sum_p sum_s  M_E[a, e] * M_phi[b, p] * density[e, p, c, s] * w_sky[c, s]
        # a: energy bin, b: phi bin, c: pixel; e, p, s: Em, phi and sky nodes.
        return np.einsum("ae,bp,epcs,cs->abc", self._em_matrix, self._phi_matrix,
                         density, sky_weights, optimize=True)

    def compute(self, pixels=None, subpixel_level=None, points_per_chunk=2_000_000, verbose=True):
        """Integrate the NF density over every (Em, phi) bin of the given pixels.

        Pixels are processed in chunks of about ``points_per_chunk`` density
        evaluations to bound memory (about 50 bytes per evaluation, i.e.
        ~100 MB for the default). Results are written into ``probability``
        chunk by chunk, so an interrupted run keeps the chunks already done.

        Parameters
        ----------
        pixels : array_like of int or None
            Pixel indices (in ``scheme``); None means all pixels.
        subpixel_level : int or None
            Sub-pixel level k for this call; None uses the default of the instance.
        points_per_chunk : int
            Target number of density evaluations per chunk (at least one pixel).
        verbose : bool
            Print one line per chunk with progress and throughput.

        Returns
        -------
        np.ndarray, shape (n_em_bins, n_phi_bins, n_pixels)
            P_i for the given pixels, in the order given.
        """
        pixels = self._check_pixels(pixels)
        k = self.subpixel_level if subpixel_level is None else _check_positive_int(
            subpixel_level, "subpixel_level", minimum=0)

        points_per_pixel = self._evaluations_per_pixel(k)
        pixels_per_chunk = max(1, points_per_chunk // points_per_pixel)
        n_chunks = -(-len(pixels) // pixels_per_chunk)    # ceiling division

        if verbose:
            print(f"{len(pixels)} pixels, k = {k}: {points_per_pixel * len(pixels):,} evaluations "
                  f"in {n_chunks} chunks of up to {pixels_per_chunk} pixels")

        start_time = time.perf_counter()
        for i, start in enumerate(range(0, len(pixels), pixels_per_chunk)):
            chunk = pixels[start:start + pixels_per_chunk]
            self.probability[:, :, chunk] = self._integrate_pixels(chunk, k)
            self.subpixel_level_used[chunk] = k

            if verbose:
                elapsed = time.perf_counter() - start_time
                done = start + len(chunk)
                print(f"  chunk {i + 1}/{n_chunks}: {done}/{len(pixels)} pixels, {elapsed:.0f} s, "
                      f"{done * points_per_pixel / elapsed:,.0f} evaluations/s")

        return self.probability[:, :, pixels].copy()

    def expected_counts(self):
        """Expected counts N_i = R * T * P_i for all pixels.

        Returns
        -------
        np.ndarray, shape (n_em_bins, n_phi_bins, npix)
            NaN for pixels not computed yet.
        """
        return self.model_counts * self.probability

    def to_histogram(self):
        """Template P_i as a histpy Histogram.

        Axes follow cosipy's binned data (``BinnedData.get_binned_data`` with
        local PsiChi binning), without the Time axis: "Em" [keV], "Phi" [deg],
        and "PsiChi" (HealpixAxis in the spacecraft frame). Only allowed once
        every pixel has been computed.

        Returns
        -------
        histpy.Histogram, shape (n_em_bins, n_phi_bins, npix)
        """
        missing = np.isnan(self.probability).any(axis=(0, 1))
        if missing.any():
            raise RuntimeError(f"{missing.sum()} of {self.npix} pixels are not computed yet.")

        return Histogram(self._axes(SpacecraftFrame()), contents=self.probability)

    def _axes(self, coordsys):
        # Em, Phi, PsiChi axes of the template, with PsiChi in the given frame.
        return Axes([Axis(self.em_edges_kev, unit=u.keV, label="Em"),
                     Axis(self.phi_edges_deg, unit=u.deg, label="Phi"),
                     HealpixAxis(nside=self.nside, scheme=self.scheme,
                                 coordsys=coordsys, label="PsiChi")])

    def get_binned_background(self, frame, ori_file, tmin, tmax, pixels=None, scatt_nside=32):
        """Expected background counts in [tmin, tmax], binned in Em x Phi x PsiChi.

        N_i = R * P_i * (live time), with R = model_counts / model_livetime and
        the live time taken from the orientation file. Requested pixels not
        computed yet are computed first with the default sub-pixel level (all
        pixels take hours).

        Parameters
        ----------
        frame : str
            "local" (spacecraft frame) or "galactic".
        ori_file : str or Path
            Orientation file (.ori or .fits) with attitude and live time.
        tmin, tmax : float
            Time window [s] (unix seconds, as the event TimeTags).
        pixels : array_like of int or None
            Local frame only: pixels to compute (in ``scheme``) if not computed
            yet; pixels never computed are NaN in the result. None means all
            pixels. The Galactic frame needs every pixel, so pixels must be None.
        scatt_nside : int
            Attitude grouping for the Galactic rotation (see ``galactic_conversion_matrix``).

        Returns
        -------
        histpy.Histogram, shape (n_em_bins, n_phi_bins, npix)
            Axes "Em" [keV], "Phi" [deg], "PsiChi" (NSIDE and scheme of the template,
            spacecraft or Galactic frame): the axes of cosipy's BinnedData after
            ``.project("Em", "Phi", "PsiChi")``.
        """
        if frame not in ("local", "galactic"):
            raise ValueError(f"frame must be 'local' or 'galactic', got {frame!r}.")
        if frame == "galactic" and pixels is not None:
            raise ValueError("The Galactic frame mixes all spacecraft-frame pixels; use pixels=None.")

        needed = self._check_pixels(pixels)                     # None -> all pixels
        missing = needed[np.isnan(self.probability[:, :, needed]).any(axis=(0, 1))]
        if missing.size:
            self.compute(missing)

        history = SpacecraftHistory.open(ori_file).select_interval(
            Time(tmin, format="unix"), Time(tmax, format="unix"))
        rate = self.model_counts / self.model_livetime          # events / s

        if frame == "local":
            livetime = float(history.livetime.to_value(u.s).sum())
            return Histogram(self._axes(SpacecraftFrame()), contents=rate * livetime * self.probability)

        W = galactic_conversion_matrix(history, self.nside, self.nest, scatt_nside)
        counts = rate * (self.probability @ W.T)                # (Em, Phi, local) -> (Em, Phi, Galactic)
        return Histogram(self._axes("galactic"), contents=counts)
