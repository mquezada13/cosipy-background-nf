"""Integration nodes and weights of the binned NF template.

Pure geometry: nothing in this module depends on the NF model. Each function
returns nodes (where the density is evaluated) and weights (the dEm, dphi or
dOmega of the integral), so that

    integral of f over a bin  ~=  sum over nodes of  f(node) * weight

Em and Phi
    Gauss-Legendre nodes in every bin (``binned_gl_nodes``) and the matrix that
    adds weighted node values into bins (``node_to_bin_matrix``).
Sky (PsiChi)
    Centres and areas of HEALPix sub-pixels (``healpix_subpixel_nodes``), with
    an extra refinement close to the poles (``polar_pixel_nodes``).
"""

import numpy as np
from numpy.typing import ArrayLike
import healpy as hp

# ---------------------------------------------------------------------------
# Em and Phi: Gauss-Legendre nodes per bin
# ---------------------------------------------------------------------------


def gl_nodes(
        a: float, 
        b: float, 
        n: int
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes and weights on one interval [a, b].

    Parameters
    ----------
    a, b : float
        Interval limits.
    n : int
        Number of nodes.

    Returns
    -------
    nodes, weights : np.ndarray (n,)
        Node positions and weights; the weights sum to b - a.
    """
    x, w = np.polynomial.legendre.leggauss(n)     # standard nodes and weights on [-1, 1]
    # Map [-1, 1] linearly onto [a, b]
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def binned_gl_nodes(
        edges: ArrayLike, 
        n_per_bin: int,
        breaks: ArrayLike = ()
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes for every bin; bins are split at break points.

    A bin with break points inside is cut into pieces, and each piece gets its
    own ``n_per_bin`` nodes (useful where the density has a kink or a line).

    Parameters
    ----------
    edges : array_like (n_bins + 1,)
        Bin edges.
    n_per_bin : int
        Nodes per bin (per piece, if the bin is split).
    breaks : array_like, optional
        Break points, in the unit of ``edges``.

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
            x, w = gl_nodes(a, b, n_per_bin)
            nodes.append(x)
            weights.append(w)
            bin_index.append(np.full(n_per_bin, i))

    return np.concatenate(nodes), np.concatenate(weights), np.concatenate(bin_index)


def node_to_bin_matrix(
        weights: np.ndarray, 
        bin_index: np.ndarray, 
        n_bins: int
        ) -> np.ndarray:
    """
    Matrix that adds weighted node values into bins.

    ``matrix @ values`` gives, for every bin, the sum of ``weight * value``
    over the nodes of that bin, i.e. the integral over the bin.

    Returns
    -------
    np.ndarray (n_bins, n_nodes)
        Row = bin, column = node, value = the node's weight (0 if the node is
        in another bin).
    """
    matrix = np.zeros((n_bins, len(weights)))
    matrix[bin_index, np.arange(len(weights))] = weights
    return matrix


def as_breaks(
        breaks: ArrayLike | None
        ) -> np.ndarray:
    """
    Break points as a 1D float array (None = no break points).
    """
    return np.atleast_1d(np.asarray([] if breaks is None else breaks, dtype=np.float64))


# ---------------------------------------------------------------------------
# Sky: HEALPix sub-pixel nodes
# ---------------------------------------------------------------------------


def near_pole(
        nside: int, 
        pixels: ArrayLike, 
        nest: bool, 
        cap_rad: float
        ) -> np.ndarray:
    """
    Which pixels have their centre within ``cap_rad`` of a pole.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE.
    pixels : array_like of int
        Pixel indices.
    nest : bool
        True for NESTED ordering, False for RING.
    cap_rad : float
        Angular distance from the nearer pole [rad].

    Returns
    -------
    np.ndarray of bool (n_pixels,)
    """
    colatitude, _ = hp.pix2ang(nside, pixels, nest=nest)
    return np.minimum(colatitude, np.pi - colatitude) < cap_rad   # distance to the nearer pole


def ring_from_pole(
        nside: int,
        pixels: ArrayLike,
        nest: bool
        ) -> np.ndarray:
    """
    HEALPix ring of each pixel, counted from the nearer pole: 1 = the ring touching the pole.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE.
    pixels : array_like of int
        Pixel indices.
    nest : bool
        True for NESTED ordering, False for RING.

    Returns
    -------
    np.ndarray of int (n_pixels,)
        1 to 2 * nside (the equator ring has 2 * nside).
    """
    all_colatitudes = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0]
    ring_colatitudes = np.unique(np.round(all_colatitudes, 12))                   # the 4 * nside - 1 rings, north to south
    colatitude, _ = hp.pix2ang(nside, np.atleast_1d(pixels), nest=nest)
    from_north = np.searchsorted(ring_colatitudes, np.round(colatitude, 12)) + 1  # 1 = northernmost ring
    return np.minimum(from_north, len(ring_colatitudes) + 1 - from_north)


def healpix_subpixel_nodes(
        nside: int, 
        level: int, 
        pixels: ArrayLike,
        nest: bool = True
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sky nodes: centres of the 4**level sub-pixels of each pixel.

    In NESTED numbering, pixel p at ``nside`` has the 4**level children
    p * 4**level, ..., p * 4**level + 4**level - 1 at ``nside * 2**level``,
    and they cover p exactly.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE of the pixels (a power of 2).
    level : int
        Sub-pixel level (0 = the pixel itself).
    pixels : array_like of int
        Pixel indices.
    nest : bool, default True
        Ordering of ``pixels``: True for NESTED, False for RING.

    Returns
    -------
    colatitude, longitude : np.ndarray (n_pixels, 4**level)
        Sub-pixel centres [rad].
    weights : np.ndarray (n_pixels, 4**level)
        Sub-pixel areas [sr]; they sum to the pixel area.
    """
    pixels = np.atleast_1d(np.asarray(pixels, dtype=np.int64))
    parents_nest = pixels if nest else hp.ring2nest(nside, pixels)         # sub-pixels are defined in NESTED numbering

    n_sub = 4**level                                                       # sub-pixels per pixel
    nside_sub = nside * 2**level                                           # finer grid of the sub-pixels
    children = parents_nest[:, None] * n_sub + np.arange(n_sub)[None, :]   # sub-pixel indices, (n_pixels, n_sub)

    colatitude, longitude = hp.pix2ang(nside_sub, children.ravel(), nest=True)   # centre of each sub-pixel
    weights = np.full(children.shape, hp.nside2pixarea(nside_sub))         # all equal in HEALPix
    return colatitude.reshape(children.shape), longitude.reshape(children.shape), weights


def polar_pixel_nodes(
        nside: int, 
        pixel: int, 
        nest: bool, 
        level: int, 
        pole_level: int,
        pole_colat_rad: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sky nodes of one pixel at ``level``, refined to ``pole_level`` close to a pole.

    The pixel is split into its 4**level sub-pixels; those with centre within
    ``pole_colat_rad`` of a pole are replaced by their own children at
    ``pole_level``. Together they still cover the pixel exactly.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE of the pixel (a power of 2).
    pixel : int
        Pixel index.
    nest : bool
        Ordering of ``pixel``: True for NESTED, False for RING.
    level : int
        Sub-pixel level of the pixel.
    pole_level : int
        Sub-pixel level close to the pole (>= ``level``).
    pole_colat_rad : float
        Sub-pixels with centre within this angle of a pole are refined [rad].

    Returns
    -------
    colatitude, longitude, weights : np.ndarray (n_sky,)
        Node positions [rad] and weights [sr]; the weights sum to the pixel area.
    """
    parent_nest = pixel if nest else hp.ring2nest(nside, pixel)
    children = parent_nest * 4**level + np.arange(4**level)                 # NESTED children at nside * 2**level
    close = near_pole(nside * 2**level, children, True, pole_colat_rad)     # sub-pixels close to the pole

    m = 4**(pole_level - level)                                             # children of each of them at pole_level
    fine = (children[close][:, None] * m + np.arange(m)).ravel()

    # Group a: sub-pixels not refined; group b: the fine sub-pixels near the pole
    colat_a, lon_a = hp.pix2ang(nside * 2**level, children[~close], nest=True)
    colat_b, lon_b = hp.pix2ang(nside * 2**pole_level, fine, nest=True)
    weights = np.concatenate([np.full(len(colat_a), hp.nside2pixarea(nside * 2**level)),
                              np.full(len(colat_b), hp.nside2pixarea(nside * 2**pole_level))])
    return np.concatenate([colat_a, colat_b]), np.concatenate([lon_a, lon_b]), weights
