"""Spacecraft frame -> Galactic frame for binned templates.

A template that is constant in the spacecraft frame is seen in the Galactic
frame through the attitude history. Only the PsiChi direction rotates; Em and
Phi do not depend on the frame.
"""

import numpy as np
import healpy as hp
import astropy.units as u
from scipy.sparse import coo_array, csr_array
from cosipy.spacecraftfile import SpacecraftHistory     # cosipy's spacecraft_file.py


def galactic_conversion_matrix(
        history: SpacecraftHistory,
        nside: int,
        nest: bool,
        scatt_nside: int = 32,
        attitudes_per_chunk: int = 512,
        sparse: bool = False
        ) -> np.ndarray | csr_array:
    """Live time each Galactic pixel spends in each spacecraft-frame pixel.

    W[g, l] is the live time [s] during which the centre of Galactic pixel g
    lies in spacecraft-frame pixel l. For a spacecraft-frame rate r [1/s], the
    Galactic counts are ``W @ r``. Same nearest-pixel method as cosipy's
    Galactic responses.

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
    sparse : bool, default False
        False: dense array (fine up to NSIDE 16, ~75 MB). True: scipy sparse
        CSR array, needed at higher NSIDE (dense would be ~1.2 GB at NSIDE 32).

    Returns
    -------
    np.ndarray or scipy.sparse.csr_array (npix, npix)
        W [s]; each row sums to the total live time.
    """
    # Attitudes grouped on a HEALPix grid, each with its live time
    scatt = history.get_scatt_map(nside=scatt_nside, target_coord=None, earth_occ=False)   # cosipy's spacecraft_file.py
    livetime = np.asarray(scatt.weights.to_value(u.s), dtype=np.float64)
    rotation = scatt.attitudes.rot.inv().as_matrix()            # (n_att, 3, 3): Galactic -> spacecraft frame

    npix = hp.nside2npix(nside)
    galactic_pix = np.arange(npix)
    galactic_dirs = np.array(hp.pix2vec(nside, galactic_pix, nest=nest))   # (3, npix), Galactic Cartesian

    W = csr_array((npix, npix)) if sparse else np.zeros(npix * npix)

    # For each attitude: rotate every Galactic pixel centre into the spacecraft frame,
    # find the local pixel it lands in, and add the attitude's live time to W[g, l]
    for a in range(0, len(rotation), attitudes_per_chunk):
        local_dirs = np.einsum("cij,jp->cip", rotation[a:a + attitudes_per_chunk], galactic_dirs)
        local_pix = hp.vec2pix(nside, local_dirs[:, 0], local_dirs[:, 1], local_dirs[:, 2], nest=nest)   # (n_att, npix)
        chunk_livetime = np.repeat(livetime[a:a + attitudes_per_chunk], npix)   # one value per (attitude, Galactic pixel)

        if sparse:
            # (row, column, value) triplets; entries with the same (row, column) are added up
            rows = np.broadcast_to(galactic_pix, local_pix.shape).ravel()
            W = W + coo_array((chunk_livetime, (rows, local_pix.ravel())), shape=(npix, npix)).tocsr()
        else:
            W += np.bincount((galactic_pix * npix + local_pix).ravel(), weights=chunk_livetime,
                             minlength=npix * npix)

    return W if sparse else W.reshape(npix, npix)
