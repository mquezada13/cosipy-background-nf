"""Tests of src/nf_template (run from the repository root: .venv/bin/python -m pytest tests).

The NF model files are ~210 MB, so the template tests use ``AnalyticTemplate``:
the same class with an analytic density instead of the flow. The integral,
binning, chunking and outputs are the real code. Tests that need the
orientation file are skipped if it is missing.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import healpy as hp
import astropy.units as u
import pytest
from histpy import Axes, Axis, HealpixAxis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # so that "src.nf_template" is found
from src.nf_template import NFBinnedTemplate, galactic_conversion_matrix   # noqa: E402
from src.nf_template.quadrature import (gl_nodes, binned_gl_nodes, node_to_bin_matrix,   # noqa: E402
                                        healpix_subpixel_nodes, polar_pixel_nodes, near_pole,
                                        ring_from_pole)

ORI_FILE = ROOT / "data/DC4_orientation.ori"
TMIN = 1835487300.0                              # start of the DC4 orientation file [unix s]


# ---------------------------------------------------------------------------
# A template with an analytic density (no model file)
# ---------------------------------------------------------------------------

class AnalyticTemplate(NFBinnedTemplate):
    """NFBinnedTemplate with ``density(em, phi, chi, psi)`` instead of the NF model."""

    def __init__(self, density, **settings):
        self._analytic_density = density
        self.n_points = 0                        # density evaluations made so far
        super().__init__("no model file", **settings)

    def _load_model(self, model_path):
        self.model_version = "3.00"
        self.model_counts = 1000.0
        self.model_livetime = 100.0
        self.rate = 10.0

    def _evaluate_density(self, em, phi, chi, psi):
        self.n_points += len(em)
        return self._analytic_density(em, phi, chi, psi)


def make_axes(em_edges_kev, phi_edges_deg, nside, scheme="ring"):
    return Axes([Axis(np.asarray(em_edges_kev, dtype=float) * u.keV, label="Em"),
                 Axis(np.asarray(phi_edges_deg, dtype=float) * u.deg, label="Phi"),
                 HealpixAxis(nside=nside, scheme=scheme, label="PsiChi")])


EM_EDGES = [100.0, 200.0, 500.0]
PHI_EDGES = np.arange(0, 181, 30)
E_RANGE = EM_EDGES[-1] - EM_EDGES[0]


def constant_density(em, phi, chi, psi):
    """Normalized constant density over [100, 500] keV x [0, pi] x full sky."""
    return np.full(len(em), 1.0 / (E_RANGE * np.pi * 4 * np.pi))


# Pole settings scaled to NSIDE 4 (its pixel centres are > 4 deg from the poles), small levels for speed
POLE_SETTINGS = dict(polar_rings=1, polar_subpixel_level=2, pole_colat_deg=8.0, pole_subpixel_level=3,
                     near_pole_rings=3, near_pole_subpixel_level=1)


# ---------------------------------------------------------------------------
# Quadrature (pure geometry)
# ---------------------------------------------------------------------------

def test_gl_nodes_integrate_polynomials_exactly():
    # n Gauss-Legendre nodes integrate polynomials of degree <= 2n - 1 exactly
    x, w = gl_nodes(2.0, 5.0, 4)
    assert np.isclose(np.sum(w * x**7), (5.0**8 - 2.0**8) / 8, rtol=1e-13)


def test_binned_gl_weights_sum_to_bin_widths():
    edges = np.array([0.0, 1.0, 3.0])
    nodes, weights, bin_index = binned_gl_nodes(edges, 3, breaks=[2.0])
    assert len(nodes) == 3 * 3                                   # bin 1 is split at 2.0: two pieces
    matrix = node_to_bin_matrix(weights, bin_index, 2)
    assert np.allclose(matrix.sum(axis=1), np.diff(edges))
    assert np.all((nodes[bin_index == 1] > 1.0) & (nodes[bin_index == 1] < 3.0))


@pytest.mark.parametrize("nest", [True, False])
def test_subpixel_weights_cover_the_sky(nest):
    nside, level = 4, 2
    pixels = np.arange(hp.nside2npix(nside))
    _, _, weights = healpix_subpixel_nodes(nside, level, pixels, nest=nest)
    assert np.allclose(weights.sum(axis=1), hp.nside2pixarea(nside))
    assert np.isclose(weights.sum(), 4 * np.pi)


def test_ring_from_pole():
    # 1 = the ring touching the nearer pole (4 pixels at each pole), up to 2 * nside at the equator
    for nside in (1, 2, 4, 8, 16):
        for nest in (False, True):
            pixels = np.arange(hp.nside2npix(nside))
            ring = ring_from_pole(nside, pixels, nest)
            colatitude = hp.pix2ang(nside, pixels, nest=nest)[0]
            to_pole = np.minimum(colatitude, np.pi - colatitude)
            assert ring.min() == 1 and ring.max() == 2 * nside and np.sum(ring == 1) == 8
            assert all(np.ptp(to_pole[ring == r]) < 1e-9 for r in np.unique(ring))   # one distance per ring
            assert np.all(np.diff(ring[np.argsort(to_pole)]) >= 0)                   # rings ordered by distance


def test_rings_select_the_511_kev_pixels():
    # At NSIDE 16, 1 and 3 rings are exactly the former angles: centre within 4 and 10 deg of a pole
    pixels = np.arange(hp.nside2npix(16))
    ring = ring_from_pole(16, pixels, False)
    assert np.array_equal(ring <= 1, near_pole(16, pixels, False, np.deg2rad(4.0)))
    assert np.array_equal(ring <= 3, near_pole(16, pixels, False, np.deg2rad(10.0)))
    levels = AnalyticTemplate(constant_density).subpixel_levels(16, pixels, False, 2)          # defaults
    assert np.array_equal(levels, np.where(ring <= 1, 5, np.where(ring <= 3, 3, 2)))
    off = AnalyticTemplate(constant_density, polar_rings=0, near_pole_rings=0)                 # corrections off
    assert np.all(off.subpixel_levels(16, pixels, False, 2) == 2)


def test_polar_nodes_cover_the_pixel():
    nside = 16
    polar = np.flatnonzero(near_pole(nside, np.arange(hp.nside2npix(nside)), False, np.deg2rad(4.0)))
    assert len(polar) == 8
    for pixel in polar:
        _, _, weights = polar_pixel_nodes(nside, pixel, False, 5, 7, np.deg2rad(1.0))
        assert np.isclose(weights.sum(), hp.nside2pixarea(nside), rtol=1e-12)


# ---------------------------------------------------------------------------
# The integral
# ---------------------------------------------------------------------------

def test_constant_density_integrates_to_one():
    template = AnalyticTemplate(constant_density, **POLE_SETTINGS)
    axes = make_axes(EM_EDGES, PHI_EDGES, nside=4)
    P = template.compute(axes, subpixel_level=1)

    # Each bin: density x dEm x dphi x pixel area; all bins together: 1
    bin_volume = (np.diff(EM_EDGES)[:, None, None] * np.deg2rad(np.diff(PHI_EDGES))[None, :, None]
                  * hp.nside2pixarea(4))
    assert np.allclose(P, bin_volume / (E_RANGE * np.pi * 4 * np.pi), rtol=1e-12)
    assert np.isclose(P.sum(), 1.0, rtol=1e-12)


def test_polynomial_in_energy_is_exact():
    # Density proportional to Em**3: 4 GL nodes per bin integrate it exactly
    template = AnalyticTemplate(lambda em, phi, chi, psi: em**3)
    axes = make_axes(EM_EDGES, [0, 180], nside=2)
    P = template.compute(axes, subpixel_level=0, n_em_per_bin=2, n_phi_per_bin=1)
    expected = np.diff(np.asarray(EM_EDGES)**4) / 4 * np.pi * hp.nside2pixarea(2)
    assert np.allclose(P[:, 0, :], expected[:, None], rtol=1e-12)


def test_n_evaluations_matches_compute():
    template = AnalyticTemplate(constant_density, **POLE_SETTINGS)
    axes = make_axes(EM_EDGES, PHI_EDGES, nside=4)
    pixels = np.arange(hp.nside2npix(4))
    planned = template.n_evaluations(axes, pixels, subpixel_level=1, em_breaks_kev=[150.0])
    template.compute(axes, pixels, subpixel_level=1, em_breaks_kev=[150.0], points_per_chunk=5_000)
    assert template.n_points == planned
    assert set(np.unique(template.subpixel_level_used)) == {1, 2}    # normal/near-pole level 1, polar level 2


def test_chunk_size_does_not_change_the_result():
    template = AnalyticTemplate(lambda em, phi, chi, psi: np.cos(psi)**2 + 0.1 * np.sin(chi) + em / 1000,
                                **POLE_SETTINGS)
    axes = make_axes(EM_EDGES, PHI_EDGES, nside=4)
    small = template.compute(axes, subpixel_level=1, points_per_chunk=1_000)
    large = template.compute(axes, subpixel_level=1, points_per_chunk=10_000_000)
    assert np.allclose(small, large, rtol=1e-12)


def test_changed_binning_discards_results(caplog):
    template = AnalyticTemplate(constant_density)
    template.compute(make_axes(EM_EDGES, PHI_EDGES, nside=4), [0, 1])
    with caplog.at_level(logging.WARNING):
        template.compute(make_axes(EM_EDGES, PHI_EDGES, nside=2), [0])
    assert "discarded" in caplog.text
    assert np.sum(template.subpixel_level_used >= 0) == 1


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def test_input_checks():
    template = AnalyticTemplate(constant_density)
    axes = make_axes(EM_EDGES, PHI_EDGES, nside=4)
    with pytest.raises(ValueError):
        template.compute(axes, subpixel_level=-1)
    with pytest.raises(ValueError):
        template.compute(axes, pixels=[hp.nside2npix(4)])            # pixel out of range
    with pytest.raises(ValueError):
        template.compute(axes, pixels=[3, 3])                         # repeated pixel
    with pytest.raises(ValueError):
        template.compute(make_axes(EM_EDGES, [0, 200], nside=4))      # Phi beyond 180 deg
    with pytest.raises(ValueError):
        template.get_binned_background(axes, "galactic", "unused.ori", pixels=[0])
    with pytest.raises(ValueError):
        template.get_binned_background(axes, "icrs", "unused.ori")
    with pytest.raises(ValueError):
        AnalyticTemplate(constant_density, seam_delta_deg=2.0)
    with pytest.raises(RuntimeError):
        AnalyticTemplate(constant_density).to_histogram()             # no binning yet


# ---------------------------------------------------------------------------
# Outputs that need the orientation file
# ---------------------------------------------------------------------------

needs_ori = pytest.mark.skipif(not ORI_FILE.is_file(), reason="orientation file not available")


@pytest.fixture(scope="module")
def history():
    """One day of the DC4 orientation file."""
    from src.nf_template.template import _open_history
    return _open_history(ORI_FILE, TMIN, TMIN + 86400.0)


@needs_ori
def test_galactic_matrix_rows_and_sparse(history):
    livetime = history.livetime.to_value(u.s).sum()
    dense = galactic_conversion_matrix(history, 4, nest=False)
    sparse = galactic_conversion_matrix(history, 4, nest=False, sparse=True)
    assert np.allclose(dense.sum(axis=1), livetime, rtol=1e-10)
    assert np.allclose(sparse.toarray(), dense, rtol=1e-12, atol=0)


@needs_ori
@pytest.mark.parametrize("frame", ["local", "galactic"])
def test_free_norm_background_matches_counts(history, frame):
    template = AnalyticTemplate(lambda em, phi, chi, psi: 1.0 + np.cos(psi)**2, **POLE_SETTINGS)
    axes = make_axes(EM_EDGES, PHI_EDGES, nside=4)
    counts = template.get_binned_background(axes, frame, history)
    background = template.get_free_norm_background(axes, frame, history)
    assert np.allclose(background.expectation().contents, counts.contents, rtol=1e-12)
    if frame == "local":
        livetime = history.livetime.to_value(u.s).sum()
        assert np.allclose(counts.contents, template.rate * livetime * template.probability, rtol=1e-14)
