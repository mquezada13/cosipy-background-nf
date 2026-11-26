"""Helpers for comparing a binned background template with binned events (notebook 10).

Residuals are r_i = (N_i - mu_i) / sqrt(mu_i), with N_i the observed counts (simulation or NF samples)
and mu_i the template (expected counts). Pure Poisson noise gives mean 0, spread 1, |r| > 2 in 4.6%.
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import COLORS

# Legend labels of the residual-class plots (the class default says "Observed" / "Fitted expectation")
RA_LABELS = {"Observed": "simulation", "Fitted expectation": "FINAL template"}


# ------------------------------------------------------------------ statistics

def chi2_text(res):
    """chi2/dof = mean of r_i^2 (nothing fitted, so dof = n); noise only gives 1 +- sqrt(2/n)."""
    r = np.asarray(res).ravel()
    n = r.size
    return f"chi2/dof {np.mean(r**2):.3f} (noise only: 1 ± {np.sqrt(2 / n):.3f}, {n} values)"


def residual_summary(res):
    """Mean, spread, share beyond +-2 and chi2/dof of residuals. Noise only: 0, 1, 4.6%, 1."""
    r = np.asarray(res).ravel()
    return f"mean {r.mean():+.3f}, std {r.std():.3f}, |r| > 2 in {np.mean(np.abs(r) > 2):5.1%}, {chi2_text(r)}"


def stats_table(analyses, labels):
    """Pearson statistics of the residual class, one row per entry of `analyses` ({key: ResidualAnalysis})."""
    return pd.concat({labels[key]: ra.statistics_table("pearson") for key, ra in analyses.items()}).droplevel(1)


# ------------------------------------------------------------------ sky geometry

def psi_rings(nside):
    """
    HEALPix rings of equal colatitude (psi in the local frame, 90 deg - b in the Galactic frame).

    Returns ring_psi (colatitude of each ring [deg]), ring_index (ring of each pixel),
    n_pix_ring (pixels per ring) and psi_edges (ring boundaries [deg], for plotting).
    """
    psi_pix = np.rad2deg(hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0])
    ring_psi, ring_index = np.unique(psi_pix.round(6), return_inverse=True)
    n_pix_ring = np.bincount(ring_index)
    psi_edges = np.concatenate([[0], 0.5 * (ring_psi[1:] + ring_psi[:-1]), [180]])
    return ring_psi, ring_index, n_pix_ring, psi_edges


def chi_line_pixels(nside, chi_deg=(0, 90, 180, 270)):
    """Boolean mask of the pixels crossed by the lines of constant chi (RING ordering)."""
    psi_line = np.linspace(0, np.pi, 5000)
    pixels = np.unique(np.concatenate([hp.ang2pix(nside, psi_line, np.full_like(psi_line, np.deg2rad(chi)))
                                       for chi in chi_deg]))
    mask = np.zeros(hp.nside2npix(nside), bool)
    mask[pixels] = True
    return mask


def phi_ring(counts, ring_index):
    """Regroup counts (Em, Phi, PsiChi) into (Phi, psi ring): sum over Em and over the pixels of each ring."""
    counts_phi_pix = counts.sum(axis=0)
    n_rings = ring_index.max() + 1
    return np.array([np.bincount(ring_index, weights=row, minlength=n_rings) for row in counts_phi_pix])


def smooth(m, fwhm_deg, nest=False):
    """Gaussian smoothing of a HEALPix map (healpy smooths in RING ordering)."""
    m_ring = hp.reorder(m, n2r=True) if nest else m
    m_smooth = hp.smoothing(m_ring, fwhm=np.deg2rad(fwhm_deg))
    return hp.reorder(m_smooth, r2n=True) if nest else m_smooth


# ------------------------------------------------------------------ plotting

def ref_lines(ax):
    """Reference lines on a residual panel: 0 (solid) and +-2 (dotted)."""
    for y, ls in ((0, "-"), (2, ":"), (-2, ":")):
        ax.axhline(y, color=COLORS["ref"], ls=ls, lw=0.8)


def tidy_legends(figure, fontsize=8, rename=None):
    """Legends of a PlottingResiduals figure: smaller, white frame, fixed corner; rename = {old label: new label}."""
    for ax in figure.axes:
        if ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            if rename:
                labels = [rename.get(label, label) for label in labels]
            corner = "upper left" if ax.get_ylabel() == "Probability density" else "upper right"
            ax.legend(handles, labels, fontsize=fontsize, loc=corner, framealpha=0.9)
    if figure._suptitle is not None:
        figure._suptitle.set_fontsize(16)
    return figure


def in_grid(make_figure, analyses, labels, plotter, n_cols=2, width=22, rename=RA_LABELS):
    """
    One residual figure per entry of `analyses`, placed in a grid with n_cols per row.

    plotter(analysis) builds the plotting object (e.g. PlottingResiduals); make_figure(plotting, name)
    draws one figure with show=False and returns it.
    """
    images = []
    for key, analysis in analyses.items():
        f = tidy_legends(make_figure(plotter(analysis), labels[key]), rename=rename)
        f.canvas.draw()
        images.append(np.asarray(f.canvas.buffer_rgba()))
        plt.close(f)
    n_rows = -(-len(images) // n_cols)
    height_per_row = width / n_cols * images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height_per_row * n_rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, image in zip(np.ravel(axes), images):
        ax.imshow(image)
    fig.tight_layout()
    plt.show()
