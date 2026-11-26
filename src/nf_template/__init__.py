"""Binned COSI background templates from an NF background model (version 3).

Integrates the density of a time-independent normalizing-flow background
model over each Compton-data-space bin, instead of sampling events. Uses the
installed cosipy, with Pascal's version-3 NF code (branch scipascal_area-renormalization).

Modules
-------
template
    ``NFBinnedTemplate``: computes P_i, the expected counts, and the binned
    background model for the fit (``get_free_norm_background``).
quadrature
    Integration nodes and weights for Em, Phi and the sky; no NF model inside.
galactic
    ``galactic_conversion_matrix``: spacecraft frame -> Galactic frame.

Example (DC4 511 keV imaging binning, from the repository root)
---------------------------------------------------------------
    from cosipy.spacecraftfile import SpacecraftHistory
    from src.nf_template import NFBinnedTemplate

    template = NFBinnedTemplate("model/nfbackground_part-01_v3-00.fits")
    axes = binned_data.axes        # any histpy Axes with "Em", "Phi", "PsiChi"
    history = SpacecraftHistory.open("data/DC4_orientation.ori")

    print(template.n_evaluations(axes, [255, 444]))   # cost before running
    rows = template.compute(axes, [255, 444])         # P_i of two pixels, shape (1, 60, 2)

    counts = template.get_binned_background(          # expected counts, all pixels
        axes, "galactic", history,
        tmin=1835487300.036206, tmax=1843467254.887687, em_breaks_kev=[511.0])

    background = template.get_free_norm_background(   # cosipy's binned background, for the fit
        axes, "galactic", history,
        tmin=1835487300.036206, tmax=1843467254.887687, em_breaks_kev=[511.0])
"""

from .template import NFBinnedTemplate
from .galactic import galactic_conversion_matrix
from .quadrature import near_pole, ring_from_pole

__all__ = ["NFBinnedTemplate", "galactic_conversion_matrix", "near_pole"]
