"""Project-wide matplotlib style for the COSI NF-background notebooks.

One call styles every plot that follows: serif Computer Modern text, thick
black frame, inward major and minor ticks on all four sides, large labels,
frameless legends, and the project color palette (NF purple first).

Usage, in the imports cell of a notebook::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))     # project root, so `src` is importable
    from src.plot_style import COLORS, set_global_style
    set_global_style()

Call ``set_global_style()`` after importing cosipy: importing cosipy pulls in a
library that installs its own matplotlib style, which would otherwise win.

Optional helpers for a single axis:
    apply_standard_formatting(ax, ...)   log scales with log minor ticks, limits, labels, title
    apply_ordered_legend(ax, ...)        legend without duplicate labels, lines before markers
"""

import numpy as np
import matplotlib
from cycler import cycler
from matplotlib import font_manager
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# Color cycle of the plots made so far (NF purple first).
PALETTE = ["#9b59b6", "#2ecc71", "#34495e", "#95a5a6", "#e74c3c", "#3498db"]

# Fixed roles, so the same quantity has the same color in every notebook.
COLORS = {
    "nf": PALETTE[0],      # NF model / integral
    "sim": "black",        # simulated events
    "nf_samples": PALETTE[4],  # events sampled from the NF (red)
    "ref": "gray",         # reference lines (0, +-2 sigma, ...)
}

# Font sizes of the style.
LABEL_SIZE = 20
XLABEL_SIZE = 23           # used by apply_standard_formatting for x labels
TICK_LABEL_SIZE = 19
TITLE_SIZE = 17
LEGEND_SIZE = 13

_CM_FONT = font_manager.FontProperties(
    fname=matplotlib.get_data_path() + "/fonts/ttf/cmr10.ttf"
)


def set_global_style():
    """Apply the project style to all following plots (idempotent)."""
    matplotlib.style.use("default")          # start clean, independent of what was imported before
    tick = {"direction": "in", "labelsize": TICK_LABEL_SIZE,
            "major.size": 8, "major.width": 1.5, "minor.size": 4, "minor.width": 1.2,
            "minor.visible": True}
    rc = {
        # text
        "font.family": "serif",
        "font.serif": [_CM_FONT.get_name()],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,         # cmr10 has no Unicode minus glyph
        "axes.formatter.use_mathtext": False,
        "text.usetex": False,
        # frame
        "axes.linewidth": 1.5,
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        # sizes
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": False,
        # colors and lines
        "axes.prop_cycle": cycler(color=PALETTE),
        "lines.linewidth": 1.5,
        "patch.linewidth": 1.5,
        # output
        "figure.dpi": 100,
        "pdf.fonttype": 42,
        # ticks on all four sides
        "xtick.top": True,
        "ytick.right": True,
    }
    rc.update({f"xtick.{k}": v for k, v in tick.items()})
    rc.update({f"ytick.{k}": v for k, v in tick.items()})
    matplotlib.rcParams.update(rc)


def apply_standard_formatting(ax, xlog=False, ylog=False, xlim=None, ylim=None,
                              xlabel=None, ylabel=None, title=None):
    """Per-axis extras on top of set_global_style: log axes, limits, labels, title."""
    for axis, log, set_scale in ((ax.xaxis, xlog, ax.set_xscale), (ax.yaxis, ylog, ax.set_yscale)):
        if log:
            set_scale("log")
            axis.set_major_locator(LogLocator(base=10.0))
            axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
            axis.set_minor_formatter(NullFormatter())
        else:
            axis.set_minor_locator(AutoMinorLocator())

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=XLABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE)


def apply_ordered_legend(ax, loc="best", ncol=1):
    """Frameless legend without duplicate labels; lines and steps before markers."""
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    lines, marks = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        is_line = (isinstance(h, Patch)
                   or (isinstance(h, Line2D) and h.get_marker() in ("None", None, "")))
        (lines if is_line and not isinstance(h, ErrorbarContainer) else marks).append((h, l))
    ordered = lines + marks
    ax.legend([h for h, _ in ordered], [l for _, l in ordered],
              fontsize=LEGEND_SIZE, frameon=False, loc=loc, ncol=ncol)
