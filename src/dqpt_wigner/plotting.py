"""Publication-oriented plotting utilities.

The style requests Times New Roman and falls back to a metrically compatible
Times-style serif when that font is not installed on the host system.
"""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager


_TIMES_CANDIDATES = (
    "Times New Roman",
    "Times",
    "Tinos",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "Liberation Serif",
    "DejaVu Serif",
)


def _resolve_times_family() -> str:
    """Return the first installed Times-compatible family.

    The resolver checks ``Times New Roman`` first.  Therefore a user who has
    the Microsoft font installed obtains it automatically.  Tinos is a
    metrically compatible fallback used on many Linux systems.
    """
    for family in _TIMES_CANDIDATES:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Serif"


SELECTED_SERIF_FONT = _resolve_times_family()


def configure_publication_style() -> str:
    """Apply one consistent PRL-oriented plotting style and return its font."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [SELECTED_SERIF_FONT],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "figure.titlesize": 10.0,
            "mathtext.fontset": "stix",
            "mathtext.rm": "STIXGeneral",
            "mathtext.it": "STIXGeneral:italic",
            "mathtext.bf": "STIXGeneral:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
        }
    )
    return SELECTED_SERIF_FONT


configure_publication_style()
