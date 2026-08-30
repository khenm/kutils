""" "Global matplotlib style application (idempotent)."""

from __future__ import annotations

import matplotlib as mpl

from kutils.style.palettes import diverging_cmap, sequential_cmap

_APPLIED = False


def apply_style(*, usetex: bool = False, base_fontsize: float = 11.0) -> None:
    """Set global rcParams for lab figures.

    `usetex=True` renders with a real LaTeX install (slower, needs `latex`);
    the default uses mathtext. Sizes derive from `base_fontsize`.
    """
    global _APPLIED

    # Register the custom colormaps first so "lab_sequential" is a valid
    # name by the time it's used as the default image.cmap below.
    sequential_cmap()
    diverging_cmap()

    mpl.rcParams.update(
        {
            "text.usetex": usetex,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
            "axes.formatter.use_mathtext": True,
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize * 1.1,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "image.cmap": "lab_sequential",
        }
    )

    _APPLIED = True
