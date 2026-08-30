"""Global matplotlib style application.

Call ``apply_style()`` once, at the top of a plotting script, before
creating any figures. It sets rcParams for the whole process; it does not
need to be called per-figure and calling it more than once is harmless
(idempotent).
"""

from __future__ import annotations

import matplotlib as mpl

from lab_utils.style.palettes import diverging_cmap, sequential_cmap

_APPLIED = False


def apply_style(*, usetex: bool = False, base_fontsize: float = 11.0) -> None:
    """Set global rcParams for lab figures.

    Parameters
    ----------
    usetex:
        If True, render text with a real LaTeX installation
        (``text.usetex = True``). Slower and requires a working ``latex``
        binary on PATH. If False (default), uses matplotlib's built-in
        ``mathtext`` renderer, which supports the ``$...$`` math subset
        used in almost all figure labels/legends without external
        dependencies — good enough for equations like axis labels
        (``$\\mathcal{L}$``, ``$\\theta$``) that don't need custom LaTeX
        packages. Switch to ``usetex=True`` only if a figure needs a
        specific LaTeX package or font not covered by mathtext.
    base_fontsize:
        Base font size in points; axis/tick/legend sizes are derived from
        this so a figure's type stays internally consistent when you bump
        it up for a poster or down for a dense multi-panel figure.
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
