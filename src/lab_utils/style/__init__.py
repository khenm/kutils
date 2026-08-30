"""Generic plotting theme and color system for lab_utils.

Usage:

    from lab_utils.style import apply_style, palette

    apply_style()  # set rcParams once, at script start

    COLORS = {
        "ours-small": palette.blue[0],
        "ours-xl": palette.blue[3],
        "baseline": palette.yellow[2],
    }

Color is assigned by hand per plot from ``palette.<family>[<shade>]`` — it
is not auto-cycled — so the same series name gets the same color across
every figure in a paper regardless of what else is in that particular plot.
"""

from lab_utils.style.apply import apply_style
from lab_utils.style.palettes import (
    check_series_encodings,
    diverging_cmap,
    palette,
    sequential_cmap,
)
from lab_utils.style.presets import (
    savefig_dual,
    style_bar_plot,
    style_heatmap,
    style_line_plot,
)
from lab_utils.style.tokens import CATEGORICAL, DIVERGING, SEQUENTIAL, UNSAFE_FAMILY_PAIRS

__all__ = [
    "apply_style",
    "palette",
    "sequential_cmap",
    "diverging_cmap",
    "check_series_encodings",
    "style_line_plot",
    "style_bar_plot",
    "style_heatmap",
    "savefig_dual",
    "CATEGORICAL",
    "SEQUENTIAL",
    "DIVERGING",
    "UNSAFE_FAMILY_PAIRS",
]
