""" "Plotting theme and color system. Colors are assigned by hand per plot
from ``palette.<family>[<shade>]`` (never auto-cycled), so a series keeps
its color across every figure."""

from kutils.style.apply import apply_style
from kutils.style.palettes import (
    check_series_encodings,
    diverging_cmap,
    palette,
    sequential_cmap,
)
from kutils.style.presets import (
    savefig_dual,
    style_bar_plot,
    style_heatmap,
    style_line_plot,
)
from kutils.style.tokens import CATEGORICAL, DIVERGING, SEQUENTIAL, UNSAFE_FAMILY_PAIRS

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
