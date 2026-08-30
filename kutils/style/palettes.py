"""Palette access, colormap construction, and safety checks.

Color is assigned by hand, per plot, from ``palette.<family>[<shade>]`` —
never auto-cycled from ``axes.prop_cycle``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from matplotlib.colors import Colormap, LinearSegmentedColormap

from kutils.style.tokens import (
    CATEGORICAL,
    DIVERGING,
    SEQUENTIAL,
    UNSAFE_FAMILY_PAIRS,
)

SEQUENTIAL_CMAP_NAME = "lab_sequential"
DIVERGING_CMAP_NAME = "lab_diverging"


@dataclass(frozen=True)
class _CategoricalPalette:
    blue: tuple[str, str, str, str]
    green: tuple[str, str, str, str]
    red: tuple[str, str, str, str]
    yellow: tuple[str, str, str, str]

    def family_of(self, hex_color: str) -> str | None:
        """Return which family a hex color belongs to, if any."""
        hex_color = hex_color.lower()
        for name, shades in (
            ("blue", self.blue),
            ("green", self.green),
            ("red", self.red),
            ("yellow", self.yellow),
        ):
            if hex_color in {s.lower() for s in shades}:
                return name
        return None


palette = _CategoricalPalette(
    blue=CATEGORICAL["blue"],
    green=CATEGORICAL["green"],
    red=CATEGORICAL["red"],
    yellow=CATEGORICAL["yellow"],
)


def sequential_cmap() -> Colormap:
    """Return the sequential (magnitude-only) colormap, building it if needed."""
    return _get_or_register_cmap(SEQUENTIAL_CMAP_NAME, SEQUENTIAL)


def diverging_cmap() -> Colormap:
    """Return the diverging (signed) colormap, building it if needed."""
    return _get_or_register_cmap(DIVERGING_CMAP_NAME, DIVERGING)


def _get_or_register_cmap(name: str, colors: tuple[str, ...]) -> Colormap:
    import matplotlib as mpl

    try:
        return mpl.colormaps[name]
    except KeyError:
        cmap = LinearSegmentedColormap.from_list(name, colors, N=256)
        mpl.colormaps.register(cmap, name=name)
        return cmap


def check_series_encodings(
    series: dict[str, str],
    *,
    markers: dict[str, object] | None = None,
    linestyles: dict[str, object] | None = None,
) -> None:
    """Warn when series from an unsafe family pair (red/green) appear in
    one figure without a distinct marker/linestyle. Call once per figure."""
    families = {name: palette.family_of(color) for name, color in series.items()}
    for fam_a, fam_b in UNSAFE_FAMILY_PAIRS:
        names_a = [n for n, f in families.items() if f == fam_a]
        names_b = [n for n, f in families.items() if f == fam_b]
        if not names_a or not names_b:
            continue
        for name_a in names_a:
            for name_b in names_b:
                if _has_secondary_encoding(name_a, name_b, markers, linestyles):
                    continue
                warnings.warn(
                    f"'{name_a}' ({fam_a}) and '{name_b}' ({fam_b}) are colored from a "
                    f"family pair that collapses under red-green color-vision deficiency, "
                    f"and no distinct marker or linestyle was given to tell them apart. "
                    f"Pass distinguishing `markers=` or `linestyles=` for this pair, or "
                    f"pick different families.",
                    stacklevel=2,
                )


def _has_secondary_encoding(
    name_a: str,
    name_b: str,
    markers: dict[str, object] | None,
    linestyles: dict[str, object] | None,
) -> bool:
    if markers and name_a in markers and name_b in markers and markers[name_a] != markers[name_b]:
        return True
    return bool(
        linestyles
        and name_a in linestyles
        and name_b in linestyles
        and linestyles[name_a] != linestyles[name_b]
    )
