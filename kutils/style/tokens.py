"""Raw color tokens for the lab plotting theme."""

from __future__ import annotations

# 4 hue families x 4 shades
CATEGORICAL: dict[str, tuple[str, str, str, str]] = {
    "blue": ("#B9DDF1", "#65ADDB", "#2878B8", "#174A7E"),
    "green": ("#BFE5C8", "#72C283", "#359653", "#176638"),
    "red": ("#F5C1BE", "#EC8179", "#CF4540", "#912D2A"),
    "yellow": ("#F8E7A1", "#F2C94C", "#D99B24", "#9B6818"),
}

UNSAFE_FAMILY_PAIRS: tuple[tuple[str, str], ...] = (("red", "green"),)

# sequential heatmap colors
SEQUENTIAL: tuple[str, ...] = (
    "#F8E7A1",
    "#E8DEA6",
    "#D2D5AC",
    "#B7CBB2",
    "#9ABEB7",
    "#79ADBA",
    "#5894AE",
    "#356F98",
    "#174A7E",
)

# diverging heatmap colors
DIVERGING: tuple[str, ...] = (
    "#174A7E",
    "#2878B8",
    "#65ADDB",
    "#B9DDF1",
    "#DCECF3",
    "#F7F7F4",
    "#F5E1DE",
    "#F5C1BE",
    "#EC8179",
    "#CF4540",
    "#912D2A",
)
