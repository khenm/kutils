"""Raw color tokens for the lab plotting theme.

These are the only hardcoded hex values in the style system — everything
else in :mod:`lab_utils.style` is derived from this file. Do not import
these directly in plotting code; go through :mod:`lab_utils.style.palettes`
instead, so color usage stays centralized and swappable.

Verification
------------
The categorical, sequential, and diverging sets below were checked against
simulated color-vision deficiency (deuteranomaly, protanomaly, tritanomaly
at severity=100, via the ``colorspacious`` package) by measuring pairwise
CIE76 (dE) distances in CIELab space. Rule of thumb used: dE < 10 is risky,
dE > 15 is comfortable.

Results:

- Sequential (9-step) and diverging (11-step) ramps: safe under all three
  simulated CVD types. Worst adjacent-step dE stayed >= 3.7 (tritanomaly on
  the sequential ramp) with L* strictly monotonic in every condition, and
  the diverging ramp's two arms stay well separated from its center
  (dE > 60 in every condition).
- Categorical (4 families x 4 shades): blue and yellow are safe against
  every other family under every simulated condition. The red and green
  families collide under deuteranomaly and protanomaly specifically —
  worst case dE ~= 3.46 (green shade 3 vs. red shade 2, protanomaly) — this
  is the well-known red/green hue collapse for red-green color vision
  deficiency, not a defect specific to these hex values. See
  ``UNSAFE_FAMILY_PAIRS`` below and ``palettes.check_series_encodings``,
  which is the mitigation: never rely on color alone to distinguish a
  red-family series from a green-family series in the same figure — pair
  them with distinct markers or linestyles.
"""

from __future__ import annotations

# Categorical: 4 hue families x 4 shades. Shade index 0 -> 3 is light -> dark;
# by convention this maps small -> large model/setting scale, but that
# mapping is a convention for callers to apply, not enforced here.
CATEGORICAL: dict[str, tuple[str, str, str, str]] = {
    "blue": ("#B9DDF1", "#65ADDB", "#2878B8", "#174A7E"),
    "green": ("#BFE5C8", "#72C283", "#359653", "#176638"),
    "red": ("#F5C1BE", "#EC8179", "#CF4540", "#912D2A"),
    "yellow": ("#F8E7A1", "#F2C94C", "#D99B24", "#9B6818"),
}

# Family pairs known to collide under simulated color-vision deficiency.
# check_series_encodings() uses this to warn when both appear in one figure
# without a secondary (marker/linestyle) encoding.
UNSAFE_FAMILY_PAIRS: tuple[tuple[str, str], ...] = (("red", "green"),)

# Sequential: single-hue-ish ramp, light -> dark, for magnitude-only heatmaps
# and colorbars where there's no meaningful zero/center point.
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

# Diverging: symmetric around a near-white center, for signed quantities
# (residuals, correlations, differences) where the sign matters.
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
