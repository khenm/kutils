import re
import warnings

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from kutils.style import (  # noqa: E402 - after importorskip
    apply_style,
    check_series_encodings,
    diverging_cmap,
    palette,
    sequential_cmap,
    style_bar_plot,
    style_heatmap,
    style_line_plot,
)

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_all_tokens_are_valid_hex():
    for family in (palette.blue, palette.green, palette.red, palette.yellow):
        assert len(family) == 4
        for hex_color in family:
            assert HEX_RE.match(hex_color)


def test_family_of_roundtrip():
    assert palette.family_of(palette.blue[2]) == "blue"
    assert palette.family_of(palette.red[0]) == "red"
    assert palette.family_of("#000000") is None


def test_apply_style_is_idempotent():
    apply_style()
    apply_style()
    apply_style(usetex=False, base_fontsize=12.0)


def test_colormaps_registered_and_usable():
    seq = sequential_cmap()
    div = diverging_cmap()
    assert seq(0.0) != seq(1.0)
    assert div(0.0) != div(1.0)
    # calling again returns the same registered cmap (matplotlib hands back a
    # copy on lookup, so compare identity by name + sampled color, not `is`)
    seq_again = sequential_cmap()
    assert seq_again.name == seq.name
    assert seq_again(0.5) == seq(0.5)


def test_check_series_encodings_warns_on_unsafe_pair_without_secondary_encoding():
    series = {"a": palette.red[1], "b": palette.green[2]}
    with pytest.warns(UserWarning, match="color-vision deficiency"):
        check_series_encodings(series)


def test_check_series_encodings_silent_with_distinct_markers():
    series = {"a": palette.red[1], "b": palette.green[2]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_series_encodings(series, markers={"a": "o", "b": "s"})


def test_check_series_encodings_silent_for_safe_pairs():
    series = {"a": palette.blue[1], "b": palette.yellow[2]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_series_encodings(series)


def test_style_helpers_run_against_real_axes():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4], color=palette.blue[3])
    style_line_plot(ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [1, 2], color=[palette.blue[1], palette.yellow[1]])
    style_bar_plot(ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(np.random.rand(4, 4), cmap=sequential_cmap())
    style_heatmap(ax, im, colorbar_label="value")
    plt.close(fig)


def test_savefig_dual(tmp_path):
    import matplotlib.pyplot as plt

    from kutils.style import savefig_dual

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "figs" / "demo"
    savefig_dual(fig, out)
    plt.close(fig)

    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".png").exists()
