"""Per-plot-type styling helpers and figure export.

These are small, optional conveniences layered on top of ``apply_style()``
— call them on an already-styled ``Axes``/``Figure`` to handle the bits
that differ by plot type (heatmaps need a colorbar and no grid; bar charts
want the grid behind horizontal gridlines only; etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def style_line_plot(ax: Any) -> None:
    """Apply line/curve-plot conventions: light grid, ticks pointing out."""
    ax.grid(True, which="major", axis="both")
    ax.tick_params(direction="out")


def style_bar_plot(ax: Any) -> None:
    """Apply bar-chart conventions: horizontal-only grid behind the bars."""
    ax.grid(True, which="major", axis="y")
    ax.grid(False, which="major", axis="x")
    ax.set_axisbelow(True)


def style_heatmap(ax: Any, image: Any, *, colorbar_label: str | None = None) -> None:
    """Apply heatmap conventions: no grid, attached colorbar."""
    ax.grid(False)
    fig = ax.get_figure()
    cbar = fig.colorbar(image, ax=ax)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)


def savefig_dual(fig: Any, path: str | Path, *, cache_data: dict[str, Any] | None = None) -> None:
    """Save a figure as both .pdf (for the paper) and .png (for quick viewing),
    and optionally cache the underlying array data alongside via
    ``lab_utils.utils.cache`` so the figure can be rebuilt without rerunning
    the analysis that produced it.

    ``path`` should be given without suffix, e.g. ``results/figs/loss_curve``;
    both ``results/figs/loss_curve.pdf`` and ``.png`` are written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))

    if cache_data is not None:
        from lab_utils.utils.cache import save_artifact

        save_artifact(path.with_suffix(".npz"), cache_data)
