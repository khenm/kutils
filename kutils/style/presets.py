"""Per-plot-type styling helpers and figure export."""

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
    """Save a figure as `path.pdf` + `path.png` (no suffix in `path`),
    and optionally cache the data via `kutils.utils.cache`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))

    if cache_data is not None:
        from kutils.utils.cache import save_artifact

        save_artifact(path.with_suffix(".npz"), cache_data)
