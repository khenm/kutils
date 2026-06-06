import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_loss_curve(
    steps: list[int],
    losses: list[float],
    title: str = "Training Loss",
    xlabel: str = "Step",
    ylabel: str = "Loss",
    smooth_window: int = 1,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 4))

    if smooth_window > 1:
        smoothed = np.convolve(losses, np.ones(smooth_window) / smooth_window, mode="valid")
        ax.plot(steps[smooth_window - 1:], smoothed, linewidth=1, label="Smoothed")

    ax.plot(steps, losses, alpha=0.3, linewidth=0.5, label="Raw")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def log_figure_to_wandb(wandb_run, fig: Image.Image, name: str, step: int) -> None:
    import wandb
    wandb_run.log({name: wandb.Image(fig)}, step=step)
