""" "Centralized logging: loguru setup, a tqdm-safe wrapper, timing utils,
and uniform experiment states."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from enum import StrEnum
from functools import partial

from loguru import logger
from tqdm import tqdm as tqdm_original


class ExperimentState(StrEnum):
    """Standard experiment states for uniform logging."""

    INITIALIZING = "Initializing"
    PARSING_ARGS = "Parsing arguments"
    SETTING_UP = "Setting up environment"
    LOADING_DATA = "Loading data"
    GENERATING_DATA = "Generating synthetic data"
    COMPUTING = "Computing"
    TRAINING = "Training"
    VALIDATING = "Validating"
    AGGREGATING = "Aggregating results"
    SAVING_RESULTS = "Saving results"
    ANALYZING = "Analyzing results"
    GENERATING_PLOTS = "Generating plots"
    COMPLETED = "Completed"
    FAILED = "Failed"
    SKIPPED = "Skipped"


def setup_experiment_logging(name: str, level: str = "INFO") -> None:
    """Configure loguru for console-only logging (call once at start)."""
    # Remove default handler to avoid duplicates
    logger.remove()

    # Console handler with color and clean format
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    logger.debug(f"Logging configured for experiment: {name}")


def log_config(config: dict) -> None:
    """Log a config dict as indented key: value lines."""
    logger.info("Experiment config:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def get_loguru_safe_tqdm():
    """A tqdm wrapper that writes to stderr, so progress bars don't clash
    with loguru's console handler."""
    return partial(tqdm_original, file=sys.stderr)


@contextmanager
def log_timing(operation: str, level: str = "INFO"):
    """Log "Starting/Completed (took Ys)" around a block; re-raises
    exceptions after logging the failure."""
    start = time.perf_counter()
    logger.log(level.upper(), f"Starting: {operation}")
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.success(f"Completed: {operation} (took {elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Failed: {operation} after {elapsed:.2f}s - {e}")
        raise
