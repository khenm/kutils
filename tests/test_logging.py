"""Tests for kutils.utils.logging."""

import time

import pytest
from loguru import logger

from kutils.utils.logging import (
    ExperimentState,
    get_loguru_safe_tqdm,
    log_config,
    log_timing,
    setup_experiment_logging,
)


def test_setup_logging(capsys):
    setup_experiment_logging("test", level="DEBUG")
    logger.info("test message")
    logger.debug("debug message")
    captured = capsys.readouterr()
    assert "test message" in captured.err
    assert "debug message" in captured.err


def test_setup_logging_info_level(capsys):
    setup_experiment_logging("test_info", level="INFO")
    logger.debug("hidden debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.success("success message")
    captured = capsys.readouterr()
    assert "info message" in captured.err
    assert "warning message" in captured.err
    assert "error message" in captured.err
    assert "success message" in captured.err
    assert "hidden debug message" not in captured.err


def test_log_config(capsys):
    setup_experiment_logging("test_config", level="INFO")
    log_config({"lr": 1e-3, "batch_size": 64})
    captured = capsys.readouterr()
    assert "Experiment config:" in captured.err
    assert "lr: 0.001" in captured.err
    assert "batch_size: 64" in captured.err


def test_log_timing_success(capsys):
    setup_experiment_logging("test_timing", level="INFO")
    with log_timing("test operation"):
        time.sleep(0.01)
    captured = capsys.readouterr()
    assert "Starting: test operation" in captured.err
    assert "Completed: test operation" in captured.err


def test_log_timing_failure(capsys):
    setup_experiment_logging("test_timing_fail", level="INFO")
    with pytest.raises(ValueError), log_timing("failing operation"):
        raise ValueError("intentional error")
    captured = capsys.readouterr()
    assert "Starting: failing operation" in captured.err
    assert "Failed: failing operation" in captured.err


def test_experiment_state_enum():
    assert ExperimentState.INITIALIZING == "Initializing"
    assert ExperimentState.COMPLETED == "Completed"
    assert ExperimentState.TRAINING == "Training"
    assert ExperimentState.SAVING_RESULTS == "Saving results"
    assert ExperimentState.FAILED == "Failed"
    assert ExperimentState.SKIPPED == "Skipped"
    for state in ExperimentState:
        assert isinstance(state.value, str)


def test_tqdm_wrapper():
    setup_experiment_logging("test_tqdm", level="INFO")
    tqdm = get_loguru_safe_tqdm()
    items = list(range(10))
    result = list(tqdm(items, desc="Test"))
    assert result == items


def test_multiple_setup_calls():
    setup_experiment_logging("test1", level="INFO")
    logger.info("test1 message")
    setup_experiment_logging("test2", level="DEBUG")
    logger.debug("test2 message")
    assert len(logger._core.handlers) == 1  # pyright: ignore[reportAttributeAccessIssue]
