"""Tests for lab_utils.utils.seed."""

import random

import numpy as np
import torch

from lab_utils.utils.seed import capture_rng_state, restore_rng_state, set_seed


def test_torch_rng_reproducible():
    set_seed(7)
    a = torch.randn(4)
    set_seed(7)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_python_rng_reproducible():
    set_seed(7)
    a = [random.random() for _ in range(3)]
    set_seed(7)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_different_seeds_differ():
    set_seed(1)
    a = torch.randn(4)
    set_seed(2)
    b = torch.randn(4)
    assert not torch.equal(a, b)


def _next_values():
    return (
        torch.randn(3),
        np.random.random(3),
        [random.random() for _ in range(3)],
    )


def _assert_same_streams(a, b):
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_rng_capture_restore_continues_stream():
    set_seed(7)
    _ = _next_values()  # advance to the state we'll capture
    state = capture_rng_state()
    expected = _next_values()  # what restoring the state must reproduce
    _ = _next_values()  # advance past it
    restore_rng_state(state)

    after = _next_values()
    _assert_same_streams(expected, after)


def test_rng_state_is_weights_only_safe():
    import io

    set_seed(7)
    _ = _next_values()
    state = capture_rng_state()
    expected = _next_values()
    # torch.load(weights_only=True) must be able to unpickle this.
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)
    restore_rng_state(loaded)
    after = _next_values()
    _assert_same_streams(expected, after)
