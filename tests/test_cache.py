"""Tests for kutils.utils.cache."""

import numpy as np
import pytest

from kutils.utils.cache import (
    cached,
    fingerprint,
    load_artifact,
    save_artifact,
)


def test_fingerprint_is_deterministic():
    assert fingerprint(1, 2, a="x") == fingerprint(1, 2, a="x")


def test_fingerprint_kwarg_order_does_not_matter():
    assert fingerprint(a=1, b=2) == fingerprint(b=2, a=1)


def test_fingerprint_positional_order_matters():
    assert fingerprint(1, 2) != fingerprint(2, 1)


def test_fingerprint_distinguishes_different_inputs():
    assert fingerprint({"lr": 1e-3}) != fingerprint({"lr": 1e-2})


def test_save_load_npy_roundtrip(tmp_path):
    arr = np.arange(12).reshape(3, 4).astype(np.float32)
    path = save_artifact(tmp_path / "arr.npy", arr)
    assert path.exists()
    loaded = load_artifact(path)
    np.testing.assert_array_equal(loaded, arr)


def test_save_load_npz_roundtrip(tmp_path):
    data = {"a": np.array([1, 2, 3]), "b": np.array([[1.0, 2.0]])}
    path = save_artifact(tmp_path / "data.npz", data)
    loaded = load_artifact(path)
    assert set(loaded.keys()) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"], data["a"])
    np.testing.assert_array_equal(loaded["b"], data["b"])


def test_save_npz_rejects_non_dict(tmp_path):
    with pytest.raises(ValueError):
        save_artifact(tmp_path / "data.npz", [1, 2, 3])


def test_save_load_json_roundtrip(tmp_path):
    obj = {"lr": 1e-3, "seed": 0, "tags": ["a", "b"]}
    path = save_artifact(tmp_path / "obj.json", obj)
    assert load_artifact(path) == obj


def test_save_load_pkl_roundtrip(tmp_path):
    obj = {"nested": (1, 2, {"x"})}
    path = save_artifact(tmp_path / "obj.pkl", obj)
    assert load_artifact(path) == obj


def test_save_artifact_unsupported_suffix_raises(tmp_path):
    with pytest.raises(ValueError):
        save_artifact(tmp_path / "obj.txt", "hello")


def test_load_artifact_unsupported_suffix_raises(tmp_path):
    path = tmp_path / "obj.txt"
    path.write_text("hello")
    with pytest.raises(ValueError):
        load_artifact(path)


def test_save_artifact_creates_parent_dirs(tmp_path):
    path = save_artifact(tmp_path / "nested" / "dir" / "obj.json", {"x": 1})
    assert path.exists()


def test_cached_computes_on_miss_and_persists(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"x": 1}

    result = cached({"seed": 0}, compute, cache_dir=tmp_path, name="thing", ext=".json")
    assert result == {"x": 1}
    assert len(calls) == 1
    assert (tmp_path / f"thing-{fingerprint({'seed': 0})}.json").exists()


def test_cached_hits_without_recomputing(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"x": 1}

    cached({"seed": 0}, compute, cache_dir=tmp_path, name="thing", ext=".json")
    result = cached({"seed": 0}, compute, cache_dir=tmp_path, name="thing", ext=".json")
    assert result == {"x": 1}
    assert len(calls) == 1


def test_cached_different_keys_miss_independently(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    r1 = cached({"seed": 0}, compute, cache_dir=tmp_path, name="thing", ext=".json")
    r2 = cached({"seed": 1}, compute, cache_dir=tmp_path, name="thing", ext=".json")
    assert r1 != r2
    assert len(calls) == 2


def test_cached_supports_array_results(tmp_path):
    def compute():
        return np.array([1.0, 2.0, 3.0])

    result = cached("k", compute, cache_dir=tmp_path, name="arr", ext=".npy")
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    cached_again = cached("k", compute, cache_dir=tmp_path, name="arr", ext=".npy")
    np.testing.assert_array_equal(cached_again, result)
