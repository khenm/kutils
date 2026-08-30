"""Tests for kutils.utils.manifest."""

import json
import subprocess

from kutils.utils.manifest import write_summary


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_write_summary_completed(tmp_path):
    path = write_summary(
        tmp_path / "run",
        run_name="260830-1402-baseline",  # cmt: 260830-1402_baseline
        config={"lr": 1e-3},
        status="completed",
        elapsed_seconds=12.345,
        metrics={"val/accuracy": 0.9},
    )
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["run_name"] == "260830-1402-baseline"  # cmt: run_name should match above
    assert data["status"] == "completed"
    assert data["elapsed_seconds"] == 12.35
    assert data["config"] == {"lr": 1e-3}
    assert data["metrics"] == {"val/accuracy": 0.9}


def test_write_summary_creates_output_dir(tmp_path):
    target = tmp_path / "nested" / "run_dir"
    assert not target.exists()
    write_summary(
        target,
        run_name="run",
        config={},
        status="failed",
        elapsed_seconds=0.1,
    )
    assert target.is_dir()
    assert (target / "summary.json").exists()


def test_write_summary_defaults_empty_metrics(tmp_path):
    path = write_summary(tmp_path, run_name="run", config={}, status="failed", elapsed_seconds=1.0)
    data = json.loads(path.read_text())
    assert data["metrics"] == {}


def test_write_summary_records_git_revision(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "code.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    sha = _git(repo, "rev-parse", "HEAD").strip()

    monkeypatch.chdir(repo)
    path = write_summary(
        repo / "out", run_name="r", config={}, status="completed", elapsed_seconds=1.0
    )
    data = json.loads(path.read_text())
    assert data["git"] == {"commit": sha, "dirty": False}


def test_write_summary_marks_dirty_tree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "code.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "code.py").write_text("x = 2\n")  # uncommitted change

    monkeypatch.chdir(repo)
    path = write_summary(
        repo / "out", run_name="r", config={}, status="completed", elapsed_seconds=1.0
    )
    data = json.loads(path.read_text())
    assert data["git"]["dirty"] is True


def test_write_summary_git_none_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_summary(tmp_path, run_name="r", config={}, status="completed", elapsed_seconds=1.0)
    data = json.loads(path.read_text())
    assert data["git"] is None


def test_write_summary_records_kutils_commit_from_uv_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text(
        "version = 1\n\n"
        "[[package]]\n"
        'name = "kutils"\n'
        'source = { git = "https://github.com/khenm/kutils.git#abc123def456" }\n'
    )
    path = write_summary(
        tmp_path / "out", run_name="r", config={}, status="completed", elapsed_seconds=1.0
    )
    data = json.loads(path.read_text())
    assert data["kutils_commit"] == "abc123def456"


def test_write_summary_kutils_commit_none_without_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_summary(tmp_path, run_name="r", config={}, status="completed", elapsed_seconds=1.0)
    data = json.loads(path.read_text())
    assert data["kutils_commit"] is None


def test_write_summary_includes_runtime_env(tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(tmp_path)
    path = write_summary(tmp_path, run_name="r", config={}, status="completed", elapsed_seconds=1.0)
    data = json.loads(path.read_text())
    assert data["env"]["torch"] == torch.__version__
    assert "image" in data["env"]
