"""Run manifest: summary.json written once per run.

Records config, wall-clock timing, metrics, what code produced the run
(paper repo git revision + pinned lab-utils commit), and the runtime
environment (torch/CUDA/cuDNN/driver/image) — all best-effort.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any


def runtime_env() -> dict[str, Any]:
    """Runtime environment: torch/CUDA/cuDNN versions, image tag, GPU driver.

    Best-effort: None for whatever isn't available (no torch, no GPU, no
    LAB_IMAGE env). Never raises.
    """
    env: dict[str, Any] = {"image": os.environ.get("LAB_IMAGE")}
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                env["nvidia_driver"] = proc.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return env


def git_revision(directory: Path | None = None) -> dict[str, Any] | None:
    """Git revision of the repo containing `directory` (default: cwd).

    Returns {"commit": <SHA>, "dirty": bool} — dirty means uncommitted
    changes, which make a bare SHA misleading. None if git is unavailable or
    not in a repo. Never raises.
    """
    if shutil.which("git") is None:
        return None
    root = directory or Path.cwd()
    try:
        toplevel = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if toplevel.returncode != 0:
            return None
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if commit.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        dirty = status.returncode == 0 and bool(status.stdout.strip())
        return {"commit": commit.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return None


def locked_dependency_rev(directory: Path | None = None, name: str = "lab-utils") -> str | None:
    """Pinned git rev of `name` recorded in uv.lock, or None. Never raises."""
    root = directory or Path.cwd()
    lock_path = root / "uv.lock"
    if not lock_path.exists():
        return None
    try:
        with open(lock_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    for package in data.get("package", []):
        if package.get("name") == name:
            source = package.get("source") or {}
            git = source.get("git", "")
            if isinstance(git, str) and "#" in git:
                return git.split("#", 1)[1]
    return None


def build_summary(
    *,
    run_name: str,
    config: dict[str, Any],
    status: str,
    elapsed_seconds: float,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """The run-summary dict — single source, also usable as model-card kwargs."""
    return {
        "run_name": run_name,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "config": config,
        "metrics": metrics or {},
        "git": git_revision(),
        "lab_utils_commit": locked_dependency_rev(),
        "env": runtime_env(),
    }


def write_summary(
    output_dir: str | Path,
    *,
    run_name: str,
    config: dict[str, Any],
    status: str,
    elapsed_seconds: float,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write (or overwrite) summary.json for a run.

    Args:
        output_dir: The run's output directory (created if missing).
        run_name: Resolved run identifier, e.g. "260830-1402-baseline".
        config: Resolved experiment config as a plain dict.
        status: "completed" | "failed".
        elapsed_seconds: Wall-clock time for the run.
        metrics: Final metrics, if the caller has any.

    Returns:
        Path to the written summary.json.

    Always includes ``git``, ``lab_utils_commit``, and ``env`` (all
    best-effort, None when unavailable).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        run_name=run_name,
        config=config,
        status=status,
        elapsed_seconds=elapsed_seconds,
        metrics=metrics,
    )

    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path
