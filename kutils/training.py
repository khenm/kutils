""" "Generic training loop powered by Lightning Fabric; per-step math lives
in a `TrainingRecipe` (`StandardRecipe` by default)."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import lightning
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from kutils.utils.seed import capture_rng_state, restore_rng_state


@dataclass
class TrainingConfig:
    """Typed configuration for `FabricTrainer`: how a run executes.

    Loop parameters (`max_epochs`, `log_every`, `save_every_epoch`) go to
    `fit` per run. Build directly or via `from_mapping`.
    """

    accelerator: str = "auto"
    precision: str = "bf16-mixed"
    devices: str | int | list[int] = "auto"
    num_nodes: int = 1
    checkpoint_dir: Path = Path("checkpoints")
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TrainingConfig:
        """Build from a plain dict, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        kwargs = {key: value for key, value in data.items() if key in known}
        if isinstance(kwargs.get("checkpoint_dir"), str):
            kwargs["checkpoint_dir"] = Path(kwargs["checkpoint_dir"])
        return cls(**kwargs)


@dataclass
class TrainStepContext:
    """Everything a recipe needs to run one training step."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    batch: Any
    fabric: lightning.Fabric
    epoch: int
    global_step: int


@dataclass
class ValidateStepContext:
    """Everything a recipe needs to run one validation step."""

    model: nn.Module
    batch: Any
    fabric: lightning.Fabric
    epoch: int


class TrainingRecipe(Protocol):
    """Per-step logic; returns per-batch metrics (``num_`` keys are summed,
    ``num_correct``/``num_total`` derive ``accuracy``)."""

    def train_step(self, ctx: TrainStepContext) -> dict[str, torch.Tensor]: ...

    def validate_step(self, ctx: ValidateStepContext) -> dict[str, torch.Tensor]: ...


class StandardRecipe:
    """Default recipe: autocast → loss → backward → optimizer step."""

    def __init__(self, loss_fn: nn.Module | None = None):
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def train_step(self, ctx: TrainStepContext) -> dict[str, torch.Tensor]:
        x, y = ctx.batch
        x = x.to(ctx.fabric.device)
        y = y.to(ctx.fabric.device)
        with ctx.fabric.autocast():
            logits = ctx.model(x)
            loss = self.loss_fn(logits, y)
        ctx.fabric.backward(loss)
        ctx.optimizer.step()
        ctx.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def validate_step(self, ctx: ValidateStepContext) -> dict[str, torch.Tensor]:
        x, y = ctx.batch
        x = x.to(ctx.fabric.device)
        y = y.to(ctx.fabric.device)
        with ctx.fabric.autocast():
            logits = ctx.model(x)
            loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        return {
            "loss": loss.detach(),
            "num_correct": (preds == y).sum().float(),
            "num_total": torch.tensor(float(y.size(0))),
        }


class TaskRecipe(StandardRecipe):
    """Recipe for homogeneous task batches in mixture training.

    Batches carry dict ``inputs`` / ``targets`` plus a ``task_id`` (the
    pattern's `Batch`): forward on ``inputs[input_key]``, loss from
    ``loss_by_task[batch.task_id]`` applied to ``targets[target_key]``.
    The step-based mixture loop runs `FabricTrainer.fit(max_epochs=1)`
    over a finite ``max_steps`` stream of such batches — see the
    research-lab data layer (`src.data.samplers.mixture`).
    """

    def __init__(
        self,
        loss_by_task: Mapping[str, nn.Module],
        *,
        input_key: str = "x",
        target_key: str = "y",
    ):
        super().__init__()
        self.loss_by_task = dict(loss_by_task)
        self.input_key = input_key
        self.target_key = target_key

    def _unpack(self, batch: Any, fabric: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if not (
            hasattr(batch, "inputs") and hasattr(batch, "targets") and hasattr(batch, "task_id")
        ):
            raise TypeError(
                "TaskRecipe expects batches with inputs/targets/task_id "
                f"(e.g. a Batch dataclass), got {type(batch).__name__}"
            )
        x = batch.inputs[self.input_key].to(fabric.device)
        y = batch.targets[self.target_key].to(fabric.device)
        return x, y, batch

    def train_step(self, ctx: TrainStepContext) -> dict[str, torch.Tensor]:
        x, y, batch = self._unpack(ctx.batch, ctx.fabric)
        loss_fn = self.loss_by_task[batch.task_id]
        with ctx.fabric.autocast():
            logits = ctx.model(x)
            loss = loss_fn(logits, y)
        ctx.fabric.backward(loss)
        ctx.optimizer.step()
        ctx.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def validate_step(self, ctx: ValidateStepContext) -> dict[str, torch.Tensor]:
        x, y, batch = self._unpack(ctx.batch, ctx.fabric)
        loss_fn = self.loss_by_task[batch.task_id]
        with ctx.fabric.autocast():
            logits = ctx.model(x)
            loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        return {
            "loss": loss.detach(),
            "num_correct": (preds == y).sum().float(),
            "num_total": torch.tensor(float(y.size(0))),
        }


@dataclass
class CheckpointInfo:
    """Context passed to `on_checkpoint` callbacks when a checkpoint is written."""

    path: Path
    epoch: int
    global_step: int
    elapsed_seconds: float
    epoch_complete: bool


class HubPushCallback:
    """Throttled `push_to_hub` (at most once per `min_interval_seconds`);
    `summary_fn` output is passed as `model_card_kwargs`."""

    def __init__(
        self,
        model: Any,
        repo_id: str,
        *,
        summary_fn: Callable[[], dict[str, Any]] | None = None,
        min_interval_seconds: int = 600,
    ):
        self.model = model
        self.repo_id = repo_id
        self.summary_fn = summary_fn
        self.min_interval_seconds = min_interval_seconds
        self._last_push = float("-inf")

    def __call__(self, info: CheckpointInfo) -> None:
        now = time.monotonic()
        if now - self._last_push < self.min_interval_seconds:
            return
        kwargs: dict[str, Any] = {"commit_message": f"checkpoint after step {info.global_step}"}
        if self.summary_fn is not None:
            kwargs["model_card_kwargs"] = self.summary_fn()
        self.model.push_to_hub(self.repo_id, **kwargs)
        self._last_push = now


def _check_metrics_finite(
    metrics: dict[str, torch.Tensor], *, epoch: int, global_step: int, phase: str
) -> None:
    """Yell immediately if training produced a non-finite metric (e.g. NaN loss)."""
    for key, value in metrics.items():
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            raise RuntimeError(
                f"non-finite {phase} metric {key!r} at epoch {epoch}, step {global_step} "
                f"— training diverged"
            )


def _aggregate(metrics_list: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    """Aggregate per-batch metrics: mean non-``num_`` keys, sum ``num_`` keys."""
    keys: set[str] = set().union(*(m.keys() for m in metrics_list)) if metrics_list else set()
    result: dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
        if key.startswith("num_"):
            result[key] = sum(v.float().sum().item() for v in values)
        else:
            result[key] = sum(v.float().mean().item() for v in values) / len(values)
    if "num_correct" in result and "num_total" in result:
        result["accuracy"] = result["num_correct"] / max(result["num_total"], 1e-8)
    return result


def _config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Plain, JSON-safe config (so `weights_only=True` loads keep working)."""
    return {
        "accelerator": config.accelerator,
        "precision": config.precision,
        "devices": config.devices,
        "num_nodes": config.num_nodes,
        "checkpoint_dir": str(config.checkpoint_dir),
        "wandb_project": config.wandb_project,
        "wandb_run_name": config.wandb_run_name,
    }


class FabricTrainer:
    """Generic training loop powered by Lightning Fabric."""

    def __init__(self, config: TrainingConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = TrainingConfig.from_mapping(config)
        self.config = config
        self.fabric = lightning.Fabric(
            accelerator=config.accelerator,
            precision=cast(Any, config.precision),
            devices=config.devices,
            num_nodes=config.num_nodes,
        )
        self.fabric.launch()

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0
        self.metrics_history: list[dict[str, float]] = []
        self._wandb = None
        self.scheduler = None
        self._fit_start = time.monotonic()
        self._last_save_time = self._fit_start
        self._resumed_epoch_complete = True

    def _init_wandb(self) -> None:
        if self.fabric.global_rank == 0 and self.config.wandb_project:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                config=_config_to_dict(self.config),
                name=self.config.wandb_run_name,
            )
            self._wandb = wandb
        else:
            self._wandb = None

    def _log(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics_history.append(metrics)
        if step is None:
            step = self.global_step
        self.fabric.log_dict(metrics, step=step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def _finish_wandb(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

    def setup_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        if scheduler is not None:
            model, optimizer = self.fabric.setup(model, optimizer, cast(Any, scheduler))
            self.scheduler = scheduler
        else:
            model, optimizer = self.fabric.setup(model, optimizer)
            self.scheduler = None
        return model, optimizer

    def setup_loader(self, loader: DataLoader) -> DataLoader:
        return cast(DataLoader, self.fabric.setup_dataloaders(loader))

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 10,
        recipe: TrainingRecipe | None = None,
        loss_fn: nn.Module | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        log_every: int = 100,
        save_every_epoch: int = 1,
        save_every_seconds: float | None = None,
        on_checkpoint: list[Any] | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, float]:
        """Run the training loop and return the final epoch's metrics.

        Epoch saves (`epoch_complete=True`) continue at `epoch + 1` on
        resume; time-based saves (`save_every_seconds`) redo their epoch.
        `on_checkpoint` fires on every save; `resume_from` may be a
        checkpoint dir or the `last_checkpoint.txt` pointer.
        """
        recipe = recipe or StandardRecipe(loss_fn=loss_fn)
        self._init_wandb()

        start_epoch = 0
        if resume_from is not None:
            epoch, epoch_complete = self._restore_checkpoint(
                resume_from, model, optimizer, scheduler
            )
            start_epoch = epoch if not epoch_complete else epoch + 1
            logger.info(
                f"Resuming from {resume_from}: epoch {start_epoch} "
                f"({'redo' if not epoch_complete else 'continue'})"
            )
            if start_epoch >= max_epochs:
                logger.warning(
                    f"Checkpoint already covers max_epochs={max_epochs}; nothing to train."
                )
                self._finish_wandb()
                return {}

        model, optimizer = self.setup_model(model, optimizer, scheduler)
        train_loader = self.setup_loader(train_loader)
        if val_loader is not None:
            val_loader = self.setup_loader(val_loader)

        self.fabric.barrier()
        model.train()

        self._fit_start = time.monotonic()
        self._last_save_time = self._fit_start

        final_metrics: dict[str, float] = {}
        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch
            train_metrics = self._run_train_epoch(
                recipe,
                model,
                optimizer,
                train_loader,
                epoch,
                log_every,
                save_every_seconds=save_every_seconds,
                on_checkpoint=on_checkpoint,
            )
            self._log({f"train/{k}": v for k, v in train_metrics.items()})

            if val_loader is not None:
                val_metrics = self._run_validate(recipe, model, val_loader, epoch)
                self._log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
                final_metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            else:
                final_metrics = dict(train_metrics)

            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % save_every_epoch == 0 and self.fabric.global_rank == 0:
                self._save_loop_checkpoint(
                    model, optimizer, epoch, epoch_complete=True, on_checkpoint=on_checkpoint
                )

        self._save_loop_checkpoint(
            model,
            optimizer,
            max_epochs,
            epoch_complete=True,
            on_checkpoint=on_checkpoint,
            path=self.checkpoint_dir / "final",
        )
        self._finish_wandb()
        return final_metrics

    def _run_train_epoch(
        self,
        recipe: TrainingRecipe,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        epoch: int,
        log_every: int,
        *,
        save_every_seconds: float | None = None,
        on_checkpoint: list[Any] | None = None,
    ) -> dict[str, float]:
        batch_metrics: list[dict[str, torch.Tensor]] = []
        for batch_idx, batch in enumerate(loader):
            ctx = TrainStepContext(
                model=model,
                optimizer=optimizer,
                batch=batch,
                fabric=self.fabric,
                epoch=epoch,
                global_step=self.global_step,
            )
            metrics = recipe.train_step(ctx)
            _check_metrics_finite(metrics, epoch=epoch, global_step=self.global_step, phase="train")
            self.global_step += 1
            if batch_idx % log_every == 0:
                self._log(
                    {
                        **{f"train/batch_{k}": float(v.item()) for k, v in metrics.items()},
                        "epoch": epoch,
                        "batch": batch_idx,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
            batch_metrics.append(metrics)
            if (
                save_every_seconds is not None
                and time.monotonic() - self._last_save_time >= save_every_seconds
            ):
                self._save_loop_checkpoint(
                    model, optimizer, epoch, epoch_complete=False, on_checkpoint=on_checkpoint
                )
                self._last_save_time = time.monotonic()
        return _aggregate(batch_metrics)

    def _save_loop_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        *,
        epoch_complete: bool,
        on_checkpoint: list[Any] | None,
        path: str | Path | None = None,
    ) -> None:
        if path is None:
            path = (
                self.checkpoint_dir / f"epoch_{epoch:04d}"
                if epoch_complete
                else self.checkpoint_dir / "latest"
            )
        path = Path(path)
        self.save_checkpoint(
            model, optimizer, epoch, path, scheduler=self.scheduler, epoch_complete=epoch_complete
        )
        if on_checkpoint:
            info = CheckpointInfo(
                path=path,
                epoch=epoch,
                global_step=self.global_step,
                elapsed_seconds=time.monotonic() - self._fit_start,
                epoch_complete=epoch_complete,
            )
            for callback in on_checkpoint:
                callback(info)

    @torch.no_grad()
    def _run_validate(
        self,
        recipe: TrainingRecipe,
        model: nn.Module,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        model.eval()
        batch_metrics: list[dict[str, torch.Tensor]] = []
        for batch in loader:
            ctx = ValidateStepContext(model=model, batch=batch, fabric=self.fabric, epoch=epoch)
            metrics = recipe.validate_step(ctx)
            _check_metrics_finite(
                metrics, epoch=epoch, global_step=self.global_step, phase="validate"
            )
            batch_metrics.append(metrics)
        model.train()
        return _aggregate(batch_metrics)

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        recipe: TrainingRecipe | None = None,
    ) -> dict[str, float]:
        """One held-out pass over `loader` (e.g. the test set); never in `fit`."""
        recipe = recipe or StandardRecipe()
        return self._run_validate(recipe, model, self.setup_loader(loader), self.current_epoch)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: str | Path,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        epoch_complete: bool = True,
    ) -> None:
        """Save model/optimizer/scheduler/RNG state; `epoch_complete=True`
        means resume continues at `epoch + 1` (else the epoch is redone)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "global_step": self.global_step,
            "config": _config_to_dict(self.config),
            "epoch_complete": epoch_complete,
            "rng": capture_rng_state(),
        }
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()

        self.fabric.save(path / "checkpoint.pt", state)

        if hasattr(model, "save_pretrained"):
            cast(Any, model).save_pretrained(str(path))

        # Pointer to the most recent checkpoint, so resume doesn't need to
        # know/glob the exact folder name.
        (self.checkpoint_dir / "last_checkpoint.txt").write_text(str(path))

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        path: str | Path,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> int:
        """Load a checkpoint, restoring model/optimizer (and scheduler/RNG
        when present). Returns the checkpoint's epoch."""
        path = Path(path)
        state = {
            "model": model,
            "optimizer": optimizer,
        }
        checkpoint_path = path / "checkpoint.pt"

        if hasattr(model, "from_pretrained") and not checkpoint_path.exists():
            loaded_model = cast(Any, model).from_pretrained(str(path))
            model.load_state_dict(loaded_model.state_dict())
            self._resumed_epoch_complete = True
            return 0

        remainder = self.fabric.load(str(checkpoint_path), state)
        if scheduler is not None and "scheduler" in remainder:
            scheduler.load_state_dict(remainder["scheduler"])
        if "rng" in remainder:
            restore_rng_state(remainder["rng"])
        self.global_step = remainder.get("global_step", 0)
        self._resumed_epoch_complete = remainder.get("epoch_complete", True)
        return remainder.get("epoch", 0)

    def _restore_checkpoint(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ) -> tuple[int, bool]:
        path = Path(path)
        if path.name == "last_checkpoint.txt":
            if not path.exists():
                raise FileNotFoundError(f"resume_from: no pointer file at {path}")
            resolved = path.read_text().strip()
            if not resolved:
                raise FileNotFoundError(f"resume_from: pointer file is empty at {path}")
            path = Path(resolved)
        if not (path / "checkpoint.pt").exists() and not hasattr(model, "from_pretrained"):
            raise FileNotFoundError(f"resume_from: no checkpoint.pt at {path}")
        epoch = self.load_checkpoint(model, optimizer, path, scheduler=scheduler)
        return epoch, self._resumed_epoch_complete
