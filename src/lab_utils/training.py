import os
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FabricTrainer:
    """Generic training loop powered by Lightning Fabric."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.fabric = L.Fabric(
            accelerator=config.get("accelerator", "auto"),
            precision=config.get("precision", "bf16-mixed"),
            devices=config.get("devices", "auto"),
            num_nodes=config.get("num_nodes", 1),
        )
        self.fabric.launch()

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0
        self.metrics_history: list[dict[str, float]] = []

    def _init_wandb(self) -> None:
        if self.fabric.global_rank == 0 and self.config.get("wandb_project"):
            import wandb
            wandb.init(
                project=self.config["wandb_project"],
                config=self.config,
                name=self.config.get("wandb_run_name"),
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
            model, optimizer = self.fabric.setup(model, optimizer, scheduler)
            self.scheduler = scheduler
        else:
            model, optimizer = self.fabric.setup(model, optimizer)
            self.scheduler = None
        return model, optimizer

    def setup_loader(self, loader: DataLoader) -> DataLoader:
        return self.fabric.setup_dataloaders(loader)

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 10,
        loss_fn: nn.Module | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        log_every: int = 100,
        save_every_epoch: int = 1,
    ) -> None:
        self._init_wandb()

        model, optimizer = self.setup_model(model, optimizer, scheduler)
        train_loader = self.setup_loader(train_loader)
        if val_loader is not None:
            val_loader = self.setup_loader(val_loader)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.fabric.barrier()
        model.train()

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            train_loss = self._train_epoch(
                model, optimizer, train_loader, loss_fn, epoch, log_every
            )
            self._log({"train/epoch_loss": train_loss, "epoch": epoch})

            if val_loader is not None:
                val_metrics = self._validate(model, val_loader, loss_fn)
                self._log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % save_every_epoch == 0 and self.fabric.global_rank == 0:
                self.save_checkpoint(
                    model, optimizer, epoch, self.checkpoint_dir / f"epoch_{epoch:04d}"
                )

        self.save_checkpoint(model, optimizer, max_epochs, self.checkpoint_dir / "final")
        self._finish_wandb()

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int,
        log_every: int,
    ) -> float:
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(self.fabric.device)
            y = y.to(self.fabric.device)

            with self.fabric.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)

            self.fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % log_every == 0:
                self._log({
                    "train/batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "lr": optimizer.param_groups[0]["lr"],
                })

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.fabric.device)
            y = y.to(self.fabric.device)

            with self.fabric.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        model.train()
        num_batches = max(len(loader), 1)
        return {
            "loss": total_loss / num_batches,
            "accuracy": correct / max(total, 1),
        }

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: str | Path,
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.fabric.save(path / "checkpoint.pt", {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.config,
        })

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(path))

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        path: str | Path,
    ) -> int:
        path = Path(path)
        state = {
            "model": model,
            "optimizer": optimizer,
        }
        checkpoint_path = path / "checkpoint.pt"

        if hasattr(model, "from_pretrained") and not checkpoint_path.exists():
            model = model.from_pretrained(str(path))
            return 0

        remainder = self.fabric.load(str(checkpoint_path), state)
        self.global_step = remainder.get("global_step", 0)
        return remainder.get("epoch", 0)
