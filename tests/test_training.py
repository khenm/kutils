"""FabricTrainer: checkpoint round-trip and a fit smoke run.

CPU-only, no wandb, no network. This is the safety net for weight
persistence — the path that must never be silently wrong.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lab_utils.models.components import MLP
from lab_utils.training import FabricTrainer, TrainingConfig

CPU_CONFIG = {
    "accelerator": "cpu",
    "precision": "32-true",
    "devices": 1,
}


def _make_trainer(tmp_path, **overrides) -> FabricTrainer:
    config = {**CPU_CONFIG, "checkpoint_dir": str(tmp_path / "ckpt"), **overrides}
    return FabricTrainer(config)


def _make_model_optimizer(seed: int = 0):
    torch.manual_seed(seed)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_checkpoint_roundtrip_restores_weights_and_optimizer(tmp_path):
    model, optimizer = _make_model_optimizer(seed=0)
    # A few steps so AdamW state (exp_avg, step counters) exists.
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(torch.randn(4, 8)).pow(2).mean()
        loss.backward()
        optimizer.step()

    trainer = _make_trainer(tmp_path)
    trainer.global_step = 7
    trainer.save_checkpoint(model, optimizer, epoch=2, path=tmp_path / "ckpt" / "final")

    saved_weights = {k: v.clone() for k, v in model.state_dict().items()}
    saved_optimizer = optimizer.state_dict()

    # Fresh model/optimizer must restore exactly, epoch/step included.
    model2, optimizer2 = _make_model_optimizer(seed=1)
    trainer2 = _make_trainer(tmp_path)
    epoch = trainer2.load_checkpoint(model2, optimizer2, tmp_path / "ckpt" / "final")

    assert epoch == 2
    assert trainer2.global_step == 7
    for key, value in saved_weights.items():
        assert torch.equal(model2.state_dict()[key], value), f"weight mismatch: {key}"
    assert torch.equal(
        optimizer2.state_dict()["state"][0]["exp_avg"],
        saved_optimizer["state"][0]["exp_avg"],
    )


def test_fit_smoke_runs_and_writes_checkpoints(tmp_path):
    torch.manual_seed(0)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 4, (64,)))
    loader = DataLoader(dataset, batch_size=16)
    trainer = _make_trainer(tmp_path)
    metrics = trainer.fit(
        model,
        optimizer,
        loader,
        max_epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1,
        save_every_epoch=1,
    )

    assert (tmp_path / "ckpt" / "epoch_0000" / "checkpoint.pt").exists()
    assert (tmp_path / "ckpt" / "final" / "checkpoint.pt").exists()
    assert "loss" in metrics, "fit() must return metrics"
    assert len(trainer.metrics_history) > 0


def test_training_config_from_mapping(tmp_path):
    config = TrainingConfig.from_mapping(
        {
            "accelerator": "cpu",
            "precision": "32-true",
            "devices": 1,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "run_name": "ignored",
            "output_dir": "ignored",
        }
    )
    assert config.accelerator == "cpu"
    assert config.precision == "32-true"
    assert config.checkpoint_dir == tmp_path / "ckpt"


def test_evaluate_returns_held_out_metrics(tmp_path):
    from lab_utils.training import StandardRecipe

    torch.manual_seed(0)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    dataset = TensorDataset(torch.randn(32, 8), torch.randint(0, 4, (32,)))
    loader = DataLoader(dataset, batch_size=16)

    trainer = _make_trainer(tmp_path)
    metrics = trainer.evaluate(model, loader, recipe=StandardRecipe())

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_checkpoint_roundtrip_includes_scheduler(tmp_path):
    model, optimizer = _make_model_optimizer(seed=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()  # give the scheduler something to track
    scheduler.step()

    trainer = _make_trainer(tmp_path)
    trainer.save_checkpoint(
        model, optimizer, epoch=1, path=tmp_path / "ckpt" / "final", scheduler=scheduler
    )
    saved = scheduler.state_dict()

    model2, optimizer2 = _make_model_optimizer(seed=1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.5)
    trainer2 = _make_trainer(tmp_path)
    epoch = trainer2.load_checkpoint(
        model2, optimizer2, tmp_path / "ckpt" / "final", scheduler=scheduler2
    )

    assert epoch == 1
    saved_lr = saved["_step_count"]
    loaded_lr = scheduler2.state_dict()["_step_count"]
    assert saved_lr == loaded_lr


def test_checkpoint_roundtrip_restores_rng_stream(tmp_path):
    model, optimizer = _make_model_optimizer(seed=0)
    torch.randn(5)  # advance the global stream
    trainer = _make_trainer(tmp_path)
    trainer.save_checkpoint(model, optimizer, epoch=0, path=tmp_path / "ckpt" / "final")

    expected = torch.randn(3)  # what the restored stream must produce next
    torch.randn(10)  # advance away

    trainer2 = _make_trainer(tmp_path)
    trainer2.load_checkpoint(model, optimizer, tmp_path / "ckpt" / "final")
    assert torch.equal(expected, torch.randn(3)), "load must restore the RNG stream"


def test_checkpoint_epoch_complete_flag(tmp_path):
    model, optimizer = _make_model_optimizer(seed=0)
    trainer = _make_trainer(tmp_path)
    trainer.save_checkpoint(
        model, optimizer, epoch=0, path=tmp_path / "ckpt" / "mid", epoch_complete=False
    )
    trainer2 = _make_trainer(tmp_path)
    epoch = trainer2.load_checkpoint(model, optimizer, tmp_path / "ckpt" / "mid")
    assert epoch == 0
    assert trainer2._resumed_epoch_complete is False


def test_time_based_checkpointing_writes_latest(tmp_path):
    torch.manual_seed(0)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 4, (64,)))
    loader = DataLoader(dataset, batch_size=16)

    trainer = _make_trainer(tmp_path)
    trainer.fit(
        model,
        optimizer,
        loader,
        max_epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=100,
        save_every_seconds=0.001,
    )
    latest = tmp_path / "ckpt" / "latest" / "checkpoint.pt"
    assert latest.exists(), "time-based saves must write checkpoint_dir/latest"
    assert not (tmp_path / "ckpt" / "epoch_0000" / "checkpoint.pt").exists()


def test_on_checkpoint_callback_fires_for_all_saves(tmp_path):
    from lab_utils.training import CheckpointInfo

    torch.manual_seed(0)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 4, (64,)))
    loader = DataLoader(dataset, batch_size=16)

    infos: list[CheckpointInfo] = []
    trainer = _make_trainer(tmp_path)
    trainer.fit(
        model,
        optimizer,
        loader,
        max_epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=1,
        save_every_seconds=0.001,
        on_checkpoint=[infos.append],
    )

    assert infos, "callbacks must fire on checkpoint writes"
    flags = {info.epoch_complete for info in infos}
    assert flags == {True, False}, "both epoch-boundary and time-based saves notify"
    assert next(i for i in infos if i.epoch_complete).path.name == "epoch_0000"
    assert next(i for i in infos if not i.epoch_complete).path.name == "latest"
    assert any(i.path.name == "final" for i in infos), "the final save must notify too"
    assert all(info.global_step >= 0 for info in infos)


def test_resume_continues_after_complete_epoch(tmp_path):
    torch.manual_seed(0)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 4, (64,)))
    loader = DataLoader(dataset, batch_size=16)

    trainer = _make_trainer(tmp_path)
    trainer.fit(
        model,
        optimizer,
        loader,
        max_epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=1,
    )

    # epoch_0001 has epoch_complete=True -> resume continues at epoch 2.
    model2 = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    trainer2 = _make_trainer(tmp_path)
    trainer2.fit(
        model2,
        optimizer2,
        loader,
        max_epochs=3,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=1,
        resume_from=tmp_path / "ckpt" / "epoch_0001",
    )

    assert (tmp_path / "ckpt" / "epoch_0002" / "checkpoint.pt").exists(), (
        "resume must train epoch 2 and save it"
    )
    assert (tmp_path / "ckpt" / "final" / "checkpoint.pt").exists()


def test_resume_redoes_partial_epoch(tmp_path):
    torch.manual_seed(0)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.randn(32, 8), torch.randint(0, 4, (32,)))
    loader = DataLoader(dataset, batch_size=16)

    trainer = _make_trainer(tmp_path)
    trainer.fit(
        model,
        optimizer,
        loader,
        max_epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=100,
        save_every_seconds=0.001,
    )
    assert (tmp_path / "ckpt" / "latest" / "checkpoint.pt").exists()
    assert not (tmp_path / "ckpt" / "epoch_0000" / "checkpoint.pt").exists()

    # latest has epoch_complete=False -> resume redoes epoch 0 from its start.
    model2 = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    trainer2 = _make_trainer(tmp_path)
    trainer2.fit(
        model2,
        optimizer2,
        loader,
        max_epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=1,
        resume_from=tmp_path / "ckpt" / "latest",
    )

    assert (tmp_path / "ckpt" / "epoch_0000" / "checkpoint.pt").exists(), (
        "redo must re-run epoch 0 and save its boundary checkpoint"
    )


def test_resume_missing_checkpoint_raises(tmp_path):
    model, optimizer = _make_model_optimizer(seed=0)
    trainer = _make_trainer(tmp_path)
    with pytest.raises(FileNotFoundError):
        trainer.fit(
            model,
            optimizer,
            DataLoader(TensorDataset(torch.randn(8, 8), torch.zeros(8)), batch_size=8),
            max_epochs=1,
            resume_from=tmp_path / "ckpt" / "nope",
        )


def test_hub_push_callback_throttles(monkeypatch, tmp_path):
    from pathlib import Path as _Path

    from lab_utils.training import CheckpointInfo, HubPushCallback

    class FakeModel:
        def __init__(self):
            self.pushes = []

        def push_to_hub(self, repo_id, **kwargs):
            self.pushes.append((repo_id, kwargs))

    clock = {"t": 0.0}
    monkeypatch.setattr("lab_utils.training.time.monotonic", lambda: clock["t"])

    model = FakeModel()
    callback = HubPushCallback(
        model,
        "khenm/test",
        summary_fn=lambda: {"run_name": "r"},
        min_interval_seconds=600,
    )
    info = CheckpointInfo(
        path=_Path("/ckpt/latest"),
        epoch=0,
        global_step=10,
        elapsed_seconds=5.0,
        epoch_complete=False,
    )

    callback(info)  # first checkpoint -> push
    callback(info)  # same instant -> throttled
    clock["t"] = 100  # < 600s -> still throttled
    callback(info)
    clock["t"] = 700  # >= 600s since first push -> push
    callback(info)

    assert len(model.pushes) == 2
    assert model.pushes[0][0] == "khenm/test"
    assert "step 10" in model.pushes[0][1]["commit_message"]
    assert model.pushes[0][1]["model_card_kwargs"] == {"run_name": "r"}, (
        "summary_fn must be passed as model_card_kwargs so pushes aren't blank"
    )


def test_last_checkpoint_pointer_file(tmp_path):
    torch.manual_seed(0)
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.randn(32, 8), torch.randint(0, 4, (32,)))
    loader = DataLoader(dataset, batch_size=16)

    trainer = _make_trainer(tmp_path)
    trainer.fit(
        model,
        optimizer,
        loader,
        max_epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=1,
    )

    pointer = tmp_path / "ckpt" / "last_checkpoint.txt"
    assert pointer.exists(), "every save must update the pointer file"
    assert Path(pointer.read_text().strip()) == tmp_path / "ckpt" / "final"


def test_resume_via_pointer_file(tmp_path):
    # A mid-run save (as a crash would leave), then resume via the pointer.
    model, optimizer = _make_model_optimizer(seed=0)
    trainer = _make_trainer(tmp_path)
    trainer.save_checkpoint(model, optimizer, epoch=1, path=tmp_path / "ckpt" / "epoch_0001")
    pointer = tmp_path / "ckpt" / "last_checkpoint.txt"
    assert Path(pointer.read_text().strip()) == tmp_path / "ckpt" / "epoch_0001"

    dataset = TensorDataset(torch.randn(32, 8), torch.randint(0, 4, (32,)))
    loader = DataLoader(dataset, batch_size=16)
    model2, optimizer2 = _make_model_optimizer(seed=1)
    trainer2 = _make_trainer(tmp_path)
    trainer2.fit(
        model2,
        optimizer2,
        loader,
        max_epochs=3,
        loss_fn=nn.CrossEntropyLoss(),
        log_every=1000,
        save_every_epoch=1,
        resume_from=pointer,
    )

    assert (tmp_path / "ckpt" / "epoch_0002" / "checkpoint.pt").exists(), (
        "resume via the pointer file must continue at epoch 2"
    )


def test_nonfinite_loss_raises_immediately(tmp_path):
    class NanRecipe:
        def train_step(self, ctx):
            return {"loss": torch.tensor(float("nan"))}

        def validate_step(self, ctx):
            return {"loss": torch.tensor(0.0)}

    model, optimizer = _make_model_optimizer(seed=0)
    loader = DataLoader(
        TensorDataset(torch.randn(16, 8), torch.zeros(16, dtype=torch.long)), batch_size=8
    )
    trainer = _make_trainer(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(
            model,
            optimizer,
            loader,
            max_epochs=1,
            recipe=NanRecipe(),
            log_every=1000,
            save_every_epoch=100,
        )
