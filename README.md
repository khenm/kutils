# lab-utils

Shared research utilities used across every paper cloned from `research-lab`.

- **Training**: `FabricTrainer` — a generic training loop on Lightning
  Fabric. Configured by the typed `TrainingConfig`; per-step math lives in a
  `TrainingRecipe` (`StandardRecipe` by default). `fit()` returns the final
  epoch's metrics.
- **Checkpointing**: `save_checkpoint` / `load_checkpoint` persist model,
  optimizer, LR scheduler, and RNG state plus an `epoch_complete` flag —
  epoch-boundary saves mean "continue at epoch+1", time-based mid-epoch
  saves (`save_every_seconds`) mean "redo that epoch". `resume_from` restores
  everything (a `last_checkpoint.txt` pointer tracks the most recent save).
  `BaseModel` gives any model `save_pretrained` / `from_pretrained` /
  `push_to_hub` for free (HF Hub); `HubPushCallback` throttles Hub uploads on
  every checkpoint, and `generate_model_card` appends training provenance.
- **Reproducibility**: `set_seed` (Lightning's `seed_everything`, including
  DataLoader workers) plus `capture_rng_state` / `restore_rng_state` for
  exact resume; every run's manifest (`write_summary`) records config,
  metrics, git revision, and the runtime environment.
- **Statistics**: `lab_utils.stats` (`summarize`, `bootstrap_ci`,
  `permutation_test` — scipy-backed) for multi-seed results.
- **Datasets**: `BaseDataset` for from-scratch datasets,
  `train_val_test_split` (fixed test split across runs), and
  `lab_utils.datasets.hf` (`load_hf_dataset`, `HFDatasetAdapter`) for the
  HuggingFace Hub. Requires the `hf` extra.
- **Models**: `lab_utils.models.pretrained.PretrainedBackbone` wraps a
  pretrained `transformers` or `timm` model plus a task head. Requires the
  `backbones` extra.
- **Losses**: `LossRegistry` for named loss functions (register / clear /
  reset for test isolation).
- **Logging**: `lab_utils.utils.logging` — structured loguru setup, a
  tqdm-safe wrapper, an `ExperimentState` enum, and a `log_timing` context
  manager.
- **Caching**: `lab_utils.utils.cache` (`save_artifact` / `load_artifact` /
  `cached`) for persisting anything expensive to recompute.
- **Plotting style**: `lab_utils.style` — a CVD-verified color system,
  colormaps, and per-plot-type helpers, all in one theme. Requires the
  `plotting` extra (matplotlib); importing `lab_utils` itself never pulls in
  matplotlib.

## Install

```bash
pip install lab-utils
# or, for real datasets / pretrained backbones / figures:
pip install "lab-utils[hf,backbones,plotting]"
```

Type-checked: the wheel ships a PEP 561 `py.typed` marker.
