# Changelog

All notable changes to kutils are documented here. Follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.1] - 2026-08-30

### Changed

- `load_local` now forwards `capability` entries (minus reserved metadata
  keys: `architecture`, `depth`, `width`, `parameters`, ...) to the model
  constructor as keyword arguments — parametrized local architectures no
  longer need a paper-side provider.
- HF-style `encode`/`encode_tensor` extracted from the vision/text
  transformer adapters into a shared `HFSequenceAdapter` base
  (`kutils.models.adapters.base`); the vision adapter keeps its timm-style
  fallback.

### Added

- `kutils/AGENTS.md` documents the model zoo.

## [0.6.0] - 2026-08-30

### Added

- Model zoo: `kutils.models` now provides a declarative, provider-agnostic
  model interface:
  - `schemas.py`: `ModelSpec` (strict TOML spec files — unknown keys raise),
    `ModelInfo` (scientific metadata with raw capability variables kept
    verbatim), `RepresentationOutput` (uniform adapter output; global/token/
    spatial/logits/layer outputs stay structurally distinct), and the
    `RepresentationModel` protocol (`preprocess` / `encode` / `model_info`).
  - `registry.py`: `ModelRegistry` with `register_provider` /
    `register_adapter` / `reset` (LossRegistry-style test isolation), a
    module-level default registry, and built-in adapter families.
  - `factory.py`: `build_model(spec)` wires a provider loader to an adapter;
    it contains no experimental logic.
  - `adapters/`: base + families (vision transformer, text transformer,
    multimodal, CNN, custom) that translate backend output conventions into
    `RepresentationOutput` without erasing structure.
  - `checkpoints/`: `hash_checkpoint` (chunked sha256) and `load_checkpoint`
    (torch or safetensors state dicts with strict key validation).
  - Optional backends (transformers/timm/OpenCLIP/torchvision) are imported
    lazily by their provider loaders — no new extras, no new mandatory
    dependencies.

### Changed

- **Breaking**: `PretrainedBackbone` now takes a `ModelSpec`
  (`PretrainedBackbone(model_spec, output_dim, freeze_backbone=False)`)
  instead of `backbone`/`model_name` strings. The backbone is constructed
  through the model factory (provider loader + adapter), removing the
  inline transformers/timm switch; head, freeze, checkpointing and Hub
  behavior are unchanged.

## [0.5.0] - 2026-08-30

### Changed

- Package renamed `lab-utils` → `kutils` and the source layout flattened
  (`src/lab_utils/` → `kutils/`). Imports become `from kutils...`; the git
  repo is now `github.com/khenm/kutils`.

## [0.4.0] - 2026-08-30

### Added

- `kutils.style`: generic plotting theme and color system, optional via
  `uv sync --extra plotting` (importing `kutils` itself still never
  requires matplotlib).
  - `apply_style()` sets global rcParams (fonts, mathtext, spines, grid,
    savefig DPI) once per process; idempotent.
  - `palette`: 4-family (blue/green/red/yellow) x 4-shade categorical
    palette, manually assigned per plot (not auto-cycled).
  - `sequential_cmap()` / `diverging_cmap()`: custom matplotlib colormaps
    registered under `"lab_sequential"` / `"lab_diverging"`.
  - `check_series_encodings()`: warns when a figure colors a red-family
    and a green-family series without a distinguishing marker/linestyle —
    red/green is the one hue pair in this palette that collapses under
    simulated red-green color-vision deficiency.
  - `style_line_plot()` / `style_bar_plot()` / `style_heatmap()`:
    per-plot-type styling helpers.
  - `savefig_dual()`: writes `.pdf` + `.png` together, with optional array
    caching via `kutils.utils.cache`.
  - All three palettes (categorical, 9-step sequential, 11-step diverging)
    verified against simulated deuteranomaly/protanomaly/tritanomaly
    (`colorspacious`, CIE76 dE in CIELab); see `style/tokens.py` docstring
    for the full verification writeup and the one known residual risk.
- CI installs `--extra plotting` too, so `tests/test_style.py` runs.

### Removed

- `kutils.utils.metrics` and `kutils.utils.plotting` — unused (accuracy
  lives in the recipe; figures go through `kutils.style`), and `plotting`
  imported Pillow without declaring it.
- `MultiLoss` from `kutils.losses` — exported but unused and untested.
- Kept: `utils.logging.log_config` (used for config banners).

## [0.3.0] - 2026-08-30

### Added

- NaN/Inf guard: `fit` raises immediately when a training/validation metric
  (e.g. loss) is non-finite, instead of surfacing 20 minutes later.
- `set_seed` now delegates to Lightning's `seed_everything(workers=True)`,
  which also seeds DataLoader worker processes (real datasets with
  augmentation are now deterministic, not just synthetic ones).
- `kutils.stats` now wraps `scipy.stats.bootstrap` /
  `permutation_test` (public API unchanged; scipy auto-enumerates when the
  relabeling count is small). New `scipy>=1.10` dependency.
- Tests: pretrained-backbone (timm branch + stubbed transformers branch,
  no downloads), `HFDatasetAdapter` (duck-typed dataset, no `datasets`
  needed), `MLP`/`ResidualBlock`, and the NaN guard.
- pyright type checking (basic mode) in CI, with `pyright` in the dev extra;
  `ruff format --check` now enforced in CI too.
- New test coverage runs in CI via `--extra hf --extra backbones`.
- Coverage tracking: CI runs `pytest --cov=kutils` (term + `coverage.xml`
  artifact via pytest-cov); no gate yet, but the number is visible.
- `integration` test marker: heavy tests are skipped by default
  (`-m "not integration"`) and run with `RUN_INTEGRATION=1`.
- PEP 561: the wheel now ships a `py.typed` marker, declaring the package
  type-checked (verified inside the built wheel).


- `kutils.utils.cache` — generic artifact caching, independent of
  training metrics: `save_artifact`/`load_artifact` (format picked from the
  file suffix: `.npy`, `.npz`, `.json`, `.pkl`) and `cached(key, compute_fn,
  ...)`, a compute-once-reuse-forever wrapper keyed by a fingerprint of
  `key`. Use it for anything worth persisting and reloading later — a
  distance matrix, a metrics history, a fitted decomposition — not just
  scalar training metrics.
- Tests: fingerprint determinism/sensitivity, save/load round-trips for all
  four supported formats, and `cached` hit/miss behavior (incl. array
  results).
- `kutils.utils.manifest.runtime_env()` — best-effort runtime environment
  (torch/CUDA/cuDNN versions, `LAB_IMAGE` tag, NVIDIA driver), recorded as
  the `env` block in every run manifest.
- `kutils.datasets.utils.train_val_test_split` — train/val/test split with
  a fixed `test_seed`, so the same examples are held out across every run
  and seed.
- `FabricTrainer.evaluate(model, loader, recipe)` — one held-out pass after
  training, never called inside `fit`, so a test set can't be iterated on by
  the training loop.
- `kutils.utils.seed.capture_rng_state()` / `restore_rng_state()` — plain,
  checkpoint-safe RNG state (torch CPU+CUDA, numpy, random), so a resumed
  run continues the exact RNG stream instead of replaying it.
- Checkpointing: `save_checkpoint`/`load_checkpoint` now persist the LR
  scheduler and RNG state plus an `epoch_complete` flag; `fit()` gains
  `save_every_seconds` (time-based mid-epoch saves to `checkpoint_dir/latest`),
  `on_checkpoint` callbacks (`CheckpointInfo`), and `resume_from` (continues
  at `epoch + 1` or redoes `epoch` based on `epoch_complete`). Every save
  updates a `checkpoint_dir/last_checkpoint.txt` pointer file, and
  `on_checkpoint` fires on the final save too. CUDA RNG restore is
  per-device best-effort (different GPU count on the resuming machine is
  skipped, not fatal).
- `HubPushCallback` — throttled `push_to_hub` via the `on_checkpoint` hook
  (independent minimum interval, so frequent local checkpoints don't mean
  frequent Hub uploads); an optional `summary_fn` provides `model_card_kwargs`
  so periodic pushes carry the same provenance as the final one.
- `BaseModel.generate_model_card` now appends a training-provenance section
  from the same dict `write_summary` builds, degrading gracefully when keys
  are missing; its signature mirrors the Hugging Face base class exactly.
- Tests: split disjointness/fixedness across seeds, `evaluate` metrics,
  manifest `env` capture, RNG capture/restore round trips, scheduler/RNG/
  `epoch_complete` checkpoint round trips, time-based saves, callback
  firing, resume continue/redo, Hub throttle, model-card provenance.

## [0.2.0] - 2026-08-30

### Added

- `kutils.utils.seed.set_seed()` — seeds Python/NumPy/PyTorch RNGs (CPU +
  CUDA) so `config.seed` in a paper actually makes runs reproducible. Call
  once at the top of an entry point, before any model/dataset construction.
- `TrainingConfig` — typed configuration for `FabricTrainer` (replaces the
  loose `dict[str, Any]`). Build directly or via `TrainingConfig.from_mapping`
  from a paper config dict; `ExperimentConfig.to_training_config()` converts.
- `TrainingRecipe` protocol + `StandardRecipe` — pluggable per-step logic.
  `FabricTrainer.fit(...)` now accepts `recipe=` and returns the final
  epoch's metrics instead of `None`.
- `LossRegistry.clear()` / `LossRegistry.reset()` — registry isolation for
  tests and papers.
- The run manifest (`write_summary`) now records what code produced the run:
  the paper repo's git commit + dirty flag, and the pinned kutils commit
  from `uv.lock` (both best-effort, `null` when unavailable).
- `kutils.stats` — run-level statistics for multi-seed results:
  `summarize` (mean/std), `bootstrap_ci` (seeded percentile bootstrap), and
  `permutation_test` (two-condition label-shuffle test, exact when the
  number of relabelings is small, add-one p-value estimator).
- Tests: checkpoint save/load round-trip, fit smoke run, seed determinism,
  `TrainingConfig.from_mapping`, `LossRegistry` isolation, manifest git
  capture, and stats (incl. a type-1 error calibration check on the
  permutation test).

### Changed

- `FabricTrainer.__init__` accepts `TrainingConfig`; a plain dict is still
  accepted and converted via `TrainingConfig.from_mapping`.
- `FabricTrainer.fit` returns `dict[str, float]` of final metrics (train loss,
  plus val loss/accuracy when a `val_loader` is given).
- Checkpoints store the trainer config as plain JSON-safe data (not a
  dataclass), so `torch`'s default `weights_only=True` checkpoint loading
  keeps working.
- Per-batch metrics returned by a recipe are mean-aggregated per epoch; keys
  prefixed `num_` are summed, and `num_correct`/`num_total` derive an
  `accuracy` metric.

### Removed

- Nothing was removed outright; 0.1.0 was never released, so the API break is
  intentional and clean (no compat shims for pre-1.0 APIs).
