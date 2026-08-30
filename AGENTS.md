# AGENTS.md — kutils conventions

Shared utilities consumed by every paper cloned from research-lab.

## Scope

- Only genuinely-shared code lives here; paper code goes in the paper's
  `src/`. The public API is the contract papers compile against.

## Model zoo

- `kutils.models` is a declarative, provider-agnostic model interface:
  `ModelSpec` (strict TOML, unknown keys raise) → provider loaders
  (`providers/`) → adapter families (`adapters/`) → one uniform
  `RepresentationOutput`, wired by `factory.build_model`. Adapters
  standardize *access* without erasing structure (tokens vs spatial maps vs
  pooled embeddings stay distinct); pooling/normalization is the
  experiment's job, never the adapter's.
- `PretrainedBackbone` builds its backbone through the factory — never
  re-implement provider logic in training code.
- `ModelRegistry` (module-level default) is the extension point: papers
  register providers/adapters via `register_provider` / `register_adapter`;
  `reset()` restores built-ins. Test isolation via the autouse
  `isolated_model_registry` fixture (tests/conftest.py).
- Optional backends (transformers/timm/OpenCLIP/torchvision) are imported
  lazily inside provider loaders, with an actionable missing-package error —
  **no extras, no new mandatory deps**. A paper adds the backend package to
  its own pyproject only when it uses that provider.
- `ModelInfo` keeps raw capability measures verbatim; derived capability
  ordering is the experiment's business (never inferred from names).
- `checkpoints/`: `hash_checkpoint` (sha256) + `load_checkpoint` (strict
  state-dict loading, torch/safetensors).

## API & versioning

- 0.x: breaking changes allowed, but each lands with a CHANGELOG entry and a
  version bump; papers pinning a tag must update deliberately. No compat
  shims before 1.0 — break cleanly.

## Trainer

- `FabricTrainer` owns the loop; per-step math lives in `TrainingRecipe`
  (`StandardRecipe` default). Prefer recipe hooks over growing `fit()`'s
  argument list.
- Checkpoints store config as plain JSON-safe data (`weights_only=True`
  loads) — never dataclass/object instances.

## Caching

- `kutils.utils.cache` is the one place to persist/reload expensive
  artifacts (format by suffix: `.npy`/`.npz`/`.json`/`.pkl`). Don't build
  second, narrower cache helpers.
- `cached(key, ...)` fingerprints `key`, not the call site — include every
  input that affects the result, or a stale fingerprint silently wins.

## Plotting style

`kutils.style` is the single source for figure colors and rcParams —
never hardcode a hex or call `plt.style.use` in a paper script. Optional
extra (`plotting`); importing `kutils` never pulls matplotlib.

- `apply_style()` — once at script start: fonts, spines, grid, DPI, and the
  `lab_sequential`/`lab_diverging` colormaps. Idempotent.
- `palette.<family>[<shade>]` — 4 families × 4 shades; assign colors by hand
  per plot (never auto-cycle) so a series keeps its color across figures.
- `sequential_cmap()` / `diverging_cmap()` — magnitude vs signed data.
- `check_series_encodings(series, markers=..., linestyles=...)` — per figure;
  **red/green collide under red-green CVD** (verified — see `tokens.py`) and
  it warns when both appear without a distinct marker/linestyle.
- `style_line_plot`/`style_bar_plot`/`style_heatmap` — per-plot conventions.
- `savefig_dual(fig, path, cache_data=...)` — writes `.pdf`+`.png`, and
  optionally caches the data via `kutils.utils.cache`.
- Labels use mathtext by default (`apply_style(usetex=False)`); `usetex=True`
  only for custom LaTeX packages. For paper (not figure) math, see
  `research-lab/paper/notation.tex`.

## Tests

- CPU-only (`accelerator="cpu"`, `precision="32-true"`), fast, network-free,
  no wandb.
- `tests/test_training.py` (checkpoint round-trip + fit smoke) is the
  weight-persistence safety net — keep it green before trainer changes.
- `LossRegistry` tests rely on the autouse `isolated_loss_registry` fixture
  (tests/conftest.py).

## Commands

- `uv sync --extra dev` · `uv run pytest -v` · `uv run ruff check .` ·
  `uv run pyright`
