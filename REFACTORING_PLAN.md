# Geometric Representation Learning — Refactoring Plan (v2)

**Status**: specification, not source code.
**Companion**: `TODO_INTEGRATION_PLAN.md` (how `ToDo.md` items land here).
**Supersedes**: v1 of this file.

---

## 0. How to use this document (read first)

### 0.1 Ground rules

> **This document contains NO copy-pasteable code.**
> Every code-shaped block is an **ILLUSTRATIVE SIGNATURE SKETCH** — it names types and
> responsibilities. It is not valid, complete, or importable. Write the real thing against
> the **Contract** and **Done-when** clauses, not against the sketch.

Each unit of work below is specified as:

| Field | Meaning |
|---|---|
| **Responsibility** | The one job this unit has (if you can't state it in one sentence, split it) |
| **Contract** | Inputs, outputs, tensor shapes, error behavior |
| **Invariants** | What must remain true after the change |
| **Done-when** | Objective, checkable completion criteria |

### 0.2 Why this rule exists (evidence)

v1 of this plan was written as ready-to-paste code. It was pasted. The result is four
modules currently in the working tree that **cannot be imported**:

| File | Defect | Failure |
|---|---|---|
| `src/graphs/builders.py` | literal `...` after keyword args | **SyntaxError** — file does not parse |
| `config/registry.py` | `EquiformerBlock`, `MeanPooling`, `FoldingDecoder` never imported | **NameError** at import |
| `config/factories.py` | imports `src.learning.config.models` — path does not exist | **ImportError** |
| `src/learning/losses/composer.py` | same non-existent import path | **ImportError** |

None of these are in git. They are plan text that became broken source. **A plan that reads
like code will be pasted like code.** Hence: specify contracts, not bodies.

---

## 1. Current-state audit

### 1.1 Structural findings (evidence-based)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| F1 | **No packaging** — no `pyproject.toml`, `setup.py`, `conftest.py`, or `__init__.py` outside two dirs | repo scan | Imports depend on CWD; the `config/` vs `src.learning.config` ambiguity that broke §0.2 is a direct symptom |
| F2 | **`config/` is two different things** — `root.py` (a path utility) and the new experiment schema | `config/root.py` vs `config/models.py` | Generic top-level name; unclear ownership |
| F3 | **`helpers.py` is a junk drawer** (361 lines): dataset loading + graph building + VTP writing + an encoder test harness | `src/learning/helpers.py` | Violates SRP; everything imports it, so everything is coupled |
| F4 | **Broken contract in `helpers.verify_encoder_behaviour`** — calls `encoder.encode(...)`, which **no encoder implements** (only `TrainingStepper` defines `encode`) | `helpers.py:281,311,345` vs `grep "def encode" src/` | Dead code that looks live |
| F5 | **Two training scripts duplicate 19 module-level constants** and the whole `main()` skeleton | `equivariant_gnn_train.py`, `perceiver_train.py` | Every new experiment = a new copied script |
| F6 | **Encoder output contract diverges** — `GroupEncoder` → `mu: [B, D]`; `GroupPerceiverEncoder` → `latent: [B, K, d]` | both `forward` returns | `FoldingDecoder` **raises** unless `K == 1` (`folding_decoder.py:57-61`) — an incompatibility discovered at runtime, not build time |
| F7 | **Loader parameter explosion** — `ResamplingGraphLoader.__init__` takes **17 params**, forwarding ~12 straight to `build_fn` | `loaders.py:41-45` | The loader knows graph-construction details it should not |
| F8 | **`TrainingLogger` has 5 responsibilities** — metric history, JSON persistence, matplotlib plotting, VTP geometry export, checkpointing | `train_logs.py` | Cannot change one without touching all |
| F9 | **`TrainingOrchestrator.run` hardcodes cadence policy** (`log_every`/`save_every`/`val_every` + what happens at each) | `E3_end2end.py:159-189` | Adding early stopping / LR schedule / new artifact = editing the loop |
| F10 | **Dead weights always constructed** — `supernode_conv` is built even when `supergraph is None`; the code says so | `group_encoder.py:53` comment | Unused params enter the optimizer |
| F11 | **Global RNG side effects in a library function** — `load_dataset` calls `random.seed(seed)` | `helpers.py:46-47` | A loader mutates process-global state; reproducibility becomes order-dependent |

### 1.2 Efficiency findings

| # | Finding | Evidence | Fix direction |
|---|---|---|---|
| E1 | **Metrics persistence is O(n²)** — `log_metrics` rewrites the *entire* history to JSON on *every* call. With `LOG_EVERY=1`, `NUM_STEPS=3001` → 3001 full-file rewrites of a growing file | `train_logs.py:53-64` | Append-only JSONL (§5.4) |
| E2 | **Eager heavy imports in the registry** — importing the registry pulls `equiformer_v3` (`so3.py` 602 L, `transformer_block.py` 754 L) even for an `se3` run | `config/registry.py:2-6` | Lazy `"module:qualname"` resolution (§4.2) |
| E3 | **Python-loop attention readout** — `for b in range(num_graphs)` per forward pass | `group_encoder.py:190-193` | Batch via padded/segment ops (tracked, not urgent) |
| E4 | **Unused submodules in the optimizer** | F10 | Config-driven construction |

### 1.3 Corrections to v1 of this plan

| v1 claimed | Reality | Consequence for the plan |
|---|---|---|
| `GroupEncoder` stacks **2** `EquiLayer`s | It builds **1** (`group_encoder.py:41-47`) | `EncoderConfig.layers` (a 2-element list) is decorative — `factories.build_encoder` only ever reads `layers[0]` |
| Training runs **epochs** (`num_epochs`, `checkpoint_every`) | Loaders are **infinite generators**; the orchestrator runs `num_steps` with `log_every`/`save_every`/`val_every` | The whole `ExperimentConfig` time axis is wrong and must be step-based |
| `ExperimentRunner.from_yaml` | **PyYAML is not in `requirements.txt`** | Add the dependency or use JSON |
| `EncoderConfig` covers the encoder | Missing `area_pool`, `verbose`, `latent_mode`; `use_supernodes` duplicated in `DataConfig` | Schema must be reconciled against the real constructor |
| `build_equivariant_transformer()` is "underutilized abstraction" | It is the **correct** existing seam and already works | Keep it; wrap, don't replace |

---

## 2. Target architecture

### 2.1 The dependency rule (the single most important structural decision)

```
L4  entry        scripts/ , configs/*.yaml
L3  orchestration trainers/ , callbacks/ , runner/
L2  assembly      config schemas , registry , factories
L1  components    models/ , modules/ , layers/ , losses/ , loader/
L0  primitives    geometry/ , graphs/ , vtk/ , transforms/
```

> **Rule: imports point downward only. L1 components must never import L2 config.**

A component takes *plain arguments*; the assembly layer translates config into those
arguments. This is what keeps components reusable outside this project's config system and
testable without constructing a config object.

**The current `composer.py` violates this** (`src/learning/losses/composer.py` imports
`LossConfig` from the config layer — L1 importing L2). Fix: the composer accepts a plain
sequence of `(name, weight, kwargs)`; the config layer converts `LossConfig` into it. The
`losses/` package then has zero knowledge of the config system.

Apply the same test everywhere: *"could I use this class in another project without copying
the config package?"* If no, the dependency is inverted.

### 2.2 Target layout

```
pyproject.toml            # NEW — makes `src` importable; kills the CWD dependency (F1)
conftest.py               # NEW — pytest rootdir anchor
configs/                  # NEW — experiment YAML/JSON files
  baseline.yaml
  ablations/*.yaml
src/
  paths.py                # was config/root.py  (a path utility, not experiment config)
  learning/
    config/               # L2 — schemas, validation, (de)serialization
    registry.py           # L2 — lazy name -> component table
    factories.py          # L2 — config -> constructed object
    models/               # L1 — encoders, decoders, latent heads
    losses/               # L1 — loss fns + composer (NO config import)
    data/                 # L1 — dataset loading, graph builders, loaders
    trainers/             # L3 — stepper, orchestrator
    callbacks/            # L3 — logging, checkpointing, visualization, validation
    runner.py             # L3 — config -> full experiment
scripts/train.py          # L4 — thin CLI; ~60 lines, no hyperparameters
```

`src/learning/helpers.py` is dissolved (F3): dataset → `data/`, graph building → `data/`,
VTP writing → `callbacks/` or `src/vtk/`, `verify_encoder_behaviour` → `tests/` (and fixed
per F4).

### 2.3 Patterns — where each applies and what it buys

| Pattern | Applied to | Problem it solves |
|---|---|---|
| **Protocol / ABC** | `Encoder`, `Decoder`, `GraphBuilder`, `LatentHead`, `Callback` | Makes the contracts in §3 explicit and checkable — fixes F6 |
| **Strategy** | graph construction, latent head (VAE ↔ AE), decoder | Swap behavior by config, no `if/elif` in callers |
| **Registry (lazy)** | transformers, decoders, latent heads, graph builders, losses | Plugin-style extension; lazy resolution fixes E2 |
| **Factory** | `build_encoder` / `build_decoder` / `build_loader` | One place translates config → objects (L2) |
| **Composite** | `LossComposer` | Loss terms compose without editing the trainer |
| **Parameter Object** | `GraphSpec` | Collapses the 17-param loader (F7) |
| **Adapter / ACL** | `EncoderOutput` | Already present — the seam that decouples encoders from the trainer |
| **Observer (callbacks)** | orchestrator events | Splits F8's five responsibilities; makes F9's policy pluggable |
| **Dependency Injection** | encoder/decoder/RNG into stepper | Already partly present; extend to RNG to fix F11 |
| **Value Object** | `ExperimentConfig` → content hash | Run identity for ablations (§5.1) |

**Deliberately NOT used**: inheritance hierarchies for models (compose instead), a DI
container (over-engineering at this size), an event bus (callbacks suffice).

---

## 3. Contracts

These are the load-bearing part of this plan. Everything else is scheduling.

### 3.1 Encoder

- **Responsibility**: geometry → latent description. Nothing else.
- **Contract**: `forward(graph, supergraph | None) -> EncoderOutput`
- **Invariants**:
  - The **only** consumption point for the latent is `EncoderOutput.sample(deterministic=)`.
    Callers must not read `.mu` directly. (Today the trainer does — `E3_end2end.py:66`.)
  - Probabilistic encoders set `mu`/`logvar`; deterministic ones set `latent`.
  - Latents are invariant to rotation/translation; pose travels in `rotation`/`translation`.
- **Declared shape** — resolves F6: an encoder must expose its latent token count
  (`n_tokens = 1` for `GroupEncoder`, `K` for `GroupPerceiverEncoder`).

### 3.2 Decoder

- **Contract**: `forward(latent) -> [B, num_samples, 3]`
- **Declared requirement**: `expects_tokens: int | None` (`FoldingDecoder` → `1`; `None` = any).
- **Invariant**: **the runner validates encoder `n_tokens` against decoder
  `expects_tokens` at build time.** Today this mismatch raises mid-training
  (`folding_decoder.py:57-61`). A config error must fail before the first step.

### 3.3 LatentHead (new seam — see `TODO_INTEGRATION_PLAN.md`)

| Implementation | Emits | Regularizer |
|---|---|---|
| `GaussianLatentHead` | `EncoderOutput(mu, logvar)` | `kl` |
| `DeterministicLatentHead` | `EncoderOutput(latent=z)`, `mu=None` | `frobenius` (or none) |

- **Invariant**: both emit the same `[B, latent_dim]` shape, so the decoder contract is
  unaffected and `latent_mode` is a pure ablation switch.
- **Consolidation**: `reparameterize` currently exists in **three** places (trainer,
  `LatentVAEHead`, `EncoderOutput.sample`). Exactly one survives: `EncoderOutput.sample`.

### 3.4 GraphBuilder + GraphSpec

- **Responsibility**: `(vertices, mask, areas, normals, rng) -> (graph, supergraph | None)`.
- **Parameter Object**: `GraphSpec` holds `r_max`, `r_supergraph`, `dropout_rate`,
  `n_supernodes`, sampling modes, `recompute_area` — the 12 params the loader currently
  forwards (F7).
- **Invariant**: a builder is **stateless w.r.t. randomness** — the RNG is passed in, never
  stored, never global (fixes F11). Same seed + same spec ⇒ same graph.
- **Done-when**: `ResamplingGraphLoader` takes `(vertices, mask, builder, rng, batch_size,
  two_view)` — 6 params, down from 17 — and knows nothing about radii.

### 3.5 LossComposer

- **Contract**: `compute(values: dict[str, Tensor]) -> (total: Tensor, breakdown: dict[str, float])`
- **Invariants**:
  - Imports nothing from the config layer (§2.1).
  - A term whose value is `None` is **skipped**, not an error (this is how `kl` disappears in
    AE mode and `contrastive` disappears in single-view/validation).
  - Non-finite total raises immediately (preserve the existing guard).
  - `total` is a **scalar** tensor. *(The current draft seeds it with `torch.zeros(1)` — a
    shape-`[1]` tensor. Use a true scalar.)*
  - `breakdown` keys are stable across train and val so the two are directly comparable.

### 3.6 Callback

- **Contract**: hooks — `on_train_start`, `on_step_end(step, breakdown, batch, pred)`,
  `on_validation_end(step, breakdown)`, `on_train_end`.
- **Responsibility split** (dissolves F8): `MetricsRecorder` (JSONL), `MetricsPlotter`,
  `CheckpointWriter`, `GeometryVisualizer`, `ValidationRunner`.
- **Invariant**: the orchestrator owns *the loop*; callbacks own *what happens at events*.
  Cadence is per-callback config, not orchestrator arguments (fixes F9).

---

## 4. Phases

Ordering is chosen so that **every phase leaves the repo runnable**.

### Phase 0 — Stop the bleeding (do this first)

- **Goal**: a trustworthy baseline before any restructuring.
- **Changes**:
  1. **Resolve the four broken files** from §0.2 — either delete them or finish them. They
     are currently the top import hazard in the tree.
  2. Add `pyproject.toml` + `conftest.py`; make `src` a real package (F1). Decide **one**
     canonical config location and move `config/root.py` → `src/paths.py` (F2).
  3. Add `PyYAML` to `requirements.txt` if YAML configs are chosen (§1.3).
  4. **Characterization tests**: pin current behavior — a short seeded training run's loss
     sequence, encoder output shapes, graph node/edge counts. These are the safety net for
     every later phase.
- **Done-when**: `pytest` passes from any CWD; every module in `src/` imports cleanly;
  a seeded 20-step run reproduces bit-identically twice.
- **Risk**: none — additive.

### Phase 1 — Config schema & validation

- **Responsibility**: the schema is the *only* description of an experiment.
- **Changes**: reconcile the schema against real constructor signatures (§1.3): step-based
  time axis (`num_steps`, `log_every`, `save_every`, `val_every` — **not** epochs), add
  `area_pool`, `latent_mode`; remove the decorative `layers` list or make `build_encoder`
  actually consume it; de-duplicate `use_supernodes` between `EncoderConfig` and `DataConfig`
  (it belongs to the graph spec).
- **Validation is a first-class feature**, not a nicety. It must reject:
  - `latent_mode="deterministic"` + a `kl` term (`enc.kl()` is `None`)
  - encoder `n_tokens` ≠ decoder `expects_tokens` (§3.2)
  - a loss term naming a metric the trainer never produces
  - `num_samples` not a perfect square (the folding grid requires it)
  - **warn**: `frobenius` + `contrastive` both strong — the variance hinge pushes latent
    scale up while Frobenius pushes it down
- **Done-when**: every constant in both training scripts (§F5) has a schema home;
  config → serialize → deserialize → config round-trips to an equal object.

### Phase 2 — Registry & factories

- **Registry design**: map `name -> "module:qualname"` **strings**, resolved on first use.
  This avoids v1's two failures at once: undefined names at import (§0.2) and pulling
  `equiformer_v3` into every run (E2).
- **Categories**: `transformer`, `decoder`, `latent_head`, `graph_builder`, `readout`, `loss`.
- **Contract**: `create(category, name, **kwargs)`; `list(category)` for discoverability;
  unknown name raises with the available names in the message.
- **Factories**: pure functions, config → object. The **only** place that reads config
  fields and calls constructors.
- **Done-when**: adding a new decoder requires touching *one* registration line and *zero*
  existing files; `list()` powers `--help`.

### Phase 3 — Loss composition

- **Changes**: `frobenius_latent_loss(Z)` = ‖Z‖²_F / B — the AE analogue of KL (mean over B
  keeps the weight batch-size independent, matching `kl_divergence_loss`); composer per §3.5
  with the L1/L2 inversion fixed.
- **Done-when**: adding a loss term = one registry entry + one config line, with no trainer
  edit; `breakdown` is identical in structure for train and val.

### Phase 4 — Latent seam & trainer

Full detail in `TODO_INTEGRATION_PLAN.md`. Summary:

- Inject a `LatentHead` into `GroupEncoder` (§3.3); unify the three `reparameterize` copies.
- Route the trainer through `EncoderOutput.sample(deterministic=)` — this is what makes AE
  mode work with **zero** trainer branching, and it fixes validation currently injecting
  random noise under `no_grad` (`E3_end2end.py:135`→`67`).
- `train_step`/`eval_step` both return composer breakdowns; loss weights move from
  constructor args to `LossConfig`.
- **Verbose validation**: `run_validation` currently computes `avg_recon_loss`/`avg_kl_loss`
  and then logs only `val_loss` (`E3_end2end.py:219-226`) — the split is computed and
  discarded. Log `val/<term>` for every term.
- **Done-when**: `latent_mode="gaussian"` reproduces the pre-refactor loss curve (Phase 0
  characterization test); `latent_mode="deterministic"` trains without any trainer code path
  being conditional on the mode.

### Phase 5 — Graph & data pipeline

- `GraphSpec` + `GraphBuilder` per §3.4; loader drops to 6 params.
- Dissolve `helpers.py` (F3); fix or relocate `verify_encoder_behaviour` (F4).
- **Done-when**: swapping `radius` → `knn` is a config line; the loader has no radius/dropout
  parameters.

### Phase 6 — Orchestrator & callbacks

- Extract the five `TrainingLogger` responsibilities into callbacks (§3.6, F8).
- Orchestrator keeps only the loop; cadence moves into callbacks (F9).
- **Done-when**: early stopping can be added as a new callback with no orchestrator edit.

### Phase 7 — Experiment runner, CLI, sweeps

- `runner.py`: config → components → data → orchestrator → results.
- `scripts/train.py`: thin CLI (`train.py configs/x.yaml --set training.lr=1e-4 --seed 0`).
  **No hyperparameters in the script** — this is what retires F5.
- Run manifests, overrides, sweeps — §5.
- **Done-when**: both existing training scripts are reproducible as YAML configs and deleted.

### Phase 8 — Migration & cleanup

- Delete the compat shims, retire duplicated scripts, write `MIGRATION.md`.

---

## 5. Ablation & experiment infrastructure

This is a first-class requirement, not a byproduct of the refactor.

### 5.1 Run identity

Canonicalize the config → stable hash → `run_id`. Two runs with the same `run_id` are the
same experiment; a diff of two configs *is* the description of the ablation. Output directory
is named by `run_id` + human label, so runs never silently overwrite each other.

### 5.2 Run manifest

Every run writes, before training starts: resolved config (fully expanded, all defaults
materialized), git SHA + dirty flag, package versions, device, seed, and the CLI overrides
used. Without this, ablation results are not attributable.

### 5.3 Overrides & sweeps

- Layered resolution: **defaults → config file → CLI `--set key.path=value`**.
- A sweep is a base config + a list of override sets; each element gets its own `run_id`.
- **Invariant**: an override must fail loudly on an unknown key path. Silent typo'd
  overrides are the classic source of "the ablation showed no effect."

### 5.4 Structured metrics

Replace the O(n²) rewrite (E1) with **append-only JSONL**, one record per event:
`{step, split, term, value}`. Benefits: constant-time writes, crash-safe, trivially loadable
for cross-run comparison, and per-term train/val series come out for free.

### 5.5 Determinism contract

Seed → all RNGs, explicitly passed (F11), never global. A run must be re-runnable from its
manifest alone. State this as a test, not a hope (Phase 0).

### 5.6 The ablation matrix this unlocks

| Axis | Values | Config field |
|---|---|---|
| Latent mode | `gaussian` (VAE) / `deterministic` (AE) | `encoder.latent_mode` |
| Regularizer | `kl` / `frobenius` / none | which term is in `LossConfig` |
| Contrastive | on / off | `LossConfig` |
| Readout | `mean` / `attention` | `encoder.readout` |
| Transformer | `se3` / `equiformer` / none | `encoder.transformer_type` |
| Graph | `radius` / `knn` | `data.graph_builder` |
| Decoder | `folding` / `sphere_folding` | `decoder.decoder_type` |

---

## 6. Testing strategy

| Layer | What it proves | When |
|---|---|---|
| **Characterization** | The refactor changed nothing it shouldn't — pinned loss sequences, shapes, graph counts | **Phase 0, before any change** |
| **Contract tests** | Every `Encoder`/`Decoder`/`GraphBuilder`/`LatentHead` implementation satisfies §3, run against *all* registered implementations | Per phase |
| **Config validation** | Illegal combinations are rejected *before* training | Phase 1 |
| **Parity harness** | Old path and new path produce identical models from equivalent settings | Phases 2–4 |
| **Equivariance** | Existing rotation/translation invariance tests still pass | Continuous |

The existing 10 test files are the regression net — they must keep passing at every phase
boundary. Note that `test_trainer_e2e.py` and `test_validation.py` exercise exactly the code
Phase 4 changes; expect to update their *assertions on return-tuple arity* when
`train_step`/`eval_step` start returning breakdowns.

**One intentional behavior change**: validation becomes deterministic (Phase 4). VAE val
curves will shift slightly on the first refactored run. That is the fix, not a regression —
record it in the characterization baseline before it happens.

---

## 7. Sequencing & risk

| Phase | Depends on | Risk | Note |
|---|---|---|---|
| 0 Foundations | — | **None** | Do not skip. Everything after assumes the safety net. |
| 1 Config | 0 | Low | Pure addition |
| 2 Registry/factories | 1 | Low | Parallel to old paths |
| 3 Losses | 1 | Low | |
| 4 Latent/trainer | 1,2,3 | **Medium** | Touches the training path; guarded by Phase 0 tests |
| 5 Graph/data | 1,2 | Medium | `helpers.py` has many importers |
| 6 Callbacks | 1 | Low | |
| 7 Runner/CLI | all | Low | |
| 8 Cleanup | 7 | Low | Deletions only |

Phases 2, 3, 5, 6 are mutually independent after Phase 1 — they can proceed in any order or
in parallel.

---

## 8. Anti-goals

- **Do not** paste this document's sketches into `.py` files (§0.2).
- **Do not** add a config framework dependency (Hydra/OmegaConf) yet — dataclasses plus a
  small override resolver cover the need at this scale.
- **Do not** build abstractions with one implementation and no second use case in sight. The
  seams specified here each have ≥2 concrete implementations *today* (two encoders, two
  decoders, two transformers, two latent modes, two graph builders).
- **Do not** refactor `equiformer_v3/` — it is vendored.
- **Do not** chase E3 (Python-loop readout) before correctness work lands.

---

## 9. Done-when — the whole refactor

- [ ] Every module imports cleanly; `pytest` passes from any working directory
- [ ] A full experiment runs from one config file with **zero** hyperparameters in code
- [ ] Both existing training scripts are reproduced as configs and deleted
- [ ] Adding a loss term / decoder / graph builder touches one registration line
- [ ] VAE ↔ AE and KL ↔ Frobenius are one-line config switches
- [ ] Every loss component is logged for **both** train and validation
- [ ] Illegal config combinations fail before step 0, with an actionable message
- [ ] Each run emits a manifest sufficient to re-run it exactly
- [ ] No L1 component imports the config layer
- [ ] Characterization tests from Phase 0 still pass
