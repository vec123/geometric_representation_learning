# Implementation Instructions — Learning by Doing

A task-by-task guide to implementing `REFACTORING_PLAN.md`.
Each task is small, independently verifiable, and teaches **one** idea.

---

## How to use this guide

Work top to bottom. Each task has the same six parts:

| Part | What it gives you |
|---|---|
| 🎯 **Goal** | The concrete outcome, in one sentence |
| 🧩 **The pattern** | Its real name, so you can look it up elsewhere |
| 🔍 **Why here** | The specific mess in *this* repo that motivates it |
| 🔨 **Do this** | Numbered, concrete steps — the part you act on |
| ✅ **Verify** | A command that proves it worked |
| 🎓 **What you learned** | When to reach for this pattern — **and when not to** |

**Rules of engagement**

1. **Type the skeletons, don't paste them.** Every code block here is a *skeleton*: valid
   Python with `TODO` bodies. It compiles; it does not work. You write the bodies. Typing it
   is how the API surface gets into your head.
2. **Run the Verify step before moving on.** If it fails, fix it before continuing. Tasks
   build on each other.
3. **Commit after each task.** One task = one commit. If a task goes wrong you lose an hour,
   not a day.
4. **Don't skip Task 2.** It is the safety net for everything after it.

**Progress**

```
Track A — Safety net      [ ] T0  [ ] T1  [ ] T2
Track B — Practice        [ ] T3  [ ] T4
Track C — The spine       [ ] T5  [ ] T6  [ ] T7
Track D — The features    [ ] T8  [ ] T9  [ ] T10  [ ] T11
Track E — Orchestration   [ ] T12 [ ] T13
```

---
---

# Track A — Build a safety net

You cannot refactor code whose behavior you cannot check. Track A takes ~half a day and
makes every later task reversible.

---

## T0 — Triage the four broken files

**⏱ 20 min · 🟢 easy**

### 🎯 Goal
Get the working tree to a state where every file in `src/` and `config/` can actually be imported.

### 🔍 Why here
Six files were created by pasting code blocks out of v1 of the refactoring plan. Two are fine.
**Four cannot be imported.** Right now `pytest` collection or any broad import will trip over them.

| File | Status | Problem |
|---|---|---|
| `config/irreps.py` | ✅ **keep** | Valid and useful as-is |
| `config/models.py` | ✅ **keep** | Valid; you'll refine it in T5 |
| `src/graphs/builders.py` | ❌ **delete** | `SyntaxError` — the file does not parse |
| `config/registry.py` | ❌ **delete** | `NameError` — uses `EquiformerBlock`, `MeanPooling`, `FoldingDecoder` without importing them |
| `config/factories.py` | ❌ **delete** | `ImportError` — imports `src.learning.config.models`, which does not exist |
| `src/learning/losses/composer.py` | ❌ **delete** | `ImportError` — same phantom path |

You will rebuild all four properly in T4, T6, T7, and T8. Deleting now is not losing work —
it is removing a decoy.

### 🔨 Do this

1. Look at `src/graphs/builders.py` line 39-42 and find the actual syntax error yourself
   before deleting it. It is worth seeing once:

   ```python
   # This is a SyntaxError: a bare `...` after keyword arguments
   graph = get_graphs_from_vertices(vertices, mask, r_max=self.r_max, ...)
   #                                                                  ^ positional after keyword

   # This is perfectly legal: `...` as a function BODY
   def build(self, vertices, mask):
       ...
   ```
   Same three characters, completely different meaning. Skeletons in this guide use the
   second form, which is why they are safe to type.

2. Delete the four broken files.

3. Confirm the two survivors are genuinely fine — read `config/models.py` and note that
   `EncoderConfig.layers` defaults to a **list of two** `EncoderLayerConfig`. Remember this;
   T5 deals with it.

### ✅ Verify

```bash
python -c "import ast,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('src').rglob('*.py')]; print('all parse')"
python -c "import config.models, config.irreps; print('config ok')"
```

### 🎓 What you learned

**Practice: distinguish a plan from an artifact.** Documentation that *looks* like code
invites being used as code. When you read a design doc, the burden is on you to check that
imports resolve and names exist before adopting a snippet.

**When this matters most:** any time you copy code from a doc, a blog post, or an LLM. Run
it in isolation before wiring it in.

---

## T1 — Make the project a real package

**⏱ 45 min · 🟢 easy**

### 🎯 Goal
`import src.learning...` and `pytest` both work from *any* directory, not just the repo root.

### 🧩 The pattern
**Explicit packaging / removing ambient dependencies.** Right now the project depends on an
invisible piece of state: your current working directory.

### 🔍 Why here
`scripts/equivariant_gnn_train.py:39` does `from config.root import get_project_root`. That
works only because Python puts the script's launch directory on `sys.path`. There is no
`pyproject.toml`, no `setup.py`, no `conftest.py`, and almost no `__init__.py`. Consequences:

- Tests do `sys.path` surgery by hand (see the top of `tests/learning/test_models.py`).
- `cd scripts && python equivariant_gnn_train.py` breaks.
- Two different import paths for the same module can coexist — which is precisely how
  `src.learning.config.models` vs `config.models` slipped into the pasted files in T0.

### 🔨 Do this

1. Create `pyproject.toml` at the repo root. Declare the project name, `requires-python =
   ">=3.8"`, and configure setuptools to find packages. Read your `requirements.txt` and move
   the pinned dependencies in (keep the pins — this project is pinned to `torch==2.4.1+cu121`
   and `e3nn==0.6.0` for a reason).

2. Add `PyYAML` as a dependency. The plan calls for YAML configs in T13 and it is currently
   **not** installed.

3. Create an empty `conftest.py` at the repo root. Its mere existence makes pytest treat the
   root as `rootdir` and put it on `sys.path`.

4. Move `config/root.py` → `src/paths.py`. It is a filesystem utility, not experiment
   configuration; leaving it in `config/` is what makes that package's purpose ambiguous.
   Update the 6 importers (`scripts/*.py`, `tests/**`) — grep for `config.root`.

5. Install the project in editable mode:
   ```bash
   pip install -e . --no-build-isolation
   ```
   or
   ```bash
   pip install -e .
   ```
   to use a seperately built environment
6. Delete the hand-rolled `sys.path` manipulation from the top of the test files.

### ✅ Verify

```bash
cd scripts && python -c "import src.learning.models.group_encoder; print('ok from scripts/')" && cd ..
python -m pytest tests -q
```
Both must pass. The first one is the real test — it would have failed before.

### 🎓 What you learned

**Practice: eliminate ambient state.** A dependency you cannot see in the code (CWD, an env
var, global RNG state) is a dependency you cannot reason about. Packaging turns "it works if
you launch it right" into "it works."

**When to do this:** the moment a project has more than one entry point, or any tests.

**When not to bother:** a single-file script. Packaging a 50-line utility is ceremony.

---

## T2 — Pin current behavior with characterization tests

**⏱ 2 h · 🟡 medium · ⚠️ do not skip**

### 🎯 Goal
A test suite that fails loudly if any later task changes what the model actually computes.

### 🧩 The pattern
**Characterization Test** (Michael Feathers, *Working Effectively with Legacy Code*) — also
called a golden test or approval test. You are not asserting the code is *correct*. You are
asserting it *does today what it did yesterday*.

### 🔍 Why here
Tracks C, D, and E rewire the training path: the latent head, the loss assembly, the trainer,
the orchestrator. Every one of those can silently change numbers. Without a baseline you will
not know whether a shifted loss curve is a bug you introduced or the intended effect of the
refactor.

### 🔨 Do this

1. Create `tests/characterization/test_baseline.py`.

2. Write a fixture that builds a **fully seeded, tiny** setup — seed everything
   (`random`, `numpy`, `torch`, and the `torch.Generator` the loaders use), 4-6 shapes,
   `latent_dim=5`, ~20 training steps. Look at `scripts/equivariant_gnn_train.py:112-130` for
   how seeding is currently done and mirror it exactly.

3. Pin these, each as its own test:

   | What to pin | Why it catches regressions |
   |---|---|
   | Loss value sequence over 20 steps | Any change to loss assembly or the latent path |
   | `mu` / `logvar` shapes out of the encoder | Contract drift (T9, T10) |
   | Decoder output shape `[B, num_samples, 3]` | Decoder contract (T7) |
   | Graph `num_nodes` / `num_edges` for a fixed seed | Graph builder refactor (T3, T4) |
   | Number of trainable parameters | Accidentally added or dropped submodules |

4. Store the expected numbers in a JSON file next to the test, not inline. When an *intended*
   change lands, you regenerate the file and the git diff shows exactly what moved.

5. Use `pytest.approx` with a tolerance (`rel=1e-5`). Exact float equality across devices is
   a losing battle.

### ✅ Verify

```bash
python -m pytest tests/characterization -q          # passes
python -m pytest tests/characterization -q          # passes again — identical numbers
```
Run it **twice**. If the two runs disagree, you have unseeded randomness — find it before
continuing. (Candidate: `src/learning/helpers.py:46-47`, where `load_dataset` calls the
process-global `random.seed`.)

### 🎓 What you learned

**Practice: characterize before you change.** The tests you write here are deliberately
*not* about correctness. They are a tripwire.

**When to use:** before refactoring any code you did not write, or wrote long enough ago that
you no longer hold it in your head.

**When NOT to use:** when the current behavior is known-wrong. Pinning a bug makes it
permanent. Two cases in this repo are known-wrong and must **not** be pinned:
- validation reparameterizes with random noise (`E3_end2end.py:135` → `67`) — T10 fixes it
- `run_validation` discards `avg_recon_loss` / `avg_kl_loss` (`E3_end2end.py:219-226`) — T11 fixes it

Write those two down as *expected* diffs now, so future-you doesn't mistake them for damage.

---
---

# Track B — Practice two patterns on low-risk code

The graph/data path is a good place to learn: it is well isolated and your T2 tests cover it.

---

## T3 — Collapse the 17-parameter loader with a Parameter Object

**⏱ 1.5 h · 🟡 medium**

### 🎯 Goal
`ResamplingGraphLoader.__init__` goes from 17 parameters to 6.

### 🧩 The pattern
**Parameter Object** (Fowler, *Refactoring*). When several parameters always travel together,
they are not separate parameters — they are one concept that has not been named yet. The
smell is called a **Data Clump**.

### 🔍 Why here
Open `src/learning/loader/loaders.py:41-45`:

```python
def __init__(self, vertices, mask, build_fn,
             key=None, r_max=0.2, r_supergraph=0.6, dropout_rate=0.8,
             use_supernodes=False, two_view=False, n_supernodes=15,
             sampling_mode_graph="uniform", sampling_mode_supernodes="uniform",
             areas=None, normals=None, recompute_area=True, batch_size=None):
```

Now look at `_draw` (line 89-101): **twelve** of those are forwarded, untouched, straight into
`build_fn`. The loader does not *use* `r_max`. It does not know what a radius is. It is acting
as a postman. And the same twelve appear again in `build_training_graph`
(`src/learning/helpers.py:128-143`) and again as constants in both training scripts.

That is the data clump: one concept — "how to build a graph from geometry" — smeared across
three signatures.

### 🔨 Do this

1. Create `src/learning/data/graph_spec.py`.

2. Define the parameter object. Note it is a **frozen** dataclass — a spec is a value, not
   mutable state:

   ```python
   # SKELETON — type this, fill in the rest yourself.
   from dataclasses import dataclass

   @dataclass(frozen=True)
   class GraphSpec:
       """Everything needed to turn raw geometry into the graph the encoder consumes.

       Grouped because these twelve values always travel together (loader ->
       build_training_graph -> get_graphs_from_vertices).
       """
       r_max: float = 0.2
       r_supergraph: float = 0.6
       dropout_rate: float = 0.8
       # TODO: n_supernodes, use_supernodes, sampling_mode_graph,
       # TODO: sampling_mode_supernodes, recompute_area, area_k, noise_std
   ```
   Get the full field list by reading `build_training_graph`'s signature
   (`helpers.py:128-143`). Use its defaults.

3. Handle the range-sampling wrinkle: `r_max` and `dropout_rate` may each be a fixed float
   **or** a `(low, high)` tuple sampled per step (see `_sample`, `loaders.py:63-69`). Keep
   that capability — type the fields as `float | tuple[float, float]` and give `GraphSpec` a
   method that resolves a concrete spec given an RNG.

4. Change `ResamplingGraphLoader.__init__` to
   `(vertices, mask, build_fn, spec, rng=None, two_view=False, batch_size=None)`.

5. Update the two call sites in `scripts/equivariant_gnn_train.py:194-224`.

### ✅ Verify

```bash
python -m pytest tests/characterization tests/graphs -q
```
Graph node/edge counts must be **unchanged** — this is a pure restructuring.

### 🎓 What you learned

**Practice: name the clump.** Long parameter lists are usually a missing concept. Once
`GraphSpec` exists, it becomes the obvious thing to put in a config file (T5), to hash for a
run id (T13), and to log in a manifest. None of that was possible when it was twelve loose
floats.

**When to use:** three or more parameters that repeatedly appear together across signatures.

**When NOT to use:** parameters that merely *co-occur* in one function but aren't a
coherent concept. Bundling unrelated arguments into a bag object makes signatures *less*
informative — you have replaced a clear list with an opaque blob. The test is: can you give
the group an honest name? "GraphSpec" — yes. "Options" or "Params" — that's a smell, not a
concept.

---

## T4 — Introduce GraphBuilder with the Strategy pattern

**⏱ 2 h · 🟡 medium**

### 🎯 Goal
Switching from radius-based to kNN graph construction is a config value, not an edit.

### 🧩 The pattern
**Strategy.** Define a family of interchangeable algorithms behind one interface; let the
caller pick at runtime. Closely related: *Replace Conditional with Polymorphism*.

### 🔍 Why here
Today graph construction is a function with a boolean: `build_training_graph(...,
use_supernodes=False, ...)` branches internally (`helpers.py:158-166`). Adding kNN means
adding another flag and another branch. Two flags = four paths, most untested.

Strategy replaces "one function with flags" with "several small classes, one interface."

### 🔨 Do this

1. Create `src/learning/data/builders.py` — this is the honest rewrite of the file you
   deleted in T0.

2. Define the interface. Note the RNG is a **parameter**, never stored — this is what makes
   builders reproducible:

   ```python
   # SKELETON — type this yourself.
   from abc import ABC, abstractmethod

   class GraphBuilder(ABC):
       """Strategy: geometry -> (graph, supergraph|None)."""

       @abstractmethod
       def build(self, vertices, mask, rng, areas=None, normals=None):
           """Returns (graph, supergraph). `supergraph` may be None.

           `rng` is passed in, never stored: same rng state + same spec => same graph.
           """
           ...
   ```

3. Implement `RadiusGraphBuilder(spec: GraphSpec)`. Its `build` should call the **existing**
   `get_graphs_from_vertices` and `build_super_graph` from `src/graphs/graphs.py`. You are
   *moving* logic out of `build_training_graph`, not writing new geometry code.

4. Leave `KNNGraphBuilder` as a stub that raises `NotImplementedError` with a clear message.
   A second implementation is what proves the abstraction is real — but a stub is honest
   until someone needs it.

5. Make `ResamplingGraphLoader` accept a `GraphBuilder` instead of `build_fn` + `spec`. The
   loader now calls `builder.build(...)` and knows nothing about radii or supernodes.

### ✅ Verify

```bash
python -m pytest tests/characterization tests/graphs -q
python -c "from src.learning.data.builders import RadiusGraphBuilder; print(RadiusGraphBuilder.__mro__)"
```

### 🎓 What you learned

**Practice: push variation behind an interface.** The caller states *what* it wants (a graph);
the strategy decides *how*. New algorithms arrive as new classes — existing code is untouched.
That is the **Open/Closed Principle**: open for extension, closed for modification.

**When to use:** you have (or clearly will have) two or more interchangeable ways to do one
job, selected at runtime.

**When NOT to use:** when there is exactly one implementation and no concrete second one in
sight. An interface with a single implementer is indirection with no payoff — it makes the
reader jump through a file to find the only thing that could have happened. Wait for the
second case. (This repo genuinely has second cases: two encoders, two decoders, two
transformers, two latent modes. That is why these particular abstractions are justified.)

---
---

# Track C — Build the configuration spine

Now the payoff structure: config → registry → factory. This is what turns experiments into
data instead of code.

---

## T5 — Reconcile the config schema with reality

**⏱ 2 h · 🟡 medium**

### 🎯 Goal
Every hyperparameter in both training scripts has exactly one home in a dataclass, and the
schema matches the constructors it feeds.

### 🧩 The pattern
**Value Object** + **fail-fast validation at the boundary**.

### 🔍 Why here
`config/models.py` (kept in T0) was written from the *plan*, not from the *code*, so it
disagrees with reality in four ways:

| Schema says | Code actually does |
|---|---|
| `layers` = list of **2** `EncoderLayerConfig` | `GroupEncoder` builds **1** `EquiLayer` (`group_encoder.py:41-47`) |
| — | `GroupEncoder` takes `area_pool`; the schema has no such field |
| `num_epochs`, `checkpoint_every` | Loaders are infinite generators; the orchestrator runs **steps** (`run(num_steps, log_every, save_every, val_every)`) |
| `use_supernodes` on `DataConfig` | Also implied by `EncoderConfig`; it belongs to `GraphSpec` (T3) |

A schema that lies is worse than no schema.

### 🔨 Do this

1. Open `scripts/equivariant_gnn_train.py:49-105` and list all ~28 constants. Every one needs
   a home. This list is your acceptance criterion.

2. Fix the time axis: replace `num_epochs`/`checkpoint_every` with `num_steps`, `log_every`,
   `save_every`, `val_every`, matching `TrainingOrchestrator.run`.

3. Fix the encoder config: add `area_pool`, add `latent_mode` (`"gaussian" | "deterministic"`
   — you'll use it in T9). Decide the `layers` question: either make it a single
   `EncoderLayerConfig`, or keep the list **and** make T7's factory actually build one
   `EquiLayer` per entry. Do not leave it decorative.

4. Embed `GraphSpec` (T3) in `DataConfig` rather than re-listing its fields.

5. Add a `validate()` method to `ExperimentConfig`. It must **reject**:
   - `latent_mode="deterministic"` together with a `kl` loss term (KL is undefined without a posterior)
   - `num_samples` that is not a perfect square (`FoldingDecoder` builds a square grid — `folding_decoder.py`)
   - a loss term whose name the trainer never produces
   - encoder token count ≠ decoder `expects_tokens` (see T7)

   and **warn** on: `frobenius` and `contrastive` both large — one pushes latent norm down
   while the other pushes per-dimension spread up.

### ✅ Verify

```bash
python -m pytest tests/config -q
python -c "from config.models import ExperimentConfig; c = ExperimentConfig(); c.validate(); print('valid config ok')"
```
Also write a test asserting an *invalid* combination raises. Validation you never see fire is
validation you don't know works.

### 🎓 What you learned

**Practice: validate at the boundary, fail before the expensive part.** A config error should
surface in the first 50 ms, not at step 300 of a GPU run. The cost of a bug scales with how
late you find it.

**When to use:** any time input crosses into your system from outside (a file, a CLI, a user).

**When NOT to use:** don't re-validate the same invariant at every internal layer. Validate
once at the edge, then trust your own types. Defensive checks scattered through internals are
noise that hides the one check that matters.

---

## T6 — A lazy component Registry

**⏱ 1.5 h · 🟡 medium**

### 🎯 Goal
`"equiformer"` in a config file resolves to a class, without importing equiformer for runs
that don't use it.

### 🧩 The pattern
**Registry** (a name→factory table) with **lazy import**. This is how plugin systems work.

### 🔍 Why here
Two problems, one design.

*Correctness:* the registry you deleted in T0 failed because it imported classes eagerly at
module scope and referenced three names it never imported. Any registry that does
`from ... import X` at the top for every component is one typo away from being unimportable —
and is prone to circular imports, since components may want to reference the registry.

*Cost:* eagerly importing every transformer pulls in `equiformer_v3` — `so3.py` (602 lines),
`transformer_block.py` (754 lines) — on **every** run, including `se3` runs that never touch it.

Storing `"module:qualname"` **strings** and importing on first use fixes both.

### 🔨 Do this

1. Create `src/learning/registry.py`.

2. Sketch:

   ```python
   # SKELETON — type this yourself.
   from importlib import import_module

   class Registry:
       """Maps (category, name) -> "module:QualName", resolved on first use."""

       _entries: dict = {}

       @classmethod
       def register(cls, category: str, name: str, target: str) -> None:
           """`target` is a STRING like "src.learning.models.folding_decoder:FoldingDecoder".
           Storing a string (not the class) is what keeps the import lazy."""
           ...

       @classmethod
       def create(cls, category: str, name: str, **kwargs):
           """Resolve, import, instantiate. Unknown name -> ValueError listing valid names."""
           ...

       @classmethod
       def available(cls, category: str) -> list:
           """Names in a category — powers --help and error messages."""
           ...
   ```

3. Register the components that exist today. Categories: `transformer`, `decoder`,
   `latent_head`, `graph_builder`, `readout`. Verify each target string against the real file
   — e.g. `FoldingDecoder` and `SphereFoldingDecoder` both live in
   `src/learning/models/folding_decoder.py`.

4. Make the error message for an unknown name list the available ones. This is a small thing
   that saves real time.

### ✅ Verify

```bash
python -c "import sys; from src.learning.registry import Registry; d = Registry.create('decoder', 'folding', num_samples=256, latent_dim=5); assert 'equiformer_v3' not in ' '.join(sys.modules), 'lazy import broken'; print('lazy registry ok:', Registry.available('decoder'))"
```
That assertion is the whole point of the task — it proves you didn't pay for what you didn't use.

### 🎓 What you learned

**Practice: indirection through data.** A registry turns "which class?" from a code decision
into a data lookup, which means a config file can make it.

**When to use:** an open-ended set of interchangeable implementations, especially when you
want people to add more without editing your code.

**When NOT to use:** a small, closed, stable set. If there are exactly two options and there
always will be, a plain dict literal or an `if/elif` is clearer — a registry makes you chase
a string through an indirection layer to learn what actually runs. Registries also defeat
"find all references" in your IDE and push errors from import time to runtime. Pay that price
only when extensibility is a real requirement.

---

## T7 — Factories, and the dependency rule

**⏱ 1.5 h · 🟡 medium**

### 🎯 Goal
One module translates config into constructed objects. Components stay ignorant of config.

### 🧩 The pattern
**Factory Function** + **Dependency Inversion** (the "D" in SOLID).

### 🔍 Why here
This is the most important architectural rule in the whole refactor:

> **Components (L1) must never import config (L2).**

Layers, per `REFACTORING_PLAN.md` §2.1:
```
L2  assembly     config schemas, registry, factories
L1  components   models/, modules/, losses/, data/     <- takes plain arguments
L0  primitives   geometry/, graphs/, vtk/
```

The deleted `composer.py` violated this: a loss module importing `LossConfig`. Once a
component imports the config package, you can no longer use it without that package, test it
without constructing a config, or reuse it in another project.

The litmus test: **"could I use this class in a different project without copying the config
package?"**

### 🔨 Do this

1. Create `src/learning/factories.py`.

2. Sketch:

   ```python
   # SKELETON — type this yourself.
   def build_encoder(cfg):
       """EncoderConfig -> encoder module. The ONLY place encoder config fields are read."""
       ...

   def build_decoder(cfg):
       """DecoderConfig -> decoder module."""
       ...

   def build_from_config(cfg):
       """ExperimentConfig -> (encoder, decoder). Validates the pairing (step 4)."""
       ...
   ```

3. `build_encoder` reads config fields and passes **plain values** to `GroupEncoder`. Compare
   with the real signature (`group_encoder.py:16-25`) — every parameter must be accounted for,
   including `area_pool`, which the old pasted factory dropped.

4. **Add the build-time compatibility check.** `FoldingDecoder` raises if handed more than one
   latent token (`folding_decoder.py:57-61`), but `GroupPerceiverEncoder` can emit `[B, K, d]`.
   Today that mismatch explodes mid-training. Have encoders expose `n_tokens` and decoders
   expose `expects_tokens`, and compare them here, before returning.

5. Delete any config import that snuck into an L1 module.

### ✅ Verify

```bash
# The dependency rule, enforced mechanically:
grep -rn "import config\|from config" src/learning/models src/learning/losses src/learning/modules src/learning/data && echo "VIOLATION" || echo "dependency rule holds"
python -m pytest tests/characterization -q
```
Consider promoting that grep into a test so the rule can't silently rot.

### 🎓 What you learned

**Practice: depend on abstractions, not on your own configuration system.** Config is an
*input format*; it should sit at the outside of your system, and the inside should never
know it exists.

**When to use factories:** whenever construction takes more than a couple of arguments or
involves choices — keep that knowledge in one place instead of at every call site.

**When NOT to use:** don't wrap a one-line constructor in a factory. `build_thing()` that
only does `return Thing()` is a layer that costs a file-jump and buys nothing.

---
---

# Track D — Deliver the features

Everything so far was structure. Now the structure pays: composable losses, an auto-encoder
mode, and validation you can actually read.

---

## T8 — LossComposer (Composite pattern)

**⏱ 1.5 h · 🟡 medium**

### 🎯 Goal
Adding a loss term is a config line. The trainer never changes.

### 🧩 The pattern
**Composite** — treat a collection of things as one thing. Many weighted terms behave like a
single loss.

### 🔍 Why here
`E3_end2end.py:82-94`:

```python
loss = recon_loss
if kl is not None and self.kl_weight:
    loss = loss + self.kl_weight * kl
if self.contrastive_weight:
    loss = loss + self.contrastive_weight * contrastive
```
Every new term = a new constructor argument, a new `if`, a new return-tuple slot. Three terms
already; the Frobenius term in T9 would make four.

### 🔨 Do this

1. Create `src/learning/losses/composer.py` — the honest rewrite of the file deleted in T0.

2. **Do not import the config package** (T7's rule). The composer takes a plain sequence of
   term specs. If you want a small dataclass for a term, define it *in the losses package*
   and let the config layer convert into it.

3. Sketch:

   ```python
   # SKELETON — type this yourself.
   class LossComposer:
       """Composite: weighted sum of named terms -> (total, per-term breakdown)."""

       def __init__(self, terms):
           """`terms`: sequence of (name, weight, kwargs). No config import here."""
           ...

       def compute(self, values):
           """`values`: {name: Tensor | None}. Returns (total_scalar, {name: float}).

           A term whose value is None is SKIPPED, not an error — that is how `kl`
           vanishes in auto-encoder mode and `contrastive` vanishes during validation.
           """
           ...
   ```

4. Three invariants to get right:
   - `total` must be a **scalar** tensor. (The deleted draft seeded it with `torch.zeros(1)` —
     shape `[1]`, not scalar.)
   - Keep the non-finite guard that exists today; it is what stops a NaN run training silently
     to all-NaN weights.
   - `breakdown` keys must be identical in structure for train and val, so the two are directly
     comparable in T11.

5. Add `frobenius_latent_loss(Z)` to `src/learning/losses/losses.py`: `‖Z‖²_F / B`, where
   `Z = [z_i]` is the batch latent matrix `[B, D]`. Mean over `B` (not sum) keeps the weight
   batch-size independent — matching how `kl_divergence_loss` already reduces. You need this
   in T9.

### ✅ Verify

```bash
python -m pytest tests/learning/test_losses.py tests/characterization -q
```
Add a unit test: a composer with `{"recon": t, "kl": None}` must return `recon` alone and a
breakdown with one key.

### 🎓 What you learned

**Practice: replace conditional accumulation with composition.** The `if weight:` chain
encodes policy in the trainer. Moving it to data means the trainer expresses *mechanism*
only — and mechanism is what you want to be stable.

**When to use:** a variable-length collection of things combined uniformly.

**When NOT to use:** when the terms are *not* uniform — if one "term" needs the optimizer, or
must run before backward, it isn't a peer and forcing it into the composite will contort the
interface. Keep genuinely different things different.

---

## T9 — LatentHead: the auto-encoder mode (Strategy, again)

**⏱ 2.5 h · 🔴 harder**

### 🎯 Goal
`latent_mode: deterministic` gives a non-variational auto-encoder regularized by the Frobenius
norm, with **no** conditional logic in the trainer.

### 🧩 The pattern
**Strategy** — the same pattern as T4, now at a deeper seam. Recognizing a familiar pattern in
unfamiliar code is the skill being practiced here.

### 🔍 Why here
This is ToDo item #3. The obstacle is that "be a VAE" is currently hardwired in **three**
places:

| Location | What it hardwires |
|---|---|
| `group_encoder.py:89-90, 194-199` | `mu_net` + `var_net` → `mu, logvar` |
| `modules/latent_vae.py` | `LatentVAEHead` — a second copy, for the Perceiver encoder |
| `E3_end2end.py:66-67, 140-143` | trainer reads `.mu` and reparameterizes — a third copy |

`reparameterize` is implemented three times. Meanwhile
`EncoderOutput.sample()` (`encoder_output.py:27-47`) **already** returns `latent` when
`mu is None`, and `EncoderOutput.kl()` **already** returns `None` in that case. The
deterministic path exists in the contract and nothing uses it.

### 🔨 Do this

1. Create `src/learning/models/latent_heads.py`.

2. Two strategies behind one interface:

   ```python
   # SKELETON — type this yourself.
   import torch.nn as nn

   class GaussianLatentHead(nn.Module):
       """VAE head — today's behavior. -> EncoderOutput(mu=..., logvar=...)"""
       def forward(self, scalars, weights, batch, num_graphs):
           ...

   class DeterministicLatentHead(nn.Module):
       """Auto-encoder head. -> EncoderOutput(latent=z), mu=None, logvar=None.

       Same [B, latent_dim] output shape as the Gaussian head, so the decoder
       contract is unaffected and latent_mode becomes a pure ablation switch.
       """
       def forward(self, scalars, weights, batch, num_graphs):
           ...
   ```

3. Move the existing `mu_net`/`var_net` logic out of `GroupEncoder` and into
   `GaussianLatentHead` **unchanged**. This step must be behavior-preserving — your T2 tests
   are the proof.

4. `DeterministicLatentHead` reuses the same scalar-pooling path; it just drops the `var_net`
   and the sampling. Keep the `[B, latent_dim]` output shape — the assert at
   `group_encoder.py:169` and the decoder both depend on it.

5. `GroupEncoder` now selects its head via the registry (T6) from `cfg.latent_mode`.

6. Register both heads under the `latent_head` category.

7. Refactor `LatentVAEHead` into the token-set variant of `GaussianLatentHead` — that removes
   the second copy.

### ✅ Verify

```bash
# gaussian mode must be bit-identical to before:
python -m pytest tests/characterization -q
# deterministic mode must build and step:
python -m pytest tests/learning -k "deterministic or latent_head" -q
```

### 🎓 What you learned

**Practice: a feature request is often an abstraction request.** "Add an auto-encoder mode"
sounds like new functionality. It is really a request to name a variation point that was
hardcoded. Once named, the feature is small.

**When to use Strategy for a feature:** when the new behavior is a *sibling* of existing
behavior — same inputs, same outputs, different rule.

**When NOT to:** when the "variant" has a genuinely different interface. Forcing a
non-conforming case into a shared interface produces methods that some implementations must
stub out or reject — a violation of the **Liskov Substitution Principle**, and a sign you
needed two interfaces, not one.

---

## T10 — Consolidate the latent seam in the trainer

**⏱ 1 h · 🟡 medium · ⚠️ intentional behavior change**

### 🎯 Goal
The trainer stops reading `.mu`, the duplicate `reparameterize` disappears, and auto-encoder
mode works end to end without a single `if`.

### 🧩 The pattern
**Adapter / Anti-Corruption Layer** — `EncoderOutput` is the boundary object that keeps the
trainer ignorant of what kind of encoder it holds.

### 🔍 Why here
`E3_end2end.py:65-67`:
```python
enc = self.encode(graph, super_graph)
mu = enc.mu                                # VAE-specific
latent = self.reparameterize(mu, enc.logvar)
```
In auto-encoder mode `enc.mu` is `None` and this crashes. The fix is not to add a branch —
it is to use the seam that already exists.

### 🔨 Do this

1. In `_encode_decode`, replace those lines with a single call to
   `enc.sample(deterministic=...)`. It handles both modes: VAE → reparameterized sample
   (training) or `mu` (eval); AE → the deterministic latent.

2. Pass `deterministic=True` on the validation path. **This is an intentional behavior
   change**: `eval_step` currently injects random noise under `no_grad`
   (`E3_end2end.py:135` → `67`). Validation should be deterministic. Expect VAE validation
   curves to shift slightly — you flagged this in T2 as an expected diff.

3. Delete `TrainingStepper.reparameterize` (line 140-143). One implementation survives:
   `EncoderOutput.sample`.

4. In the two-view contrastive path (line 103-112), use
   `enc.sample(deterministic=True)` instead of `mu_a` / `mu_b`, so contrastive works in AE
   mode too.

5. Build the loss dict and hand it to the T8 composer. `kl` comes from `enc.kl()` — `None` in
   AE mode, so the composer skips it. `frobenius` gets the latent matrix `Z`.

6. Move `kl_weight` / `contrastive_weight` out of `TrainingStepper.__init__` (line 34-36).
   Weights live in the loss config now.

### ✅ Verify

```bash
python -m pytest tests/characterization tests/learning/test_trainer_e2e.py -q
grep -n "def reparameterize" src/learning/trainers/E3_end2end.py && echo "STILL THERE" || echo "consolidated"
grep -n "enc\.mu\|\.mu\b" src/learning/trainers/E3_end2end.py    # should find nothing
```

### 🎓 What you learned

**Practice: use the seam you already have.** The single most valuable move in this whole
refactor is not new code — it is routing through `EncoderOutput.sample()`, a method that was
already written, already correct, and already handled both cases. Before building an
abstraction, check whether someone left you one.

**Practice: duplication of *logic* is worse than duplication of *text*.** Three
`reparameterize` implementations meant three places to fix a bug — and indeed one of them
(the trainer's) had the eval-noise bug the other two didn't.

**When NOT to consolidate:** when two similar-looking pieces of code are similar by
coincidence rather than by shared meaning. Merging those couples things that should evolve
separately, and the next change has to tear them apart again.

---

## T11 — Verbose, split validation loss

**⏱ 45 min · 🟢 easy**

### 🎯 Goal
Validation logs every loss component, not one scalar.

### 🔍 Why here
This is ToDo item #1, and it is nearly free after T8. Look at
`E3_end2end.py:203-226` — the code **already computes** `avg_recon_loss` and `avg_kl_loss`,
assembles them into a `metrics` dict... and then logs only:

```python
self.logger.log_metrics({"val_loss": avg_val_loss}, step)
```
The split is computed and thrown away.

### 🔨 Do this

1. Have `eval_step` return the composer breakdown (T8) instead of a fixed tuple.

2. In `run_validation`, accumulate **every** key across validation batches.

3. Log with a consistent prefix — `val/recon`, `val/kl`, `val/frobenius` — mirroring
   `train/recon`, `train/kl`. Symmetry is what makes the plot readable.

4. `TrainingLogger.plot_metrics` already plots every series in its history, so per-term
   train-vs-val curves appear with no plotting changes.

### ✅ Verify

```bash
python -m pytest tests/learning/test_validation.py -q
```
Then run a short training job and confirm `metrics.json` contains `val/recon` and `val/kl`
keys, not just `val_loss`.

### 🎓 What you learned

**Practice: observability is a design property, not an afterthought.** You could not have
logged per-term validation before T8, because the terms weren't separable — they were fused
in an `if` chain. Good structure makes good instrumentation cheap.

**When debugging an ablation, aggregate metrics lie.** A flat total loss can hide recon
improving while KL degrades. Always split what you sum.

---
---

# Track E — Orchestration and experiments

---

## T12 — Callbacks (Observer pattern)

**⏱ 2.5 h · 🔴 harder**

### 🎯 Goal
The training loop contains only the loop. Logging, checkpointing, visualization, and
validation become pluggable.

### 🧩 The pattern
**Observer** (a.k.a. hooks, callbacks, listeners) — an object emits events; interested
parties subscribe. It's **Inversion of Control**: the loop no longer decides what happens,
only *when*.

### 🔍 Why here
Two overloaded classes.

`TrainingLogger` (`train_logs.py`) does **five** jobs: metric history, JSON persistence,
matplotlib plotting, VTP geometry export, and checkpointing. You cannot change the metrics
format without touching the class that also writes 3D geometry.

`TrainingOrchestrator.run` (`E3_end2end.py:159-189`) hardcodes cadence *policy*: four
`if step % N == 0` blocks. Adding early stopping or an LR schedule means editing the loop.

There is also a real performance bug hiding here: `log_metrics` calls `_save_metrics`, which
rewrites the **entire** history to JSON on **every** call (`train_logs.py:53-64`). With
`LOG_EVERY=1` and `NUM_STEPS=3001`, that is 3001 full rewrites of a growing file — quadratic
work for something that should be an append.

### 🔨 Do this

1. Create `src/learning/callbacks/base.py`:

   ```python
   # SKELETON — type this yourself.
   class Callback:
       """Observer. Override only the hooks you care about; defaults do nothing."""
       def on_train_start(self, ctx): ...
       def on_step_end(self, ctx, step, breakdown, batch, pred): ...
       def on_validation_end(self, ctx, step, breakdown): ...
       def on_train_end(self, ctx): ...
   ```
   Default no-op bodies matter: a subclass overrides two hooks, not eight.

2. Split `TrainingLogger` into one callback per responsibility: `MetricsRecorder`,
   `MetricsPlotter`, `CheckpointWriter`, `GeometryVisualizer`, `ValidationRunner`.

3. Give each callback its **own** cadence (`every_n_steps`), instead of the orchestrator
   holding `log_every` / `save_every` / `val_every`. Policy moves to the party that
   implements it.

4. Fix E1 while you're in `MetricsRecorder`: write **append-only JSONL**, one record per
   event — `{"step":…, "split":…, "term":…, "value":…}`. Constant-time writes, crash-safe,
   and trivially loadable for cross-run comparison in T13.

5. Reduce `TrainingOrchestrator.run` to: get batch → step → fire hooks.

### ✅ Verify

```bash
python -m pytest tests/learning/test_logger.py tests/characterization -q
```
Then prove the pattern works: write an `EarlyStoppingCallback` **without editing the
orchestrator**. If you can't, the abstraction isn't finished.

### 🎓 What you learned

**Practice: separate mechanism from policy.** The loop is mechanism (stable). What to do at
step 100 is policy (volatile). Keeping volatile things out of stable things is most of what
architecture *is*.

**When to use Observer:** several independent things must react to the same events, and you
want to add reactions without touching the emitter.

**When NOT to use:** with one listener and no prospect of a second, a direct call is clearer.
Callbacks also make control flow **non-local** — reading `run()` no longer tells you what
happens, and ordering bugs between callbacks are genuinely hard to debug. Don't reach for
this until you actually have several listeners. (Here you have five.)

---

## T13 — Experiment runner, CLI, and run manifests

**⏱ 3 h · 🔴 harder**

### 🎯 Goal
`python scripts/train.py configs/baseline.yaml --set encoder.latent_mode=deterministic`
runs a complete experiment. No hyperparameter lives in a `.py` file.

### 🧩 The pattern
**Composition Root** — one place, at the outermost edge, where the whole object graph is
assembled. Plus **reproducibility-by-manifest**.

### 🔍 Why here
`scripts/equivariant_gnn_train.py` and `scripts/perceiver_train.py` duplicate **19**
module-level constants and the entire `main()` skeleton. Every new experiment currently means
copying a 276-line script. That is the root cause the whole refactor has been working toward.

### 🔨 Do this

1. Create `src/learning/runner.py`: config → components (T7) → data (T4) → orchestrator
   (T12) → run. It reads config; nothing below it does.

2. Create `configs/baseline.yaml` reproducing `equivariant_gnn_train.py` exactly. Your T2
   characterization tests verify "exactly."

3. Create `scripts/train.py` — a thin CLI: config path, `--seed`, `--device`, and
   `--set key.path=value` overrides. Target ~60 lines and **zero** hyperparameters.

4. **Override resolution**: defaults → file → `--set`. An unknown key path must raise. A
   silently-ignored typo'd override is the classic reason an ablation "shows no effect."

5. **Run identity**: hash the fully-resolved config to a short `run_id`. Same config ⇒ same
   id. The diff between two configs *is* the description of the ablation.

6. **Run manifest**: before training starts, write the resolved config, git SHA + dirty flag,
   package versions, device, seed, and the overrides used. Without this, results are not
   attributable to inputs — and an unattributable result is not a result.

7. Once `configs/` reproduces both scripts, delete them.

### ✅ Verify

```bash
python scripts/train.py configs/baseline.yaml --set training.num_steps=20
python scripts/train.py configs/ablations/ae_frobenius.yaml --set training.num_steps=20
python scripts/train.py configs/baseline.yaml --set encoder.nonexistent=1   # must FAIL loudly
```
Then the real test: run the same config twice and confirm identical `run_id` and identical
metrics.

### 🎓 What you learned

**Practice: push construction to the edges, keep the core pure.** Everything inside receives
what it needs; only the composition root decides. This is why the core became testable —
tests are just an alternative composition root.

**Practice: an experiment is data, not code.** Once a run is fully described by a config plus
a seed, experiments become diffable, sweepable, reproducible, and comparable. That is the
entire point of Tracks A–D.

**When NOT to go further:** stop here. Don't adopt a config framework (Hydra, OmegaConf) yet —
dataclasses plus a small override resolver cover this project's needs, and a framework brings
its own concepts to learn and debug. Add one when you feel concrete pain, not in anticipation.

---
---

# Reference

## Pattern cheat-sheet

| Pattern | Task | Use when | Avoid when |
|---|---|---|---|
| Characterization Test | T2 | Changing code you can't fully predict | Current behavior is known-wrong |
| Parameter Object | T3 | 3+ params travel together and have an honest name | The group is coincidental — "Options" isn't a concept |
| Strategy | T4, T9 | 2+ interchangeable algorithms, chosen at runtime | Only one implementation exists |
| Registry (lazy) | T6 | Open-ended plugin set; import cost matters | Small closed set — a dict is clearer |
| Factory | T7 | Construction is non-trivial or config-driven | It would just wrap one constructor |
| Dependency Inversion | T7 | Always, between layers | — |
| Composite | T8 | Uniform collection combined as one | The parts aren't really peers |
| Adapter / ACL | T10 | Shielding a core from varying external shapes | Only one external shape exists |
| Observer | T12 | Several independent reactions to one event | One listener; non-local flow costs more than it saves |
| Composition Root | T13 | Any app with more than one entry point | Single-purpose script |

## Two ideas that show up in almost every task

**Open/Closed Principle** — code should be open to extension, closed to modification. Every
registry entry, strategy, and callback in this guide exists so that *adding* a capability
doesn't mean *editing* working code. Test: "to add a second decoder, how many existing files
must I change?" The target is one registration line.

**Separate mechanism from policy** — mechanism is *how* (the training loop, the weighted sum);
policy is *what and when* (which losses, which cadence, which latent mode). Policy changes
constantly; mechanism shouldn't. Nearly every task here moves policy out of mechanism and into
data.

## If you get stuck

| Symptom | Likely cause |
|---|---|
| `ImportError` on your own module | T1 incomplete — reinstall with `pip install -e .` |
| Characterization tests fail after a "pure" refactor | You changed behavior. Diff the pinned JSON — it names the metric that moved |
| Same config, different results across runs | Unseeded RNG. Check `helpers.py:46-47` and any `torch.randn` without a generator |
| Circular import involving the registry | You stored a class instead of a `"module:qualname"` string (T6) |
| Config field silently ignored | The factory never reads it (T7). Add a test asserting the field reaches the constructor |

## Task → plan section map

| Task | `REFACTORING_PLAN.md` |
|---|---|
| T0, T1, T2 | §0.2, §1.1 F1–F2, Phase 0 |
| T3, T4 | §3.4, Phase 5 |
| T5 | §1.3, Phase 1 |
| T6, T7 | §2.1, §2.3, Phase 2 |
| T8 | §3.5, Phase 3 |
| T9, T10, T11 | §3.3, Phase 4, `TODO_INTEGRATION_PLAN.md` |
| T12 | §3.6, Phase 6 |
| T13 | §5, Phase 7 |
