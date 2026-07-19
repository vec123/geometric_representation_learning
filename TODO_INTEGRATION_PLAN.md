# ToDo → Refactor Integration Plan

How the concepts in `ToDo.md` map onto the architecture in `REFACTORING_PLAN.md`,
with reusability, maintainability, and **ablation studies** as the design targets.

The `ToDo.md` items:

1. **Val loss more verbose — split the components.**
2. **Change structure for reusability.** (the refactor itself — see `REFACTORING_PLAN.md`)
3. **Auto-encoder mode (not variational), regularized by the Frobenius norm of the latent
   code matrix `Z = [z_i]`.**

Item 2 is the whole refactor. Items **1** and **3** are new *behavior* that must slot into
the refactor cleanly. The insight below is that **#3 is the design driver**: doing it right
forces one clean abstraction (a swappable *latent head* + a swappable *regularizer*) that
also makes #1 fall out for free and turns "VAE vs AE" into a one-line ablation.

---

## 0. The one abstraction that unifies all three: the latent seam

Today the latent step is hardwired VAE in three duplicated places:

| Location | What it does |
|---|---|
| `GroupEncoder` (`group_encoder.py:89-90, 194-199`) | `mu_net` + `var_net` → `mu, logvar` |
| `LatentVAEHead` (`latent_vae.py`) | tokens → `mu, logvar, kl` (for the Perceiver encoder) |
| `TrainingStepper` (`E3_end2end.py:66-67, 140-143`) | reads `enc.mu`, reparameterizes |

`reparameterize` is implemented **three times** (trainer, `LatentVAEHead`, `EncoderOutput.sample`),
and the trainer is bolted to `mu`/`logvar` specifically. That is exactly what blocks a
non-variational mode.

**Introduce a `LatentHead` strategy** (Phase 2 of the refactor, new registry category
`latent_head`). Two implementations, one contract — both return an `EncoderOutput`:

```
# interface sketch — not final code
class LatentHead(nn.Module):
    def forward(self, pooled_scalars, batch) -> EncoderOutput: ...

class GaussianLatentHead(LatentHead):   # == today's VAE behavior
    # mu_net + var_net -> EncoderOutput(mu=..., logvar=...)
    # regularizer: KL   (enc.kl() already handles this)

class DeterministicLatentHead(LatentHead):   # NEW: auto-encoder
    # code_net -> EncoderOutput(latent=z)     (mu=None, logvar=None)
    # regularizer: Frobenius (or none)
```

Registered:

```
ComponentRegistry.register('latent_head', 'gaussian',      GaussianLatentHead)
ComponentRegistry.register('latent_head', 'deterministic', DeterministicLatentHead)
```

`GroupEncoder` stops hardcoding `mu_net`/`var_net` and instead holds
`self.latent_head = ComponentRegistry.create('latent_head', cfg.latent_mode, ...)`.
`LatentVAEHead` becomes the token-set flavor of `GaussianLatentHead` (dedupe, reuse).

**Backward compat:** `latent_mode="gaussian"` reproduces the current mu/logvar path exactly.

---

## Concept 3 — Auto-encoder mode + Frobenius regularization

### 3.1 Encoder side — `DeterministicLatentHead`
Emits `EncoderOutput(latent=z)` with `mu=None`. Then, with **zero** trainer branching:
- `EncoderOutput.sample()` already returns `latent` when `mu is None` (`encoder_output.py:41-42`).
- `EncoderOutput.kl()` already returns `None` when `mu is None` (`encoder_output.py:57-58`).

### 3.2 Trainer side — route through `sample()`, delete the duplicate reparameterize
`_encode_decode` (`E3_end2end.py:65-67`) must change from:

```
mu = enc.mu
latent = self.reparameterize(mu, enc.logvar)      # VAE-only, random even in eval
```

to:

```
code   = enc.sample(deterministic=eval_mode)      # VAE: reparam (train) / mu (eval)
                                                  # AE : returns latent, unconditionally
```

This single change:
- makes AE mode work end-to-end with **no** `if vae/ae` in the trainer,
- removes `TrainingStepper.reparameterize` (consolidate onto `EncoderOutput.sample`),
- **fixes a latent eval bug**: `eval_step` currently reparameterizes with *random* noise
  under `no_grad` (`E3_end2end.py:135`→`67`) — validation should be deterministic
  (`deterministic=True`).

The two-view/contrastive path (`E3_end2end.py:103-112`) switches from `mu_a`/`mu_b` to
`code_a`/`code_b = enc.sample(deterministic=True)`, so contrastive works in **both** modes.

### 3.3 Loss side — Frobenius as a composable term
Add to `losses.py`:

```
def frobenius_latent_loss(Z, squared=True, reduction="mean"):
    # Z = [z_i], the batch latent matrix, shape [B, D]
    # ||Z||_F^2 / B   (mean squared code norm) — the AE analogue of KL's role:
    # bound code magnitude so the decoder can't exploit unbounded latent scale.
```

Design choices to lock in (expose via `LossTermConfig.kwargs` so they're ablatable):
- **Normalize by B** (mean, not sum) → weight is batch-size independent, matching how
  `kl_divergence_loss` already does `.mean()` over B.
- **Squared** `||Z||_F^2` by default (smoother regularizer than the bare norm).
- Optionally per-dim / row-normalized variants later — all live behind the same kwarg.

Register it in the composer (Phase 3):

```
LossComposer.AVAILABLE_LOSSES['frobenius'] = frobenius_latent_loss
```

The trainer passes `Z = code` (the point estimate `enc.sample(deterministic=True)`) into the
`losses_dict`, so Frobenius is meaningful in **both** modes — you can even ablate
"VAE + Frobenius". The composer's breakdown then logs it automatically (→ Concept 1).

### 3.4 Config side — the ablation switch
- `EncoderConfig.latent_mode: str = "gaussian"  # "gaussian" | "deterministic"`
- Regularizer is **not** a field — it is just which term appears in `LossConfig`:

```yaml
# VAE (today)
encoder: { latent_mode: gaussian }
losses:  [{name: recon, weight: 1.0}, {name: kl, weight: 0.1}]

# AE + Frobenius (ToDo #3)
encoder: { latent_mode: deterministic }
losses:  [{name: recon, weight: 1.0}, {name: frobenius, weight: 1.0e-3}]

# AE + Frobenius + contrastive
encoder: { latent_mode: deterministic }
losses:  [{name: recon, weight: 1.0}, {name: frobenius, weight: 1.0e-3},
          {name: contrastive, weight: 0.05}]
```

Switching VAE↔AE or KL↔Frobenius is a **config edit, no trainer change** — this is the core
ablation enablement.

### 3.5 Guardrail — config validation (Phase 1)
Add a consistency check in `ExperimentConfig` validation:
- `latent_mode="deterministic"` **+** a `kl` term with weight>0 → error (`enc.kl()` is `None`).
- `latent_mode="gaussian"` **+** no regularizer at all → warn (unbounded posterior).
- `frobenius` present with weight>0 but no code produced → error.

Keeps the illegal quadrants of the ablation matrix from silently mis-training.

---

## Concept 1 — Verbose, split validation loss (falls out of the composer)

The refactor's `LossComposer.compute(losses_dict)` already returns
`(total, breakdown)` where `breakdown = {"recon": ..., "kl": ..., "frobenius": ...}`.
Route **both** `train_step` and `eval_step` through the composer and the split is free.

- `eval_step` returns `(pred, total, breakdown)` — the same term names as training, minus
  training-only terms (contrastive is skipped because it's absent from the val `losses_dict`).
- `run_validation` accumulates **every** key across val batches and logs `val/<term>` for
  each, plus `val/total` — replacing the single `{"val_loss": ...}` at
  `E3_end2end.py:226` (and using the recon/kl means it currently computes and discards).
- Symmetric logging: `train/recon` vs `val/recon`, `train/frobenius` vs `val/frobenius`, …
  so `plot_metrics` draws per-term train-vs-val curves — which is exactly what you need to
  see *which component* drives val behavior in an ablation.

This is the smallest item and needs no new abstraction — only that the composer becomes the
single place both paths compute loss.

---

## How it lands on the existing phases

| Refactor phase | Additions for ToDo items |
|---|---|
| **Phase 1 — Config** | `EncoderConfig.latent_mode`; `frobenius` as a valid loss name; mode↔loss validation (3.5) |
| **Phase 2 — Registry/Factories** | new `latent_head` category (`gaussian`/`deterministic`); `build_encoder` wires the head; `GroupEncoder` takes an injected head instead of inline `mu_net`/`var_net` |
| **Phase 3 — Loss composer** | add `frobenius_latent_loss`; composer breakdown → verbose train **and** val logging |
| **Phase 4 — Trainer** | route through `EncoderOutput.sample(deterministic=…)`; drop the 3× duplicated reparameterize; build `losses_dict`; both `train_step`/`eval_step` return breakdowns; contrastive uses `code_a/code_b` |
| **Logging** | symmetric `train/<term>` vs `val/<term>`; fix the discarded val components |

No new files beyond the refactor's plan except one loss function and one head module; both
land inside modules the refactor already creates (`losses/composer.py`, a
`models/latent_heads.py`).

---

## The ablation matrix this unlocks

All config-driven, all one-line, all logged per-term on train **and** val:

```
latent_mode   ∈ { gaussian, deterministic }
regularizer   ∈ { kl, frobenius, none }          # = which term is in LossConfig
contrastive   ∈ { on, off }
readout       ∈ { mean, attention }              # already in config
transformer   ∈ { se3, equiformer, none }        # already in config
decoder       ∈ { folding, … }                   # already in config
```

---

## Caveats worth writing down before implementing

1. **Frobenius ↔ contrastive tension.** The contrastive variance hinge
   (`contrastive_alignment_loss`, `losses.py:86-90`) pushes per-dim std **up** to
   `std_target`; Frobenius pushes code norm **down**. Combined, one is a floor and the other
   a ceiling on latent scale — pick weights so they don't fight. Flag in the config validation
   as a warning, not an error.
2. **Frobenius normalization convention.** Mean-over-B, squared, is the recommended default
   (batch-size robust, matches KL's reduction). Keep it in `kwargs` so it's ablatable rather
   than baked in.
3. **Eval determinism.** Making `eval_step` deterministic (3.2) is a *behavior change* for
   existing VAE runs (val was stochastic). It's a fix, but call it out so val curves shifting
   slightly on the first refactored run isn't mistaken for a regression.
4. **`latent_dim` vs code shape.** `GroupEncoder` asserts `n_scalars == latent_dim`
   (`group_encoder.py:169`). The deterministic head must emit the same `[B, latent_dim]`
   shape the decoder expects, so `DeterministicLatentHead` reuses the existing scalar-pool →
   linear path; only the `var_net`/reparameterize tail is dropped.

---

## Backward-compatibility summary

- Default `latent_mode="gaussian"` + `losses=[recon, kl]` reproduces the current model and
  training path (same `mu_net`/`var_net`, same KL).
- AE mode, Frobenius, and per-term val logging are **purely additive**.
- The only intentional behavior change is deterministic validation (caveat 3), which is a
  correctness fix.
