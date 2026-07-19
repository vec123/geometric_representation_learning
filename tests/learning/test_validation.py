"""Tests for the validation path added to the training loop.

Covers the contract that makes validation trustworthy and useful:

    * ``TrainingStepper.eval_step`` computes a finite loss WITHOUT moving any weights
      (no optimizer step, no gradient leak from being called next to training);
    * ``TrainingOrchestrator`` runs validation at the ``val_every`` cadence, logs a
      ``val_loss``, and asks the logger to save the validation VTPs;
    * end-to-end, a run leaves validation prediction VTPs under ``vtk/validation`` and a
      ``metrics.png`` + ``metrics.json`` behind.

Batches are the 4-tuple ``(graph, super_graph, true_verts, padding_mask)`` the real
loaders yield; ``super_graph`` is None (the full-graph path). Run with pytest, or:
    python tests/learning/test_validation.py
"""

import os
import sys
import math
import tempfile

import torch


from torch_geometric.data import Data

from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
from src.learning.logger.train_logs import TrainingLogger
from src.learning.models.group_encoder import GroupEncoder
from src.learning.models.folding_decoder import FoldingDecoder


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_batch():
    """A small 2-graph batch as ``(graph, super_graph, true_verts, padding_mask)``."""
    torch.manual_seed(0)

    def ring(offset, n):
        s = torch.arange(n) + offset
        d = (torch.arange(n) + 1) % n + offset
        return torch.stack([torch.cat([s, d]), torch.cat([d, s])])

    edge_index = torch.cat([ring(0, 6), ring(6, 6)], dim=1)
    pos = torch.randn(12, 3)
    batch = torch.tensor([0] * 6 + [1] * 6)
    x = torch.ones(12, 1)  # 1x0e input scalar per node

    graph = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
    true_verts = torch.randn(2, 10, 3)
    padding_mask = torch.ones(2, 10, dtype=torch.bool)
    return graph, None, true_verts, padding_mask


def make_models():
    cfg = {
        'input_irreps': '1x0e',
        'intermediate_irreps': '4x0e + 2x1o',
        'output_irreps': '4x0e + 2x1o',
    }
    encoder = GroupEncoder(latent_dim=4, irreps_cfg=cfg, verbose=False)
    decoder = FoldingDecoder(num_samples=16, latent_dim=4, n_freqs=2, verbose=False)
    return encoder, decoder


# --------------------------------------------------------------------------- #
# TrainingStepper.eval_step
# --------------------------------------------------------------------------- #
def test_eval_step_does_not_update_parameters():
    """Validation must compute a finite loss WITHOUT moving any weights."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-2, device='cpu')
    graph, super_graph, true_verts, mask = make_batch()

    before = decoder.fold2.weight.detach().clone()
    pred, loss, recon, kl = stepper.eval_step(graph, super_graph, true_verts, mask)
    after = decoder.fold2.weight.detach()

    assert isinstance(loss, float) and math.isfinite(loss), f"bad val loss: {loss}"
    assert pred.shape == (2, 16, 3), f"unexpected decoded shape {tuple(pred.shape)}"
    assert torch.allclose(before, after), "eval_step must NOT update parameters"


# --------------------------------------------------------------------------- #
# TrainingOrchestrator validation cadence (isolated from the real stepper)
# --------------------------------------------------------------------------- #
class _DummyModule:
    """Stands in for encoder/decoder so run_validation can toggle eval()/train()."""
    def eval(self):
        pass

    def train(self):
        pass


class _RecordingStepper:
    def __init__(self):
        self.calls = 0
        self.eval_calls = 0
        self.encoder = _DummyModule()
        self.decoder = _DummyModule()

    def train_step(self, *batch):
        self.calls += 1
        # (pred, loss, recon, kl, contrastive)
        return None, 0.5, 0.4, 0.0, 0.0

    def eval_step(self, *batch):
        self.eval_calls += 1
        # (pred, loss, recon, kl)
        return None, 0.25, 0.2, 0.0


class _RecordingLogger:
    def __init__(self):
        self.logged, self.saved, self.visualized = [], [], []
        self.val_logged, self.val_visualized = [], []
        self.plotted = 0

    def log_metrics(self, metrics, step):
        self.logged.append(step)
        if "val_loss" in metrics:
            self.val_logged.append(step)

    def save_checkpoint(self, state, step):
        self.saved.append(step)

    def visualize_batch(self, batch, pred, step, subdir="vtk"):
        self.visualized.append(step)

    def visualize_val_batch(self, batch, pred, step):
        self.val_visualized.append(step)

    def plot_metrics(self, *args, **kwargs):
        self.plotted += 1


class _InfiniteLoader:
    """Iterable that yields a fixed (empty) 4-tuple batch forever."""
    def __iter__(self):
        while True:
            yield (None, None, None, None)


def test_orchestrator_runs_validation_at_cadence():
    stepper = _RecordingStepper()
    logger = _RecordingLogger()
    orch = TrainingOrchestrator(
        stepper=stepper, logger=logger,
        dataloader=_InfiniteLoader(), val_loader=_InfiniteLoader())

    orch.run(num_steps=4, log_every=1, save_every=10, val_every=2)

    assert stepper.eval_calls == 2, f"expected val at steps 0,2 -> 2 eval calls, got {stepper.eval_calls}"
    assert logger.val_logged == [0, 2], f"val-loss log cadence wrong: {logger.val_logged}"
    assert logger.val_visualized == [0, 2], f"val VTP cadence wrong: {logger.val_visualized}"


def test_orchestrator_skips_validation_without_val_loader():
    stepper = _RecordingStepper()
    logger = _RecordingLogger()
    orch = TrainingOrchestrator(stepper=stepper, logger=logger, dataloader=_InfiniteLoader())

    orch.run(num_steps=4, log_every=1, save_every=10, val_every=2)

    assert stepper.eval_calls == 0, "no val_loader -> validation must not run"
    assert logger.val_visualized == [], "no val_loader -> no validation VTPs"
    assert logger.plotted >= 1, "run() should still emit a final metrics plot"


# --------------------------------------------------------------------------- #
# End-to-end: validation artifacts on disk
# --------------------------------------------------------------------------- #
class _OneBatchLoader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


def test_validation_writes_vtps_and_metrics_plot():
    """A real run must write validation prediction VTPs under vtk/validation and leave a
    metrics.png + metrics.json behind — the artifacts requested."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-3, device='cpu')

    with tempfile.TemporaryDirectory() as tmp:
        logger = TrainingLogger(log_dir=tmp)
        orch = TrainingOrchestrator(
            stepper=stepper, logger=logger,
            dataloader=_OneBatchLoader(make_batch()),
            val_loader=_OneBatchLoader(make_batch()))
        orch.run(num_steps=2, log_every=1, save_every=1, val_every=1)

        val_dir = os.path.join(tmp, "vtk", "validation")
        assert os.path.isdir(val_dir), "validation VTP subdir was not created"
        val_vtps = [f for f in os.listdir(val_dir) if f.endswith(".vtp")]
        assert any(f.startswith("pred_shape") for f in val_vtps), \
            f"no validation prediction VTPs written: {val_vtps}"

        assert os.path.exists(os.path.join(tmp, "metrics.png")), "metrics plot not written"
        assert os.path.exists(os.path.join(tmp, "metrics.json")), "metrics history not written"


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #
def _run_all():
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {str(exc)[:140]}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")


if __name__ == '__main__':
    _run_all()
