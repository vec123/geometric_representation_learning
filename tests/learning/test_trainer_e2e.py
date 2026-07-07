"""Acceptance tests for the training loop (TrainingStepper + TrainingOrchestrator).

These wire the REAL GroupEncoder + FoldingDecoder through the trainer and define the
contract a working training step must satisfy:

    * one optimizer step runs, returns a finite scalar loss and decoded points, and
      actually updates the model parameters (gradient flowed end-to-end);
    * the orchestrator drives the stepper once per step and logs / checkpoints /
      visualizes at the configured cadence.

The stepper takes a ``device`` argument; these tests pass ``device='cpu'`` so the
logic runs deterministically regardless of hardware. Run with pytest, or:
    python tests/learning/test_trainer_e2e.py
"""

import os
import sys
import math

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torch_geometric.data import Data

from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
from src.learning.models.group_encoder import GroupEncoder
from src.learning.models.folding_decoder import FoldingDecoder


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_batch():
    """A small 2-graph batch: node features + geometry, and a padded target cloud."""
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
    return graph, true_verts, padding_mask


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
# TrainingStepper
# --------------------------------------------------------------------------- #
def test_training_step_returns_finite_loss_and_points():
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-2, device='cpu')
    graph, true_verts, mask = make_batch()

    loss, points = stepper.train_step(graph, true_verts, mask)

    assert isinstance(loss, float) and math.isfinite(loss), f"bad loss: {loss}"
    assert points.shape == (2, 16, 3), f"unexpected decoded shape {tuple(points.shape)}"


def test_training_step_updates_parameters():
    """A real gradient must flow through encoder -> decoder and move the weights."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-2, device='cpu')
    graph, true_verts, mask = make_batch()

    before = decoder.fold2.weight.detach().clone()
    stepper.train_step(graph, true_verts, mask)
    after = decoder.fold2.weight.detach()

    assert not torch.allclose(before, after), "optimizer did not update decoder params"


def test_training_step_is_stable_over_multiple_steps():
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-3, device='cpu')
    graph, true_verts, mask = make_batch()

    losses = [stepper.train_step(graph, true_verts, mask)[0] for _ in range(5)]
    assert all(math.isfinite(l) for l in losses), f"non-finite losses: {losses}"


# --------------------------------------------------------------------------- #
# TrainingOrchestrator (cadence, isolated from the real stepper)
# --------------------------------------------------------------------------- #
class _RecordingStepper:
    def __init__(self):
        self.calls = 0

    def train_step(self, *batch):
        self.calls += 1
        return 0.5, None  # (loss, pred)


class _RecordingLogger:
    def __init__(self):
        self.logged, self.saved, self.visualized = [], [], []

    def log_metrics(self, metrics, step):
        self.logged.append(step)

    def save_checkpoint(self, state, step):
        self.saved.append(step)

    def visualize_results(self, aux, step):
        self.visualized.append(step)


class _InfiniteLoader:
    """Iterable that yields a fixed (empty) batch forever."""
    def __iter__(self):
        while True:
            yield (None, None, None)


def test_orchestrator_drives_stepper_and_logs_at_cadence():
    stepper = _RecordingStepper()
    logger = _RecordingLogger()
    orch = TrainingOrchestrator(stepper=stepper, logger=logger, dataloader=_InfiniteLoader())

    orch.run(num_steps=4, log_every=2, save_every=2)

    assert stepper.calls == 4, f"expected 4 steps, got {stepper.calls}"
    assert logger.logged == [0, 2], f"log cadence wrong: {logger.logged}"
    assert logger.saved == [0, 2], f"save cadence wrong: {logger.saved}"
    assert logger.visualized == [0, 2], f"visualize cadence wrong: {logger.visualized}"


# --------------------------------------------------------------------------- #
# Full integration: orchestrator + real stepper + a real dataloader
# --------------------------------------------------------------------------- #
class _OneBatchLoader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


def test_orchestrator_runs_real_training_end_to_end():
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-3, device='cpu')
    logger = _RecordingLogger()
    loader = _OneBatchLoader(make_batch())

    orch = TrainingOrchestrator(stepper=stepper, logger=logger, dataloader=loader)
    orch.run(num_steps=2, log_every=1, save_every=1)

    assert logger.logged == [0, 1], f"expected two logged steps, got {logger.logged}"


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
