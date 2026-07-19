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


from torch_geometric.data import Data

from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
from src.learning.losses.composer import LossComposer, LossTerm
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
    # 4-tuple the trainer steps on: (graph, super_graph, true_verts, mask). super_graph is
    # None (the full-graph path); the encoder handles a missing supergraph.
    return graph, None, true_verts, padding_mask


def make_models():
    layers_cfg = [{
        'in_irreps': '1x0e',
        'target_irreps': '4x0e + 2x1o',
        'spatial_sh_lmax': 1,
        'interaction_sh_lmax': 4,
    }]
    encoder = GroupEncoder(layers_cfg=layers_cfg, latent_dim=4,
                            output_irreps='4x0e + 2x1o', verbose=False)
    decoder = FoldingDecoder(num_samples=16, latent_dim=4, n_freqs=2, verbose=False)
    return encoder, decoder


# --------------------------------------------------------------------------- #
# TrainingStepper
# --------------------------------------------------------------------------- #
def test_training_step_returns_finite_loss_and_points():
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-2, device='cpu')

    pred, loss, breakdown = stepper.train_step(*make_batch())

    assert isinstance(loss, float) and math.isfinite(loss), f"bad loss: {loss}"
    assert pred.shape == (2, 16, 3), f"unexpected decoded shape {tuple(pred.shape)}"
    # Post-T10 a term that can't contribute is ABSENT rather than reported as 0.0.
    assert "contrastive" not in breakdown, "single-view step must carry no contrastive term"


def test_training_step_updates_parameters():
    """A real gradient must flow through encoder -> decoder and move the weights."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-2, device='cpu')

    before = decoder.fold2.weight.detach().clone()
    stepper.train_step(*make_batch())
    after = decoder.fold2.weight.detach()

    assert not torch.allclose(before, after), "optimizer did not update decoder params"


def test_training_step_is_stable_over_multiple_steps():
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-3, device='cpu')
    batch = make_batch()

    # train_step returns (pred, loss, breakdown); loss is index 1.
    losses = [stepper.train_step(*batch)[1] for _ in range(5)]
    assert all(math.isfinite(l) for l in losses), f"non-finite losses: {losses}"


def test_two_view_contrastive_step():
    """A six-tuple (two views) runs the contrastive path: finite loss, a positive
    contrastive term, and the weights still move."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(
        encoder, decoder, learning_rate=1e-2, device='cpu',
        composer=LossComposer([LossTerm("recon", 1.0), LossTerm("contrastive", 0.1)]))
    graph, super_graph, true_verts, mask = make_batch()

    before = decoder.fold2.weight.detach().clone()
    pred, loss, breakdown = stepper.train_step(
        graph, super_graph, true_verts, mask, graph, super_graph)
    after = decoder.fold2.weight.detach()

    assert math.isfinite(loss), f"bad loss: {loss}"
    assert pred.shape == (2, 16, 3), f"unexpected decoded shape {tuple(pred.shape)}"
    assert breakdown["contrastive"] > 0.0, "two-view step should carry a positive contrastive term"
    assert not torch.allclose(before, after), "optimizer did not update decoder params"


# --------------------------------------------------------------------------- #
# TrainingOrchestrator (cadence, isolated from the real stepper)
# --------------------------------------------------------------------------- #
class _RecordingStepper:
    def __init__(self):
        self.calls = 0

    def train_step(self, *batch):
        self.calls += 1
        # (pred, loss, breakdown) -- the arity the orchestrator unpacks post-T10.
        return None, 0.5, {"recon": 0.4}


class _RecordingLogger:
    def __init__(self):
        self.logged, self.saved, self.visualized = [], [], []

    def log_metrics(self, metrics, step):
        self.logged.append(step)

    def save_checkpoint(self, state, step):
        self.saved.append(step)

    def visualize_batch(self, batch, pred, step, *args, **kwargs):
        self.visualized.append(step)

    def plot_metrics(self, *args, **kwargs):
        pass


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
