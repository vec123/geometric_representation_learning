"""Checkpoint-writing callback (INSTRUCTIONS.md T12)."""

import os

import torch

from src.learning.callbacks.base import Callback


class CheckpointWriter(Callback):
    """Saves encoder / decoder / optimizer state on its own cadence.

    NOTE on key compatibility: T9 moved the latent machinery into a submodule, so
    ``mu_net.*`` / ``var_net.*`` / ``readout_pool.*`` are now ``latent_head.mu_net.*``
    etc. Checkpoints written before T9 will NOT load into a current encoder without
    remapping those three prefixes. Nothing here migrates them.
    """

    def __init__(self, every_n_steps=100, directory="checkpoints", verbose=True):
        super().__init__(every_n_steps)
        self.directory = directory
        self.verbose = verbose

    def on_step_end(self, ctx, step, metrics, batch, pred):
        if self._due(step):
            self.save(ctx, step)

    def save(self, ctx, step):
        checkpoint_dir = os.path.join(ctx.log_dir, self.directory)
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"step_{step}.pt")
        torch.save({
            "encoder": ctx.stepper.encoder.state_dict(),
            "decoder": ctx.stepper.decoder.state_dict(),
            "optimizer": ctx.stepper.optimizer.state_dict(),
        }, path)
        if self.verbose:
            print(f"  checkpoint -> {path}")