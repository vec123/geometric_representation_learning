from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class EncoderOutput:
    """Standard container for encoder outputs, decoupling encoders from the trainer.

    All fields are optional so a range of encoders satisfy one contract:
      * deterministic encoder -> set ``latent``
      * VAE encoder           -> set ``mu`` / ``logvar`` (the trainer reparameterizes)
      * equivariant encoder   -> additionally set ``rotation`` / ``translation`` (pose)

    ``aux`` carries anything extra (e.g. the raw equivariant vectors) without
    widening the contract.
    """

    latent: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    rotation: Optional[torch.Tensor] = None
    translation: Optional[torch.Tensor] = None
    aux: dict = field(default_factory=dict)
