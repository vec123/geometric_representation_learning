import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.learning.modules.transformers.layers import PerceiverLayer


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        qk_channels,
        v_channels,
        num_heads,
        widening_factor,
    ):
        super().__init__()

        self.encoder = PerceiverLayer(
            input_kv_dim=input_dim,
            input_q_dim=latent_dim,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            widening_factor=widening_factor,
            is_cross_attention=True,
        )

    def forward(self, input, latent_embeddings):
        return self.encoder(input_kv=input, input_q=latent_embeddings)
    


class Perceiver(nn.Module):
    def __init__(self, latent_dim, qk_channels, v_channels, num_heads, widening_factor):
        super().__init__()

        self.layer = PerceiverLayer(
            input_kv_dim=None,
            input_q_dim=latent_dim,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            widening_factor=widening_factor,
            is_cross_attention=False,
        )

    def forward(self, latent):
        return self.layer(input_kv=latent, input_q=latent)