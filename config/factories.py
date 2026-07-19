import torch.nn as nn
from src.learning.models.group_encoder import GroupEncoder
from src.learning.models.folding_decoder import FoldingDecoder
from src.learning.config.models import (
    EncoderConfig, DecoderConfig, ExperimentConfig
)

def build_encoder(config: EncoderConfig) -> nn.Module:
    """Instantiate encoder from config."""
    return GroupEncoder(
        latent_dim=config.latent_dim,
        irreps_cfg={
            "input_irreps": config.layers[0].in_irreps,
            "intermediate_irreps": config.layers[0].target_irreps,
            "output_irreps": f"{config.latent_dim}x0e + 2x1o",
        },
        sh_lmax=config.layers[0].sh_lmax,
        readout=config.readout,
        readout_heads=config.readout_heads,
        supernode_sh_lmax=config.supernode_sh_lmax,
        transformer_type=config.transformer_type,
        transformer_cfg=config.transformer_cfg,
        verbose=config.verbose if hasattr(config, 'verbose') else False,
    )

def build_decoder(config: DecoderConfig) -> nn.Module:
    """Instantiate decoder from config."""
    if config.decoder_type == "folding":
        return FoldingDecoder(
            num_samples=config.num_samples,
            latent_dim=config.latent_dim,
            n_freqs=config.n_freqs,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown decoder type: {config.decoder_type}")

def build_experiment(config: ExperimentConfig) -> tuple:
    """Build encoder, decoder, and training setup from ExperimentConfig."""
    encoder = build_encoder(config.encoder)
    decoder = build_decoder(config.decoder)
    return encoder, decoder