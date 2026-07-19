from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class EncoderLayerConfig:
    """Single equivariant layer configuration."""
    in_irreps: str = "1x0e"
    target_irreps: str = "32x0e + 32x0o + 16x1e + 16x1o"
    sh_lmax: int = 1
    interaction_sh_lmax: int = 4

@dataclass
class EncoderConfig:
    """Encoder model configuration."""
    latent_dim: int = 5
    layers: List[EncoderLayerConfig] = field(default_factory=lambda: [
        EncoderLayerConfig(),
        EncoderLayerConfig(),
    ])
    irreps_preset: str = "standard"  # or explicit in_irreps/intermediate/output
    
    # Supernode aggregation
    use_supernodes: bool = True
    n_supernodes: int = 15
    supernode_sh_lmax: int = 4
    
    # Transformer refinement
    transformer_type: Optional[str] = "se3"  # "se3", "equiformer", None
    transformer_cfg: dict = field(default_factory=dict)
    
    # Readout
    readout: str = "mean"  # "mean" or "attention"
    readout_heads: int = 1

@dataclass
class DecoderConfig:
    """Decoder model configuration."""
    num_samples: int = 256
    latent_dim: int = 5
    n_freqs: int = 4
    hidden_dim: int = 128
    decoder_type: str = "folding"  # "folding", custom, etc.

@dataclass
class LossTermConfig:
    """Single loss term (recon, kl, contrastive, etc.)."""
    name: str
    weight: float = 1.0
    kwargs: dict = field(default_factory=dict)

@dataclass
class LossConfig:
    """Loss composition configuration."""
    terms: List[LossTermConfig] = field(default_factory=lambda: [
        LossTermConfig(name="recon", weight=1.0),
    ])
    
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 1e-5
    device: Optional[str] = None  # "cuda", "cpu", None (auto)
    losses: LossConfig = field(default_factory=LossConfig)
    verbose: bool = False
    
@dataclass
class DataConfig:
    """Data loading configuration."""
    data_path: str = "DATA_ROOT"
    parts: List[str] = field(default_factory=lambda: ["mouth", "nose"])
    load_fields: bool = True
    val_fraction: float = 0.2
    shuffle: bool = True
    seed: Optional[int] = None
    
    # Graph building
    graph_builder: str = "radius"  # "radius", "knn", "fully_connected"
    r_max: float = 0.2
    r_supergraph: float = 0.6
    dropout_rate: float = 0.8
    sampling_mode_graph: str = "uniform"
    sampling_mode_supernodes: str = "uniform"

@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default_experiment"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    num_epochs: int = 100
    checkpoint_every: int = 10
    seed: Optional[int] = None