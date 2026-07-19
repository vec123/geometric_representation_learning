"""Experiment configuration schema (INSTRUCTIONS.md T5).
field-by-field against the constructors:
GroupEncoder 
(src/learning/models/group_encoder.py:16-25), 
EquiLayer
(src/learning/layers/equivariant/Self_Spatial_layer.py:13-14), 
FoldingDecoder /
SphereFoldingDecoder 
(src/learning/models/folding_decoder.py), 
GraphSpec
(src/learning/data/graph_spec.py),
TrainingOrchestrator.run
(src/learning/trainers/E3_end2end.py:159).

Every constant has exactly one field here 
(see that file for defaults this schema mirrors).
"""

import warnings
from dataclasses import dataclass, field, replace
from typing import Optional, List, Tuple, Union

import torch

VALID_LOSS_TERMS = {"recon", "kl", "contrastive", "frobenius"}
VALID_DECODER_TYPES = {"folding", "sphere_folding"}
_LARGE_WEIGHT_THRESHOLD = 1.5


@dataclass
class EncoderLayerConfig:
    """Mirrors EquiLayer's constructor (Self_Spatial_layer.py:13-14).

    GroupEncoder builds exactly ONE of these (group_encoder.py:41-47) 
    single field on EncoderConfig
    """
    in_irreps: str = "1x0e"
    target_irreps: str = "32x0e + 32x0o + 16x1e + 16x1o"
    spatial_sh_lmax: int = 1

    # GroupEncoder hardcodes interaction_sh_lmax=4 when it builds its EquiLayer
    # (group_encoder.py:45) and does not expose it as a constructor parameter --
    # this field has NO EFFECT until GroupEncoder itself is changed to accept it.
    # Kept (not dropped) because it IS a real EquiLayer parameter; T7's factory
    # should know it can't forward this one, rather than silently pretending to.
    interaction_sh_lmax: int = 4


@dataclass
class EncoderConfig:
    """Matches GroupEncoder's real constructor (group_encoder.py:16-25)."""
    encoder_type: str = "group_encoder"   # registry key (T6); only one exists today
    latent_dim: int = 5
    layer: EncoderLayerConfig = field(default_factory=EncoderLayerConfig)
    readout: str = "mean"          # "mean" | "attention"
    readout_heads: int = 1         # only read when readout == "attention"
    supernode_sh_lmax: int = 4
    transformer_type: Optional[str] = "se3"   # "se3" | "equiformer" | None
    transformer_cfg: dict = field(default_factory=dict)
    area_pool: bool = False
    latent_mode: str = "gaussian"  # "gaussian" | "deterministic" -- T9's LatentHead strategy

    @property
    def output_irreps(self) -> str:
        """Not a free choice: GroupEncoder.forward asserts exactly ``latent_dim``
        scalars (0e) and 2 vectors (1o) come out of the final projection
        (group_encoder.py:169, 215). Computing it here removes an entire class of
        invalid config (an output_irreps that disagrees with latent_dim)."""
        return f"{self.latent_dim}x0e + 2x1o"

    @property
    def n_tokens(self) -> int:
        """Tokens emitted per shape. GroupEncoder's readout ("mean" or "attention")
        always collapses to exactly one token regardless of which is chosen -- this
        schema doesn't yet model a multi-token encoder (e.g. GroupPerceiverEncoder,
        scripts/perceiver_train.py). Give this a real encoder_type-aware value when
        such an encoder joins the registry (T6, T9)."""
        return 1


@dataclass
class DecoderConfig:
    """Matches FoldingDecoder / SphereFoldingDecoder's shared constructor shape
    (folding_decoder.py). ``hidden_dim`` was dropped: both decoders hardcode their
    hidden width to 128 internally and take no such argument -- it was decorative.
    """
    num_samples: int = 256
    latent_dim: int = 5
    n_freqs: int = 4
    decoder_type: str = "sphere_folding"  # "folding" | "sphere_folding" (registry key, T6)

    @property
    def expects_tokens(self) -> int:
        """Both decoder types hard-require ``latent.shape[1] == 1`` and raise
        otherwise (folding_decoder.py:57-61, 187-191); this schema doesn't yet
        model a decoder that accepts more than one token."""
        return 1


@dataclass
class LossTermConfig:
    """Single named term for LossComposer (T8): name, weight, extra kwargs."""
    name: str
    weight: float = 1.0
    kwargs: dict = field(default_factory=dict)


@dataclass
class LossConfig:
    terms: List[LossTermConfig] = field(default_factory=lambda: [
        LossTermConfig(name="recon", weight=1.0),
    ])


@dataclass
class TrainingConfig:
    """Optimization + the run's time axis.

    ``num_steps`` / ``log_every`` / ``save_every`` / ``val_every`` replace v1's
    ``num_epochs`` / ``checkpoint_every`` (which lived on ExperimentConfig): the
    loaders here (OneBatchLoader, ResamplingGraphLoader) are infinite generators
    with no notion of an epoch, and TrainingOrchestrator.run is literally
    ``run(num_steps, log_every, save_every, val_every)`` (E3_end2end.py:159).
    """
    learning_rate: float = 1e-3   # v1 had 1e-5; equivariant_gnn_train.py uses 1e-3
    device: Optional[str] = None  # "cuda" | "cpu" | None (auto -- TrainingStepper._resolve_device)
    losses: LossConfig = field(default_factory=LossConfig)
    verbose: bool = False

    num_steps: int = 3001
    log_every: int = 1
    save_every: int = 100
    val_every: int = 100

    batch_size: Optional[int] = None   # mini-batch over SHAPES; None = full batch every step


RangeOrFixed = Union[float, Tuple[float, float]]


@dataclass(frozen=True)
class GraphSpec:
    """Parameter Object for graph construction (T3), embedded in DataConfig below.

    Names the data clump that used to travel, unbundled, across three signatures:
    ``ResamplingGraphLoader.__init__`` -> ``build_training_graph`` ->
    ``get_graphs_from_vertices``. Consumed directly by L1 components
    (``RadiusGraphBuilder``, ``src/learning/data/builders.py``) as well as by this
    config -- it is plain data, not config machinery, so components importing it
    from here does not pull in the rest of ExperimentConfig.

    ``r_max`` / ``dropout_rate`` may each be a fixed float or a ``(low, high)``
    range, sampled fresh per call via :meth:`resolve`.
    """

    r_max: RangeOrFixed = 0.1
    r_supergraph: float = 0.2
    dropout_rate: RangeOrFixed = 0.8
    n_supernodes: int = 10
    use_supernodes: bool = False
    sampling_mode_graph: str = "uniform"
    sampling_mode_supernodes: str = "uniform"
    recompute_area: bool = False
    area_k: int = 8
    noise_std: float = 0.0

    def __post_init__(self):
        for name in ("r_max", "dropout_rate"):
            value = getattr(self, name)
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise ValueError(f"{name} range must be (low, high), got {value!r}")
                low, high = value
                if low > high:
                    raise ValueError(f"{name} range must have low <= high, got {value!r}")

    def resolve(self, rng: torch.Generator = None) -> "GraphSpec":
        """Return a new GraphSpec with every range field sampled down to a fixed value.

        ``rng`` is a parameter, never stored on the spec itself: same rng state +
        same spec => same resolved spec, which is what keeps graph construction
        reproducible.
        """
        return replace(
            self,
            r_max=self._sample(self.r_max, rng),
            dropout_rate=self._sample(self.dropout_rate, rng),
        )

    @staticmethod
    def _sample(value: RangeOrFixed, rng: torch.Generator) -> float:
        """Fixed value passes through; a (low, high) range is sampled uniformly."""
        if isinstance(value, (tuple, list)):
            low, high = value
            u = torch.rand((), generator=rng).item()
            return low + u * (high - low)
        return value


@dataclass
class DataConfig:
    """Dataset loading + graph-construction policy.

    Graph-construction knobs are NOT re-listed here: they live on ``GraphSpec``
    above (T3) and are embedded as one field. v1 also had ``use_supernodes`` here
    (duplicating a concern EncoderConfig implied too); it now has exactly one
    home, on ``GraphSpec``.
    """
    data_path: str = "DATA_ROOT"
    parts: Optional[List[str]] = field(default_factory=lambda: ["mouth", "nose"])
    load_fields: bool = True
    val_fraction: float = 0.2
    shuffle: bool = True
    seed: Optional[int] = None

    graph_builder: str = "radius"   # registry key: "radius" | "knn" (T4, T6)
    graph_spec: GraphSpec = field(default_factory=GraphSpec)
    resample_graph: bool = True     # True: rebuild the graph each step; False: build once, reuse


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration -- the object which is turned into a run."""
    name: str = "default_experiment"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: Optional[int] = None

    def validate(self) -> None:
        """Fail fast, at the boundary, before anything expensive gets built."""
        term_names = [t.name for t in self.training.losses.terms]

        unknown_terms = set(term_names) - VALID_LOSS_TERMS
        if unknown_terms:
            raise ValueError(
                f"unknown loss term(s) {sorted(unknown_terms)}; the trainer only "
                f"ever produces {sorted(VALID_LOSS_TERMS)}"
            )

        if self.encoder.latent_mode == "deterministic" and "kl" in term_names:
            raise ValueError(
                "encoder.latent_mode='deterministic' has no posterior, so a 'kl' "
                "loss term is undefined. Use latent_mode='gaussian' for KL, or "
                "drop the 'kl' term (e.g. add a 'frobenius' term instead)."
            )

        if self.decoder.decoder_type not in VALID_DECODER_TYPES:
            raise ValueError(
                f"unknown decoder.decoder_type {self.decoder.decoder_type!r}; "
                f"expected one of {sorted(VALID_DECODER_TYPES)}"
            )
        if self.decoder.decoder_type == "folding":
            n = self.decoder.num_samples
            root = int(round(n ** 0.5))
            if root * root != n:
                raise ValueError(
                    f"decoder.num_samples={n} is not a perfect square; FoldingDecoder "
                    f"folds a square grid (grid_size = sqrt(num_samples)). "
                    f"decoder_type='sphere_folding' has no such requirement if a "
                    f"non-square sample count is needed."
                )

        if self.encoder.n_tokens != self.decoder.expects_tokens:
            raise ValueError(
                f"encoder.encoder_type={self.encoder.encoder_type!r} emits "
                f"{self.encoder.n_tokens} token(s) per shape but "
                f"decoder.decoder_type={self.decoder.decoder_type!r} expects "
                f"{self.decoder.expects_tokens} (see T7's build-time compatibility check)."
            )

        frobenius_w = next((t.weight for t in self.training.losses.terms if t.name == "frobenius"), 0.0)
        contrastive_w = next((t.weight for t in self.training.losses.terms if t.name == "contrastive"), 0.0)
        if frobenius_w > _LARGE_WEIGHT_THRESHOLD and contrastive_w > _LARGE_WEIGHT_THRESHOLD:
            warnings.warn(
                f"both 'frobenius' (weight={frobenius_w}) and 'contrastive' "
                f"(weight={contrastive_w}) are large: frobenius pushes latent norm "
                f"DOWN while the contrastive variance hinge pushes per-dimension "
                f"spread UP. Tune weights if training destabilizes.",
                stacklevel=2,
            )
