"""Parameter Object for graph construction (INSTRUCTIONS.md T3).

``GraphSpec`` names the data for -> ``get_graphs_from_vertices``
(``src/graphs/graphs.py``). 
Deliberately narrow: only the values that
describe HOW to build a graph from geometry live here. 
Things like
``vertices``, ``mask``,
``areas``, ``normals`` 
and random key are per-call data, not spec, and stay out.
"""

from dataclasses import dataclass, replace
from typing import Tuple, Union
import torch

RangeOrFixed = Union[float, Tuple[float, float]]

@dataclass(frozen=True)
class GraphSpec:
    """Everything needed to turn raw geometry into the graph the encoder consumes.

    A spec is a value, not mutable state 
    -- frozen, and equal specs compare equal.

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
        reproducible. A spec with no range fields resolves to itself unchanged
        (``rng`` may be ``None`` in that case).
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
