
import torch

class OneBatchLoader:
    """Yields the same prebuilt (graph, true_verts, mask) batch each step."""
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


class ResamplingGraphLoader:
    """Rebuilds the encoder graph from raw geometry each step with a fresh key.

    The reconstruction target (``true_verts``, ``mask``) is fixed, but every step draws
    a new dropout mask / node sampling (and optionally a jittered radius / dropout rate),
    so the encoder fits the underlying geometry rather than one frozen edge set. The
    ``key`` generator's state advances as it is consumed, giving a different graph each
    step while staying reproducible from the seed.

    ``r_max`` and ``dropout_rate`` may each be a fixed float or a ``(low, high)`` range
    that is sampled uniformly per step.
    """
    def __init__(self, vertices, mask, build_fn,
                  key=None, r_max=0.2, dropout_rate=0.8, use_supernodes = False):
        
        self.vertices = vertices
        self.mask = mask
        self.build_fn = build_fn
        self.key = key
        self.r_max = r_max
        self.dropout_rate = dropout_rate
        self.use_supernodes = use_supernodes

    def _sample(self, value):
        """Return value as-is if scalar, else draw uniformly from a (low, high) range."""
        if isinstance(value, (tuple, list)):
            low, high = value
            u = torch.rand((), generator=self.key).item()
            return low + u * (high - low)
        return value

    def __iter__(self):
        while True:
            graph, supergraph = self.build_fn(
                self.vertices, self.mask, key=self.key,
                r_max=self._sample(self.r_max),
                dropout_rate=self._sample(self.dropout_rate),
                use_supernodes= self.use_supernodes
            )
            yield (graph, supergraph, self.vertices, self.mask)