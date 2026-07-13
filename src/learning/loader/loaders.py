
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

    ``two_view`` makes the contrastive path OPTIONAL: when False (default) each step
    yields the ordinary single-view ``(graph, super, true_verts, mask)`` four-tuple; when
    True it draws TWO independent graphs (different vertex dropout / sampling) of the same
    shapes and yields ``(graph_a, super_a, true_verts, mask, graph_b, super_b)`` -- view A
    in the first four slots so the logger (which reads ``batch[:4]``) is unaffected. The
    six-tuple is what ``TrainingStepper.train_step`` turns into a "same shape -> same
    encoding" alignment loss.
    """
    def __init__(self, vertices, mask, build_fn,
                  key=None, r_max=0.2, r_supergraph = 0.6, dropout_rate=0.8,
                  use_supernodes = False, two_view = False, n_supernodes = 15,
                  sampling_mode_graph = "uniform", sampling_mode_supernodes = "uniform"):

        self.vertices = vertices
        self.mask = mask
        self.build_fn = build_fn
        self.key = key
        self.r_max = r_max
        self.r_supergraph = r_supergraph
        self.dropout_rate = dropout_rate
        self.use_supernodes = use_supernodes
        self.two_view = two_view
        self.n_supernodes = n_supernodes
        self.sampling_mode_graph = sampling_mode_graph
        self.sampling_mode_supernodes = sampling_mode_supernodes

    def _sample(self, value):
        """Return value as-is if scalar, else draw uniformly from a (low, high) range."""
        if isinstance(value, (tuple, list)):
            low, high = value
            u = torch.rand((), generator=self.key).item()
            return low + u * (high - low)
        return value

    def _draw(self):
        """Build one fresh view (new dropout / node sampling) from the fixed geometry."""
        return self.build_fn(
            self.vertices, self.mask, key=self.key,
            r_max=self._sample(self.r_max),
            r_supergraph=self.r_supergraph,
            dropout_rate=self._sample(self.dropout_rate),
            n_supernodes=self.n_supernodes,
            use_supernodes=self.use_supernodes,
            sampling_mode_graph=self.sampling_mode_graph,
            sampling_mode_supernodes=self.sampling_mode_supernodes,
        )

    def __iter__(self):
        while True:
            graph_a, super_a = self._draw()
            if self.two_view:
                graph_b, super_b = self._draw()
                yield (graph_a, super_a, self.vertices, self.mask, graph_b, super_b)
            else:
                yield (graph_a, super_a, self.vertices, self.mask)