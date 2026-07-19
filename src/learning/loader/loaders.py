
import torch

class OneBatchLoader:
    """Yields the same prebuilt (graph, true_verts, mask) batch each step."""
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


class ResamplingGraphLoader:
    """Rebuilds the encoder graph from raw geometry each step with a fresh rng.

    Graph-construction policy (radius, dropout rate, supernode settings, ...) is
    collapsed into a single ``GraphSpec`` (src/learning/data/graph_spec.py) instead of
    the ~10 loose keyword arguments this constructor used to forward untouched into
    ``build_fn`` (INSTRUCTIONS.md T3). ``spec.resolve(rng)`` is called fresh each draw,
    so a ranged field (e.g. ``r_max=(0.2, 0.3)``) is still sampled per step exactly as
    before -- see ``GraphSpec.resolve``.

    ``vertices`` / ``mask`` / ``areas`` / ``normals`` stay as direct arguments: they are
    the loader's fixed per-shape geometry (parallel arrays, indexed identically), not a
    construction choice, so they don't belong on the spec.

    The ``rng`` generator's state advances as it is consumed, giving a different graph
    each step while staying reproducible from the seed.

    ``two_view`` makes the contrastive path OPTIONAL: when False (default) each step
    yields the ordinary single-view ``(graph, super, true_verts, mask)`` four-tuple; when
    True it draws TWO independent graphs (different vertex dropout / sampling) of the same
    shapes and yields ``(graph_a, super_a, true_verts, mask, graph_b, super_b)`` -- view A
    in the first four slots so the logger (which reads ``batch[:4]``) is unaffected. The
    six-tuple is what ``TrainingStepper.train_step`` turns into a "same shape -> same
    encoding" alignment loss.

    ``batch_size`` selects mini-batching over SHAPES (dim 0 of ``vertices``): when set and
    smaller than the dataset, each step draws a fresh random subset of that many shapes and
    builds the graph from only those. The yielded ``true_verts`` / ``mask`` are sliced to
    the SAME subset so the decoder's reconstruction target matches the encoder's input; in
    ``two_view`` mode both views use the same subset. ``None`` (default) keeps the original
    full-batch behaviour (every shape, every step).
    """
    def __init__(self, vertices, mask, build_fn, spec,
                 rng=None, two_view=False, batch_size=None,
                 areas=None, normals=None):

        self.vertices = vertices
        self.mask = mask
        self.build_fn = build_fn
        self.spec = spec
        self.rng = rng
        self.two_view = two_view
        self.batch_size = batch_size
        self.areas = areas
        self.normals = normals

    def _pick_indices(self):
        """Draw a fresh random subset of shape indices, or None for the full batch."""
        n = self.vertices.shape[0]
        if self.batch_size is None or self.batch_size >= n:
            return None
        return torch.randperm(n, generator=self.rng)[:self.batch_size]

    def _draw(self, idx=None):
        """Build one fresh view (new dropout / node sampling) from the fixed geometry.

        ``idx`` (if given) restricts the graph to that subset of shapes; the matching
        sliced ``(vertices, mask)`` are returned alongside so the caller can use them as
        the reconstruction target.
        """
        verts = self.vertices if idx is None else self.vertices[idx]
        mask = self.mask if idx is None else self.mask[idx]
        areas = self.areas if (idx is None or self.areas is None) else self.areas[idx]
        normals = self.normals if (idx is None or self.normals is None) else self.normals[idx]

        # Resolved fresh each draw: a ranged field (r_max/dropout_rate as (low, high))
        # is sampled now; a fixed field passes through untouched. Same rng, same order
        # (r_max then dropout_rate) as the pre-GraphSpec _sample calls -- draw-for-draw
        # reproducible with the old code.
        spec = self.spec.resolve(self.rng)
        graph, super_graph = self.build_fn(
            verts, mask, key=self.rng,
            r_max=spec.r_max,
            r_supergraph=spec.r_supergraph,
            dropout_rate=spec.dropout_rate,
            n_supernodes=spec.n_supernodes,
            use_supernodes=spec.use_supernodes,
            sampling_mode_graph=spec.sampling_mode_graph,
            sampling_mode_supernodes=spec.sampling_mode_supernodes,
            areas=areas,
            normals=normals,
            recompute_area=spec.recompute_area,
            area_k=spec.area_k,
            # spec.noise_std has no home yet: build_training_graph hardcodes noise_std=0.0
            # internally and takes no such argument (see GraphSpec's docstring). Wire it
            # through once T4's GraphBuilder calls get_graphs_from_vertices directly.
        )
        return graph, super_graph, verts, mask

    def __iter__(self):
        while True:
            idx = self._pick_indices()
            graph_a, super_a, verts, mask = self._draw(idx)
            if self.two_view:
                # Same shape subset, fresh sampling -> "same shape -> same encoding".
                graph_b, super_b, _, _ = self._draw(idx)
                yield (graph_a, super_a, verts, mask, graph_b, super_b)
            else:
                yield (graph_a, super_a, verts, mask)