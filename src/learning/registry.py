"""Lazy component Registry.

Maps ``(category, name)``
-> a ``"module.path:ClassName"`` STRING, resolved
(imported) only on the first ``Registry.create()`` call for that entry.

Storing strings instead of live imports means:
nothing is imported until something asks for it by name.

Two known failure modes of stringly-typed targets, and how they are handled:

1. A rename/move refactor silently rots a target string (it still parses).
   Guard: tests/learning/test_registry.py resolves EVERY registered string,
   so rot fails CI instead of the first run that asks for it by name.
2. Failures move from import time to run time.  Accepted: resolution happens
   at config-load (first ``create()``), i.e. within the first second of a run,
   and ``resolve()`` raises an error naming the exact broken registration.
"""

from importlib import import_module


class Registry:
    """Maps (category, name) -> "module:QualName",
      resolved on first use."""

    _entries: dict = {}

    @classmethod
    def register(cls, category: str, name: str, target: str) -> None:
        """`target` is a STRING like "src.learning.models.folding_decoder:FoldingDecoder".
        Storing a string (not the class) is what keeps the import lazy."""
        cls._entries[(category, name)] = target

    @classmethod
    def resolve(cls, category: str, name: str):
        """Import and return the target class WITHOUT instantiating it.

        A broken target string (module moved, class renamed) surfaces here as an
        ImportError that names the registration, not a bare AttributeError deep
        in importlib -- so the config-load failure points at the fix.
        """
        key = (category, name)
        if key not in cls._entries:
            raise ValueError(
                f"no {category!r} registered under name {name!r}; "
                f"available: {cls.available(category)}"
            )
        target = cls._entries[key]
        module_path, qualname = target.split(":")
        try:
            module = import_module(module_path)
        except ImportError as err:
            raise ImportError(
                f"registry entry ({category!r}, {name!r}) -> {target!r}: "
                f"module {module_path!r} failed to import -- moved or renamed?"
            ) from err
        try:
            return getattr(module, qualname)
        except AttributeError as err:
            raise ImportError(
                f"registry entry ({category!r}, {name!r}) -> {target!r}: "
                f"module {module_path!r} has no attribute {qualname!r} -- "
                f"class renamed without updating the registration?"
            ) from err

    @classmethod
    def create(cls, category: str, name: str, **kwargs):
        """Resolve, import, instantiate. Unknown name -> ValueError listing valid names."""
        target_cls = cls.resolve(category, name)
        return target_cls(**kwargs)

    @classmethod
    def available(cls, category: str) -> list:
        """Names in a category -- powers --help and error messages."""
        return sorted(name for (cat, name) in cls._entries if cat == category)

    @classmethod
    def entries(cls) -> dict:
        """Snapshot of every registration -- powers the anti-rot test."""
        return dict(cls._entries)


# --------------------------------------------------------------------------- #
# Registrations: one line per component that exists TODAY.
# Each target string is checked against the real class it points to by
# tests/learning/test_registry.py::test_every_registered_target_resolves.
# --------------------------------------------------------------------------- #
Registry.register("encoder", "group_encoder",
                   "src.learning.models.group_encoder:GroupEncoder")

Registry.register("decoder", "folding",
                   "src.learning.models.folding_decoder:FoldingDecoder")

Registry.register("decoder", "sphere_folding",
                   "src.learning.models.folding_decoder:SphereFoldingDecoder")

Registry.register("transformer", "se3",
                   "src.learning.modules.equivariant.transformer:SE3Transformer")
Registry.register("transformer", "equiformer",
                   "src.learning.modules.equivariant.equiformer:EquiformerTransformer")

Registry.register("graph_builder", "radius",
                   "src.learning.data.builders:RadiusGraphBuilder")
Registry.register("graph_builder", "knn",
                   "src.learning.data.builders:KNNGraphBuilder")

Registry.register("latent_head", "gaussian",
                   "src.learning.models.latent_heads:GaussianLatentHead")
Registry.register("latent_head", "deterministic",
                   "src.learning.models.latent_heads:DeterministicLatentHead")

Registry.register("readout", "attention",
                   "src.learning.modules.transformers.perceiver_encoder:PerceiverReducer")
# readout="mean" has no class of its own -- the LatentHead base computes it inline
# as a weighted global_add_pool (latent_heads.py:_reduce), not through a component.
# Nothing to register until that path is extracted into a Strategy class of its own.

