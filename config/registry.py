from typing import Type, Dict, Callable, Any
from src.learning.modules.equivariant.transformer import (
    SE3AttentionLayer,
    build_equivariant_transformer
)
from src.learning.modules.transformers.perceiver_decoder import PerceiverReducer

class ComponentRegistry:
    """Registry for pluggable components (transformers, readouts, decoders, etc.)."""
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, category: str, name: str, component_class: Type, **kwargs):
        """Register a component."""
        if category not in cls._registry:
            cls._registry[category] = {}
        cls._registry[category][name] = {
            'class': component_class,
            'kwargs': kwargs,
        }
    
    @classmethod
    def create(cls, category: str, name: str, **override_kwargs):
        """Instantiate a registered component."""
        if category not in cls._registry:
            raise ValueError(f"Unknown category: {category}")
        if name not in cls._registry[category]:
            raise ValueError(f"Unknown {category}: {name}")
        
        entry = cls._registry[category][name]
        merged_kwargs = {**entry['kwargs'], **override_kwargs}
        return entry['class'](**merged_kwargs)
    
    @classmethod
    def list(cls, category: str):
        """List available components in a category."""
        return list(cls._registry.get(category, {}).keys())

# Pre-register built-in components
ComponentRegistry.register('transformer', 'se3', SE3AttentionLayer)
ComponentRegistry.register('transformer', 'equiformer', EquiformerBlock)  # existing class
ComponentRegistry.register('readout', 'mean', MeanPooling)  # wrapper
ComponentRegistry.register('readout', 'attention', PerceiverReducer)
ComponentRegistry.register('decoder', 'folding', FoldingDecoder)