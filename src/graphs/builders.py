from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import numpy as np
from torch_geometric.data import Data
from src.graphs.graphs import get_graphs_from_vertices, build_super_graph

class GraphBuilder(ABC):
    """Abstract base for graph construction strategies."""
    
    @abstractmethod
    def build(self, vertices: torch.Tensor, mask: torch.Tensor, 
              areas: Optional[torch.Tensor] = None,
              normals: Optional[torch.Tensor] = None) -> Tuple[Data, Optional[Data]]:
        """Build graph(s) from vertices and optional fields.
        
        Returns: (graph, supergraph) - supergraph can be None
        """
        pass

class RadiusGraphBuilder(GraphBuilder):
    """Build graphs with radius-based edges."""
    
    def __init__(self, r_max: float = 0.2, r_supergraph: float = 0.6,
                 dropout_rate: float = 0.8, n_supernodes: int = 15):
        self.r_max = r_max
        self.r_supergraph = r_supergraph
        self.dropout_rate = dropout_rate
        self.n_supernodes = n_supernodes
    
    def build(self, vertices, mask, areas=None, normals=None):
        # Move existing build logic here
        graph = get_graphs_from_vertices(
            vertices, mask, r_max=self.r_max, 
            dropout_rate=self.dropout_rate, ...
        )
        supergraph = build_super_graph(
            graph, r=self.r_supergraph, n_supernodes=self.n_supernodes, ...
        )
        return graph, supergraph

class KNNGraphBuilder(GraphBuilder):
    """Build graphs with k-nearest neighbors."""
    
    def __init__(self, k: int = 8, n_supernodes: int = 15):
        self.k = k
        self.n_supernodes = n_supernodes
    
    def build(self, vertices, mask, areas=None, normals=None):
        # Implement kNN graph building
        pass

class GraphBuilderFactory:
    """Factory for GraphBuilder instances."""
    
    _builders = {
        'radius': RadiusGraphBuilder,
        'knn': KNNGraphBuilder,
    }
    
    @classmethod
    def create(cls, builder_type: str, **kwargs) -> GraphBuilder:
        if builder_type not in cls._builders:
            raise ValueError(f"Unknown builder: {builder_type}")
        return cls._builders[builder_type](**kwargs)