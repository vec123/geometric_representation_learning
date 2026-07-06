import torch
import torch.nn as nn
from e3nn import o3
from src.learning.modules.equivariant.interaction import (
    SelfInteraction,
    SpatialConvolution,
)
from src.learning.modules.equivariant.layer_norm import (
    EquivariantLayerNorm
)

class EquiLayer(nn.Module):
    def __init__(self, in_irreps, target_irreps, sh_lmax = 1, verbose=True):
        super().__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sh_lmax = sh_lmax
        self.verbose = verbose

        # 1. Self Interaction
        self.self_int = SelfInteraction(
            in_irreps=self.in_irreps,
            target_irreps=self.target_irreps,
            sh_lmax=self.sh_lmax,
            verbose=verbose
        )

        # 2. Spatial Convolution
        self.spatial_conv = SpatialConvolution(
            in_irreps=self.target_irreps,
            target_irreps=self.target_irreps,
            sh_lmax=self.sh_lmax,
            verbose=verbose
        )

        # 3. Residual Projection (if irreps mismatch)
        self.res_proj = None
        if self.in_irreps != self.target_irreps:
            self.res_proj = o3.Linear(self.in_irreps, self.target_irreps)

        # 4. Layer Norm
        self.norm = EquivariantLayerNorm(self.target_irreps, verbose=verbose)

    def forward(self, x, pos, edge_index):
        # x is node features, pos is node positions, edge_index for graph structure
        
        # Self Interaction
        h = self.self_int(x)
        
        # Spatial Convolution
        msg = self.spatial_conv(h, pos, edge_index)
        
        # Skip Connection
        if self.res_proj is not None:
            skip = self.res_proj(x)
        else:
            skip = x
            
        res = msg + skip
        
        if self.verbose:
            print("------Skip connection--------")
            print("x.shape: ", x.shape)
            print("msg.shape: ", msg.shape)
            print("skip.shape: ", skip.shape)
            print("res.shape: ", res.shape)
            print("-------Finished--------")

        # Layer Norm
        h_norm = self.norm(res)

        if self.verbose:
            print("--------------Layer --------------")
            print("in.irreps : ", self.in_irreps)
            print("target.irreps : ", self.target_irreps)
            print("out.shape: ", h_norm.shape)
            print("------------Finished-------------")
            
        return h_norm