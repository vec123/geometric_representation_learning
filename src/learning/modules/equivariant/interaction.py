import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from src.learning.modules.equivariant.irreps_utils import (
    scalar_features,
    expand_per_irrep_gate,
)

class GatingBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # Defining the layers in the constructor
        self.dense1 = nn.Linear(input_dim, hidden_dim) # Note: requires input dimension
        self.dense2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # Implementation of the MLP gate logic
        x = self.dense1(x)
        x = F.silu(x)
        x = self.dense2(x)
        
        return x
  

class SelfInteraction(nn.Module):
    def __init__(self, in_irreps, target_irreps, sh_lmax = 1, verbose=True):
        super().__init__()
        # self.in_irreps = o3.Irreps(in_irreps)
        self.in_irreps = self.limit_irreps(in_irreps, sh_lmax)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sh_lmax = sh_lmax
        self.verbose = verbose

        # Tensor Product: V ⊗ V (equivariant self-interaction / "square").
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.in_irreps,
            self.in_irreps,  # Output irreps of the interaction
            internal_weights=True
        )

        # Concatenated features V ⊕ (V ⊗ V); order preserved to match torch.cat.
        self.concat_irreps = self.in_irreps + self.in_irreps

        # MLP gating: one gate per irrep, computed from the invariant scalars.
        self.num_scalars = self.concat_irreps.count("0e")
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.num_scalars, 64),
            nn.SiLU(),
            nn.Linear(64, self.concat_irreps.num_irreps)  # one gate per irrep
        )

        # Final Linear projection: (V ⊕ V⊗V) -> target.
        self.linear_out = o3.Linear(self.concat_irreps, self.target_irreps)

    def limit_irreps(self, irreps_str, max_l):
        """Filters an irreps string to keep only l <= max_l."""
        irreps = o3.Irreps(irreps_str)
        filtered = [ir for ir in irreps if ir.ir.l <= max_l]
        return o3.Irreps(filtered)

    def forward(self, node_features):
        # Step 1: V ⊗ V, then concatenate with V -> V ⊕ (V ⊗ V).
        interaction = self.tp(node_features, node_features)
        v_concat = torch.cat([node_features, interaction], dim=-1)

        # Step 2: per-irrep gate from the invariant scalars (keeps equivariance).
        scalars = scalar_features(v_concat, self.concat_irreps)
        gate = torch.sigmoid(self.gate_mlp(scalars))
        gate = expand_per_irrep_gate(gate, self.concat_irreps)
        gated = v_concat * gate

        # Step 3: linear projection to the target irreps.
        out = self.linear_out(gated)
        if self.verbose:
            print("--------------SelfInteraction --------------")
            print("target_irreps: ", self.target_irreps)
            print("out.shape: ", out.shape)
            print("--------------Finished --------------")

        return out
    

class SpatialConvolution(MessagePassing):
    def __init__(self, in_irreps, target_irreps, sh_lmax=4, verbose=True):
        super().__init__(aggr='add') # "add" aggregation
        self.verbose = verbose
        self.in_irreps = o3.Irreps(in_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sh_lmax = sh_lmax

        # Spherical Harmonics
        self.sh = o3.SphericalHarmonics([l for l in range(1, sh_lmax + 1)], normalize=True)
        
        # Tensor Product for messages; weights supplied per-edge by the radial net.
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.sh.irreps_out, self.in_irreps,
            shared_weights=False, internal_weights=False,
        )
        self.num_weights = self.tp.weight_numel
        # Radial network: message weights as a function of edge length (scalar -> weights).
        self.weight_net = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, self.num_weights)
        )

        # Message features x_j ⊕ (x_j ⊗ Y); order preserved to match torch.cat.
        self.geo_irreps = self.in_irreps + self.in_irreps
        self.lin_msg = o3.Linear(self.geo_irreps, target_irreps)

        # Gating: one gate per message irrep, from sender + receiver invariant scalars.
        self.num_scalars = self.in_irreps.count("0e")
        self.gate_net = GatingBlock(
            input_dim=2 * self.num_scalars,
            hidden_dim=64,
            out_dim=self.geo_irreps.num_irreps,
        )


    def forward(self, x, pos, edge_index):
        # 1. In-degree of each receiver node (edge_index[1]) for 1/k normalization.
        _, col = edge_index
        ones = torch.ones(col.size(0), device=x.device)
        degree = scatter(ones, col, dim=0, dim_size=x.size(0), reduce='add')

        # 2. Propagate (messages aggregated at the receivers).
        out = self.propagate(edge_index, x=x, pos=pos)

        # 3. Normalize by 1/k (per-node invariant scalar); guard isolated nodes.
        norm = 1.0 / torch.clamp(degree, min=1.0)
        if self.verbose:
            print("--------------SpatialConvolution --------------")
            print("target_irreps: ", self.target_irreps)
            print("out.shape: ", out.shape)
            print("--------------Finished --------------")
        return out * norm.view(-1, 1)  # Scaling the aggregated features

    def message(self, x_i, x_j, pos_i, pos_j):
        # x_j: sender features, x_i: receiver features
        rel_pos = pos_j - pos_i
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_sh = self.sh(rel_pos)

        # Equivariant message: (x_j ⊗ Y) with radial weights, concatenated with x_j.
        weights = self.weight_net(dist)
        tp_msg = self.tp(x_j, edge_sh, weights)
        geo_features = torch.cat([x_j, tp_msg], dim=-1)

        # Per-irrep gate from sender + receiver invariant scalars (keeps equivariance).
        x_i_0e = scalar_features(x_i, self.in_irreps)
        x_j_0e = scalar_features(x_j, self.in_irreps)
        gate = torch.sigmoid(self.gate_net(torch.cat([x_i_0e, x_j_0e], dim=-1)))
        gate = expand_per_irrep_gate(gate, self.geo_irreps)

        return self.lin_msg(geo_features * gate)

    def update(self, aggr_out, x):
        # aggr_out is the sum of messages (v_tilde)
        # Normalization by degree (k)
        # Assuming degree is pre-computed or retrieved via count
        return aggr_out + x # Simplified residual