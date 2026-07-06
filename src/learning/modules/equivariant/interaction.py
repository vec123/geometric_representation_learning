import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter



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
    def __init__(self, in_irreps, target_irreps, sh_lmax=4, verbose=True):
        super().__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sh_lmax = sh_lmax
        self.verbose = verbose

        # 1. Tensor Product: V_i ⊗ V_i
        # Using Full TensorProduct to combine features with themselves
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.in_irreps, 
            self.in_irreps, # Output irreps restricted by logic if needed
            shared_weights=False
        )
        
        # 2. Gating Block
        # We need to know the number of scalars and norms to set up the gate
        # This assumes your GatingBlock is defined for these dimensions
        self.gate_net = GatingBlock(input_dim=32, hidden_dim=self.in_irreps.num_irreps, out_dim=self.in_irreps.num_irreps)
        
        # 3. Final Projection
        self.linear_out = o3.Linear(self.in_irreps, self.target_irreps)

    def forward(self, node_features):
        # 1. Tensor Product
        # v_sq represents the squared interaction
        v_sq = self.tp(node_features, node_features)
        
        # 2. Skip connection and Gating setup
        # In PyTorch e3nn, we typically use o3.concatenate
        v_intermediate = o3.concatenate([node_features, v_sq], dim=-1)
        
        # 3. Gating logic
        scalars = v_intermediate.slice_by_irreps("0e")
        vectors = v_intermediate.slice_by_irreps("1o")
        v_lengths = vectors.norm() 
        
        gate_input = torch.cat([scalars.array, v_lengths.array], dim=-1)
        gate = self.gate_net(gate_input)
        
        # Gating
        gated = v_intermediate * gate
        
        # 4. Final Projection
        v_out = self.linear_out(gated)
        
        if self.verbose:
            print("--------------SelfInteraction --------------")
            print("target_irreps: ", self.target_irreps)
            print("in.irreps: ", node_features.irreps)
            print("v_intermediate.irreps: ", v_intermediate.irreps)
            print("v_out.irreps: ", v_out.irreps)
            print("--------------Finished --------------")
            
        return v_out
    

class SpatialConvolution(MessagePassing):
    def __init__(self, in_irreps, target_irreps, sh_lmax=4, verbose=True):
        super().__init__(aggr='add') # "add" aggregation
        self.verbose = verbose
        self.in_irreps = o3.Irreps(in_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sh_lmax = sh_lmax

        # Spherical Harmonics
        self.sh = o3.SphericalHarmonics([l for l in range(1, sh_lmax + 1)], normalize=True)
        
        # Tensor Product for messages
        self.tp = o3.FullyConnectedTensorProduct(self.in_irreps, self.sh.irreps_out, self.in_irreps)
        
        # Linear projection for message
        self.lin_msg = o3.Linear(self.in_irreps + self.in_irreps, target_irreps)
        
        # Gating logic: takes scalars and norms
        self.gate_net = GatingBlock(input_dim=32, hidden_dim=self.in_irreps.num_irreps, out_dim=self.in_irreps.num_irreps)

    def forward(self, x, pos, edge_index):
        # 1. Compute Degrees: Count edges per node
        # edge_index[1] contains the receiver indices
        row, col = edge_index
        degree = torch.ones(x.size(0), device=x.device)
        degree = scatter(degree, row, reduce='add') # k = number of incoming edges
        
        # 2. Propagate
        out = self.propagate(edge_index, x=x, pos=pos)
        
        # 3. Apply Normalization (1/k)
        # Avoid division by zero for isolated nodes
        norm = 1.0 / torch.clamp(degree, min=1.0)
        if self.verbose:
            print("--------------SpatialConvolution --------------")
            print("target_irreps: ", self.target_irreps)
            print("in.irreps: ", x.irreps)
            print("out.irreps: ", out.irreps)
            print("--------------Finished --------------")
        return out * norm.view(-1, 1) # Scaling the aggregated features

    def message(self, x_i, x_j, pos_i, pos_j):
        # x_j: sender features, x_i: receiver features
        rel_pos = pos_j - pos_i
        edge_sh = self.sh(rel_pos)
        
        # Tensor Product path
        tp_msg = self.tp(x_j, edge_sh)
        geo_features = o3.concatenate([x_j, tp_msg], dim=-1)
        
        # Gating
        gate_in = torch.cat([x_i.slice_by_irreps("0e").array, x_j.slice_by_irreps("0e").array], dim=-1)
        gate = self.gate_net(gate_in)
        
        return self.lin_msg(geo_features * gate)

    def update(self, aggr_out, x):
        # aggr_out is the sum of messages (v_tilde)
        # Normalization by degree (k)
        # Assuming degree is pre-computed or retrieved via count
        return aggr_out + x # Simplified residual