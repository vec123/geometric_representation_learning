import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from e3nn import o3
from src.learning.layers.equivariant.Self_Spatial_layer import EquiLayer

class GroupEncoder(nn.Module):
    def __init__(self, latent_dim: int = 5, irreps_cfg: dict = None, verbose: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.verbose = verbose
        
        # Define EquiLayers (Assuming stack of EquiLayer modules)
        # In PyG, we stack these in a ModuleList
        in_irreps_str = irreps_cfg.get("input_irreps", "1x0e")
        intermediate_irreps_str = irreps_cfg.get("intermediate_irreps", "32x0e + 32x0o + 16x1e + 16x1o")
        output_irreps_str = irreps_cfg.get("output_irreps", f"{latent_dim}x0e + 2x1o")
        
        self.layers = nn.ModuleList([
            EquiLayer(in_irreps=in_irreps_str, target_irreps=intermediate_irreps_str, verbose=verbose),
            EquiLayer(in_irreps=intermediate_irreps_str, target_irreps=intermediate_irreps_str, verbose=verbose),
            EquiLayer(in_irreps=intermediate_irreps_str, target_irreps=output_irreps_str, verbose=verbose)
        ])
        
        # MLP Heads
        self.mu_net = nn.Linear(latent_dim, latent_dim)
        self.var_net = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Softplus())
        self.weight_net = nn.Linear(latent_dim, 1)

    def forward(self, x, pos, edge_index, batch_idx):
        # 1. Message Passing
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, edge_index)
            if self.verbose:
                print(f"Layer {i} output shape: {x.shape}")

        # 2. Global Pooling (VAE Latent Space)
        # Using invariant scalars for pooling
        scalars = x.slice_by_irreps("0e").array
        
        # Weighted mean pooling
        weights = torch.softmax(self.weight_net(scalars), dim=0)
        mu = global_mean_pool(weights * self.mu_net(scalars), batch_idx)
        logvar = torch.log(global_mean_pool(weights * self.var_net(scalars), batch_idx) + 1e-8)
        
        # 3. Equivariant Output (Rotation & Translation)
        vectors = x.slice_by_irreps("1o").array
        # Weighted sum for rotation vectors
        vec_graph = global_mean_pool(weights * vectors, batch_idx)
        
        # Extract v1, v2 for rotation matrix (2 vectors -> 3x3 R)
        v1, v2 = vec_graph[:, 0, :], vec_graph[:, 1, :]
        rot_matrix = self.get_rotation_matrix_from_two_vectors(v1, v2)
        
        # Translation: center of mass
        transl = global_mean_pool(pos, batch_idx)
        
        return (mu, logvar), rot_matrix, vec_graph, transl

    def get_rotation_matrix_from_two_vectors(self, v1, v2):
        u = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
        
        # Gram-Schmidt
        dot = torch.einsum('bi,bi->b', u, v2).unsqueeze(-1)
        w_raw = v2 - dot * u
        w = w_raw / (torch.norm(w_raw, dim=-1, keepdim=True) + 1e-8)
        
        # Cross product for 3rd basis vector
        last_v = torch.cross(u, w, dim=-1)
        return torch.stack([u, w, last_v], dim=-1)