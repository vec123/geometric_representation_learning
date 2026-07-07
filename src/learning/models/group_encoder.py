import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from e3nn import o3
from src.learning.layers.equivariant.Self_Spatial_layer import EquiLayer
from src.learning.modules.equivariant.irreps_utils import scalar_features, vector_features
from src.learning.models.encoder_output import EncoderOutput

class GroupEncoder(nn.Module):
    def __init__(self, latent_dim: int = 5, 
                 irreps_cfg: dict = None, 
                 sh_lmax: int =1, 
                 verbose: bool = False):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.verbose = verbose
        
        # Define EquiLayers (Assuming stack of EquiLayer modules)
        # In PyG, we stack these in a ModuleList
        in_irreps_str = irreps_cfg.get("input_irreps", "1x0e")
        intermediate_irreps_str = irreps_cfg.get("intermediate_irreps", "32x0e + 32x0o + 16x1e + 16x1o")
        output_irreps_str = irreps_cfg.get("output_irreps", f"{latent_dim}x0e + 2x1o")
        self.out_irreps = o3.Irreps(output_irreps_str)

        self.layers = nn.ModuleList([
            EquiLayer(in_irreps=in_irreps_str,
                        target_irreps=intermediate_irreps_str,
                        spatial_sh_lmax = sh_lmax,
                        interaction_sh_lmax = 4,
                        verbose=verbose),
            EquiLayer(in_irreps=intermediate_irreps_str,
                       target_irreps=intermediate_irreps_str,
                        spatial_sh_lmax = sh_lmax,
                        interaction_sh_lmax = 4, 
                       verbose=verbose),
            EquiLayer(in_irreps=intermediate_irreps_str, 
                      target_irreps=output_irreps_str,
                      spatial_sh_lmax = sh_lmax,
                      interaction_sh_lmax = 4,
                        verbose=verbose)
        ])
        
        # MLP Heads
        self.mu_net = nn.Linear(latent_dim, latent_dim)
        self.var_net = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Softplus())
        self.weight_net = nn.Linear(latent_dim, 1)

    def forward(self, x, pos, edge_index, batch_idx):
        # 1. Message Passing
        for i, layer in enumerate(self.layers):
            if self.verbose:
                print(f"---------Layer {i} with: "
                     f" x: {x.shape}, "
                     f" pos: {pos.shape}, "
                     f" edge index: {edge_index.shape}")
                
            x = layer(x, pos, edge_index)
            if self.verbose:
                print(f"Layer {i} output shape: {x.shape}")

        #  Global Pooling (VAE Latent Space)
        # Using invariant scalars for pooling
        scalars = scalar_features(x, self.out_irreps)          # [N, #0e]

        # Pass size= to every pool so a non-contiguous batch_idx (e.g. a shape that
        # lost all its nodes to dropout) can't silently drop a row: the latent batch
        # dimension stays fixed at the number of graphs in this batch.
        num_graphs = int(batch_idx.max().item()) + 1

        # Weighted mean pooling
        weights = torch.softmax(self.weight_net(scalars), dim=0)
        mu = global_mean_pool(weights * self.mu_net(scalars), batch_idx, size=num_graphs)
        logvar = torch.log(global_mean_pool(weights * self.var_net(scalars), batch_idx, size=num_graphs) + 1e-8)

        #  Equivariant Output (Rotation & Translation)
        vectors = vector_features(x, self.out_irreps, '1o')    # [N, n_vec, 3]
        n_vec = vectors.shape[1]
        # Weighted mean over nodes, per graph, then reshape back to [B, n_vec, 3].
        vec_weighted = (weights.unsqueeze(-1) * vectors).reshape(vectors.shape[0], -1)
        vec_graph = global_mean_pool(vec_weighted, batch_idx, size=num_graphs).reshape(-1, n_vec, 3)

        # Extract v1, v2 for rotation matrix (2 vectors -> 3x3 R)
        v1, v2 = vec_graph[:, 0, :], vec_graph[:, 1, :]
        rot_matrix = self.get_rotation_matrix_from_two_vectors(v1, v2)

        # Translation: center of mass
        transl = global_mean_pool(pos, batch_idx, size=num_graphs)
        
        #return (mu, logvar), rot_matrix, vec_graph, transl
        return EncoderOutput( mu=mu, 
                             logvar=logvar, 
                             rotation=rot_matrix, 
                             translation=transl)
    

    def get_rotation_matrix_from_two_vectors(self, v1, v2):
        u = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
        
        # Gram-Schmidt
        dot = torch.einsum('bi,bi->b', u, v2).unsqueeze(-1)
        w_raw = v2 - dot * u
        w = w_raw / (torch.norm(w_raw, dim=-1, keepdim=True) + 1e-8)
        
        # Cross product for 3rd basis vector
        last_v = torch.cross(u, w, dim=-1)
        return torch.stack([u, w, last_v], dim=-1)