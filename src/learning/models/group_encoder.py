import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax
from e3nn import o3
from src.learning.layers.equivariant.Self_Spatial_layer import EquiLayer
from src.learning.modules.equivariant.interaction import BipartiteSpatialConvolution
from src.learning.modules.equivariant.transformer import build_equivariant_transformer
from src.learning.modules.equivariant.irreps_utils import scalar_features, vector_features
from src.learning.models.encoder_output import EncoderOutput
from src.learning.modules.transformers.perceiver_encoder import PerceiverReducer

class GroupEncoder(nn.Module):
    def __init__(self, latent_dim: int = 5,
                 irreps_cfg: dict = None,
                 sh_lmax: int =1,
                 readout: str = "mean",
                 readout_heads: int = 1,
                 supernode_sh_lmax: int = 4,
                 transformer_type: str = "se3",
                 transformer_cfg: dict = None,
                 area_pool: bool = False,
                 verbose: bool = False):

        super().__init__()
        self.latent_dim = latent_dim
        self.readout = readout
        self.area_pool = area_pool
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
                       verbose=verbose)
        ])
        
        self.final_linear = o3.Linear(intermediate_irreps_str, output_irreps_str)

        # MLP Heads (shared by both readouts + the pose head)
        self.mu_net = nn.Linear(latent_dim, latent_dim)
        self.var_net = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Softplus())
        self.weight_net = nn.Linear(latent_dim, 1)

        self.mu_bn = nn.BatchNorm1d(latent_dim)
        self.mu_ln = nn.LayerNorm(latent_dim)
        # Pool readout: one learned query cross-attends to all node scalars,
        # collapsing them to a single [latent_dim] token per graph (PerceiverReducer
        # with stages=[1]). The scalar width is latent_dim, so d_shared = latent_dim.
        # The mean-pool path below is kept for ablation (readout="mean").
        self.readout_pool = None
        if readout == "attention":
            self.readout_pool = PerceiverReducer(
                d_shared=latent_dim, stages=[1],
                num_heads=readout_heads, self_attend=False,
            )
        elif readout != "mean":
            raise ValueError(f"readout must be 'attention' or 'mean', got {readout!r}")

        # Optional supernode aggregation: an equivariant bipartite conv maps the
        # full-node features onto the supernodes (in_irreps == out_irreps, so the
        # scalar count is unchanged). When enabled, the readout below pools over
        # supernodes instead of nodes.
        # Attention: if supergraph is None these are dead weights    
        self.supernode_conv = BipartiteSpatialConvolution(
                in_irreps=self.out_irreps, target_irreps=self.out_irreps,
                sh_lmax=supernode_sh_lmax, verbose=verbose,
            )

        # Optional equivariant transformer refinement of the pooled features, applied
        # after supernode aggregation and before the scalars are filtered out. Selectable
        # backend ('se3' | 'equiformer'); in-place on out_irreps so nothing downstream
        # changes. Disable with transformer_type=None.
        self.equi_transformer = build_equivariant_transformer(
            transformer_type, self.out_irreps, transformer_cfg, verbose=verbose,
        )

    def forward(self,graph, supergraph):

        x = graph.x
        pos = graph.pos
        
        edge_index = graph.edge_index
        batch_idx = graph.batch
        node_area = getattr(graph, 'area', None) if self.area_pool else None
        node_normal = getattr(graph, 'normal', None)
        if node_normal is not None:
            x = torch.cat([x, node_normal], dim=-1)
        if supergraph is not None:
            super_pos=supergraph.pos
            super_batch=supergraph.batch
            super_edge_index=supergraph.edge_index

        # Message Passing on the full graph.
        for i, layer in enumerate(self.layers):
            if self.verbose:
                print(f"---------Layer {i} with: "
                     f" x: {x.shape}, "
                     f" pos: {pos.shape}, "
                     f" edge index: {edge_index.shape}")

            x = layer(x, pos, edge_index, area=node_area)
            if self.verbose:
                print(f"Layer {i} output shape: {x.shape}")

        # Optional supernode aggregation
        if supergraph is not None:
            if super_pos is None or super_batch is None or super_edge_index is None:
                raise ValueError(
                    "use_supernodes=True requires super_pos, super_batch and super_edge_index."
                )
            feat = self.supernode_conv(x, pos, super_pos, super_edge_index,
                                       area_src=node_area )  # [S, out_irreps.dim]
            pool_batch = super_batch
            pool_pos = super_pos
            pool_area = getattr(supergraph, 'area', None)          # supernode mass, if provided
        else:
            feat = x
            pool_batch = batch_idx
            pool_pos = pos
            pool_area = getattr(graph, 'area', None)               # per-vertex area, if provided

        # Area weighting turns sums over the pooled set into surface integrals; 
        #  Off unless area_pool and an 'area' attribute are present 
        # -> falls back to the uniform behavior.
        if not self.area_pool or pool_area is None:
            pool_area = None
        elif pool_area.dim() == 1:
            pool_area = pool_area.reshape(-1, 1)                   # [n_pool, 1]

        # Equivariant transformer refinement over the pooled set (supernodes or nodes),
        # on the full irreps BEFORE the invariant scalars are filtered out.
        # and before the irreps are reduced to the final encoding set
        if self.equi_transformer is not None:
            feat = self.equi_transformer(feat, pool_pos, pool_batch)

        # reduce to the final encoding set
        feat = self.final_linear(feat)

        # Global Pooling (VAE Latent Space) over the invariant scalars.
        scalars = scalar_features(feat, self.out_irreps)          # [n_nodes, #0e]
        n_saclars = scalars.shape[1]
        assert n_saclars == self.latent_dim

        # Pass size= to every pool so a non-contiguous pool_batch can't drop a row.
        num_graphs = int(pool_batch.max().item()) + 1

        # Per-graph attention weights: softmax is taken WITHIN each shape (grouped by
        # pool_batch), so the weights sum to 1 per shape and don't leak across shapes.
        # Computed once and shared by the scalar readout and the vector/pose head. Use
        # with global_add_pool so the softmax is the single normalization (a following
        # global_mean_pool would divide again -> a per-shape latent crushed toward 0).
        logit = self.weight_net(scalars)                                  # [n_pool, 1]
        if pool_area is not None:
            # Area-weighted attention: softmax(logit + log a) = a·e^logit / Σ a·e^logit,
            # so a denser sampling no longer over-counts (a per-shape scale of a cancels).
            logit = logit + torch.log(pool_area.clamp_min(1e-12))
        weights = scatter_softmax(logit, pool_batch)                     # [n_pool, 1]

        if self.readout == "attention":
            # Attention-pool: collapse each graph's tokens to one, then project to
            # mu/logvar. Per-graph (batch dim 1) so tokens only attend within a shape.
            pooled = []
            for b in range(num_graphs):
                toks = scalars[pool_batch == b].unsqueeze(0)        # [1, n_b, latent_dim]
                pooled.append(self.readout_pool(toks))              # [1, 1, latent_dim]
            pooled = torch.cat(pooled, dim=0).squeeze(1)            # [B, latent_dim]
            mu = self.mu_net(pooled)
            logvar = torch.log(self.var_net(pooled) + 1e-8)
        else:
            # Weighted sum (weights already sum to 1 per shape -> single normalization).
            mu = global_add_pool(weights * self.mu_net(scalars), pool_batch, size=num_graphs)
            logvar = torch.log(global_add_pool(weights * self.var_net(scalars), pool_batch, size=num_graphs) + 1e-8)

        #Optional: Batch and/or Layer Norm
        #mu = mu.unsqueeze(1)
        #logvar = logvar.unsqueeze(1)
        #mu = self.mu_ln(mu)

        #print("mu.shape: ", mu.shape)
        #mu = mu.squeeze(1)
        #mu = self.mu_bn(mu)
        #mu = mu.unsqueeze(1)
        #print("mu.shape: ", mu.shape)

        # Equivariant Output (Rotation & Translation)
        vectors = vector_features(feat, self.out_irreps, '1o')    #[n_nodes, n_vec, 3]
        n_vec = vectors.shape[1]
        assert n_vec == 2

        # Weighted sum over tokens (same per-shape weights), then reshape to [B, n_vec, 3].
        vec_weighted = (weights.unsqueeze(-1) * vectors).reshape(vectors.shape[0], -1)
        vec_graph = global_add_pool(vec_weighted, pool_batch, size=num_graphs).reshape(-1, n_vec, 3)

        # Extract v1, v2 for rotation matrix (2 vectors -> 3x3 R)
        v1, v2 = vec_graph[:, 0, :], vec_graph[:, 1, :]
        rot_matrix = self.get_rotation_matrix_from_two_vectors(v1, v2)

        # Translation: center of mass (over the pooled token set). Area-weighted when
        # areas are available -> the true surface centroid (a plain mean over-weights
        # densely sampled regions).
        if pool_area is not None:
            denom = global_add_pool(pool_area, pool_batch, size=num_graphs).clamp_min(1e-12)
            transl = global_add_pool(pool_area * pool_pos, pool_batch, size=num_graphs) / denom
        else:
            transl = global_mean_pool(pool_pos, pool_batch, size=num_graphs)
        
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