import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FoldingDecoder(nn.Module):
    def __init__(self, num_samples=256, latent_dim=8, n_freqs=4, verbose=True):
        super().__init__()
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.n_freqs = n_freqs
        self.verbose = verbose
        self.grid_size = int(math.sqrt(num_samples))

        # Input dimension calculation:
        # grid: 2 channels * 2 * n_freqs = 4 * n_freqs
        # latent: latent_dim
        # total = (4 * n_freqs) + latent_dim
        input_dim_1 = (4 * n_freqs) 
        input_dim_2 = (4 * n_freqs)  

        # Layers
        self.dense1_1 = nn.Linear(input_dim_1, 128)
        self.film1 = nn.Linear(latent_dim, 128*2)
        self.norm1_1 = nn.LayerNorm(128)
        self.dense1_2 = nn.Linear(128, 128)
        self.norm1_2 = nn.LayerNorm(128)
        self.fold1 = nn.Linear(128, 3)

        self.dense2_1 = nn.Linear(input_dim_2, 128)
        self.film2 = nn.Linear(latent_dim, 128*2)
        self.film3 = nn.Linear(3, 128*2)
        self.norm2_1 = nn.LayerNorm(128)
        self.dense2_2 = nn.Linear(128, 128)
        self.norm2_2 = nn.LayerNorm(128)
        self.fold2 = nn.Linear(128, 3)

        # Initialize weights (Variance Scaling)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m is self.fold1 or m is self.fold2:
                    nn.init.normal_(m.weight, std=0.01)

    def positional_encoding(self, coords):
        # coords: [batch, samples, 2]
        freqs = 2.0 ** torch.arange(self.n_freqs, device=coords.device)
        scaled = coords.unsqueeze(-1) * freqs * math.pi
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return encoded.view(coords.shape[0], coords.shape[1], -1)

    def forward(self, latent, manual_inv=False):
        # latent: [B, 1, D] -- one D-dimensional latent vector per sample. A plain
        # [B, D] is promoted to [B, 1, D] for backward compatibility.
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)
        if latent.shape[1] != 1:
            raise ValueError(
                f"FoldingDecoder expects a single latent per sample ([B, 1, D]); got "
                f"{latent.shape[1]} tokens. Reduce/pool the latent set to one token first."
            )
        batch_size = latent.shape[0]

        latent_raw = torch.zeros_like(latent) if manual_inv else latent
        latent_tiled = latent_raw.expand(-1, self.num_samples, -1)   # [B, num_samples, D]

        # Create grid
        u = torch.linspace(-1, 1, self.grid_size, device=latent.device)
        v = torch.linspace(-1, 1, self.grid_size, device=latent.device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        grid = torch.stack([uu.flatten(), vv.flatten()], dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        encoded_grid = self.positional_encoding(grid)
        
        # --- First Fold ---
        #x = torch.cat([encoded_grid, latent_tiled], dim=-1)
        #h = self.norm1_1(F.silu(self.dense1_1(x)))
        # Replace cat conditioning with FiLM conditioning
        h = self.norm1_1(F.silu(self.dense1_1(encoded_grid)))   # grid pathway, normed
        g, b = self.film1(latent_tiled).chunk(2, dim=-1)        # latent -> scale/shift
        h = (1 + g) * h + b 
        h = self.norm1_2(F.silu(self.dense1_2(h) + h))
        points_coarse = self.fold1(h)
        
        # --- Second Fold ---
        # x = torch.cat([encoded_grid, points_coarse, latent_tiled, x], dim=-1)
        # Replace cat conditioning with FiLM conditioning
        h = self.norm2_1(F.silu(self.dense2_1(encoded_grid)))
        
        g, b = self.film2(latent_tiled).chunk(2, dim=-1)  # latent -> scale/shift
        h = (1 + g) * h + b 

        g, b = self.film3(points_coarse).chunk(2, dim=-1)  # latent -> scale/shift
        h = (1 + g) * h + b 

        h = self.norm2_2(F.silu(self.dense2_2(h) + h))
        points_final = points_coarse + self.fold2(h)
        
        return points_final