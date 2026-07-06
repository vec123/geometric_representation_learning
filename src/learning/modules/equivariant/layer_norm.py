
import torch
import torch.nn as nn
from e3nn import o3



class EquivariantLayerNorm(nn.Module):
    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component', verbose=True):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.verbose = verbose
        self.normalization = normalization

        if self.affine:
            # Create a ParameterList to hold the learned scalar weights
            # For each (mul, ir) pair, we want a learnable scalar per multiplicity
            self.affine_weights = nn.ParameterList([
                nn.Parameter(torch.ones(mul)) for mul, ir in self.irreps
            ])

    def forward(self, x):
        # Assuming x is an IrrepsArray
        field_list = []
        for i, (mul, ir) in enumerate(self.irreps):
            field = x.slice_by_irreps(f"{mul}x{ir}")
            field_reshaped = field.array.reshape(field.shape[0], mul, ir.dim)

            if ir.l == 0:
                # --- SCALAR BRANCH ---
                mean = field_reshaped.mean(dim=(1, 2), keepdim=True)
                var = field_reshaped.var(dim=(1, 2), keepdim=True, unbiased=False)
                normed = (field_reshaped - mean) / torch.sqrt(var + self.eps)
            else:
                # --- VECTOR/TENSOR BRANCH (RMS Norm) ---
                sq = torch.square(field_reshaped)
                # RMS calculation: mean of squares across dimensions
                if self.normalization == 'norm':
                    rms = torch.sqrt(sq.sum(dim=-1, keepdim=True) + self.eps)
                else:
                    rms = torch.sqrt(sq.mean(dim=-1, keepdim=True) + self.eps)
                
                normed = field_reshaped / (rms + self.eps)

            # --- AFFINE ---
            if self.affine:
                # Apply learned weight: (mul,) -> (1, mul, 1) for broadcasting
                weight = self.affine_weights[i].view(1, mul, 1)
                normed = field_reshaped * weight
            else:
                normed = field_reshaped

            field_list.append(normed.reshape(field.shape[0], -1))

        final_array = torch.cat(field_list, dim=-1)

        if self.verbose:
            print("--------------EquivariantLayerNorm --------------")
            print("Input irreps: ", self.irreps)
            print("Output shape: ", final_array.shape)
            print("--------------Finished --------------")

        return o3.IrrepsArray(self.irreps, final_array)