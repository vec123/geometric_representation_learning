import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.learning.models.folding_decoder import FoldingDecoder
from src.learning.models.group_encoder import GroupEncoder


def test_folding_decoder():
    print('=== test_folding_decoder ===')
    batch_size = 2
    latent_dim = 4
    num_samples = 16

    decoder = FoldingDecoder(num_samples=num_samples, latent_dim=latent_dim, n_freqs=2, verbose=False)
    latent = torch.randn(batch_size, latent_dim, dtype=torch.float32)

    output = decoder(latent)
    print('folding_decoder output shape:', output.shape)
    assert output.shape == (batch_size, num_samples, 3)

    output_manual = decoder(latent, manual_inv=True)
    print('folding_decoder manual_inv output shape:', output_manual.shape)
    assert output_manual.shape == output.shape


def test_group_encoder_rotation():
    print('=== test_group_encoder_rotation ===')
    layer_cfg = {
        'input_irreps': '1x0e',
        'intermediate_irreps': '1x0e + 1x1o',
        'output_irreps': '4x0e + 2x1o',
    }
    encoder = GroupEncoder(latent_dim=4, irreps_cfg=layer_cfg, verbose=False)

    v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    v2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    rotation_matrix = encoder.get_rotation_matrix_from_two_vectors(v1, v2)

    print('rotation_matrix shape:', rotation_matrix.shape)
    assert rotation_matrix.shape == (2, 3, 3)

    identity = torch.eye(3, dtype=rotation_matrix.dtype, device=rotation_matrix.device).unsqueeze(0)
    product = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
    print('rotation_matrix orthonormal check:', product)
    assert torch.allclose(product, identity, atol=1e-5)


if __name__ == '__main__':
    test_folding_decoder()
    test_group_encoder_rotation()
    print('test_models.py completed successfully.')
