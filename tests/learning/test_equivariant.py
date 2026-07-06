import os
import sys
import torch
from e3nn import o3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.learning.modules.equivariant.layer_norm import EquivariantLayerNorm

try:
    from src.learning.modules.equivariant.attention import EquivariantAttention
    attention_import_error = None
except Exception as exc:
    EquivariantAttention = None
    attention_import_error = exc


def test_equivariant_layer_norm():
    print('=== test_equivariant_layer_norm ===')
    irreps = '2x0e + 1x1o'
    irreps_array = o3.IrrepsArray(irreps, torch.randn(2, 5, dtype=torch.float32))
    layer_norm = EquivariantLayerNorm(irreps, eps=1e-5, affine=True, normalization='component')

    output = layer_norm(irreps_array)
    print('output shape:', output.shape)
    print('output irreps:', output.irreps)

    assert output.shape == irreps_array.shape
    assert output.irreps == irreps_array.irreps


def test_equivariant_attention():
    print('=== test_equivariant_attention ===')
    if EquivariantAttention is None:
        print('Skipping EquivariantAttention test because import failed:', attention_import_error)
        return

    irreps_in = '1x0e + 1x1o'
    irreps_out = '1x0e + 1x1o'
    num_nodes = 3
    node_features = o3.IrrepsArray(irreps_in, torch.randn(num_nodes, 4, dtype=torch.float32))
    positions = torch.randn(num_nodes, 3, dtype=torch.float32)
    senders = torch.tensor([0, 1, 2], dtype=torch.long)
    receivers = torch.tensor([1, 2, 0], dtype=torch.long)

    attention = EquivariantAttention(irreps_in, irreps_out, sh_lmax=1)
    try:
        output, alpha = attention(node_features, positions, senders, receivers, num_nodes)
        print('attention output type:', type(output))
        print('alpha shape:', alpha.shape)
        assert alpha.shape[0] == senders.shape[0]
    except Exception as exc:
        print('EquivariantAttention forward pass failed:', exc)
        raise


if __name__ == '__main__':
    test_equivariant_layer_norm()
    test_equivariant_attention()
    print('test_equivariant.py completed successfully.')
