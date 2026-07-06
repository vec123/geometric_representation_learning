import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.vtk.io import save_vtp, load_vtp
from src.vtk.extract import extract_vtp_points_cells
from src.vtk.create import create_polydata_w_lines
from src.graphs.graphs import (
    get_graphs_from_vertices,
    sample_nodes,
    build_bipartite_graph
)
from src.transforms.padding import pad_vertex_list
from config.root import get_project_root


def load_shapes(names, project_root):
    """Loads VTP files and returns them as a padded [B, N, 3] tensor + validity mask."""
    shape_vertices = []
    for name in names:
        vtp = load_vtp(os.path.join(project_root, "tests", "data", f"{name}.vtp"))
        vertices, _ = extract_vtp_points_cells(vtp)
        shape_vertices.append(vertices)

    padded, mask = pad_vertex_list(shape_vertices)
    return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def save_graph_as_vtp(graph, batch_idx, output_dir, filename_suffix):
    """Extracts, re-indexes, and saves one shape from a homogeneous PyG graph."""
    mask = (graph.batch == batch_idx)
    pos = graph.pos[mask].detach().cpu().numpy()

    # Normalize global edge indices to local [0, num_nodes_in_batch) via the batch start.
    start_idx = mask.nonzero(as_tuple=True)[0][0]
    edge_mask = (graph.batch[graph.edge_index[0]] == batch_idx)
    edges = (graph.edge_index[:, edge_mask] - start_idx).T.detach().cpu().numpy()

    save_path = os.path.join(output_dir, f"graph_{batch_idx}_{filename_suffix}.vtp")
    save_vtp(create_polydata_w_lines(pos, edges), save_path)


def save_bipartite_as_vtp(graph, batch_idx, output_dir, filename_suffix):
    """Saves one shape's supernode aggregation: full nodes + supernodes joined by the
    bipartite edges each supernode aggregates over.

    The bipartite ``Data`` carries two node sets (``source_pos``/``source_batch`` for
    the full graph and ``pos``/``batch`` for the supernodes); we merge them into a
    single point cloud so the aggregation edges can be rendered as VTP lines.
    """
    # Source (full-graph) nodes for this shape.
    src_mask = (graph.source_batch == batch_idx)
    src_pos = graph.source_pos[src_mask]
    src_start = src_mask.nonzero(as_tuple=True)[0][0]

    # Target (super) nodes for this shape.
    tgt_mask = (graph.batch == batch_idx)
    tgt_pos = graph.pos[tgt_mask]
    tgt_start = tgt_mask.nonzero(as_tuple=True)[0][0]

    # Edges whose supernode (row 0) belongs to this shape, remapped to local indices.
    ei = graph.edge_index
    edge_mask = tgt_mask[ei[0]]
    e_super = ei[0, edge_mask] - tgt_start          # local supernode index
    e_full = ei[1, edge_mask] - src_start           # local full-node index

    # Combined point set [full ; super]; supernode indices are offset by n_full.
    n_full = src_pos.size(0)
    points = torch.cat([src_pos, tgt_pos], dim=0).detach().cpu().numpy()
    lines = torch.stack([e_full, e_super + n_full], dim=0).T.detach().cpu().numpy()

    save_path = os.path.join(output_dir, f"graph_{batch_idx}_{filename_suffix}.vtp")
    save_vtp(create_polydata_w_lines(points, lines), save_path)


def main():
    Project_ROOT = get_project_root()
    OUTPUT_DIR = os.path.join(Project_ROOT, "tests", "output_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load shapes into a padded [B, N, 3] tensor + validity mask.
    vertices, mask = load_shapes(["sample_01", "nose_0"], Project_ROOT)
    print(f"Loaded vertices shape: {vertices.shape}, mask shape: {mask.shape}")

    # 2. Full graph: radius graph over all valid nodes. No dropout/noise here so the
    #    supernodes sampled below share the same coordinate space; pass a dropout_rate
    #    to get_graphs_from_vertices to decimate the graph before construction.
    full_graph = get_graphs_from_vertices(
        vertices, masks=mask, r_max=0.1, dropout_rate=0.8, noise_std=0.0
    )
    print(f"Full graph has {full_graph.num_nodes} nodes and {full_graph.num_edges} edges.")

    # 3. Supernodes: draw n_s well-spread nodes per shape via farthest-point sampling.
    super_nodes, super_batch = sample_nodes(vertices, mask, num_samples=50, mode='fps')

    # 4. Bipartite aggregation graph: each supernode gathers the full-graph nodes within
    #    r_max (the neighbourhood a supernode aggregates through the GNN).
    super_graph = build_bipartite_graph(
        full_graph.pos, full_graph.batch, super_nodes, super_batch, r_max=0.2
    )
    print(f"Supernode graph has {super_graph.pos.shape[0]} supernodes and "
          f"{super_graph.edge_index.shape[1]} aggregation edges.")

    # 5. Inspection: one VTP per shape for both the full graph and the supernode graph.
    for j in range(vertices.shape[0]):
        save_graph_as_vtp(full_graph, j, OUTPUT_DIR, "full")
        save_bipartite_as_vtp(super_graph, j, OUTPUT_DIR, "supernodes")

    print(f"Inspection files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
