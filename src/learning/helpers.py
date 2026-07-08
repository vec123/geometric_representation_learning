
import os
import glob
import torch

from src.vtk.io import load_vtp, save_vtp
from src.vtk.create import create_polydata_w_lines
from src.vtk.extract import extract_vtp_points_cells
from src.vtk.fields import add_point_field
from src.graphs.graphs import (
    get_graphs_from_vertices, 
    build_super_graph,
    get_individual_graph,
    get_bipartite_graph)
from src.transforms.padding import pad_vertex_list
import random

def load_dataset(data_path="DATA_ROOT", parts=["mouth", "nose"]):
    """Load face-part shapes; fall back to tests/data so the script always runs."""
    vertices = []
    for part in parts:
        for file in glob.glob(os.path.join(data_path, part, "*.vtp")):
            verts, _ = extract_vtp_points_cells(load_vtp(file))
            vertices.append(verts)
            
    if not vertices:
        print("Dataset not found - falling back to tests/data shapes.")
        # Assuming you have a fallback mechanism here
        
    # --- Shuffle the list of vertices ---
    random.shuffle(vertices)
    
    print("loaded shapes:", [v.shape for v in vertices])
    
    # Now pad the shuffled list
    padded, mask = pad_vertex_list(vertices)
    
    return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
""" 
def load_dataset(data_path = "DATA_ROOT", parts = ["mouth", "nose"] ):
    vertices = []
    for part in parts:
        for file in glob.glob(os.path.join(data_path, part, "*.vtp")):
            verts, _ = extract_vtp_points_cells(load_vtp(file))
            vertices.append(verts)
    if not vertices:
        print("Dataset not found - falling back to tests/data shapes.")
    print("loaded shapes:", [v.shape for v in vertices])
    padded, mask = pad_vertex_list(vertices)
    return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
"""

def build_training_graph(vertices, mask, 
                         key, 
                         r_max=0.1, 
                         dropout_rate=0.8, 
                         n_supernodes = 10, 
                         r_supergraph = 0.2,
                         use_supernodes= False):
    """Build the graph fed to the encoder, per the USE_SUPERNODES toggle, and attach
    a constant 1x0e node feature (the encoder consumes `graph.x`).

    ``r_max`` / ``dropout_rate`` default to the module constants; the resampling loader
    overrides them (and advances ``key``) to draw a fresh graph each training step."""

    radius_graph = get_graphs_from_vertices(
            vertices, masks=mask, r_max=r_max, dropout_rate=dropout_rate, noise_std=0.0,
            key=key, sampling_mode="uniform")

    if use_supernodes:
        super_graph = build_super_graph(vertices, mask, radius_graph,
                                   num_samples = n_supernodes,
                                    r_max = r_supergraph,
                                    mode = "fps")
    else:
        super_graph = None
    radius_graph.x = torch.ones(radius_graph.num_nodes, 1)
    if super_graph is not None:
     super_graph.x = torch.ones(super_graph.num_nodes, 1)

    return radius_graph, super_graph

def save_graph_vtp(graph, 
                   output_dir = "OUTPUT_DIR", 
                   is_supernodes = False):
    for sample_idx in range(int(graph.batch.max()) + 1):
        if sample_idx <= 10:
            if is_supernodes:
                pos, edges, node_field = get_bipartite_graph(graph, sample_idx)
            else:
                pos, edges = get_individual_graph(graph, sample_idx)
            save_path = os.path.join(output_dir, f"init_graph_{sample_idx}.vtp")
            vtp = create_polydata_w_lines(pos, edges)
            if is_supernodes:
                vtp = add_point_field(vtp, field_data=node_field,  field_name="super_node")
            save_vtp(vtp, save_path)


