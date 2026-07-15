
import os
import glob
import torch
import numpy as np

from src.vtk.io import load_vtp, save_vtp
from src.vtk.create import create_polydata_w_lines
from src.vtk.extract import extract_vtp_points_cells, extract_vtp_point_fields
from src.vtk.fields import add_point_field
from src.graphs.graphs import (
    get_graphs_from_vertices, 
    build_super_graph,
    get_individual_graph,
    get_bipartite_graph)
from src.transforms.padding import pad_vertex_list
import random

def load_dataset(data_path="DATA_ROOT", parts=["mouth", "nose"], shuffle=True, verbose=True, load_fields=False):
    """Load face-part shapes and optional point fields; fall back to tests/data so the script always runs."""
    samples = []
    for part in parts if parts is not None else [None]:
        glob_path = os.path.join(data_path, part, "*.vtp") if part is not None else os.path.join(data_path, "*.vtp")
        for file in glob.glob(glob_path):
            if verbose:
                print("Loading: ", file)
            polydata = load_vtp(file)
            verts, _ = extract_vtp_points_cells(polydata)
            areas = None
            normals = None
            if load_fields:
                fields = extract_vtp_point_fields(polydata, ["area", "normal"])
                areas = fields.get("area")
                normals = fields.get("normal")
                if areas is None and verbose:
                    print(f"Warning: missing area field in {file}; using ones.")
                if normals is None and verbose:
                    print(f"Warning: missing normal field in {file}; using zeros.")
            samples.append((verts, areas, normals))

    if not samples:
        print("Dataset not found - falling back to tests/data shapes.")

    if shuffle:
        if verbose:
            print("shuffling samples")
        random.shuffle(samples)

    vertices = [s[0] for s in samples]
    padded, mask = pad_vertex_list(vertices)

    if not load_fields:
        return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

    num_shapes, max_vertices, _ = padded.shape
    padded_areas = np.ones((num_shapes, max_vertices), dtype=np.float32)
    padded_normals = np.zeros((num_shapes, max_vertices, 3), dtype=np.float32)
    for i, (_, areas, normals) in enumerate(samples):
        n = vertices[i].shape[0]
        if areas is not None:
            padded_areas[i, :n] = np.asarray(areas, dtype=np.float32)
        if normals is not None:
            padded_normals[i, :n, :] = np.asarray(normals, dtype=np.float32)

    return (
        torch.tensor(padded, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(padded_areas, dtype=torch.float32),
        torch.tensor(padded_normals, dtype=torch.float32),
    )
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

def split_dataset(*tensors, val_fraction=0.2, shuffle=True, seed=None):
    """Split shape-indexed tensors (as returned by ``load_dataset``) into train/val.

    Every passed tensor is indexed by shape on dim 0 (vertices, mask, areas, normals,
    ...) and is split with the SAME permutation, so each shape's rows stay aligned
    across all arrays. Pass the arrays in whatever order ``load_dataset`` returned them:

        train, val = split_dataset(*load_dataset(...), val_fraction=0.2, seed=0)
        (verts, mask, areas, normals), (v_verts, v_mask, v_areas, v_normals) = train, val

    ``val_fraction`` is the share of shapes held out for validation (clamped so both
    splits keep at least one shape). ``shuffle`` randomizes which shapes land in val;
    pass ``seed`` for a reproducible split. Returns ``(train_tensors, val_tensors)``,
    each a tuple in the input order."""
    if not tensors:
        raise ValueError("split_dataset needs at least one tensor to split.")

    n = tensors[0].shape[0]
    for i, t in enumerate(tensors):
        if t.shape[0] != n:
            raise ValueError(
                f"all tensors must share dim-0 length; tensor 0 has {n} shapes but "
                f"tensor {i} has {t.shape[0]}.")
    if n < 2:
        raise ValueError(f"need at least 2 shapes to split, got {n}.")

    n_val = int(round(val_fraction * n))
    n_val = max(1, min(n_val, n - 1))   # keep both splits non-empty

    if shuffle:
        g = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        perm = torch.randperm(n, generator=g)
    else:
        perm = torch.arange(n)

    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = tuple(t[train_idx] for t in tensors)
    val = tuple(t[val_idx] for t in tensors)
    return train, val

def build_training_graph(vertices, mask,
                         key, 
                         r_max=0.1, 
                         dropout_rate=0.8, 
                         n_supernodes = 10, 
                         r_supergraph = 0.2,
                         use_supernodes= False,
                         sampling_mode_graph = "uniform",
                         sampling_mode_supernodes =  "uniform",
                         features=None,
                         areas=None,
                         normals=None):
    """Build the graph fed to the encoder, per the USE_SUPERNODES toggle, and attach
    a constant 1x0e node feature (the encoder consumes `graph.x`).

    ``r_max`` / ``dropout_rate`` default to the module constants; the resampling loader
    overrides them (and advances ``key``) to draw a fresh graph each training step."""

    radius_graph = get_graphs_from_vertices(
            vertices, masks=mask, r_max=r_max, dropout_rate=dropout_rate, noise_std=0.0,
            key=key, sampling_mode=sampling_mode_graph,
            features=features, areas=areas, normals=normals)

    if use_supernodes:
        super_graph = build_super_graph(vertices, mask, radius_graph,
                                   num_samples = n_supernodes,
                                    r_max = r_supergraph,
                                    mode = sampling_mode_supernodes)
    else:
        super_graph = None

    if not hasattr(radius_graph, 'x') or radius_graph.x is None:
        radius_graph.x = torch.ones(radius_graph.num_nodes, 1)
    if super_graph is not None and (not hasattr(super_graph, 'x') or super_graph.x is None):
        super_graph.x = torch.ones(super_graph.num_nodes, 1)

    return radius_graph, super_graph

def save_graph_vtp(graph, 
                   output_dir = "OUTPUT_DIR", 
                   is_supernodes = False):
    for sample_idx in range(int(graph.batch.max()) + 1):
        if sample_idx <= 10:
            if is_supernodes:
                pos, edges, node_field = get_bipartite_graph(graph, sample_idx)
                vtp = create_polydata_w_lines(pos, edges)
                vtp = add_point_field(vtp, field_data=node_field, field_name="super_node")

                if hasattr(graph, 'area') and graph.area is not None:
                    src_mask = (graph.source_batch == sample_idx)
                    tgt_mask = (graph.batch == sample_idx)
                    src_area = graph.area[src_mask] if src_mask.any() else torch.tensor([])
                    tgt_area = graph.area[tgt_mask] if tgt_mask.any() else torch.tensor([])
                    area_field = torch.cat([src_area, tgt_area], dim=0).detach().cpu().numpy()
                    vtp = add_point_field(vtp, area_field.astype(np.float32), field_name="area")

                if hasattr(graph, 'normal') and graph.normal is not None:
                    src_mask = (graph.source_batch == sample_idx)
                    tgt_mask = (graph.batch == sample_idx)
                    src_norm = graph.normal[src_mask] if src_mask.any() else torch.empty((0, 3), device=graph.normal.device)
                    tgt_norm = torch.zeros((tgt_mask.sum().item(), 3), dtype=graph.normal.dtype, device=graph.normal.device)
                    normal_field = torch.cat([src_norm, tgt_norm], dim=0).detach().cpu().numpy()
                    vtp = add_point_field(vtp, normal_field.astype(np.float32), field_name="normal")

            else:
                pos, edges = get_individual_graph(graph, sample_idx)
                vtp = create_polydata_w_lines(pos, edges)
                if hasattr(graph, 'area') and graph.area is not None:
                    node_mask = (graph.batch == sample_idx)
                    area_field = graph.area[node_mask].detach().cpu().numpy()
                    vtp = add_point_field(vtp, area_field.astype(np.float32), field_name="area")
                if hasattr(graph, 'normal') and graph.normal is not None:
                    node_mask = (graph.batch == sample_idx)
                    normal_field = graph.normal[node_mask].detach().cpu().numpy()
                    vtp = add_point_field(vtp, normal_field.astype(np.float32), field_name="normal")

            save_path = os.path.join(output_dir, f"init_graph_{sample_idx}.vtp")
            save_vtp(vtp, save_path)


