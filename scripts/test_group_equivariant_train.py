
import os
import glob
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from src.vtk.io import load_vtp, save_vtp
from src.vtk.create import(
    create_polydata, 
    create_polydata_w_lines
)
from src.vtk.extract import extract_vtp_points_cells
from src.graphs.graphs import (
    get_graphs_from_vertices, 
    get_vertices_and_edges,
    get_individual_graph
    )

from src.learning.models.folding_decoder import FoldingDecoder
from src.learning.models.group_encoder import GroupEncoder
from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator  
from src.learning.logger.train_logs import TrainingLogger  
from src.transforms.padding import pad_vertex_list

from config.root import get_project_root

Project_ROOT = get_project_root()
SHAPE_DATA_ROOT = os.path.join(Project_ROOT, "Dataset", "vtp_samples", "Dataset_faceparts_normalized_small")
OUTPUT_DIR = os.path.join(Project_ROOT, "training_logs")

vtp_path = os.path.join(Project_ROOT, "tests", "data", "sample_01.vtp")

vertices = []
for part in ["mouth", "nose"]:
    DATASET_1 = os.path.join(SHAPE_DATA_ROOT, part, "*.vtp")
    print("looking in: ", DATASET_1)
    vtp_files = glob.glob(DATASET_1)
    print("getting vtp_files: ", vtp_files)
    for file in vtp_files:
        vtp = load_vtp(file)
        verts, _ = extract_vtp_points_cells(vtp)
        vertices.append(verts)
print("vertices: ", [v.shape for v in vertices])
shape_vertices, shape_mask = pad_vertex_list(vertices)
shape_mask = torch.tensor(shape_mask, dtype=torch.bool)


sampling_mode = "uniform"
num_samples = 20
key = torch.Generator(device='cpu') 
key.manual_seed(0)
graphs = get_graphs_from_vertices(shape_vertices,
                                          masks=shape_mask, r_max=0.1, 
                                          dropout_rate=0.8, 
                                          noise_std=0.05,
                                          key = key,
                                          sampling_mode=sampling_mode,
                                          num_samples = num_samples)
print("graphs.batch: ", torch.unique(graphs.batch))
for i in range(len(torch.unique(graphs.batch))):
    print(f"Processing graph {i}")
    V,E = get_individual_graph(graphs, index=i)
    graph_vtp = create_polydata_w_lines(V, E)
    output_path = os.path.join(OUTPUT_DIR, f"init_graph_{i}.vtp")
    save_vtp(graph_vtp, output_path)


layer_cfg = {
    "input_irreps": "1x0e",
    "intermediate_irreps": "1x0e + 1x1o",
    "output_irreps": "1x0e + 1x1o",
    }

latent_dim = 5
encoder = GroupEncoder(latent_dim=latent_dim, irreps_cfg=layer_cfg)
decoder = FoldingDecoder(num_samples=256, latent_dim=latent_dim, n_freqs=4, verbose=True)
loader = DataLoader([graphs], batch_size=1, shuffle=True)

step = TrainingStepper(encoder, decoder, learning_rate=1e-5)
logger = TrainingLogger(log_dir=OUTPUT_DIR)
trainer = TrainingOrchestrator(stepper=step, logger=logger, dataloader=loader)

trainer.run(num_steps=5, log_every=1, save_every=2)