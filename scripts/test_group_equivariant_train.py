"""End-to-end equivariant training on a real face-parts dataset.

Encodes each shape's (sampled) point graph with the equivariant GroupEncoder, decodes
a canonical point cloud with the FoldingDecoder, and logs predictions as VTPs.

Supernode toggle
----------------
USE_SUPERNODES switches how the graph fed to the encoder is built:
  * True  -> sample N_SUPERNODES nodes (fps/uniform) and run the GNN on that subset.
  * False -> use the full graph, optionally decimated by a uniform dropout.
NOTE: this samples the node SET the (homogeneous) GNN runs on. True bipartite
aggregation (supernodes gathering the full-graph neighbourhood via build_bipartite_graph)
needs the encoder's `readout` support, which is not wired yet.
"""

import os
import glob

import torch

from src.vtk.io import load_vtp, save_vtp
from src.vtk.create import create_polydata, create_polydata_w_lines
from src.vtk.extract import extract_vtp_points_cells
from src.vtk.fields import add_point_field
from src.graphs.graphs import (
    get_graphs_from_vertices, 
    build_super_graph,
    get_individual_graph,
    get_bipartite_graph)

from src.learning.models.folding_decoder import FoldingDecoder
from src.learning.models.group_encoder import GroupEncoder
from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
from src.learning.logger.train_logs import TrainingLogger
from src.transforms.padding import pad_vertex_list
from config.root import get_project_root


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
USE_SUPERNODES = False          # toggle: supernode subset (True) vs full/decimated graph (False)
N_SUPERNODES   = 50           # n_s, used when USE_SUPERNODES is True
SAMPLING_MODE  = "fps"         # 'fps' | 'uniform' | 'gaussian'
DROPOUT_RATE   = 0.8          # uniform node dropout to reduce data-sample size uniformly
NOISE_STD      = 0.00          #Optional: noise addition
R_MAX         = 0.25         #radius for graph

# Rebuild the encoder graph from geometry each step (fit geometry, not a fixed graph).
# False -> build one graph up front and reuse it every step (prebuilt path).
RESAMPLE_GRAPH   = False
# When resampling, each may be a fixed float or a (low, high) range sampled per step,
# e.g. RESAMPLE_R_MAX = (0.2, 0.3) / RESAMPLE_DROPOUT = (0.7, 0.9).
RESAMPLE_R_MAX   = R_MAX
RESAMPLE_DROPOUT = DROPOUT_RATE

LATENT_DIM     = 5
NUM_SAMPLES    = 256           # decoder output points (perfect square for the folding grid)
LEARNING_RATE  = 1e-4
NUM_STEPS      = 1000
LOG_EVERY      = 1
SAVE_EVERY     = 100

Project_ROOT = get_project_root()
SHAPE_DATA_ROOT = os.path.join(Project_ROOT, "Dataset", "vtp_samples", "Dataset_faceparts_normalized_small")
OUTPUT_DIR = os.path.join(Project_ROOT, "training_logs_fixed")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataset():
    """Load face-part shapes; fall back to tests/data so the script always runs."""
    vertices = []
    for part in ["mouth", "nose"]:
        for file in glob.glob(os.path.join(SHAPE_DATA_ROOT, part, "*.vtp")):
            verts, _ = extract_vtp_points_cells(load_vtp(file))
            vertices.append(verts)

    if not vertices:
        print("Dataset not found - falling back to tests/data shapes.")
        for name in ["sample_01", "nose_0"]:
            verts, _ = extract_vtp_points_cells(
                load_vtp(os.path.join(Project_ROOT, "tests", "data", f"{name}.vtp")))
            vertices.append(verts)

    print("loaded shapes:", [v.shape for v in vertices])
    padded, mask = pad_vertex_list(vertices)
    return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def build_training_graph(vertices, mask, key, r_max=R_MAX, dropout_rate=DROPOUT_RATE):
    """Build the graph fed to the encoder, per the USE_SUPERNODES toggle, and attach
    a constant 1x0e node feature (the encoder consumes `graph.x`).

    ``r_max`` / ``dropout_rate`` default to the module constants; the resampling loader
    overrides them (and advances ``key``) to draw a fresh graph each training step."""

    full_graph = get_graphs_from_vertices(
            vertices, masks=mask, r_max=r_max, dropout_rate=dropout_rate, noise_std=NOISE_STD,
            key=key, sampling_mode="uniform")

    if USE_SUPERNODES:
        graph = build_super_graph(vertices, mask, full_graph,
                                   num_samples = N_SUPERNODES,
                                    r_max = r_max,
                                    mode = "uniform")
    else:
        graph = full_graph
    graph.x = torch.ones(graph.num_nodes, 1)

    return graph

def save_graph_vtp(graph, output_dir = OUTPUT_DIR, is_supernodes = False):
    for sample_idx in range(int(graph.batch.max()) + 1):
        if is_supernodes:
            pos, edges, node_field = get_bipartite_graph(graph, sample_idx)
        else:
            pos, edges = get_individual_graph(graph, sample_idx)
        save_path = os.path.join(output_dir, f"init_graph_{sample_idx}.vtp")
        vtp = create_polydata_w_lines(pos, edges)
        if is_supernodes:
            vtp = add_point_field(vtp, field_data=node_field,  field_name="super_node")
        save_vtp(vtp, save_path)

class OneBatchLoader:
    """Yields the same prebuilt (graph, true_verts, mask) batch each step."""
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


class ResamplingGraphLoader:
    """Rebuilds the encoder graph from raw geometry each step with a fresh key.

    The reconstruction target (``true_verts``, ``mask``) is fixed, but every step draws
    a new dropout mask / node sampling (and optionally a jittered radius / dropout rate),
    so the encoder fits the underlying geometry rather than one frozen edge set. The
    ``key`` generator's state advances as it is consumed, giving a different graph each
    step while staying reproducible from the seed.

    ``r_max`` and ``dropout_rate`` may each be a fixed float or a ``(low, high)`` range
    that is sampled uniformly per step.
    """
    def __init__(self, vertices, mask, build_fn, key=None,
                 r_max=R_MAX, dropout_rate=DROPOUT_RATE):
        self.vertices = vertices
        self.mask = mask
        self.build_fn = build_fn
        self.key = key
        self.r_max = r_max
        self.dropout_rate = dropout_rate

    def _sample(self, value):
        """Return value as-is if scalar, else draw uniformly from a (low, high) range."""
        if isinstance(value, (tuple, list)):
            low, high = value
            u = torch.rand((), generator=self.key).item()
            return low + u * (high - low)
        return value

    def __iter__(self):
        while True:
            graph = self.build_fn(
                self.vertices, self.mask, key=self.key,
                r_max=self._sample(self.r_max),
                dropout_rate=self._sample(self.dropout_rate),
            )
            yield (graph, self.vertices, self.mask)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    key = torch.Generator(device="cpu")
    key.manual_seed(0)

    shape_vertices, shape_mask = load_dataset()
  
    graph = build_training_graph(shape_vertices, shape_mask, key)
    mode = f"supernodes(n_s={N_SUPERNODES}, {SAMPLING_MODE})" if USE_SUPERNODES else f"full(dropout={DROPOUT_RATE})"
    print(f"graph mode: {mode} | nodes={graph.num_nodes} | shapes={int(graph.batch.max()) + 1}")
    save_graph_vtp(graph,output_dir =OUTPUT_DIR, is_supernodes = USE_SUPERNODES  )

   
    layer_cfg = {
        "input_irreps": "1x0e",
        "intermediate_irreps": "32x0e + 32x1o",
        "output_irreps": f"{LATENT_DIM}x0e + 2x1o",   # latent_dim scalars + 2 vectors (rotation frame)
    }


    encoder = GroupEncoder(
        latent_dim=LATENT_DIM, 
        irreps_cfg=layer_cfg, 
        sh_lmax = 1,
        verbose=False)
    
    decoder = FoldingDecoder(
        num_samples=NUM_SAMPLES,
          latent_dim=LATENT_DIM, 
          n_freqs=4, 
          verbose=False)

    # Reconstruction target: the full padded shapes. The graph fed to the encoder is
    # either resampled from geometry each step, or the single prebuilt graph above.
    if RESAMPLE_GRAPH:
        loader = ResamplingGraphLoader(
            shape_vertices, shape_mask, build_training_graph, key=key,
            r_max=RESAMPLE_R_MAX, dropout_rate=RESAMPLE_DROPOUT)
        print(f"loader: resampling graph each step (r_max={RESAMPLE_R_MAX}, dropout={RESAMPLE_DROPOUT})")
    else:
        loader = OneBatchLoader((graph, shape_vertices, shape_mask))
        print("loader: prebuilt graph reused every step")

    stepper = TrainingStepper(encoder, decoder, learning_rate=LEARNING_RATE)
    logger = TrainingLogger(log_dir = OUTPUT_DIR)
    trainer = TrainingOrchestrator(stepper=stepper, logger=logger, dataloader=loader)

    print(f"----------training on device: {stepper.device}----------")
    trainer.run(num_steps=NUM_STEPS, log_every=LOG_EVERY, save_every=SAVE_EVERY)
    print(f"done. outputs in {OUTPUT_DIR}")
    

if __name__ == "__main__":
    main()
