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
from src.learning.models.group_perceiver_encoder import GroupPerceiverEncoder
from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
from src.learning.logger.train_logs import TrainingLogger
from src.learning.loader.loaders import OneBatchLoader, ResamplingGraphLoader

from config.root import get_project_root
from src.learning.helpers import load_dataset, build_training_graph, save_graph_vtp


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

LATENT_DIM     = 32
NUM_SAMPLES    = 256           # decoder output points (perfect square for the folding grid)
LEARNING_RATE  = 1e-4
NUM_STEPS      = 1000
LOG_EVERY      = 1
SAVE_EVERY     = 100

Project_ROOT = get_project_root()
SHAPE_DATA_ROOT = os.path.join(Project_ROOT, "Dataset", "vtp_samples", "Dataset_faceparts_normalized_small")
OUTPUT_DIR = os.path.join(Project_ROOT, "training_logs_fixed")




# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    key = torch.Generator(device="cpu")
    key.manual_seed(0)

    shape_vertices, shape_mask = load_dataset(data_path=SHAPE_DATA_ROOT,
                                            parts = ["mouth", "nose"] )
  
  
    graph = build_training_graph(shape_vertices, 
                                 shape_mask,
                                key,
                                r_max=R_MAX, 
                                dropout_rate=DROPOUT_RATE, 
                                n_supernodes = N_SUPERNODES, 
                                use_supernodes= USE_SUPERNODES)
    
    mode = f"supernodes(n_s={N_SUPERNODES}, {SAMPLING_MODE})" if USE_SUPERNODES else f"full(dropout={DROPOUT_RATE})"
    print(f"graph mode: {mode} | nodes={graph.num_nodes} | shapes={int(graph.batch.max()) + 1}")
    save_graph_vtp(graph,
                   output_dir =OUTPUT_DIR, 
                   is_supernodes = USE_SUPERNODES )


   
    layer_cfg = {
        "input_irreps": "1x0e",
        "intermediate_irreps": "32x0e + 32x1o",
        "output_irreps": f"1028x0e + 2x1o",   # latent_dim scalars + 2 vectors (rotation frame)
    }


    encoder = GroupPerceiverEncoder(
        irreps_cfg=layer_cfg, 
        n_latent=8,
        d_shared=LATENT_DIM,
        self_attn_heads=2,
        cross_attn_heads=2,
        n_self_layers=1,
        widening_factor=2,
        reduce_stages=[8,1],
        reduce_heads=2,
        vae_mode="per_token",
        sh_lmax=1,
        interaction_sh_lmax=1,
        perceiver_weight_sharing=True,
        n_perceiver_layers=2,
        verbose=False,)
    
    decoder = FoldingDecoder(
        num_samples=NUM_SAMPLES,
        latent_dim=LATENT_DIM, 
        n_freqs=4, 
        verbose=False)

    # Reconstruction target: the full padded shapes are used to define the real geometry. 
    # The graph fed to the encoder is either
    # rebuilt from geometry each step,
    # or the single prebuilt graph above.

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
