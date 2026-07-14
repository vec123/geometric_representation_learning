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
from src.learning.logger.headless import enable_headless
from src.learning.loader.loaders import OneBatchLoader, ResamplingGraphLoader
from config.root import get_project_root
from src.learning.helpers import load_dataset, build_training_graph, save_graph_vtp

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
USE_SUPERNODES = True          # toggle: supernode subset (True) vs full/decimated graph (False)
DROPOUT_RATE   = 0.9          # uniform node dropout to reduce data-sample size uniformly
N_SUPERNODES   = 15           # n_s, used when USE_SUPERNODES is True
DORPOUT_SAMPLING_MODE  = "uniform"         # 'fps' | 'uniform' | 'gaussian'
SUPERNODE_SAMPLING_MODE  = "uniform"         # 'fps' | 'uniform' | 'gaussian'
NOISE_STD      = 0.00          #Optional: noise addition
R_MAX         = 0.25         #radius for graph
R_SUPERGRPAH = 0.6
# Rebuild the encoder graph from geometry each step (fit geometry, not a fixed graph).
# False -> build one graph up front and reuse it every step (prebuilt path).
RESAMPLE_GRAPH   = True
# When resampling, each may be a fixed float or a (low, high) range sampled per step,
# e.g. RESAMPLE_R_MAX = (0.2, 0.3) / RESAMPLE_DROPOUT = (0.7, 0.9).
RESAMPLE_R_MAX   = R_MAX
RESAMPLE_DROPOUT = DROPOUT_RATE

# Contrastive objective: "same shape, different vertex sampling -> same encoding".
# When True, each step draws TWO views of the same shapes and pulls their latents
# together (this needs resampling, so it selects the two-view loader below). False ->
# ordinary single-view reconstruction only.
CONTRASTIVE            = True
CONTRASTIVE_WEIGHT     = 0.1   # weight of the alignment loss; raise if views don't align, lower if recon stalls
CONTRASTIVE_VAR_WEIGHT = 1.0   # variance-hinge weight (anti-collapse); set 0 for pure alignment

LATENT_DIM     = 5
NUM_SAMPLES    = 256           # decoder output points (perfect square for the folding grid)
LEARNING_RATE  = 1e-3
NUM_STEPS      = 3001
LOG_EVERY      = 1
SAVE_EVERY     = 100
VAL_EVERY      = 100           # run + save validation every N steps

Project_ROOT = get_project_root()
SHAPE_DATA_ROOT = os.path.join(Project_ROOT, 
                               "Dataset", "vtp_samples",
                                 "Dataset_faceparts_normalized_small")
VAL_SHAPE_DATA_ROOT =  os.path.join(
    Project_ROOT,
    "Dataset", "vtp_samples", "val_Dataset_faceparts_normalized")

OUTPUT_DIR = os.path.join(Project_ROOT,
                          "training_log_contrastive_vae")

# Headless/HPC toggle: None -> auto (mirror output to a log file when stdout is not a TTY,
# e.g. under SLURM/nohup); True/False to force. In remote mode stdout+stderr are teed to a
# timestamped, flushed log under OUTPUT_DIR for easy inspection (tail -f).
REMOTE = None



# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enable_headless(OUTPUT_DIR, remote=REMOTE, name="equivariant_gnn_train")
    key = torch.Generator(device="cpu")
    key.manual_seed(0)

    parts = ["bridge", "nose"]
    shape_vertices, shape_mask = load_dataset(data_path=SHAPE_DATA_ROOT,
                                            parts = parts )
  
    graph, supergraph = build_training_graph(shape_vertices, 
                                shape_mask,
                                key,
                                r_max = R_MAX,       
                                dropout_rate = DROPOUT_RATE, 
                                n_supernodes = N_SUPERNODES, 
                                 r_supergraph= R_SUPERGRPAH,
                                use_supernodes= USE_SUPERNODES,
                                sampling_mode_graph=DORPOUT_SAMPLING_MODE,
                                sampling_mode_supernodes=SUPERNODE_SAMPLING_MODE)

    mode = f"supernodes(n_s={N_SUPERNODES}, {SUPERNODE_SAMPLING_MODE})" if USE_SUPERNODES else f"full(dropout={DROPOUT_RATE})"
    print(f"graph mode: {mode} | nodes={graph.num_nodes} | shapes={int(graph.batch.max()) + 1}")
    save_graph_vtp(graph,
                   output_dir = os.path.join(OUTPUT_DIR, "init_graphs"),
                   is_supernodes = False )
    if supergraph is not None:
        save_graph_vtp(supergraph,
                   output_dir = os.path.join(OUTPUT_DIR, "init_supernodes"),
                   is_supernodes = True )
   
    layer_cfg = {
        "input_irreps": "1x0e",
        "intermediate_irreps": "32x0e + 32x0o + 16x1o + 16x1e+ 4x2o+ 4x2e",
        "output_irreps": f"{LATENT_DIM}x0e + 2x1o",   # latent_dim scalars + 2 vectors (rotation frame)
    }

        
    encoder = GroupEncoder(
        latent_dim=LATENT_DIM, 
        irreps_cfg=layer_cfg, 
        sh_lmax = 5,
        readout = "mean",
        readout_heads = 1,
        verbose=False)
    
    decoder = FoldingDecoder(
        num_samples=NUM_SAMPLES,
          latent_dim=LATENT_DIM, 
          n_freqs=4, 
          verbose=False)

    # Reconstruction target: the full padded shapes. The graph fed to the encoder is
    # either resampled from geometry each step, or the single prebuilt graph above.
    if CONTRASTIVE:
        loader = ResamplingGraphLoader(
            shape_vertices, shape_mask, build_training_graph, key=key,
            r_max=RESAMPLE_R_MAX,
            r_supergraph=R_SUPERGRPAH,
            dropout_rate=RESAMPLE_DROPOUT,
            use_supernodes=USE_SUPERNODES,
            two_view=True,
            n_supernodes=N_SUPERNODES,
            sampling_mode_graph=DORPOUT_SAMPLING_MODE,
            sampling_mode_supernodes=SUPERNODE_SAMPLING_MODE)
        print("loader: two-view contrastive (two fresh samplings of the same shapes per step)")
    elif RESAMPLE_GRAPH:
        loader = ResamplingGraphLoader(
            shape_vertices, shape_mask, build_training_graph, key=key,
            r_max=RESAMPLE_R_MAX,
            r_supergraph=R_SUPERGRPAH,
            dropout_rate=RESAMPLE_DROPOUT)
        print(f"loader: resampling graph each step (r_max={RESAMPLE_R_MAX}, dropout={RESAMPLE_DROPOUT})")
    else:
        loader = OneBatchLoader((graph, supergraph, shape_vertices, shape_mask))
        print("loader: prebuilt graph reused every step")

    
    val_shape_vertices, val_shape_mask = load_dataset(data_path=VAL_SHAPE_DATA_ROOT,
                                            parts = parts)

    # Build the validation encoder graph from the VALIDATION geometry (not the training
    # verts) so the graph and its reconstruction target describe the same shapes.
    val_graph, val_supergraph = build_training_graph(val_shape_vertices,
                                 val_shape_mask,
                                key,
                                r_max = R_MAX,
                                 r_supergraph= R_SUPERGRPAH,
                                dropout_rate = DROPOUT_RATE,
                                n_supernodes = N_SUPERNODES,
                                use_supernodes= USE_SUPERNODES)

    val_loader = OneBatchLoader((val_graph, val_supergraph, val_shape_vertices, val_shape_mask))


    stepper = TrainingStepper(encoder, decoder,
                               learning_rate=LEARNING_RATE,
                               kl_weight=0.0000,
                               contrastive_weight=CONTRASTIVE_WEIGHT if CONTRASTIVE else 0.0,
                               contrastive_var_weight=CONTRASTIVE_VAR_WEIGHT)
    logger = TrainingLogger(log_dir = OUTPUT_DIR)
    trainer = TrainingOrchestrator(stepper=stepper, logger=logger, 
                                   dataloader=loader, val_loader=val_loader)

    print(f"----------training on device: {stepper.device}----------")
    trainer.run(num_steps=NUM_STEPS, log_every=LOG_EVERY,
                 save_every=SAVE_EVERY, val_every=VAL_EVERY)
    print(f"done. outputs in {OUTPUT_DIR}")
    

if __name__ == "__main__":
    main()
