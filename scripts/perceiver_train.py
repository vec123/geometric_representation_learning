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
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.logger.headless import enable_headless
from src.learning.callbacks.metrics import MetricsRecorder, MetricsPlotter
from src.learning.callbacks.checkpointing import CheckpointWriter
from src.learning.callbacks.visualization import GeometryVisualizer
from src.learning.callbacks.validation import ValidationRunner
from src.learning.loader.loaders import OneBatchLoader, ResamplingGraphLoader

from src.paths import get_project_root
from src.learning.helpers import load_dataset, build_training_graph, save_graph_vtp


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
USE_SUPERNODES = True          # toggle: supernode subset (True) vs full/decimated graph (False)
N_SUPERNODES   = 10           # n_s, used when USE_SUPERNODES is True
SAMPLING_MODE  = "fps"         # 'fps' | 'uniform' | 'gaussian'
DROPOUT_RATE   = 0.8          # uniform node dropout to reduce data-sample size uniformly
NOISE_STD      = 0.00          #Optional: noise addition
R_MAX         = 0.25         #radius for graph
R_SUPERGRPAH = 0.6

# Rebuild the encoder graph from geometry each step (fit geometry, not a fixed graph).
# False -> build one graph up front and reuse it every step (prebuilt path).
RESAMPLE_GRAPH   = False
# When resampling, each may be a fixed float or a (low, high) range sampled per step,
# e.g. RESAMPLE_R_MAX = (0.2, 0.3) / RESAMPLE_DROPOUT = (0.7, 0.9).
RESAMPLE_R_MAX   = R_MAX
RESAMPLE_DROPOUT = DROPOUT_RATE

LATENT_DIM     = 4
NUM_SAMPLES    = 256           # decoder output points (perfect square for the folding grid)
LEARNING_RATE  = 1e-3
NUM_STEPS      = 101
LOG_EVERY      = 1
SAVE_EVERY     = 100
VAL_EVERY      = 20            # run + save validation every N steps

Project_ROOT = get_project_root()
SHAPE_DATA_ROOT = os.path.join(
    Project_ROOT, 
    "Dataset", "vtp_samples", "Dataset_faceparts_normalized_small")

VAL_SHAPE_DATA_ROOT =  os.path.join(
    Project_ROOT, 
    "Dataset", "vtp_samples", "val_Dataset_faceparts_normalized_small")

OUTPUT_DIR = os.path.join(Project_ROOT, "training_logs_perceiver")

# Headless/HPC toggle: None -> auto (mirror output to a log file when stdout is not a TTY,
# e.g. under SLURM/nohup); True/False to force. In remote mode stdout+stderr are teed to a
# timestamped, flushed log under OUTPUT_DIR for easy inspection (tail -f).
REMOTE = None



# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enable_headless(OUTPUT_DIR, remote=REMOTE, name="perceiver_train")
    key = torch.Generator(device="cpu")
    key.manual_seed(0)

    shape_vertices, shape_mask, shape_areas, shape_normals = load_dataset(
        data_path=SHAPE_DATA_ROOT,
        parts=["mouth", "nose"],
        load_fields=True,
    )
  
  
    graph, supergraph = build_training_graph(shape_vertices, 
                                 shape_mask,
                                key,
                                r_max = R_MAX, 
                                 r_supergraph= R_SUPERGRPAH,
                                dropout_rate = DROPOUT_RATE, 
                                n_supernodes = N_SUPERNODES, 
                                use_supernodes= USE_SUPERNODES,
                                areas=shape_areas,
                                normals=shape_normals)
    
    mode = f"supernodes(n_s={N_SUPERNODES}, {SAMPLING_MODE})" if USE_SUPERNODES else f"full(dropout={DROPOUT_RATE})"
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
        "intermediate_irreps": "32x0e + 32x1o",
        "output_irreps": f"1028x0e + 2x1o",   # latent_dim scalars + 2 vectors (rotation frame)
    }


    encoder = GroupPerceiverEncoder(
        irreps_cfg=layer_cfg, 
        n_latent=4,
        d_shared=LATENT_DIM,
        self_attn_heads=1,
        cross_attn_heads=1,
        n_self_layers=1,
        widening_factor=1,
        reduce_stages=[1],
        reduce_heads=1,
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
            r_max=RESAMPLE_R_MAX,
            r_supergraph=R_SUPERGRPAH,
            dropout_rate=RESAMPLE_DROPOUT)
        print(f"loader: resampling graph each step (r_max={RESAMPLE_R_MAX}, dropout={RESAMPLE_DROPOUT})")
    else:
        loader = OneBatchLoader((graph, supergraph, shape_vertices, shape_mask))
        print("loader: prebuilt graph reused every step")


    val_shape_vertices, val_shape_mask, val_shape_areas, val_shape_normals = load_dataset(
        data_path=VAL_SHAPE_DATA_ROOT,
        parts=["mouth", "nose"],
        load_fields=True,
    )

    # Build the validation encoder graph from the VALIDATION geometry (not the training
    # verts) so the graph and its reconstruction target describe the same shapes.
    val_graph, val_supergraph = build_training_graph(val_shape_vertices,
                                 val_shape_mask,
                                key,
                                r_max = R_MAX,
                                 r_supergraph= R_SUPERGRPAH,
                                dropout_rate = DROPOUT_RATE,
                                n_supernodes = N_SUPERNODES,
                                use_supernodes= USE_SUPERNODES,
                                areas=val_shape_areas,
                                normals=val_shape_normals)

    val_loader = OneBatchLoader((val_graph, val_supergraph, val_shape_vertices, val_shape_mask))

    # recon + 0.1*kl was TrainingStepper's implicit default before T10 moved loss
    # weights onto the composer; stated explicitly here so this script's behavior
    # is unchanged (and visible) rather than inherited from a constructor default.
    stepper = TrainingStepper(
        encoder, decoder, learning_rate=LEARNING_RATE,
        composer=LossComposer([LossTerm("recon", 1.0), LossTerm("kl", 0.1)]))
    # Each callback carries its own cadence (T12); the orchestrator just loops.
    recorder = MetricsRecorder(every_n_steps=LOG_EVERY)
    callbacks = [
        recorder,
        MetricsPlotter(recorder, every_n_steps=VAL_EVERY),
        CheckpointWriter(every_n_steps=SAVE_EVERY),
        GeometryVisualizer(every_n_steps=SAVE_EVERY),
        ValidationRunner(val_loader, every_n_steps=VAL_EVERY),
    ]
    trainer = TrainingOrchestrator(stepper=stepper, dataloader=loader,
                                   callbacks=callbacks, log_dir=OUTPUT_DIR)

    print(f"----------training on device: {stepper.device}----------")
    trainer.run(num_steps=NUM_STEPS)
    print(f"done. outputs in {OUTPUT_DIR}")
    

if __name__ == "__main__":
    main()
