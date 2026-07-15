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
import random
import numpy as np

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
from src.learning.helpers import (
    load_dataset, 
    split_dataset,
    build_training_graph, 
    save_graph_vtp)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
USE_SUPERNODES = True          # toggle: supernode subset (True) vs full/decimated graph (False)
DROPOUT_RATE   = 0.9          # uniform node dropout to reduce data-sample size uniformly
N_SUPERNODES   = 15           # n_s, used when USE_SUPERNODES is True
DROPOUT_SAMPLING_MODE  = "uniform"         # 'fps' | 'uniform' | 'gaussian'
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
                               "Dataset", "Primitives", "transitions")

parts = ["box_to_ellipse_frames", "box_to_pyramid_frames",
         "box_to_sphere_frames", "ellipse_to_box_frames", 
         "ellipse_to_pyramid_frames", "ellipse_to_sphere_frames",
         "pyramid_to_box_frames", "pyramid_to_ellipse_frames",
         "pyramid_to_sphere_frames", "sphere_to_box_frames",
         "sphere_to_ellipse_frames", 
         "sphere_to_pyramid_frames" ]

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
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)              # seeds CPU + the default CUDA generator
    torch.cuda.manual_seed_all(SEED)     # all GPUs (redundant with above but explicit)
    key = torch.Generator(device="cpu")  # you already have this
    key.manual_seed(SEED)

   
    shape_vertices, shape_mask, shape_areas, shape_normals = load_dataset(
        data_path=SHAPE_DATA_ROOT,
        parts=parts,
        load_fields=True,
        seed = 0
    )

    # Hold out a fraction of the SAME loaded set for validation. One shared permutation
    # splits all four arrays, so each shape's verts/mask/areas/normals stay aligned.
    (shape_vertices, shape_mask, shape_areas, shape_normals), \
    (val_shape_vertices, val_shape_mask, val_shape_areas, val_shape_normals) = split_dataset(
        shape_vertices, shape_mask, shape_areas, shape_normals,
        val_fraction=0.2, seed=0)
  
    graph, supergraph = build_training_graph(shape_vertices, 
                                shape_mask,
                                key,
                                r_max = R_MAX,       
                                dropout_rate = DROPOUT_RATE, 
                                n_supernodes = N_SUPERNODES, 
                                 r_supergraph= R_SUPERGRPAH,
                                use_supernodes= USE_SUPERNODES,
                                sampling_mode_graph=DROPOUT_SAMPLING_MODE,
                                sampling_mode_supernodes=SUPERNODE_SAMPLING_MODE,
                                areas=shape_areas,
                                normals=shape_normals)

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
        "input_irreps": '1x0e + 1x1o',
        "intermediate_irreps": "32x0e + 32x0o + 16x1o + 16x1e+ 4x2o+ 4x2e",
       #  "output_irreps": f"{LATENT_DIM}x0e + 2x1o",   # latent_dim scalars + 2 vectors (rotation frame)
         "output_irreps": f"{LATENT_DIM}x0e + 2x1o"
    }

    tcfg = {'num_layers': 2, 'num_heads': 2, 'hidden_channels': 8, 'sh_lmax': 2}
    transformer_type='equiformer'
    area_pool = True

    encoder = GroupEncoder(
        latent_dim=LATENT_DIM, 
        irreps_cfg=layer_cfg, 
        sh_lmax = 2,
        readout = "mean",
        readout_heads = 1,
        supernode_sh_lmax=2,
        transformer_type=transformer_type,
        transformer_cfg=tcfg, 
        area_pool=area_pool,
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
            sampling_mode_graph=DROPOUT_SAMPLING_MODE,
            sampling_mode_supernodes=SUPERNODE_SAMPLING_MODE,
            areas=shape_areas,
            normals=shape_normals)
        print("loader: two-view contrastive (two fresh samplings of the same shapes per step)")
    elif RESAMPLE_GRAPH:
        loader = ResamplingGraphLoader(
            shape_vertices, shape_mask, build_training_graph, key=key,
            r_max=RESAMPLE_R_MAX,
            r_supergraph=R_SUPERGRPAH,
            dropout_rate=RESAMPLE_DROPOUT,
            areas=shape_areas,
            normals=shape_normals)
        print(f"loader: resampling graph each step (r_max={RESAMPLE_R_MAX}, dropout={RESAMPLE_DROPOUT})")
    else:
        loader = OneBatchLoader((graph, supergraph, shape_vertices, shape_mask))
        print("loader: prebuilt graph reused every step")

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


    stepper = TrainingStepper(encoder, decoder,
                               learning_rate=LEARNING_RATE,
                               kl_weight=0.001,
                               contrastive_weight=CONTRASTIVE_WEIGHT if CONTRASTIVE else 0.0,
                               contrastive_var_weight=CONTRASTIVE_VAR_WEIGHT)
    logger = TrainingLogger(log_dir = OUTPUT_DIR)
    trainer = TrainingOrchestrator(stepper=stepper, logger=logger, 
                                   dataloader=loader, val_loader=val_loader)

    print(f"----------training on device: {stepper.device}----------")
    if stepper.device != "cuda":
        # Surface WHY the GPU gate failed instead of dying on a bare AssertionError:
        # a CPU-only torch build and a GPU node with no visible device look identical
        # otherwise. (Under `python -O` the assert below is stripped, so this print is
        # the only signal the run landed on CPU.)
        print(
            "[gpu-gate] CUDA required but not selected. "
            f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
            f"device_count={torch.cuda.device_count()}, "
            f"torch_cuda_build={torch.version.cuda}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    assert stepper.device == "cuda", (
        f"expected to train on CUDA but resolved to {stepper.device}; "
        "see the [gpu-gate] diagnostics above."
    )
    trainer.run(num_steps=NUM_STEPS, log_every=LOG_EVERY,
                 save_every=SAVE_EVERY, val_every=VAL_EVERY)
    print(f"done. outputs in {OUTPUT_DIR}")
    

if __name__ == "__main__":
    main()
