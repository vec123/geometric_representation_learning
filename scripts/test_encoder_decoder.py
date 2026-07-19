
"""
Use a trained Encoder and Decoder on patches of ultrasounds
Examine the encodings of ultrasound patches And clean training data patches.
Compute the distance to chosen latents, 
Label the points in the patch with the respective distance.
"""

import os
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
from src.paths import get_project_root
from src.learning.helpers import load_dataset, build_training_graph, save_graph_vtp


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
USE_SUPERNODES = True          # toggle: supernode subset (True) vs full/decimated graph (False)
N_SUPERNODES   = 10           # n_s, used when USE_SUPERNODES is True
SAMPLING_MODE  = "fps"         # 'fps' | 'uniform' | 'gaussian'
DROPOUT_RATE   = 0.9          # uniform node dropout to reduce data-sample size uniformly
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

LATENT_DIM     = 5
NUM_SAMPLES    = 256           # decoder output points (perfect square for the folding grid)
LEARNING_RATE  = 1e-3
NUM_STEPS      = 1001
LOG_EVERY      = 1
SAVE_EVERY     = 100
VAL_EVERY      = 100           # run + save validation every N steps

Project_ROOT = get_project_root()

VAL_SHAPE_DATA_ROOT =  os.path.join(
    Project_ROOT,
    "Dataset", "vtp_samples", "val_Dataset_faceparts_normalized")
SHUFFLE = False
categories =  [0] *10 + [1] * 10
US_PATHES_DATA_ROOT =  os.path.join(
    Project_ROOT,
    "Dataset", "US_patches", "patches", "2")


CHECKPOINT_PATH = os.path.join(Project_ROOT, 
                               "training_log_vae", 
                               "checkpoints",
                               "step_1100.pt")

OUTPUT_DIR = os.path.join(Project_ROOT,
                               f"Training_Analytics_drop_{DROPOUT_RATE}")

# Headless/HPC toggle: None -> auto (mirror output to a log file when stdout is not a TTY,
# e.g. under SLURM/nohup); True/False to force. In remote mode stdout+stderr are teed to a
# timestamped, flushed log under OUTPUT_DIR for easy inspection (tail -f).
REMOTE = None

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok= True)
    enable_headless(OUTPUT_DIR, remote=REMOTE, name="test_encoder_decoder")

    key = torch.Generator(device="cpu")
    key.manual_seed(5)

    device = "cuda"

    layers_cfg = [{
        "in_irreps": "1x0e",
        "target_irreps": "32x0e + 32x0o + 16x1o + 16x1e+ 4x2o+ 4x2e",
        "spatial_sh_lmax": 1,
        "interaction_sh_lmax": 4,
    }]
    output_irreps = f"{LATENT_DIM}x0e + 2x1o"   # latent_dim scalars + 2 vectors (rotation frame)

    encoder = GroupEncoder(
        layers_cfg=layers_cfg,
        latent_dim=LATENT_DIM,
        output_irreps=output_irreps,
        readout = "mean",
        readout_heads = 1,
        verbose=False)
    
    decoder = FoldingDecoder(
        num_samples=NUM_SAMPLES,
          latent_dim=LATENT_DIM, 
          n_freqs=4, 
          verbose=False)


    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    # Load the weights into the models
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    # Set models to evaluation mode (Crucial for inference!)
    encoder.eval()
    decoder.eval()
    # Move models to the device
    encoder.to(device)
    decoder.to(device)



    # Load Data
    parts = ["bridge", "nose"]
    shape_vertices, shape_mask = load_dataset(data_path=VAL_SHAPE_DATA_ROOT,
                                            parts = parts, shuffle = SHUFFLE)
    shape_graph, shape_supergraph = build_training_graph(shape_vertices, 
                                shape_mask,
                                key,
                                r_max=R_MAX,       
                                dropout_rate=DROPOUT_RATE, 
                                n_supernodes = N_SUPERNODES, 
                                 r_supergraph= R_SUPERGRPAH,
                                use_supernodes= USE_SUPERNODES)
    
    true_shape_loader = OneBatchLoader((shape_graph, shape_supergraph, shape_vertices, shape_mask))
    

    """
    us_patch_vertices, us_patch_mask = load_dataset(data_path=VAL_SHAPE_DATA_ROOT,
                                            parts = None )
    us_graph, us_graph = build_training_graph(shape_vertices, 
                                shape_mask,
                                key,
                                r_max=R_MAX,       
                                dropout_rate=DROPOUT_RATE, 
                                n_supernodes = N_SUPERNODES, 
                                 r_supergraph= R_SUPERGRPAH,
                                use_supernodes= USE_SUPERNODES)
    us_shape_loader = OneBatchLoader((us_graph, us_graph, us_patch_vertices, us_patch_mask))
    """

        
    stepper = TrainingStepper(encoder, decoder, learning_rate=LEARNING_RATE, kl_weight=0.0)
    logger = TrainingLogger(log_dir = OUTPUT_DIR)
    
    shape_graph = shape_graph.to(device)
    shape_supergraph = shape_supergraph.to(device)
    encoder_output = stepper.encode(shape_graph, shape_supergraph)
    mu = encoder_output.mu
    print("mu.shape: ", mu.shape)

   
    decodings, loss, recon, kl = stepper.eval_step(shape_graph, shape_supergraph, shape_vertices, shape_mask)
    print("loss: ", loss, "recon", recon,  "kl: ", kl) 
    batch = (shape_graph, shape_supergraph, shape_vertices, shape_mask)
    logger.visualize_batch(batch, decodings, step = 1, subdir = "vtk", max_num = 102)

    def project_latents_to_2d(latents: torch.Tensor, method='tsne', n_components=2, perplexity = 5):
        """
        Projects latents to 2D using various methods.
        - 'pca': Linear, good for global structure.
        - 'tsne': Nonlinear, excellent for local clusters.
        - 'umap': Nonlinear, often faster and preserves global structure better than t-SNE.
        """
        data = latents

        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            # perplexity should be lower than the number of samples
            reducer = TSNE(n_components=n_components, perplexity=perplexity, init='pca', learning_rate='auto', random_state=42)
        else:
            raise ValueError("Method must be 'pca', 'tsne', or 'umap'")

        return reducer.fit_transform(data)

    def plot_latent_scatter(projected_data, categories, save_path):
        """
        Creates a 2D scatter plot where points are colored by their category.
        categories: list or array of shape (N_samples,) containing group IDs.
        """
        plt.figure(figsize=(8, 6))

        # Create the scatter plot
        # 'c' takes the category labels; 'cmap' defines the color scheme (e.g., 'viridis', 'coolwarm')
        scatter = plt.scatter(
            projected_data[:, 0],
            projected_data[:, 1],
            c=categories,
            cmap='viridis',
            alpha=0.8,
            edgecolors='k',
            s=100
        )

        # Create a legend
        legend1 = plt.legend(*scatter.legend_elements(), title="Categories")
        plt.gca().add_artist(legend1)

        plt.title("Latent Space Projection by Category")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(save_path)

    # --- Usage ---
    # Assuming 'mu' is a torch.Size([20, 1, 5]) tensor
    projected_mu = project_latents_to_2d(mu.cpu().detach().numpy(), method='tsne', perplexity = 3)
    print("projected_mu.shape: ", projected_mu.shape)

    print("categories: ", categories)
    save_path = os.path.join(OUTPUT_DIR, "scatter_latents.png")
    plot_latent_scatter(projected_mu, categories, save_path)

    def get_outlier_indices(projected_data, categories):
        # 1. Separate the data based on your known categories
        cat0_data = projected_data[np.array(categories) == 0]
        cat1_data = projected_data[np.array(categories) == 1]

        # 2. Calculate the center (centroid) of each cluster
        centroid0 = np.mean(cat0_data, axis=0)
        centroid1 = np.mean(cat1_data, axis=0)

        outliers = []

        # 3. Check each point to see if it is closer to the "other" cluster
        for i, point in enumerate(projected_data):
            dist0 = np.linalg.norm(point - centroid0)
            dist1 = np.linalg.norm(point - centroid1)

            # If category is 0 but it's closer to centroid1 (or vice versa), mark as outlier
            if (categories[i] == 0 and dist1 < dist0) or (categories[i] == 1 and dist0 < dist1):
                outliers.append(i)
                print(f"Outlier Found at Index {i}: Category {categories[i]} is closer to the other cluster.")

        return outliers

    # Run the function
    outlier_indices = get_outlier_indices(projected_mu, categories)
    print(f"Total outlier indices: {outlier_indices}")