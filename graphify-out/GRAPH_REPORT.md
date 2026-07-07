# Graph Report - geometric_representation_learning  (2026-07-07)

## Corpus Check
- 38 files · ~13,772 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 318 nodes · 605 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 34 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6c0441b7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_test_graphs.py|test_graphs.py]]
- [[_COMMUNITY_EquiLayer|EquiLayer]]
- [[_COMMUNITY_geometry_np.py|geometry_np.py]]
- [[_COMMUNITY_TrainingStepper|TrainingStepper]]
- [[_COMMUNITY_losses.py|losses.py]]
- [[_COMMUNITY_GroupEncoder|GroupEncoder]]
- [[_COMMUNITY_process.py|process.py]]
- [[_COMMUNITY_group_transforms.py|group_transforms.py]]
- [[_COMMUNITY_convert_vtu_to_vtp_vtk|convert_vtu_to_vtp_vtk]]
- [[_COMMUNITY_add_point_field|add_point_field]]
- [[_COMMUNITY_attention.py|attention.py]]
- [[_COMMUNITY_perceiver.py|perceiver.py]]
- [[_COMMUNITY_TrainingLogger|TrainingLogger]]

## God Nodes (most connected - your core abstractions)
1. `GroupEncoder` - 20 edges
2. `TrainingStepper` - 20 edges
3. `TrainingOrchestrator` - 16 edges
4. `SpatialConvolution` - 13 edges
5. `save_vtp()` - 13 edges
6. `TrainingLogger` - 12 edges
7. `PerceiverLayer` - 12 edges
8. `create_polydata()` - 11 edges
9. `assert_close()` - 11 edges
10. `ResamplingGraphLoader` - 10 edges

## Surprising Connections (you probably didn't know these)
- `OneBatchLoader` --uses--> `GroupEncoder`  [INFERRED]
  scripts/test_group_equivariant_train.py → src/learning/models/group_encoder.py
- `OneBatchLoader` --uses--> `TrainingOrchestrator`  [INFERRED]
  scripts/test_group_equivariant_train.py → src/learning/trainers/E3_end2end.py
- `OneBatchLoader` --uses--> `TrainingStepper`  [INFERRED]
  scripts/test_group_equivariant_train.py → src/learning/trainers/E3_end2end.py
- `ResamplingGraphLoader` --uses--> `GroupEncoder`  [INFERRED]
  scripts/test_group_equivariant_train.py → src/learning/models/group_encoder.py
- `ResamplingGraphLoader` --uses--> `TrainingOrchestrator`  [INFERRED]
  scripts/test_group_equivariant_train.py → src/learning/trainers/E3_end2end.py

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "test_graphs.py"
Cohesion: 0.08
Nodes (41): get_project_root(), Returns the absolute path of the parent directory of the      folder containing, build_training_graph(), End-to-end equivariant training on a real face-parts dataset.  Encodes each shap, Build the graph fed to the encoder, per the USE_SUPERNODES toggle, and attach, save_graph_vtp(), apply_noise_and_masking(), build_bipartite_graph() (+33 more)

### Community 1 - "EquiLayer"
Cohesion: 0.06
Nodes (41): MessagePassing, EquiLayer, EquivariantAttention, GatingBlock, Filters an irreps string to keep only l <= max_l., SelfInteraction, SpatialConvolution, EquivariantLayerNorm (+33 more)

### Community 2 - "geometry_np.py"
Cohesion: 0.13
Nodes (26): build_grad(), build_grad_point_cloud(), build_tangent_frames(), compute_grads(), compute_mean_curvature(), cross(), dot(), edge_tangent_vectors() (+18 more)

### Community 3 - "TrainingStepper"
Cohesion: 0.14
Nodes (7): AttentionHead, MultiHeadAttention, MLP, PerceiverLayer, PerceiverDecoder, Perceiver, PerceiverEncoder

### Community 4 - "losses.py"
Cohesion: 0.12
Nodes (11): FoldingDecoder, GroupEncoder, expand_per_irrep_gate(), Small helpers for working with plain ``[N, irreps.dim]`` tensors in PyTorch e3nn, Gather all invariant scalar (0e) channels of ``x`` -> ``[..., #0e]``., Gather every ``target`` irrep (e.g. ``1o``) as ``[..., total_mul, ir.dim]``., Broadcast a per-irrep gate ``[..., num_irreps]`` to per-channel ``[..., dim]``., scalar_features() (+3 more)

### Community 5 - "GroupEncoder"
Cohesion: 0.40
Nodes (9): chamfer_loss(), combined_surface_loss(), geometric_clustering_loss(), kl_divergence_loss(), laplacian_loss(), pred_pos: [Batch, N_samples, 3] where N_samples is square., Highly performant Chamfer Distance using torch.cdist.     pred_pos: [B, N, 3], Computes the KL divergence between N(mean, exp(log_var)) and N(0, 1). (+1 more)

### Community 6 - "process.py"
Cohesion: 0.09
Nodes (23): EncoderOutput, Standard container for encoder outputs, decoupling encoders from the trainer., Runs a single optimization step: encode -> reparameterize -> decode -> loss., Adapt the current GroupEncoder output into the standard EncoderOutput., Drives the training loop: fetch a batch, step, log/checkpoint at cadence.      T, _resolve_device(), TrainingOrchestrator, TrainingStepper (+15 more)

### Community 7 - "group_transforms.py"
Cohesion: 0.33
Nodes (4): nodes: [Total_Nodes, 3]     n_node: [Num_Graphs]     rotations: [Num_Graphs, 3, nodes: [Total_Nodes, 3]     n_node: [Num_Graphs]     rotations: [Num_Graphs, 3, SE3_transform(), SE3_transform_numpy()

### Community 8 - "convert_vtu_to_vtp_vtk"
Cohesion: 0.33
Nodes (4): clean_vtp(), filter_vtp_largest_component(), Isolates the single largest connected component, removes topological defects,, Loads a VTP, extracts only the largest connected component, and saves it.

### Community 10 - "attention.py"
Cohesion: 0.12
Nodes (16): AbstractPositionEncoding, build_linear_positions(), build_position_encoding(), _check_or_build_spatial_positions(), FourierPositionEncoding, generate_fourier_features(), PositionEncodingProjector, Abstract Perceiver decoder. (+8 more)

### Community 11 - "perceiver.py"
Cohesion: 0.13
Nodes (10): load_dataset(), main(), OneBatchLoader, Yields the same prebuilt (graph, true_verts, mask) batch each step., Rebuilds the encoder graph from raw geometry each step with a fresh key.      Th, Return value as-is if scalar, else draw uniformly from a (low, high) range., Load face-part shapes; fall back to tests/data so the script always runs., ResamplingGraphLoader (+2 more)

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GroupEncoder` connect `losses.py` to `test_graphs.py`, `EquiLayer`, `perceiver.py`, `process.py`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `TrainingStepper` connect `process.py` to `test_graphs.py`, `perceiver.py`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `TrainingOrchestrator` connect `process.py` to `test_graphs.py`, `perceiver.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `GroupEncoder` (e.g. with `OneBatchLoader` and `ResamplingGraphLoader`) actually correct?**
  _`GroupEncoder` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TrainingStepper` (e.g. with `OneBatchLoader` and `ResamplingGraphLoader`) actually correct?**
  _`TrainingStepper` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TrainingOrchestrator` (e.g. with `OneBatchLoader` and `ResamplingGraphLoader`) actually correct?**
  _`TrainingOrchestrator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `End-to-end equivariant training on a real face-parts dataset.  Encodes each shap`, `Load face-part shapes; fall back to tests/data so the script always runs.`, `Build the graph fed to the encoder, per the USE_SUPERNODES toggle, and attach` to the rest of the system?**
  _70 weakly-connected nodes found - possible documentation gaps or missing edges._