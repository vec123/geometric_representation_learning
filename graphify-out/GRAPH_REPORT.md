# Graph Report - .  (2026-07-06)

## Corpus Check
- Corpus is ~10,247 words - fits in a single context window. You may not need a graph.

## Summary
- 217 nodes · 443 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]

## God Nodes (most connected - your core abstractions)
1. `SpatialConvolution` - 13 edges
2. `EquivariantLayerNorm` - 12 edges
3. `EquiLayer` - 11 edges
4. `GroupEncoder` - 11 edges
5. `TrainingStepper` - 11 edges
6. `assert_close()` - 11 edges
7. `SelfInteraction` - 10 edges
8. `TrainingOrchestrator` - 10 edges
9. `save_vtp()` - 10 edges
10. `norm()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_equi_layer_shape_and_equivariance()` --calls--> `EquiLayer`  [EXTRACTED]
  tests/learning/test_equivariant_modules.py → src/learning/layers/equivariant/Self_Spatial_layer.py
- `test_save_checkpoint()` --calls--> `TrainingLogger`  [EXTRACTED]
  tests/learning/test_logger.py → src/learning/logger/train_logs.py
- `test_folding_decoder()` --calls--> `FoldingDecoder`  [EXTRACTED]
  tests/learning/test_models.py → src/learning/models/folding_decoder.py
- `_encoder_inputs()` --calls--> `GroupEncoder`  [EXTRACTED]
  tests/learning/test_equivariant_modules.py → src/learning/models/group_encoder.py
- `test_group_encoder_forward()` --calls--> `GroupEncoder`  [EXTRACTED]
  tests/learning/test_models.py → src/learning/models/group_encoder.py

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.10
Nodes (35): get_project_root(), Returns the absolute path of the parent directory of the      folder containing, apply_noise_and_masking(), build_bipartite_graph(), build_radius_graph(), get_graphs_from_vertices(), get_individual_graph(), get_vertices_and_edges() (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (35): EquivariantAttention, EquivariantLayerNorm, get_slice_for_irrep(), assert_close(), _attention_graph(), _encoder_inputs(), max_err(), Acceptance tests for the equivariant modules (contract for the jax -> torch port (+27 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (26): build_grad(), build_grad_point_cloud(), build_tangent_frames(), compute_grads(), compute_mean_curvature(), cross(), dot(), edge_tangent_vectors() (+18 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (8): TrainingOrchestrator, TrainingStepper, DummyDataloader, DummyDecoder, DummyEncoder, DummyLogger, test_training_orchestrator(), test_training_stepper()

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (8): MessagePassing, EquiLayer, GatingBlock, get_slice_for_irrep(), Filters an irreps string to keep only l <= max_l., SelfInteraction, SpatialConvolution, test_self_interaction_shape()

### Community 5 - "Community 5"
Cohesion: 0.44
Nodes (10): chamfer_loss(), combined_surface_loss(), geometric_clustering_loss(), kl_divergence_loss(), laplacian_loss(), pred_pos: [Batch, N_samples, 3] where N_samples is square., Highly performant Chamfer Distance using torch.cdist.     pred_pos: [B, N, 3], Computes the KL divergence between N(mean, exp(log_var)) and N(0, 1). (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (5): FoldingDecoder, GroupEncoder, test_folding_decoder(), test_group_encoder_forward(), test_group_encoder_rotation()

### Community 7 - "Community 7"
Cohesion: 0.33
Nodes (4): nodes: [Total_Nodes, 3]     n_node: [Num_Graphs]     rotations: [Num_Graphs, 3, nodes: [Total_Nodes, 3]     n_node: [Num_Graphs]     rotations: [Num_Graphs, 3, SE3_transform(), SE3_transform_numpy()

### Community 8 - "Community 8"
Cohesion: 0.33
Nodes (4): clean_vtp(), filter_vtp_largest_component(), Isolates the single largest connected component, removes topological defects,, Loads a VTP, extracts only the largest connected component, and saves it.

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GroupEncoder` connect `Community 6` to `Community 0`, `Community 1`, `Community 4`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `TrainingStepper` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `TrainingOrchestrator` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `EquiLayer` (e.g. with `SelfInteraction` and `SpatialConvolution`) actually correct?**
  _`EquiLayer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `TrainingStepper` (e.g. with `DummyDataloader` and `DummyDecoder`) actually correct?**
  _`TrainingStepper` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Returns the absolute path of the parent directory of the      folder containing`, `Computes norm of an array of vectors along the last dimension.`, `Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after n` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0988235294117647 - nodes in this community are weakly interconnected._