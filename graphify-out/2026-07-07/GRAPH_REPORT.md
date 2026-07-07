# Graph Report - geometric_representation_learning  (2026-07-06)

## Corpus Check
- 35 files · ~11,200 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 250 nodes · 506 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3febcd73`
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

## God Nodes (most connected - your core abstractions)
1. `GroupEncoder` - 17 edges
2. `SpatialConvolution` - 13 edges
3. `EquivariantLayerNorm` - 12 edges
4. `EquiLayer` - 11 edges
5. `assert_close()` - 11 edges
6. `TrainingStepper` - 11 edges
7. `SelfInteraction` - 10 edges
8. `TrainingOrchestrator` - 10 edges
9. `save_vtp()` - 10 edges
10. `EquivariantAttention` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_equi_layer_shape_and_equivariance()` --calls--> `EquiLayer`  [EXTRACTED]
  tests/learning/test_equivariant_modules.py → src/learning/layers/equivariant/Self_Spatial_layer.py
- `_encoder_inputs()` --calls--> `GroupEncoder`  [EXTRACTED]
  tests/learning/test_equivariant_modules.py → src/learning/models/group_encoder.py
- `test_group_encoder_forward()` --calls--> `GroupEncoder`  [EXTRACTED]
  tests/learning/test_models.py → src/learning/models/group_encoder.py
- `test_group_encoder_rotation()` --calls--> `GroupEncoder`  [EXTRACTED]
  tests/learning/test_models.py → src/learning/models/group_encoder.py
- `_InfiniteLoader` --uses--> `GroupEncoder`  [INFERRED]
  tests/learning/test_trainer_e2e.py → src/learning/models/group_encoder.py

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "test_graphs.py"
Cohesion: 0.10
Nodes (35): get_project_root(), Returns the absolute path of the parent directory of the      folder containing, apply_noise_and_masking(), build_bipartite_graph(), build_radius_graph(), get_graphs_from_vertices(), get_individual_graph(), get_vertices_and_edges() (+27 more)

### Community 1 - "EquiLayer"
Cohesion: 0.09
Nodes (34): EquivariantAttention, EquivariantLayerNorm, assert_close(), _attention_graph(), _encoder_inputs(), max_err(), Acceptance tests for the equivariant modules (contract for the jax -> torch port, SI(D_in x) == D_out SI(x). (+26 more)

### Community 2 - "geometry_np.py"
Cohesion: 0.13
Nodes (26): build_grad(), build_grad_point_cloud(), build_tangent_frames(), compute_grads(), compute_mean_curvature(), cross(), dot(), edge_tangent_vectors() (+18 more)

### Community 3 - "TrainingStepper"
Cohesion: 0.15
Nodes (8): TrainingOrchestrator, TrainingStepper, DummyDataloader, DummyDecoder, DummyEncoder, DummyLogger, test_training_orchestrator(), test_training_stepper()

### Community 4 - "losses.py"
Cohesion: 0.11
Nodes (14): MessagePassing, EquiLayer, GatingBlock, Filters an irreps string to keep only l <= max_l., SelfInteraction, SpatialConvolution, expand_per_irrep_gate(), Small helpers for working with plain ``[N, irreps.dim]`` tensors in PyTorch e3nn (+6 more)

### Community 5 - "GroupEncoder"
Cohesion: 0.44
Nodes (10): chamfer_loss(), combined_surface_loss(), geometric_clustering_loss(), kl_divergence_loss(), laplacian_loss(), pred_pos: [Batch, N_samples, 3] where N_samples is square., Highly performant Chamfer Distance using torch.cdist.     pred_pos: [B, N, 3], Computes the KL divergence between N(mean, exp(log_var)) and N(0, 1). (+2 more)

### Community 6 - "process.py"
Cohesion: 0.09
Nodes (19): FoldingDecoder, GroupEncoder, test_group_encoder_forward(), test_group_encoder_rotation(), _InfiniteLoader, make_batch(), make_models(), _OneBatchLoader (+11 more)

### Community 7 - "group_transforms.py"
Cohesion: 0.33
Nodes (4): nodes: [Total_Nodes, 3]     n_node: [Num_Graphs]     rotations: [Num_Graphs, 3, nodes: [Total_Nodes, 3]     n_node: [Num_Graphs]     rotations: [Num_Graphs, 3, SE3_transform(), SE3_transform_numpy()

### Community 8 - "convert_vtu_to_vtp_vtk"
Cohesion: 0.33
Nodes (4): clean_vtp(), filter_vtp_largest_component(), Isolates the single largest connected component, removes topological defects,, Loads a VTP, extracts only the largest connected component, and saves it.

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GroupEncoder` connect `process.py` to `test_graphs.py`, `EquiLayer`, `losses.py`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **Why does `TrainingStepper` connect `TrainingStepper` to `test_graphs.py`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `TrainingOrchestrator` connect `TrainingStepper` to `test_graphs.py`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `GroupEncoder` (e.g. with `EquiLayer` and `_InfiniteLoader`) actually correct?**
  _`GroupEncoder` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `EquiLayer` (e.g. with `SelfInteraction` and `SpatialConvolution`) actually correct?**
  _`EquiLayer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Filters an irreps string to keep only l <= max_l.`, `Small helpers for working with plain ``[N, irreps.dim]`` tensors in PyTorch e3nn`, `Gather all invariant scalar (0e) channels of ``x`` -> ``[..., #0e]``.` to the rest of the system?**
  _52 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_graphs.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0988235294117647 - nodes in this community are weakly interconnected._