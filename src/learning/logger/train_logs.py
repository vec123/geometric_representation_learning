import os
import torch
from src.vtk.io import save_vtp
from src.vtk.create import create_polydata, create_polydata_w_lines
from src.vtk.fields import add_point_field
from src.graphs.graphs import get_individual_graph, get_bipartite_graph



class Logger1:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "vtk"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    def log_metrics(self, metrics, step):
        print(f"[step {step}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))

    def save_checkpoint(self, stepper, step):
        path = os.path.join(self.log_dir, "checkpoints", f"step_{step}.pt")
        torch.save({
            "encoder": stepper.encoder.state_dict(),
            "decoder": stepper.decoder.state_dict(),
            "optimizer": stepper.optimizer.state_dict(),
        }, path)
        print(f"  checkpoint -> {path}")

    def visualize_results(self, pred, step):
        pred = pred.detach().cpu().numpy()
        for i in range(pred.shape[0]):
            path = os.path.join(self.log_dir, "vtk", f"pred_shape{i}_step{step}.vtp")
            save_vtp(create_polydata(pred[i]), path)
        print(f"  saved {pred.shape[0]} prediction VTP(s) at step {step}")
        
        
        
    

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir

    def log_metrics(self, metrics, step):
        print(f"[step {step}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))

    def save_checkpoint(self, stepper, step,  directory="checkpoints"):
        checkpoint_dir = os.path.join(self.log_dir, directory)
        path = os.path.join(checkpoint_dir, f"step_{step}.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "encoder": stepper.encoder.state_dict(),
            "decoder": stepper.decoder.state_dict(),
            "optimizer": stepper.optimizer.state_dict(),
        }, path)
        print(f"  checkpoint -> {path}")

    def visualize_results(self, pred, step):
        pred = pred.detach().cpu().numpy()
        for i in range(pred.shape[0]):
            path = os.path.join(self.log_dir, "vtk", f"pred_shape{i}_step{step}.vtp")
            save_vtp(create_polydata(pred[i]), path)
        print(f"  saved {pred.shape[0]} prediction VTP(s) at step {step}")

    def visualize_batch(self, batch, pred, step):
        """Save, for this step, the input graph, the supergraph (if any), the true
        target verts, and the predictions.

        ``batch`` is the ``(graph, super_graph, true_verts, mask)`` tuple the trainer
        steps on. Graph tensors are expected on CPU (the trainer moves its own copies to
        the device), so they render directly.
        """
        graph, super_graph, true_verts, mask = batch[0], batch[1], batch[2], batch[3]
        self._save_graph_vtp(graph, step, name="input_graph", is_supernodes=False)
        if super_graph is not None:
            self._save_graph_vtp(super_graph, step, name="supergraph", is_supernodes=True)
        self._save_true_verts(true_verts, mask, step)
        self.visualize_results(pred, step)

    def _save_true_verts(self, true_verts, mask, step):
        """Save the reconstruction target as a point cloud (no edges), one VTP per shape.
        Padded entries are dropped via ``mask`` so only real vertices are written."""
        out_dir = os.path.join(self.log_dir, "vtk")
        tv = true_verts.detach().cpu()
        m = mask.detach().cpu().bool() if mask is not None else None
        for i in range(tv.shape[0]):
            pts = tv[i][m[i]] if m is not None else tv[i]        # [n_i, 3] valid verts
            save_vtp(create_polydata(pts), os.path.join(out_dir, f"true_verts_shape{i}_step{step}.vtp"))
        print(f"  saved {tv.shape[0]} true-verts VTP(s) at step {step}")

    def _save_graph_vtp(self, graph, step, name, is_supernodes=False):
        """Render one VTP per shape (with edges) for a homogeneous graph, or the merged
        full+super point set with aggregation lines for a bipartite supergraph."""
        out_dir = os.path.join(self.log_dir, "vtk")
        num_graphs = int(graph.batch.max().item()) + 1
        for i in range(num_graphs):
            if is_supernodes:
                pos, edges, node_field = get_bipartite_graph(graph, i)
            else:
                pos, edges = get_individual_graph(graph, i)
            vtp = create_polydata_w_lines(pos, edges)
            if is_supernodes:
                vtp = add_point_field(vtp, field_data=node_field, field_name="super_node")
            save_vtp(vtp, os.path.join(out_dir, f"{name}_shape{i}_step{step}.vtp"))
        print(f"  saved {num_graphs} {name} VTP(s) at step {step}")

    def log_visualizations(self, data_dict, step, sample_idx=0):
        """
        Expects data_dict: {'original': np.array, 'canonical': np.array, ...}
        """
        for name, data in data_dict.items():
            print(f"Logging visualization for {name} at step {step}")
            poly = create_polydata(data)
            path = os.path.join(self.log_dir, "vtk", f"{sample_idx}_{name}_{step}.vtp")
            save_vtp(poly, path)
    