import os 
import torch
from src.vtk.io import save_vtp
from src.vtk.create import create_polydata



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
        
    def log_visualizations(self, data_dict, step, sample_idx=0):
        """
        Expects data_dict: {'original': np.array, 'canonical': np.array, ...}
        """
        for name, data in data_dict.items():
            print(f"Logging visualization for {name} at step {step}")
            poly = create_polydata(data)
            path = os.path.join(self.log_dir, "vtk", f"{sample_idx}_{name}_{step}.vtp")
            save_vtp(poly, path)
    