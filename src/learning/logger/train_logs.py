import os 
import torch
from src.vtk.io import save_vtp
from src.vtk.create import create_polydata

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir

    def save_checkpoint(self, model, optimizer, step, directory="checkpoints"):
        path = os.path.join(self.log_dir, directory, f"step_{step}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving checkpoint at step {step} to {path}")
        torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)

    def log_visualizations(self, data_dict, step, sample_idx=0):
        """
        Expects data_dict: {'original': np.array, 'canonical': np.array, ...}
        """
        for name, data in data_dict.items():
            print(f"Logging visualization for {name} at step {step}")
            poly = create_polydata(data)
            path = os.path.join(self.log_dir, "vtk", f"{sample_idx}_{name}_{step}.vtp")
            save_vtp(poly, path)