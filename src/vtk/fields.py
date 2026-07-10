
from vtk.util import numpy_support
import torch
import numpy as np
def add_point_field(polydata, field_data, field_name="field", verbose = False):
    """Adds a scalar or vector field to the existing PolyData."""
    
    if verbose:
        print(f"adding field {field_data.shape}")
    if isinstance(field_data, torch.Tensor):
        field_data = field_data.detach().cpu().numpy()
        print(f"convert to {field_data.shape}")

    vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
    vtk_array.SetName(field_name)
    polydata.GetPointData().AddArray(vtk_array)
    return polydata

