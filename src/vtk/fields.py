
from vtk.util import numpy_support

def add_point_field(polydata, field_data, field_name="field"):
    """Adds a scalar or vector field to the existing PolyData."""
    vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
    vtk_array.SetName(field_name)
    polydata.GetPointData().AddArray(vtk_array)
    return polydata

