
from vtk.util import numpy_support


def extract_vtp_points_cells(poly_data):

    points = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    cells = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetConnectivityArray())
    return points, cells.reshape(-1, 3)