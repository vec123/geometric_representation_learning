import os
import vtk
import torch
import numpy as np
from vtk.util import numpy_support

from config.root import get_project_root
from src.graphs.graphs import sample_nodes
from src.vtk.io import save_vtp

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def center_polydata(polydata):
    com_filter = vtk.vtkCenterOfMass()
    com_filter.SetInputData(polydata)
    com_filter.SetUseScalarsAsWeights(False)
    com_filter.Update()
    center = com_filter.GetCenter()
    
    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(polydata)
    transform_filter.Update()
    return transform_filter.GetOutput()

def prepare_mesh_for_sampling(polydata):
    """
    Cleans the mesh to ensure it is a single, solid, manifold triangle mesh.
    """
    # 1. Triangulate ensures side faces are not just large quadrilaterals
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    
    # 2. Clean merges coincident points at the seams of the caps
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(tri.GetOutputPort())
    cleaner.PointMergingOn()
    
    # 3. Generate normals (sometimes required for correct surface identification)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaner.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.Update()
    
    return center_polydata(normals.GetOutput())

def create_dense_custom_mesh(vertices, faces, subdivisions=4):
    points = vtk.vtkPoints()
    for v in vertices: points.InsertNextPoint(v)
    cells = vtk.vtkCellArray()
    for f in faces:
        tri = vtk.vtkTriangle()
        for i in range(3): tri.GetPointIds().SetId(i, f[i])
        cells.InsertNextCell(tri)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    
    subdivision = vtk.vtkLinearSubdivisionFilter()
    subdivision.SetNumberOfSubdivisions(subdivisions)
    subdivision.SetInputData(polydata)
    subdivision.Update()
    return subdivision.GetOutput()

def process_and_save_primitive(polydata, filename, num_samples=512):
    points_np = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    pts_tensor = torch.tensor(points_np, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones((1, pts_tensor.shape[1]), dtype=torch.bool)
    
    sampled_pts, _ = sample_nodes(pts_tensor, mask, num_samples=num_samples, mode='fps')
    sampled_np = sampled_pts.squeeze(0).numpy()
    
    points = vtk.vtkPoints()
    for p in sampled_np: points.InsertNextPoint(p)
    new_poly = vtk.vtkPolyData()
    new_poly.SetPoints(points)
    
    verts = vtk.vtkCellArray()
    for i in range(sampled_np.shape[0]):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    new_poly.SetVerts(verts)
    
    save_vtp(new_poly, filename)
    print(f"Saved {os.path.basename(filename)} ({sampled_np.shape[0]} points).")

# -------------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------------

ROOT = get_project_root()
OUTPUT_FOLDER = os.path.join(ROOT, "Dataset", "Primitives")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
NUM_POINTS = 512

# 1. Sphere
sphere = vtk.vtkSphereSource(); sphere.SetThetaResolution(100); sphere.SetPhiResolution(100); sphere.Update()
process_and_save_primitive(prepare_mesh_for_sampling(sphere.GetOutput()), os.path.join(OUTPUT_FOLDER, "sphere.vtp"), NUM_POINTS)

# 2. Ellipsoid
transform = vtk.vtkTransform(); transform.Scale(2.0, 1.0, 0.5)
tf = vtk.vtkTransformPolyDataFilter(); tf.SetTransform(transform); tf.SetInputConnection(sphere.GetOutputPort()); tf.Update()
process_and_save_primitive(prepare_mesh_for_sampling(tf.GetOutput()), os.path.join(OUTPUT_FOLDER, "ellipse.vtp"), NUM_POINTS)


# 4. Box
box_verts = [[-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [0.5,0.5,-0.5], [-0.5,0.5,-0.5], [-0.5,-0.5,0.5], [0.5,-0.5,0.5], [0.5,0.5,0.5], [-0.5,0.5,0.5]]
box_faces = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
process_and_save_primitive(prepare_mesh_for_sampling(create_dense_custom_mesh(box_verts, box_faces)), os.path.join(OUTPUT_FOLDER, "box.vtp"), NUM_POINTS)

# 5. Pyramid
pyr_verts = [[-0.5,-0.5,0.0], [0.5,-0.5,0.0], [0.5,0.5,0.0], [-0.5,0.5,0.0], [0.0,0.0,1.0]]
pyr_faces = [[0,2,1],[0,3,2],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]
process_and_save_primitive(prepare_mesh_for_sampling(create_dense_custom_mesh(pyr_verts, pyr_faces)), os.path.join(OUTPUT_FOLDER, "pyramid.vtp"), NUM_POINTS)


