import os
import argparse
import numpy as np
from src.vtk.io import load_vtp, save_vtp
from src.vtk.fields import add_point_field
from src.vtk.extract import extract_vtp_points_cells
from src.geometry.geometry_np import vertex_areas, vertex_normals


def process_vtp(input_path, output_path, verbose=False):
    polydata = load_vtp(input_path)
    points, cells = extract_vtp_points_cells(polydata)

    if verbose:
        print(f"Loaded {input_path}: points={points.shape[0]} cells={cells.shape[0]}")

    # Compute area and normals depending on whether faces exist.
    if cells.size == 0:
        from src.preprocessing.surface_measure import SurfaceMeasure
        # Faithful quadrature areas + normals straight from the point cloud -- no mesh, so
        # an imperfect triangulation cannot distort them, and sum_i area_i ~= surface area.
        areas, normals = SurfaceMeasure(k=16)(points)
        if verbose:
            print(f"Point cloud: SurfaceMeasure areas+normals (no triangulation)")
    else:
        areas = vertex_areas(points, faces=cells)
        normals = vertex_normals(points, faces=cells)

    # Keep areas ABSOLUTE (a partition of unity): sum_i area_i ~= surface area, so the
    # area-weighted aggregation is a genuine surface integral. The encoder's area weighting
    # is scale-invariant (conv divides by Sum_j a_j; pool softmaxes log a), so there is no
    # need to rescale per shape.
    areas = np.asarray(areas, dtype=np.float32)

    # Orient normals consistently relative to the centroid.
    normals = np.asarray(normals, dtype=np.float32)
    centroid = points.mean(axis=0, keepdims=True)
    direction = np.sum(normals * (points - centroid), axis=-1, keepdims=True)
    direction = np.sign(direction)
    direction[direction == 0] = 1.0
    normals = normals * direction

    # Normalize normals to unit length for proper glyph visualization
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(norms, 1e-12, None)

    # Rebuild polydata with face connectivity when needed.
    # Always create a fresh PolyData object so old field arrays cannot persist.
    from src.vtk.create import create_polydata
    if cells.size > 0:
        polydata = create_polydata(points.astype(np.float32), faces=cells.astype(np.int64))
    else:
        polydata = create_polydata(points.astype(np.float32), faces=None)

    # Remove any old area/normal arrays from input files before saving.
    for old_field in ["area", "normal"]:
        if polydata.GetPointData().GetArray(old_field) is not None:
            polydata.GetPointData().RemoveArray(old_field)

    # Attach point fields
    polydata = add_point_field(polydata, areas.astype(np.float32), field_name="area")
    normals = normals.astype(np.float32)
    polydata = add_point_field(polydata, normals, field_name="normal")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_vtp(polydata, output_path)
    if verbose:
        print(f"Saved {output_path}")


def find_vtp_files(root_dir, recursive=True):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".vtp"):
                yield os.path.join(dirpath, fname)


def main():
    parser = argparse.ArgumentParser(description="Add per-vertex area and normal fields to .vtp point clouds.")
    parser.add_argument("root", help="Root folder containing .vtp files to process")
    parser.add_argument("--out", help="Output root folder. If omitted, overwrite the input files", default=None)
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    out_root = args.out or args.root
    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"Root path not found: {args.root}")

    for input_path in find_vtp_files(args.root, recursive=args.recursive):
        rel_path = os.path.relpath(input_path, args.root)
        output_path = os.path.join(out_root, rel_path)
        process_vtp(input_path, output_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
