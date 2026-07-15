"""Triangulate a surface point cloud into a triangle mesh.

The ``.vtp`` datasets here store only POINTS (surface samples, no connectivity), but
several downstream quantities are only well-defined on a mesh:

  * per-vertex AREA -- the integration measure ``a_i`` for area-weighted pooling
    (``sum_i a_i f_i ~ integral over the surface``);
  * consistent per-vertex NORMALS (area-weighted average of incident face normals).

``SurfaceTriangulator`` reconstructs a triangle mesh whose VERTICES ARE THE INPUT POINTS,
so those quantities map back to the original vertices 1:1 (unlike Poisson / marching-cubes
reconstruction, which resamples). The default backend is a stitched per-vertex
tangent-plane Delaunay triangulation (``scipy`` only, no heavy deps): for each point we
project its k-nearest neighbours onto the local tangent plane, Delaunay-triangulate there,
and keep the triangles incident to the centre point. The union over all points (deduped)
is the mesh. An optional Open3D ball-pivoting backend is used if ``open3d`` is importable.

Typical use (offline, once per shape)::

    tri = SurfaceTriangulator(k=16)
    out = tri.process(points)              # dict(faces, areas, normals)
    #  -> store out["areas"] alongside the point cloud for area-weighted pooling
"""

import numpy as np
from scipy.spatial import cKDTree, Delaunay


class SurfaceTriangulator:
    """Reconstruct a triangle mesh from a surface point cloud, keeping input points as
    vertices, and compute per-vertex areas / normals.

    :param k:               neighbourhood size for the local tangent-plane triangulation
    :param backend:         'local_delaunay' (scipy, default) or 'open3d' (ball pivoting)
    :param orient_normals:  flip vertex normals to point outward from the centroid
    """

    def __init__(self, k: int = 16, backend: str = "local_delaunay",
                 orient_normals: bool = True, manifold_cleanup: bool = True):
        self.k = int(k)
        self.backend = backend
        self.orient_normals = orient_normals
        self.manifold_cleanup = manifold_cleanup

    # ------------------------------------------------------------------ API #
    def triangulate(self, points) -> np.ndarray:
        """points [N,3] -> faces [F,3] int (indices into ``points``)."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected points [N,3], got {points.shape}")
        if self.backend == "open3d":
            return self._triangulate_open3d(points)
        if self.backend == "local_delaunay":
            return self._triangulate_local_delaunay(points)
        raise ValueError(f"unknown backend {self.backend!r}")

    def vertex_areas(self, points, faces) -> np.ndarray:
        """Barycentric per-vertex area (mass): each triangle gives 1/3 of its area to each
        of its vertices, so ``sum_i a_i == total mesh area``. Returns [N]."""
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        a = np.zeros(len(points))
        if len(faces) == 0:
            return a
        v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
        face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        np.add.at(a, faces.reshape(-1), np.repeat(face_area / 3.0, 3))
        return a

    def pca_normals(self, points, k: int = None) -> np.ndarray:
        """Per-vertex normal from local PCA of the k-NN neighbourhood (smallest principal
        direction). Robust and independent of face winding; outward-oriented if
        ``orient_normals``. Returns [N,3] unit vectors."""
        points = np.asarray(points, dtype=np.float64)
        n = len(points)
        k = min(k or self.k, n - 1)
        _, idx = cKDTree(points).query(points, k=k + 1)
        idx = np.atleast_2d(idx)
        vn = np.zeros_like(points)
        for i in range(n):
            P = points[idx[i]] - points[idx[i]].mean(0)
            _, _, Vt = np.linalg.svd(P, full_matrices=False)
            vn[i] = Vt[-1]                                    # smallest-variance direction
        vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
        if self.orient_normals:
            flip = np.sign((vn * (points - points.mean(0))).sum(1, keepdims=True))
            flip[flip == 0] = 1.0
            vn *= flip
        return vn

    def vertex_normals(self, points, faces) -> np.ndarray:
        """Per-vertex normal as the area-weighted average of incident face normals. Only
        meaningful with a consistently-wound mesh; for a stitched point-cloud mesh prefer
        :meth:`pca_normals`. Returns [N,3] unit vectors."""
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        vn = np.zeros_like(points)
        if len(faces) == 0:
            return vn
        v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)                       # area-weighted (unnormalised)
        np.add.at(vn, faces.reshape(-1), np.repeat(fn, 3, axis=0))
        vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
        if self.orient_normals:
            flip = np.sign((vn * (points - points.mean(0))).sum(1, keepdims=True))
            flip[flip == 0] = 1.0
            vn *= flip
        return vn

    def process(self, points) -> dict:
        """-> dict with ``faces`` [F,3] (reconstructed connectivity, for anyone who needs
        a mesh) plus ``areas`` [N] and ``normals`` [N,3] from :class:`SurfaceMeasure`.

        The areas/normals come from the point cloud directly (partition-of-unity Voronoi
        areas + PCA normals), so they do NOT depend on the triangulation quality -- an
        imperfect reconstruction can no longer distort the area/normal fields, and
        ``sum_i areas[i] ~= surface area`` by construction."""
        from src.preprocessing.surface_measure import SurfaceMeasure
        faces = self.triangulate(points)
        areas, normals = SurfaceMeasure(
            k=self.k, orient="outward" if self.orient_normals else None)(points)
        return {"faces": faces, "areas": areas, "normals": normals}

    def save_vtp(self, points, faces, path):
        """Write the reconstructed mesh (points + triangles) as a ``.vtp`` for inspection.
        Lazily imports ``vtk`` so the class stays dependency-light when unused."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        vpoints = vtk.vtkPoints()
        vpoints.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=1))
        polys = vtk.vtkCellArray()
        for f in faces:
            polys.InsertNextCell(3, [int(f[0]), int(f[1]), int(f[2])])
        poly = vtk.vtkPolyData()
        poly.SetPoints(vpoints)
        poly.SetPolys(polys)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(poly)
        writer.Write()
        return path

    # ------------------------------------------------------------- backends #
    def _triangulate_local_delaunay(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        k = min(self.k, n - 1)
        if k < 2:
            return np.zeros((0, 3), dtype=np.int64)
        _, idx = cKDTree(points).query(points, k=k + 1)       # [n, k+1], idx[:,0]==self
        idx = np.atleast_2d(idx)
        faces = set()
        for i in range(n):
            nbr = idx[i]                                       # global indices, nbr[0]==i
            P = points[nbr] - points[i]                        # centre at i
            # Local tangent plane: top-2 right singular vectors (PCA of the neighbourhood).
            try:
                _, _, Vt = np.linalg.svd(P, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            proj = P @ Vt[:2].T                                # [k+1, 2] tangent coords
            try:
                simplices = Delaunay(proj).simplices
            except Exception:
                continue                                       # degenerate patch -> skip
            for tri in simplices:
                if 0 in tri:                                   # keep the centre's "umbrella"
                    g = tuple(sorted(int(nbr[t]) for t in tri))
                    if g[0] != g[1] and g[1] != g[2] and g[0] != g[2]:
                        faces.add(g)
        if not faces:
            return np.zeros((0, 3), dtype=np.int64)
        faces = np.array(sorted(faces), dtype=np.int64)
        if self.manifold_cleanup:
            faces = self._prune_to_manifold(faces)
        return faces

    @staticmethod
    def _prune_to_manifold(faces: np.ndarray) -> np.ndarray:
        """Greedily drop faces so every undirected edge has <=2 incident faces. Removes the
        overlapping double-diagonal triangles that stitched local Delaunay produces on
        near-cocircular (e.g. gridded) samplings; a no-op on a clean manifold. Deterministic
        (faces are pre-sorted)."""
        edge_count: dict = {}
        kept = []
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            edges = ((a, b) if a < b else (b, a),
                     (b, c) if b < c else (c, b),
                     (a, c) if a < c else (c, a))
            if all(edge_count.get(e, 0) < 2 for e in edges):
                kept.append(f)
                for e in edges:
                    edge_count[e] = edge_count.get(e, 0) + 1
        return np.array(kept, dtype=np.int64) if kept else np.zeros((0, 3), dtype=np.int64)

    def _triangulate_open3d(self, points: np.ndarray) -> np.ndarray:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(self.k)
        d = np.asarray(pcd.compute_nearest_neighbor_distance())
        r = 1.5 * float(d.mean())
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([r, 2 * r]))
        return np.asarray(mesh.triangles, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Self-test: reconstruct primitives and check total area vs. the analytic value.
# --------------------------------------------------------------------------- #
def _fibonacci_sphere(n, r=1.0):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return r * np.stack([np.sin(phi) * np.cos(theta),
                         np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _torus_points(n, R=1.0, r=0.35):
    g = int(round(n ** 0.5))
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, g, endpoint=False),
                       np.linspace(0, 2 * np.pi, g, endpoint=False), indexing='ij')
    u, v = u.ravel(), v.ravel()
    return np.stack([(R + r * np.cos(v)) * np.cos(u),
                     (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], axis=1)


if __name__ == '__main__':
    tri = SurfaceTriangulator(k=16)

    P = _fibonacci_sphere(2000, r=1.3)
    out = tri.process(P)
    total, analytic = out["areas"].sum(), 4 * np.pi * 1.3 ** 2
    outward = float(np.mean(np.sum(out["normals"] * (P / np.linalg.norm(P, axis=1, keepdims=True)), axis=1)))
    print(f"[sphere] faces={len(out['faces'])}  area={total:.3f}  analytic={analytic:.3f}  "
          f"rel_err={abs(total-analytic)/analytic:.2%}  normal·outward={outward:.3f}")

    P = _torus_points(2000, R=1.0, r=0.35)
    out = tri.process(P)
    total, analytic = out["areas"].sum(), (2 * np.pi * 1.0) * (2 * np.pi * 0.35)
    print(f"[torus ] faces={len(out['faces'])}  area={total:.3f}  analytic={analytic:.3f}  "
          f"rel_err={abs(total-analytic)/analytic:.2%}")
