"""Per-vertex surface areas + normals from a (dense) surface point cloud.

Turns an unstructured surface sample into quadrature weights, so a sum over vertices
approximates a surface integral::

    sum_i  area_i * f(x_i)   ~=   integral_S  f dA

That makes vertex-wise operations (e.g. area-weighted graph-conv pooling) resolution-
and density-invariant: denser sampling gives each point a smaller area, so the weighted
sum stays put instead of over-counting dense regions. No mesh connectivity and no
known/analytic area are required.

Method (per point, local, scipy-only):
  * the k nearest neighbours define a local surface patch;
  * PCA of the patch gives the unit NORMAL (smallest-variance direction) and a tangent
    basis;
  * the neighbours are projected to that tangent plane and the point's 2D Voronoi-cell
    area is its own surface area. Voronoi cells tile the plane, so the areas form a
    partition of unity -- they sum to ~the surface area BY CONSTRUCTION, which is why
    no analytic reference is needed to be faithful.

Usage (offline, once per cloud)::

    areas, normals = SurfaceMeasure(k=16)(points)   # points [N,3]
"""

import numpy as np
from scipy.spatial import cKDTree, Voronoi


def _polygon_area(verts):
    """Area of a convex polygon given its (unordered) vertices [M,2]."""
    c = verts.mean(0)
    order = np.argsort(np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0]))
    v = verts[order]
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class SurfaceMeasure:
    """Faithful per-vertex areas (quadrature weights) and normals for a surface cloud.

    :param k:       neighbourhood size for the local patch (16-32 works for dense clouds)
    :param orient:  'outward' flips normals to point away from the centroid (fine for
                    star-convex shapes), or None to leave the PCA sign as-is.
    """

    def __init__(self, k: int = 16, orient: str = "outward"):
        self.k = int(k)
        self.orient = orient

    def __call__(self, points):
        return self.compute(points)

    def compute(self, points):
        """points [N,3] -> (areas [N], normals [N,3])."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected points [N,3], got {points.shape}")
        n = len(points)
        k = min(self.k, n - 1)
        _, idx = cKDTree(points).query(points, k=k + 1)     # [n, k+1], col 0 == self
        idx = np.atleast_2d(idx)

        areas = np.zeros(n)
        normals = np.zeros((n, 3))
        for i in range(n):
            P = points[idx[i]] - points[i]                  # centre at i -> [k+1, 3]
            _, _, Vt = np.linalg.svd(P, full_matrices=False)
            normals[i] = Vt[-1]                             # smallest-variance direction
            uv = P @ Vt[:2].T                               # project to tangent plane
            areas[i] = self._cell_area(uv)

        normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
        if self.orient == "outward":
            s = np.sign((normals * (points - points.mean(0))).sum(1))
            s[s == 0] = 1.0
            normals *= s[:, None]
        return areas, normals

    @staticmethod
    def _cell_area(uv):
        """Voronoi-cell area of the centre point (row 0, at the origin) among its
        tangent-plane neighbours. Falls back to a local-density disk estimate when the
        cell is unbounded or DEGENERATE.

        Degeneracy matters at sharp features (edges/corners): there the neighbourhood
        straddles two faces, its tangent-plane projection is near-collinear, and the
        Voronoi vertices (sliver circumcenters) fly off toward infinity, inflating the
        cell area by orders of magnitude. A valid local cell is bounded by the nearest
        ring, so any cell vertex beyond the neighbourhood radius flags a degeneracy."""
        d = np.linalg.norm(uv[1:], axis=1)
        r_max = float(d.max()) if len(d) else 0.0
        try:
            vor = Voronoi(uv)
            region = vor.regions[vor.point_region[0]]
            if region and -1 not in region:
                V = vor.vertices[region]
                if len(V) and np.linalg.norm(V, axis=1).max() <= r_max:
                    return _polygon_area(V)
        except Exception:
            pass
        # fallback: k points fill the disk out to the farthest neighbour -> area / k each.
        return float(np.pi * r_max ** 2 / len(d)) if len(d) else 0.0
