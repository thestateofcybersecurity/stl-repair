"""Self-intersection detection and hole triangulation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Broad-phase safety valve: past this many candidate pairs we report the check
# as skipped rather than grinding for minutes on a pathological mesh.
MAX_CANDIDATE_PAIRS = 5_000_000

# Triangles more than 2**MAX_SIZE_LEVELS smaller than the largest share a
# bucket; beyond that the extra buckets cost more than they save.
MAX_SIZE_LEVELS = 12


def _size_buckets(radii: np.ndarray) -> list[np.ndarray]:
    """Split triangles into power-of-two size classes.

    Real print meshes mix a few large flat faces with thousands of tiny ones.
    Searching everything at the largest triangle's radius makes the small ones
    return enormous candidate lists, so each class is searched at its own scale.
    """
    largest = float(radii.max())
    if largest <= 0:
        return [np.arange(len(radii))]

    safe = np.maximum(radii, largest * 1e-9)
    level = np.floor(np.log2(safe / largest)).astype(np.int64)
    level = np.clip(level, -MAX_SIZE_LEVELS, 0)
    return [np.flatnonzero(level == lv) for lv in np.unique(level)]


def _candidate_pairs(vertices: np.ndarray, faces: np.ndarray):
    """Triangle pairs whose bounding spheres overlap, excluding neighbours."""
    tris = vertices[faces]
    centres = tris.mean(axis=1)
    radii = np.linalg.norm(tris - centres[:, None, :], axis=2).max(axis=1)

    # Bounding boxes are far tighter than the centroid spheres the tree has to
    # search with, so each chunk is narrowed the moment it comes back rather
    # than accumulating millions of rows only to throw most of them away.
    low, high = tris.min(axis=1), tris.max(axis=1)

    def narrow(found):
        a, b = found[:, 0], found[:, 1]
        separated = (low[a] > high[b]) | (low[b] > high[a])
        return found[~separated.any(axis=1)]

    buckets = [b for b in _size_buckets(radii) if len(b)]
    trees = [cKDTree(centres[b]) for b in buckets]
    reach = [float(radii[b].max()) for b in buckets]

    collected = []
    total = 0
    for i, (idx_a, tree_a) in enumerate(zip(buckets, trees)):
        for j in range(i, len(buckets)):
            idx_b, tree_b = buckets[j], trees[j]
            span = reach[i] + reach[j]

            if i == j:
                local = tree_a.query_pairs(span, output_type="ndarray")
                if not len(local):
                    continue
                found = np.column_stack([idx_a[local[:, 0]], idx_a[local[:, 1]]])
            else:
                sparse = tree_a.sparse_distance_matrix(
                    tree_b, span, output_type="ndarray"
                )
                if not len(sparse):
                    continue
                found = np.column_stack([idx_a[sparse["i"]], idx_b[sparse["j"]]])

            total += len(found)
            if total > MAX_CANDIDATE_PAIRS:
                return None

            found = narrow(found)
            if len(found):
                collected.append(found)

    if not collected:
        return np.zeros((0, 2), dtype=np.int64)

    pairs = np.sort(np.vstack(collected), axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return pairs

    # Dedupe on a packed integer key; a row-wise unique would lexsort instead.
    stride = np.int64(len(faces))
    _, keep = np.unique(pairs[:, 0] * stride + pairs[:, 1], return_index=True)
    pairs = pairs[keep]

    # Triangles sharing a vertex always touch; that is not a defect.
    a, b = faces[pairs[:, 0]], faces[pairs[:, 1]]
    shares = (a[:, :, None] == b[:, None, :]).any(axis=(1, 2))
    return pairs[~shares]


def _plane_distances(tris: np.ndarray, other: np.ndarray):
    """Signed distance of each vertex of ``tris`` to the plane of ``other``."""
    normal = np.cross(other[:, 1] - other[:, 0], other[:, 2] - other[:, 0])
    offset = -np.einsum("ij,ij->i", normal, other[:, 0])
    dist = np.einsum("mij,mj->mi", tris, normal) + offset[:, None]
    return normal, dist


def _line_interval(tris: np.ndarray, dist: np.ndarray, direction: np.ndarray, eps):
    """Parameter range where a triangle crosses the other triangle's plane.

    Each vertex sitting on the plane contributes directly; each edge straddling
    the plane contributes its crossing point. Both are projected onto the
    intersection line so the two triangles can be compared in 1-D.
    """
    proj = np.einsum("mij,mj->mi", tris, direction)
    on_plane = np.abs(dist) <= eps[:, None]

    values = np.where(on_plane, proj, np.nan)
    candidates = [values]

    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = dist[:, i], dist[:, j]
        crosses = (di * dj < 0) & ~on_plane[:, i] & ~on_plane[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = proj[:, i] + (proj[:, j] - proj[:, i]) * di / (di - dj)
        candidates.append(np.where(crosses, t, np.nan)[:, None])

    stacked = np.hstack(candidates)
    valid = ~np.isnan(stacked).all(axis=1)
    lo = np.full(len(tris), np.nan)
    hi = np.full(len(tris), np.nan)
    if valid.any():
        lo[valid] = np.nanmin(stacked[valid], axis=1)
        hi[valid] = np.nanmax(stacked[valid], axis=1)
    return lo, hi


def count_self_intersections(vertices: np.ndarray, faces: np.ndarray):
    """Number of intersecting triangle pairs, or None if the check was skipped.

    Coplanar overlaps are deliberately not counted: adjacent flat surfaces
    produce them constantly and they are not what breaks a slicer.
    """
    if len(faces) < 2:
        return 0

    pairs = _candidate_pairs(vertices, faces)
    if pairs is None:
        return None
    if len(pairs) == 0:
        return 0

    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    eps = max(scale, 1.0) * 1e-9

    ta = vertices[faces[pairs[:, 0]]]
    tb = vertices[faces[pairs[:, 1]]]

    normal_b, dist_a = _plane_distances(ta, tb)
    normal_a, dist_b = _plane_distances(tb, ta)

    # Scale the tolerance per pair so it tracks the size of the triangles.
    eps_a = np.maximum(np.linalg.norm(normal_b, axis=1) * eps, np.finfo(float).tiny)
    eps_b = np.maximum(np.linalg.norm(normal_a, axis=1) * eps, np.finfo(float).tiny)

    # Reject when one triangle sits wholly on one side of the other's plane.
    keep = ~(
        (dist_a > eps_a[:, None]).all(axis=1)
        | (dist_a < -eps_a[:, None]).all(axis=1)
        | (dist_b > eps_b[:, None]).all(axis=1)
        | (dist_b < -eps_b[:, None]).all(axis=1)
    )
    if not keep.any():
        return 0

    ta, tb = ta[keep], tb[keep]
    dist_a, dist_b = dist_a[keep], dist_b[keep]
    normal_a, normal_b = normal_a[keep], normal_b[keep]
    eps_a, eps_b = eps_a[keep], eps_b[keep]

    direction = np.cross(normal_a, normal_b)
    coplanar = np.linalg.norm(direction, axis=1) <= eps
    if coplanar.all():
        return 0

    lo_a, hi_a = _line_interval(ta, dist_a, direction, eps_a)
    lo_b, hi_b = _line_interval(tb, dist_b, direction, eps_b)

    overlap = (hi_a >= lo_b) & (hi_b >= lo_a)
    overlap &= ~coplanar & ~np.isnan(lo_a) & ~np.isnan(lo_b)
    return int(overlap.sum())


def _signed_area_2d(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_triangle(p, a, b, c) -> bool:
    def cross(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])

    d1, d2, d3 = cross(p, a, b), cross(p, b, c), cross(p, c, a)
    has_neg = min(d1, d2, d3) < 0
    has_pos = max(d1, d2, d3) > 0
    return not (has_neg and has_pos)


def _ear_clip(points: np.ndarray) -> list[tuple[int, int, int]] | None:
    """Classic O(n^2) ear clipping on a projected, simple polygon."""
    n = len(points)
    order = list(range(n))
    if _signed_area_2d(points) < 0:
        order.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(order) > 3 and guard < n * n:
        guard += 1
        for k in range(len(order)):
            prev = order[k - 1]
            curr = order[k]
            nxt = order[(k + 1) % len(order)]
            a, b, c = points[prev], points[curr], points[nxt]

            convex = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0
            if not convex:
                continue
            if any(
                _point_in_triangle(points[o], a, b, c)
                for o in order
                if o not in (prev, curr, nxt)
            ):
                continue

            triangles.append((prev, curr, nxt))
            order.pop(k)
            break
        else:
            return None

    if len(order) != 3:
        return None
    triangles.append((order[0], order[1], order[2]))
    return triangles


def triangulate_loop(vertices: np.ndarray, loop: list[int], next_index: int):
    """Cap one boundary loop.

    Returns ``(new_vertices, new_faces)`` in global indices, where any added
    vertex is numbered from ``next_index``. Planar-ish loops are ear clipped so
    no geometry is added; awkward loops fall back to a fan from the loop
    centroid, which is robust but adds one vertex.
    """
    loop = list(dict.fromkeys(loop))  # drop accidental repeats, keep order
    if len(loop) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    if len(loop) == 3:
        return np.zeros((0, 3)), np.array([loop], dtype=np.int64)

    pts = vertices[loop]
    centred = pts - pts.mean(axis=0)
    _, _, basis = np.linalg.svd(centred, full_matrices=True)
    projected = centred @ basis[:2].T

    fan = _ear_clip(projected)
    if fan is not None:
        local = np.array(fan, dtype=np.int64)
        return np.zeros((0, 3)), np.asarray(loop, dtype=np.int64)[local]

    centroid = pts.mean(axis=0)[None, :]
    faces = [
        (next_index, loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))
    ]
    return centroid, np.array(faces, dtype=np.int64)
