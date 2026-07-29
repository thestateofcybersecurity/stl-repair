"""Deliberately broken meshes, one per defect the tool claims to detect."""

from __future__ import annotations

import numpy as np

CUBE_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)

# Outward-facing, counter-clockwise seen from outside.
CUBE_FACES = np.array(
    [
        [0, 3, 2], [0, 2, 1],  # bottom (z=0), normal -Z
        [4, 5, 6], [4, 6, 7],  # top (z=1), normal +Z
        [0, 1, 5], [0, 5, 4],  # front (y=0), normal -Y
        [2, 3, 7], [2, 7, 6],  # back (y=1), normal +Y
        [1, 2, 6], [1, 6, 5],  # right (x=1), normal +X
        [3, 0, 4], [3, 4, 7],  # left (x=0), normal -X
    ],
    dtype=np.int64,
)


def good_cube():
    return CUBE_VERTICES.copy(), CUBE_FACES.copy()


def unwelded_cube():
    """Every triangle with its own vertices, exactly as an STL file stores it."""
    v, f = good_cube()
    return v[f].reshape(-1, 3), np.arange(len(f) * 3).reshape(-1, 3)


def cube_with_hole(n_faces: int = 1):
    """Drop faces to open a square hole in the bottom."""
    v, f = good_cube()
    return v, f[n_faces:]


def cube_with_flipped_face(count: int = 1):
    v, f = good_cube()
    f[:count] = f[:count][:, ::-1]
    return v, f


def inside_out_cube():
    v, f = good_cube()
    return v, f[:, ::-1]


def cube_with_duplicate_face():
    v, f = good_cube()
    return v, np.vstack([f, f[3:4]])


def cube_with_degenerate_face():
    """Append a needle triangle with three near-collinear points."""
    v, f = good_cube()
    v = np.vstack([v, [[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.55, 0.5 + 1e-12, 0.5]]])
    extra = np.array([[8, 9, 10]], dtype=np.int64)
    return v, np.vstack([f, extra])


def cube_with_non_manifold_edge():
    """A stray fin sharing one cube edge, giving that edge three faces."""
    v, f = good_cube()
    v = np.vstack([v, [[0.5, -1.0, 0.5]]])
    extra = np.array([[0, 1, 8]], dtype=np.int64)
    return v, np.vstack([f, extra])


def two_disjoint_cubes(offset=(3.0, 0.0, 0.0)):
    v, f = good_cube()
    v2 = v + np.asarray(offset)
    return np.vstack([v, v2]), np.vstack([f, f + len(v)])


def cube_with_speck():
    """A main cube plus a tiny floating shell, the classic scan artefact."""
    v, f = good_cube()
    v2 = v * 0.02 + np.array([5.0, 0.0, 0.0])
    return np.vstack([v, v2]), np.vstack([f, f + len(v)])


def overlapping_cubes():
    """Two cubes sharing volume: shells intersect, needing a real boolean."""
    v, f = good_cube()
    v2 = v + np.array([0.5, 0.5, 0.5])
    return np.vstack([v, v2]), np.vstack([f, f + len(v)])


def self_intersecting_pair():
    """Two triangles crossing through each other, sharing no vertex."""
    v = np.array(
        [
            [0, 0, 0], [2, 0, 0], [0, 2, 0],
            [1, -1, -1], [1, -1, 1], [1, 1, 0.5],
        ],
        dtype=np.float64,
    )
    f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return v, f


def sphere(subdivisions: int = 2, radius: float = 1.0):
    """A clean reference solid with many more triangles than a cube."""
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def punctured_sphere(subdivisions: int = 2, n_faces: int = 6):
    """A sphere with a ragged, non-planar hole cut into it."""
    v, f = sphere(subdivisions)
    centre = v[f].mean(axis=1)
    order = np.argsort(-centre[:, 2])
    keep = np.ones(len(f), dtype=bool)
    keep[order[:n_faces]] = False
    return v, f[keep]
