"""Topology primitives: edge classification, boundary loops, winding, shells.

Everything here works on a plain (vertices, faces) pair so the diagnostics and
the repair passes can share one implementation and never disagree.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _csgraph_components

EDGE_CYCLE = ((0, 1), (1, 2), (2, 0))


def directed_edges(faces: np.ndarray) -> np.ndarray:
    """Every half-edge (a -> b) in face order, shape (3F, 2)."""
    return np.vstack([faces[:, [i, j]] for i, j in EDGE_CYCLE])


def edge_table(faces: np.ndarray):
    """Group half-edges by their undirected key.

    Returns (keys, inverse, counts) where ``keys[inverse[k]]`` is the undirected
    edge of half-edge ``k`` and ``counts`` is how many faces touch each edge.

    Each edge is packed into a single integer before sorting. Row-wise
    ``np.unique`` on a (3F, 2) array does a lexicographic sort that dominates
    the runtime on large meshes; sorting one integer column does not.
    """
    if len(faces) == 0:
        empty = np.zeros((0, 2), dtype=np.int64)
        return empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    de = directed_edges(faces)
    low = de.min(axis=1).astype(np.int64)
    high = de.max(axis=1).astype(np.int64)
    stride = np.int64(faces.max()) + 1

    codes, inverse, counts = np.unique(
        low * stride + high, return_inverse=True, return_counts=True
    )
    keys = np.column_stack([codes // stride, codes % stride])
    return keys, inverse.ravel(), counts


def _halfedge_groups(inverse: np.ndarray, n_faces: int):
    """Pairs of half-edges that share an undirected edge, as face indices.

    Sorting by edge id puts every group together, so consecutive entries within
    a group give the adjacency links without a Python loop.
    """
    face_of_halfedge = np.tile(np.arange(n_faces), 3)
    order = np.argsort(inverse, kind="stable")
    sorted_edges = inverse[order]
    same_group = sorted_edges[1:] == sorted_edges[:-1]
    left = face_of_halfedge[order][:-1][same_group]
    right = face_of_halfedge[order][1:][same_group]
    return left, right, order, face_of_halfedge


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
    )


def face_heights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Shortest altitude of each triangle: 2 * area / longest edge.

    Height catches slivers that area alone misses, since a long thin needle can
    still have a respectable area while being geometrically degenerate.
    """
    tris = vertices[faces]
    edges = np.linalg.norm(
        np.stack(
            [tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 1], tris[:, 0] - tris[:, 2]],
            axis=1,
        ),
        axis=2,
    )
    longest = edges.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        height = np.where(longest > 0, 2.0 * face_areas(vertices, faces) / longest, 0.0)
    return height


def degenerate_mask(vertices: np.ndarray, faces: np.ndarray, height_tol: float):
    """Faces with a repeated vertex or a vanishing altitude."""
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    return repeated | (face_heights(vertices, faces) <= height_tol)


def duplicate_mask(faces: np.ndarray) -> np.ndarray:
    """Mark the *extra* copies of any face that appears more than once.

    Two faces count as duplicates when they use the same three vertices,
    regardless of winding: a back-to-back pair is a zero-thickness sliver, not
    real surface. The first occurrence is kept, later ones are marked.
    """
    keys = np.sort(faces, axis=1)
    _, first_index, inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    dup = counts[inverse] > 1
    dup[first_index[counts > 1]] = False
    return dup


def label_graph(rows, cols, n_nodes: int) -> np.ndarray:
    """Connected-component labels for an undirected graph given as edge lists."""
    if n_nodes == 0:
        return np.zeros(0, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_nodes, n_nodes),
    )
    _, labels = _csgraph_components(graph, directed=False)
    return labels.astype(np.int64)


def connected_components(faces: np.ndarray, n_faces: int | None = None) -> np.ndarray:
    """Label each face with a shell id, connecting faces across shared edges."""
    n_faces = len(faces) if n_faces is None else n_faces
    if n_faces == 0:
        return np.zeros(0, dtype=np.int64)

    _, inverse, _ = edge_table(faces)
    left, right, _, _ = _halfedge_groups(inverse, n_faces)
    return label_graph(left, right, n_faces)


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Chain naked half-edges into closed vertex loops (one per hole)."""
    keys, inverse, counts = edge_table(faces)
    naked = counts[inverse] == 1
    if not naked.any():
        return []

    de = directed_edges(faces)[naked]

    successors: dict[int, list[int]] = {}
    for a, b in de:
        successors.setdefault(int(a), []).append(int(b))

    loops: list[list[int]] = []
    for start in list(successors):
        while successors.get(start):
            loop = [start]
            current = successors[start].pop()
            guard = 0
            limit = len(de) + 2
            while current != start and guard < limit:
                loop.append(current)
                options = successors.get(current)
                if not options:
                    loop = []
                    break
                current = options.pop()
                guard += 1
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def loop_planarity(vertices: np.ndarray, loop: list[int]) -> float:
    """Out-of-plane deviation of a loop, relative to its own size.

    0 means perfectly planar. Compared against a small threshold to split
    "planar holes" from "non-planar holes".
    """
    pts = vertices[loop]
    centred = pts - pts.mean(axis=0)
    scale = np.linalg.norm(centred, axis=1).max()
    if scale <= 0:
        return 0.0
    # Smallest singular direction is the plane normal; its magnitude is the
    # residual thickness of the point cloud.
    singular = np.linalg.svd(centred, compute_uv=False)
    return float(singular[-1] / (scale * np.sqrt(len(loop))))


def winding_flips(faces: np.ndarray) -> np.ndarray:
    """Faces whose winding disagrees with their shell.

    Returns a boolean mask of faces to reverse so every shell becomes
    internally consistent.

    Two faces sharing an edge agree when they traverse it in opposite
    directions. That makes this a 2-colouring: give every face a "kept" node
    and a "reversed" node, join the states that can coexist, and one pass of
    connected components settles the whole surface at once. A breadth-first
    walk would need a Python loop per face.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return np.zeros(0, dtype=bool)

    _, inverse, _ = edge_table(faces)
    left, right, order, face_of_halfedge = _halfedge_groups(inverse, n_faces)
    if len(left) == 0:
        return np.zeros(n_faces, dtype=bool)

    de = directed_edges(faces)
    forward = de[:, 0] < de[:, 1]
    sorted_forward = forward[order]
    same_group = inverse[order][1:] == inverse[order][:-1]
    # Same traversal direction on a shared edge means exactly one must flip.
    conflict = (sorted_forward[:-1] == sorted_forward[1:])[same_group]

    keep, flipped = left, left + n_faces
    other_keep = np.where(conflict, right + n_faces, right)
    other_flip = np.where(conflict, right, right + n_faces)

    labels = label_graph(
        np.concatenate([keep, flipped]),
        np.concatenate([other_keep, other_flip]),
        2 * n_faces,
    )

    shells = label_graph(left, right, n_faces)
    flip = np.zeros(n_faces, dtype=bool)

    # One representative per shell defines "not flipped" for that shell.
    representatives = np.full(int(shells.max()) + 1, -1, dtype=np.int64)
    representatives[shells[::-1]] = np.arange(n_faces)[::-1]

    anchor = representatives[shells]
    orientable = labels[anchor] != labels[anchor + n_faces]
    flip[orientable] = (labels[np.arange(n_faces) + n_faces] == labels[anchor])[
        orientable
    ]
    return flip


def orientation_flip_mask(vertices: np.ndarray, faces: np.ndarray, labels=None):
    """Which faces must be reversed so every shell is consistent and outward.

    Two things can be wrong at once: individual faces disagreeing with their
    neighbours, and a whole shell being inside-out. The breadth-first walk
    settles the first; the sign of the enclosed volume settles the second. Open
    shells have no meaningful volume, so those fall back to majority rule.
    """
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)

    labels = connected_components(faces) if labels is None else labels
    result = np.zeros(len(faces), dtype=bool)

    for shell in range(int(labels.max()) + 1):
        member = labels == shell
        shell_faces = faces[member]
        flip = winding_flips(shell_faces)

        corrected = shell_faces.copy()
        corrected[flip] = corrected[flip][:, ::-1]

        _, _, counts = edge_table(corrected)
        closed = not (counts == 1).any()

        if closed:
            inside_out = shell_signed_volume(vertices, corrected) < 0
        else:
            inside_out = flip.sum() > len(flip) / 2

        result[member] = ~flip if inside_out else flip

    return result


def shell_signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume; negative means the surface is inside-out."""
    if len(faces) == 0:
        return 0.0
    tris = vertices[faces]
    return float(
        np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0
    )
