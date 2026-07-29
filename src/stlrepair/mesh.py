"""STL loading, welding and saving."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import topology

# STL stores every triangle independently in float32, so vertices that should
# coincide rarely match bit for bit. This relative tolerance decides how close
# is "the same point".
DEFAULT_WELD_TOL = 1e-6


def load_stl(source, file_type: str = "stl"):
    """Read an STL into raw (vertices, faces) with no processing applied."""
    if isinstance(source, (str, Path)):
        loaded = trimesh.load(str(source), file_type=file_type, process=False)
    else:
        loaded = trimesh.load(source, file_type=file_type, process=False)

    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    return vertices, faces


def detect_format(source) -> str:
    """Return "ascii" or "binary" for an STL.

    An ASCII file starts with "solid", but so do plenty of binary ones written
    by careless exporters. The reliable test is whether the triangle count in
    the header accounts for the actual file length.
    """
    try:
        if isinstance(source, (str, Path)):
            with open(source, "rb") as handle:
                head = handle.read(84)
                size = Path(source).stat().st_size
        else:
            position = source.tell()
            source.seek(0, 2)
            size = source.tell()
            source.seek(0)
            head = source.read(84)
            source.seek(position)
    except OSError:
        return "unknown"

    if len(head) >= 84:
        count = int.from_bytes(head[80:84], "little")
        if size == 84 + count * 50:
            return "binary"
    return "ascii" if head[:5].lower() == b"solid" else "binary"


def save_stl(path, vertices: np.ndarray, faces: np.ndarray, ascii_mode: bool = False):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    data = trimesh.exchange.stl.export_stl_ascii(mesh) if ascii_mode else (
        trimesh.exchange.stl.export_stl(mesh)
    )
    mode = "w" if ascii_mode else "wb"
    with open(path, mode) as handle:
        handle.write(data)


def export_bytes(vertices: np.ndarray, faces: np.ndarray, ascii_mode: bool = False):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if ascii_mode:
        return trimesh.exchange.stl.export_stl_ascii(mesh).encode("utf-8")
    return trimesh.exchange.stl.export_stl(mesh)


def bbox_diagonal(vertices: np.ndarray) -> float:
    if len(vertices) == 0:
        return 0.0
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))


def weld(vertices: np.ndarray, faces: np.ndarray, tol: float | None = None):
    """Merge coincident vertices using a true radius query.

    A rounding grid is the usual shortcut, but it splits points that sit either
    side of a cell boundary. Clustering by actual distance avoids that.
    """
    if len(vertices) == 0:
        return vertices, faces

    if tol is None:
        tol = max(bbox_diagonal(vertices), 1.0) * DEFAULT_WELD_TOL
    if tol <= 0:
        return vertices, faces

    tree = cKDTree(vertices)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return vertices, faces

    labels = topology.label_graph(pairs[:, 0], pairs[:, 1], len(vertices))

    merged = np.zeros((labels.max() + 1, 3), dtype=np.float64)
    np.add.at(merged, labels, vertices)
    counts = np.bincount(labels, minlength=len(merged))[:, None]
    merged /= counts

    return merged, labels[faces]


def drop_unreferenced(vertices: np.ndarray, faces: np.ndarray):
    if len(faces) == 0:
        return np.zeros((0, 3)), faces
    used, remapped = np.unique(faces, return_inverse=True)
    return vertices[used], remapped.reshape(faces.shape)
