"""The eight-point mesh health check, plus a couple of printability extras."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import topology
from .geometry import count_self_intersections

# A loop whose out-of-plane residual exceeds this fraction of its own size is
# treated as a non-planar hole.
PLANARITY_TOL = 1e-4


@dataclass
class Diagnosis:
    vertex_count: int = 0
    triangle_count: int = 0

    naked_edges: int = 0
    planar_holes: int = 0
    non_planar_holes: int = 0
    non_manifold_edges: int = 0
    inverted_normals: int = 0
    duplicate_faces: int = 0
    degenerate_faces: int = 0
    disjoint_shells: int = 0

    self_intersections: int | None = None
    shell_count: int = 0
    is_watertight: bool = False
    volume: float = 0.0
    area: float = 0.0
    bounds: list = field(default_factory=list)

    # Checks in the order they are reported, mapped to their labels.
    CHECKS = (
        ("naked_edges", "Naked edges"),
        ("planar_holes", "Planar holes"),
        ("non_planar_holes", "Non-Planar holes"),
        ("non_manifold_edges", "Non-Manifold edges"),
        ("inverted_normals", "Inverted normals"),
        ("duplicate_faces", "Duplicate faces"),
        ("degenerate_faces", "Degenerate faces"),
        ("disjoint_shells", "Disjoint shells"),
    )

    @property
    def defect_total(self) -> int:
        return sum(getattr(self, name) for name, _ in self.CHECKS)

    @property
    def is_clean(self) -> bool:
        """Clean means printable.

        Disjoint shells are excluded on purpose: a model can legitimately be
        several separate bodies, so that count is reported but never treated as
        damage.
        """
        blocking = sum(
            getattr(self, name)
            for name, _ in self.CHECKS
            if name != "disjoint_shells"
        )
        return blocking == 0 and not self.self_intersections

    def to_dict(self) -> dict:
        data = asdict(self)
        data["defect_total"] = self.defect_total
        data["is_clean"] = self.is_clean
        return data

    def report_lines(self) -> list[str]:
        lines = [f"--> {getattr(self, n)} {label}" for n, label in self.CHECKS]
        if self.self_intersections is None:
            lines.append("--> ? Self-intersections (skipped, mesh too dense)")
        else:
            lines.append(f"--> {self.self_intersections} Self-intersections")
        return lines


def height_tolerance(vertices: np.ndarray) -> float:
    """Degeneracy threshold, scaled to the size of the model."""
    if len(vertices) == 0:
        return 0.0
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return max(diagonal, 1.0) * 1e-8


def diagnose(
    vertices: np.ndarray,
    faces: np.ndarray,
    check_self_intersections: bool = True,
) -> Diagnosis:
    """Inspect an already-welded mesh. Never mutates its input."""
    result = Diagnosis(vertex_count=len(vertices), triangle_count=len(faces))
    if len(faces) == 0:
        return result

    result.bounds = np.vstack([vertices.min(axis=0), vertices.max(axis=0)]).tolist()

    _, inverse, counts = topology.edge_table(faces)
    per_edge = counts
    result.naked_edges = int((per_edge == 1).sum())
    result.non_manifold_edges = int((per_edge > 2).sum())
    result.is_watertight = result.naked_edges == 0 and result.non_manifold_edges == 0

    for loop in topology.boundary_loops(faces):
        if topology.loop_planarity(vertices, loop) <= PLANARITY_TOL:
            result.planar_holes += 1
        else:
            result.non_planar_holes += 1

    result.degenerate_faces = int(
        topology.degenerate_mask(vertices, faces, height_tolerance(vertices)).sum()
    )
    result.duplicate_faces = int(topology.duplicate_mask(faces).sum())

    labels = topology.connected_components(faces)
    result.shell_count = int(labels.max()) + 1
    result.disjoint_shells = max(0, result.shell_count - 1)

    result.inverted_normals = int(
        topology.orientation_flip_mask(vertices, faces, labels).sum()
    )

    result.area = float(topology.face_areas(vertices, faces).sum())
    result.volume = (
        topology.shell_signed_volume(vertices, faces) if result.is_watertight else 0.0
    )

    if check_self_intersections:
        result.self_intersections = count_self_intersections(vertices, faces)

    return result


