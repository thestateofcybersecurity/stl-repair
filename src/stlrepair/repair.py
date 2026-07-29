"""The repair pipeline: three escalating tiers, each verified before accepting.

Tier 1  conservative  weld, clean, reorient, cap holes. Never resamples.
Tier 2  manifold      rebuild through manifold3d's exact boolean kernel.
Tier 3  voxel         re-derive the surface from a filled volume.

Nothing is taken on trust: after every tier the mesh is re-diagnosed, and a
tier that fails to improve matters is discarded rather than returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from . import mesh as mesh_io
from . import topology
from .diagnostics import Diagnosis, diagnose, height_tolerance
from .geometry import triangulate_loop

MODES = ("conservative", "auto", "force")
MAX_CLEANUP_ROUNDS = 4
MAX_NON_MANIFOLD_ROUNDS = 5


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {word}"


@dataclass
class RepairOptions:
    mode: str = "auto"
    weld_tol: float | None = None
    fill_holes: bool = True
    max_hole_edges: int = 0  # 0 means no limit
    resolve_non_manifold: bool = True
    min_shell_fraction: float = 0.0  # drop shells smaller than this share of the largest
    voxel_resolution: int = 256
    voxel_smoothing: float = 0.8
    check_self_intersections: bool = True


@dataclass
class RepairResult:
    vertices: np.ndarray
    faces: np.ndarray
    before: Diagnosis
    after: Diagnosis
    tier: str
    steps: list = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "tier": self.tier,
            "steps": self.steps,
            "elapsed_ms": self.elapsed_ms,
            "vertex_delta": self.after.vertex_count - self.before.vertex_count,
            "triangle_delta": self.after.triangle_count - self.before.triangle_count,
        }


# --------------------------------------------------------------------------
# Individual passes
# --------------------------------------------------------------------------


def remove_degenerate(vertices, faces):
    bad = topology.degenerate_mask(vertices, faces, height_tolerance(vertices))
    return faces[~bad], int(bad.sum())


def remove_duplicates(vertices, faces):
    bad = topology.duplicate_mask(faces)
    return faces[~bad], int(bad.sum())


def resolve_non_manifold(vertices, faces):
    """Trim the surplus faces from edges shared by more than two triangles.

    Extra faces on an edge are usually internal walls, stray fins or leftover
    geometry from a bad boolean. Two faces per edge is the most a surface can
    carry, so the surplus is dropped and any resulting hole is capped later.

    Which two to keep matters. Faces are ranked by how many clean two-face
    edges they already have, so well-attached surface wins and loose fins lose.
    """
    removed = 0
    for _ in range(MAX_NON_MANIFOLD_ROUNDS):
        _, inverse, counts = topology.edge_table(faces)
        if not (counts > 2).any():
            break

        face_of_halfedge = np.tile(np.arange(len(faces)), 3)
        attachment = np.bincount(
            face_of_halfedge[(counts == 2)[inverse]], minlength=len(faces)
        )

        order = np.argsort(inverse, kind="stable")
        sorted_edges = inverse[order]
        starts = np.flatnonzero(np.r_[True, sorted_edges[1:] != sorted_edges[:-1]])
        ends = np.r_[starts[1:], len(sorted_edges)]

        drop = set()
        for begin, end in zip(starts, ends):
            if end - begin <= 2:
                continue
            candidates = face_of_halfedge[order[begin:end]]
            ranked = candidates[np.argsort(-attachment[candidates], kind="stable")]
            drop.update(int(x) for x in ranked[2:])

        if not drop:
            break
        keep = np.ones(len(faces), dtype=bool)
        keep[list(drop)] = False
        faces = faces[keep]
        removed += len(drop)

    return faces, removed


def remove_small_shells(vertices, faces, fraction: float):
    """Discard shells that are tiny compared with the largest one.

    Size is measured by surface area, not triangle count: a scan speck can be a
    scaled-down copy with just as many triangles as the real body.
    """
    if fraction <= 0 or len(faces) == 0:
        return faces, 0

    labels = topology.connected_components(faces)
    areas = np.bincount(
        labels, weights=topology.face_areas(vertices, faces), minlength=labels.max() + 1
    )
    keep = (areas >= areas.max() * fraction)[labels]
    return faces[keep], int((~keep).sum())


def fix_orientation(vertices, faces):
    flip = topology.orientation_flip_mask(vertices, faces)
    if not flip.any():
        return faces, 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return faces, int(flip.sum())


def fill_holes(vertices, faces, max_edges: int = 0):
    """Cap every boundary loop, biggest first so nested cases settle sensibly."""
    loops = topology.boundary_loops(faces)
    if not loops:
        return vertices, faces, 0

    added_vertices = []
    added_faces = []
    filled = 0
    next_index = len(vertices)

    for loop in sorted(loops, key=len, reverse=True):
        if max_edges and len(loop) > max_edges:
            continue
        new_v, new_f = triangulate_loop(vertices, loop, next_index)
        if len(new_f) == 0:
            continue
        if len(new_v):
            added_vertices.append(new_v)
            next_index += len(new_v)
        added_faces.append(new_f)
        filled += 1

    if not added_faces:
        return vertices, faces, 0

    if added_vertices:
        vertices = np.vstack([vertices] + added_vertices)
    faces = np.vstack([faces] + added_faces)
    return vertices, faces, filled


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------


def conservative_pass(vertices, faces, options: RepairOptions):
    """Clean, reorient and cap without ever resampling the surface."""
    steps: list[str] = []

    for round_index in range(MAX_CLEANUP_ROUNDS):
        changed = False

        faces, n = remove_degenerate(vertices, faces)
        if n:
            steps.append(f"removed {_plural(n, 'degenerate face')}")
            changed = True

        faces, n = remove_duplicates(vertices, faces)
        if n:
            steps.append(f"removed {_plural(n, 'duplicate face')}")
            changed = True

        if options.resolve_non_manifold:
            faces, n = resolve_non_manifold(vertices, faces)
            if n:
                steps.append(f"trimmed {_plural(n, 'face')} off non-manifold edges")
                changed = True

        faces, n = remove_small_shells(vertices, faces, options.min_shell_fraction)
        if n:
            steps.append(f"discarded {_plural(n, 'face')} in undersized shells")
            changed = True

        faces, n = fix_orientation(vertices, faces)
        if n:
            steps.append(f"reversed {_plural(n, 'inverted face')}")
            changed = True

        if options.fill_holes:
            vertices, faces, n = fill_holes(vertices, faces, options.max_hole_edges)
            if n:
                steps.append(f"capped {_plural(n, 'hole')}")
                changed = True
                # Capping can introduce coincident geometry, so re-weld before
                # the next round judges the result.
                vertices, faces = mesh_io.weld(vertices, faces, options.weld_tol)

        if not changed:
            break
        if round_index == MAX_CLEANUP_ROUNDS - 1:
            steps.append("cleanup round limit reached")

    vertices, faces = mesh_io.drop_unreferenced(vertices, faces)
    return vertices, faces, steps


def manifold_pass(vertices, faces):
    """Rebuild through manifold3d, whose output is guaranteed intersection-free.

    Raises ``RuntimeError`` when the input is too broken for the kernel to
    accept, which is the signal to escalate to the voxel tier.
    """
    import manifold3d

    solid = _to_manifold(manifold3d, vertices, faces)
    status = solid.status()
    if status != manifold3d.Error.NoError:
        raise RuntimeError(f"manifold3d rejected the mesh: {status}")

    # Any boolean re-runs the exact kernel, which is what actually resolves
    # overlaps between shells and leftover self-intersections.
    parts = solid.decompose()
    if len(parts) > 1:
        rebuilt = manifold3d.Manifold.batch_boolean(
            list(parts), manifold3d.OpType.Add
        )
    else:
        rebuilt = solid

    if rebuilt.is_empty():
        raise RuntimeError("manifold3d produced an empty result")

    out = rebuilt.to_mesh()
    verts = np.asarray(out.vert_properties, dtype=np.float64)[:, :3]
    tris = np.asarray(out.tri_verts, dtype=np.int64)
    return verts, tris


def _to_manifold(manifold3d, vertices, faces):
    """Prefer the double-precision path so we do not lose accuracy on entry."""
    tri = np.ascontiguousarray(faces, dtype=np.uint32)
    if hasattr(manifold3d, "Mesh64"):
        try:
            m64 = manifold3d.Mesh64(
                vert_properties=np.ascontiguousarray(vertices, dtype=np.float64),
                tri_verts=tri,
            )
            m64.merge()
            return manifold3d.Manifold(m64)
        except (TypeError, RuntimeError):
            pass

    m32 = manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
        tri_verts=tri,
    )
    m32.merge()
    return manifold3d.Manifold(m32)


def voxel_pass(vertices, faces, options: RepairOptions, notes: list | None = None):
    from .voxel import voxel_remesh

    return voxel_remesh(
        vertices,
        faces,
        resolution=options.voxel_resolution,
        smoothing=options.voxel_smoothing,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _score(d: Diagnosis) -> tuple:
    """Lower is better. Used to decide whether a tier actually helped."""
    return (
        0 if d.is_clean else 1,
        d.naked_edges + d.non_manifold_edges,
        d.inverted_normals + d.duplicate_faces + d.degenerate_faces,
        d.self_intersections or 0,
    )


def repair(vertices, faces, options: RepairOptions | None = None) -> RepairResult:
    options = options or RepairOptions()
    if options.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {options.mode!r}")

    started = perf_counter()
    check_si = options.check_self_intersections

    vertices, faces = mesh_io.weld(vertices, faces, options.weld_tol)
    before = diagnose(vertices, faces, check_si)

    best_v, best_f, steps = conservative_pass(vertices, faces, options)
    best = diagnose(best_v, best_f, check_si)
    tier = "conservative"

    wants_more = options.mode == "force" or (
        options.mode == "auto" and not best.is_clean
    )

    if wants_more:
        for name, run in (
            ("manifold", lambda v, f: manifold_pass(v, f)),
            ("voxel", lambda v, f: voxel_pass(v, f, options, steps)),
        ):
            if best.is_clean and name == "voxel":
                break
            try:
                cand_v, cand_f = run(best_v, best_f)
                # Rebuilt geometry gets the same scrutiny as the original. The
                # voxel tier in particular can leave near-coincident vertices
                # at thin walls, which welding turns into pinch points.
                cand_v, cand_f = mesh_io.weld(cand_v, cand_f, options.weld_tol)
                cand_v, cand_f, extra = conservative_pass(cand_v, cand_f, options)
                steps.extend(f"{name}: {s}" for s in extra)
                candidate = diagnose(cand_v, cand_f, check_si)
            except Exception as exc:  # noqa: BLE001 - tier failure is expected
                steps.append(f"{name} tier failed: {exc}")
                continue

            if _score(candidate) < _score(best):
                best_v, best_f, best, tier = cand_v, cand_f, candidate, name
                steps.append(f"{name} tier accepted")
            else:
                steps.append(f"{name} tier rejected (no improvement)")

            if best.is_clean and options.mode != "force":
                break

    return RepairResult(
        vertices=best_v,
        faces=best_f,
        before=before,
        after=best,
        tier=tier,
        steps=steps,
        elapsed_ms=(perf_counter() - started) * 1000.0,
    )
