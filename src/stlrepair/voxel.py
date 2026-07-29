"""Last-resort repair: rebuild the model as a solid from a voxel grid.

This always yields a watertight, self-intersection-free result because the
surface is extracted from a filled volume rather than patched. The trade-off is
that detail is resampled at the voxel size, so it only runs when the
conservative and boolean tiers have both failed.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from . import topology

PAD = 4

# Voxelising a surface marks every cell the surface passes through, which
# inflates the solid by roughly half a cell. Extracting the isosurface half a
# cell in puts it back where it belongs.
SURFACE_OFFSET = -0.5


def voxel_remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 256,
    smoothing: float = 0.8,
    close_gaps: int = 2,
    notes: list | None = None,
):
    """Return (vertices, faces) of a watertight solid approximating the input.

    ``resolution``  voxels across the model's longest axis.
    ``close_gaps``  starting radius, in voxels, for the morphological closing
                    that bridges holes. Raised automatically until the interior
                    flood fill stops leaking out through an opening.
    ``smoothing``   gaussian blur on the distance field, in voxels.
    ``notes``       optional list; receives a line describing what was needed.
    """
    if len(faces) == 0:
        raise ValueError("cannot remesh an empty mesh")

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    longest = float(extent.max())
    if longest <= 0:
        raise ValueError("mesh has no extent")

    pitch = longest / max(int(resolution), 8)

    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    grid = trimesh.voxel.creation.voxelize_subdivide(surface, pitch=pitch, max_iter=12)

    max_close = max(close_gaps, int(resolution) // 6)
    pad = PAD + max_close
    shell = np.pad(np.asarray(grid.matrix, dtype=bool), pad, constant_values=False)

    solid, used, sealed = _fill_interior(shell, close_gaps, max_close)
    if notes is not None:
        if sealed:
            notes.append(f"voxel fill sealed openings with a {used}-voxel closing")
        else:
            notes.append(
                "voxel fill found no enclosed interior; thickened the surface "
                f"by {used} voxels instead"
            )

    # Signed distance in voxel units: negative inside, positive outside. This
    # is far smoother than a 0/1 field, so the extracted surface is too.
    sdf = ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)
    if smoothing > 0:
        sdf = ndimage.gaussian_filter(sdf, sigma=smoothing)

    level = SURFACE_OFFSET
    if sdf.min() >= level or sdf.max() <= level:
        raise ValueError("voxel field has no surface to extract")

    verts, tris, _, _ = measure.marching_cubes(
        sdf, level=level, spacing=(pitch, pitch, pitch)
    )

    origin = np.asarray(grid.transform)[:3, 3]
    verts = verts + origin - pad * pitch

    tris = tris.astype(np.int64)
    if topology.shell_signed_volume(verts, tris) < 0:
        tris = tris[:, ::-1]

    return verts, tris


def _closing_radii(start: int, limit: int) -> list[int]:
    """Radii to try, growing geometrically so wide holes are reached quickly."""
    radii, r = [], max(start, 1)
    while r <= limit:
        radii.append(r)
        r = max(r + 1, int(r * 1.6))
    if not radii:
        radii = [max(start, 1)]
    return radii


def _fill_interior(shell: np.ndarray, close_gaps: int, max_close: int):
    """Everything not reachable from outside is material.

    Dilating first bridges holes so the flood fill cannot leak inside; eroding
    afterwards puts the boundary back. The radius needed depends on how wide
    the holes are, which is not known up front, so it escalates until the fill
    actually encloses something.

    Returns ``(solid, radius_used, sealed)``. ``sealed`` is False when no
    enclosed interior could be found at any radius, which means the input is
    genuinely a surface rather than a solid.
    """
    # One distance transform serves every radius: dilating by r is the same as
    # taking every voxel within distance r of the shell.
    outward = ndimage.distance_transform_edt(~shell)

    for radius in _closing_radii(close_gaps, max_close):
        working = outward <= radius
        labels, _ = ndimage.label(~working)
        solid = ~(labels == labels[0, 0, 0])

        if not (solid & ~working).any():
            continue  # fill leaked straight through; widen the closing

        inward = ndimage.distance_transform_edt(solid)
        shrunk = inward > radius
        return ((shrunk | shell) if shrunk.any() else solid), radius, True

    thickness = max(close_gaps, 2)
    thickened = outward <= thickness
    if not thickened.any():
        raise ValueError("voxel fill produced an empty solid")
    return thickened, thickness, False
