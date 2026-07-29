"""Last-resort repair: rebuild the model as a solid from a voxel grid.

This always yields a watertight, self-intersection-free result because the
surface is extracted from a filled volume rather than patched. The trade-off is
that detail is resampled at the voxel size, so it only runs when the
conservative and boolean tiers have both failed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure

from . import topology

PAD = 4

# Voxelising a surface marks every cell the surface passes through, which
# inflates the solid by roughly half a cell. Extracting the isosurface half a
# cell in puts it back where it belongs.
SURFACE_OFFSET = -0.5

# Ceiling on the grid, so a request for a fine resolution on a large model
# cannot quietly ask for tens of gigabytes. Exceeding it lowers the resolution
# instead of failing.
MAX_GRID_CELLS = 48_000_000

# Sample points held in memory at once while rasterising. Keeps peak usage flat
# no matter how large the mesh or how fine the pitch.
SAMPLE_BUDGET = 2_000_000


def _barycentric_lattice(steps: int) -> np.ndarray:
    """Evenly spaced barycentric coordinates covering a triangle."""
    i, j = np.meshgrid(np.arange(steps + 1), np.arange(steps + 1), indexing="ij")
    keep = (i + j) <= steps
    a, b = i[keep], j[keep]
    return np.stack([a, b, steps - a - b], axis=1) / float(steps)


def rasterise_surface(vertices, faces, pitch, origin, dims):
    """Mark every voxel the surface passes through.

    Sampling density follows the size of each triangle, so the cost tracks
    surface area divided by pitch squared. Subdividing each triangle uniformly
    instead -- as the obvious library call does -- costs a power of four per
    level, which on a mesh mixing 15 mm triangles with a 0.1 mm pitch means tens
    of millions of sub-triangles and gigabytes of memory for a grid that only
    needs a few hundred thousand samples.
    """
    grid = np.zeros(dims, dtype=bool)
    if len(faces) == 0:
        return grid

    tris = vertices[faces]
    edges = np.linalg.norm(
        np.stack(
            [tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 1], tris[:, 0] - tris[:, 2]],
            axis=1,
        ),
        axis=2,
    )
    # Half a pitch between samples guarantees no voxel along the surface is
    # stepped over.
    steps = np.maximum(np.ceil(edges.max(axis=1) / (pitch * 0.5)), 1).astype(np.int64)
    upper = np.asarray(dims, dtype=np.int64) - 1

    for count in np.unique(steps):
        members = np.flatnonzero(steps == count)
        lattice = _barycentric_lattice(int(count))
        per_batch = max(1, SAMPLE_BUDGET // len(lattice))

        for start in range(0, len(members), per_batch):
            batch = tris[members[start:start + per_batch]]
            points = np.einsum("kb,mbc->mkc", lattice, batch).reshape(-1, 3)
            index = np.floor((points - origin) / pitch).astype(np.int64)
            np.clip(index, 0, upper, out=index)
            grid[index[:, 0], index[:, 1], index[:, 2]] = True

    return grid


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

    resolution = max(int(resolution), 8)
    max_close = max(close_gaps, resolution // 6)
    pad = PAD + max_close

    # Settle on a pitch the grid can actually hold. Padding counts towards the
    # total, so it has to be part of the sum rather than an afterthought.
    while True:
        pitch = longest / resolution
        dims = np.ceil(extent / pitch).astype(np.int64) + 2 * pad + 1
        if int(np.prod(dims)) <= MAX_GRID_CELLS or resolution <= 16:
            break
        resolution = max(16, int(resolution * 0.75))
        max_close = max(close_gaps, resolution // 6)
        pad = PAD + max_close
        if notes is not None:
            notes.append(f"voxel resolution reduced to {resolution} to bound memory")

    origin = vertices.min(axis=0) - pad * pitch
    shell = rasterise_surface(vertices, faces, pitch, origin, tuple(int(d) for d in dims))

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
    # Each transform returns float64; narrowing straight away and dropping the
    # intermediates halves what a large grid holds at once.
    sdf = ndimage.distance_transform_edt(~solid).astype(np.float32)
    inside = ndimage.distance_transform_edt(solid).astype(np.float32)
    sdf -= inside
    del inside, solid
    if smoothing > 0:
        sdf = ndimage.gaussian_filter(sdf, sigma=smoothing)

    level = SURFACE_OFFSET
    if sdf.min() >= level or sdf.max() <= level:
        raise ValueError("voxel field has no surface to extract")

    verts, tris, _, _ = measure.marching_cubes(
        sdf, level=level, spacing=(pitch, pitch, pitch)
    )

    verts = verts + origin

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
