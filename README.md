# stl-repair

Diagnose and repair STL files locally. No uploads, no account, no network.

A self-hosted alternative to the online STL repair services, built around the same
eight checks they report, with a web UI, a batch CLI, and a repair pipeline that
escalates only as far as it has to.

```
━━━ STL REPAIR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Zap.stl
  709.7 KB · binary STL · 14,532 triangles · 2 bodies

  PROBLEMS FOUND                                                before → after
    ✓ Non-manifold edges                                          20 →       0
    ✓ Degenerate faces                                            20 →       0
    ✓ Duplicate faces                                              1 →       0

  WORTH KNOWING
    ● Disjoint shells                                              1 →       1

  ✓ 5 other checks passed
    self-intersections · naked edges · non-planar holes · planar holes
    inverted normals

  REPAIR                                            conservative tier · 957 ms
    cleaned and closed in place, no geometry resampled

    · removed 20 degenerate faces

  RESULT
      Triangles                                       14,532 →  14,512     -20
      Vertices                                         7,260 →   7,260      +0
      Watertight                                                           yes
      Volume                                                         2,386.937
      Separate bodies                                                        2

  ✓  REPAIRED   watertight and ready to print
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## What it checks

| Check | What it means |
| --- | --- |
| Naked edges | Edges with only one triangle. The surface is open here. |
| Planar holes | Boundary loops that lie flat, the easy kind to cap. |
| Non-planar holes | Boundary loops that twist out of plane, needing a fitted patch. |
| Non-manifold edges | Edges shared by three or more triangles. Slicers choke on these. |
| Inverted normals | Triangles wound the wrong way, or a whole shell turned inside out. |
| Duplicate faces | The same triangle stored more than once, including back-to-back pairs. |
| Degenerate faces | Zero-area triangles and needle slivers. |
| Disjoint shells | Separate bodies in one file. Reported, never treated as damage. |
| Self-intersections | Triangles that pass through each other. |

Problems are listed worst-first, not in a fixed order, so the thing most likely to
break a print is the first thing on screen. Checks that pass collapse into one
line; `-v` expands them.

Two deliberate choices about honesty:

- **Disjoint shells are not damage.** A model can legitimately be several bodies,
  so the count is reported under "worth knowing" and never blocks a clean verdict.
  Use `--min-shell-fraction` if you actually want specks discarded.
- **A skipped check is never reported as passed.** The self-intersection search
  gives up on pathological meshes rather than grinding for minutes. When it does,
  it says `not checked` and the verdict admits it.

## Install

Needs Python 3.10+.

```bash
git clone https://github.com/thestateofcybersecurity/stl-repair.git
cd stl-repair
./scripts/setup
```

`scripts/setup` installs the dependencies into a local `pylibs/` folder with
`pip install --target`. Nothing outside the project directory is touched and no
`sudo` is required. If you would rather use a virtualenv, the dependencies are
listed in `requirements.txt` and the scripts respect an existing `PYTHONPATH`.

## Use

### Web UI

```bash
./scripts/serve
```

Opens on <http://127.0.0.1:8765>, bound to loopback. Drop STL files anywhere on
the page, or click the box to browse. You get a before/after 3D view with the
naked edges highlighted in red, the full report, and a download button. The
viewer is a vendored copy of three.js, so the page works with the network
unplugged.

**Batches.** Drop as many files as you like. They queue and are repaired **one at
a time**, never in parallel, because each repair is CPU and memory hungry and
running several together only slows the machine down. The queue shows per-file
progress and a download link each, plus a single *Download all + report* zip at
the end.

**Per-file report.** For bulk work you need to know what was wrong with each
file, not just that the batch finished. *View* opens a table of every check for
every file, before and after, and the report downloads as CSV or JSON. The zip
carries `report.csv` and `report.json` alongside the meshes, so the record does
not get separated from the models.

*Stop after this file* halts the run at the next boundary rather than killing it
mid-repair; the remaining files stay in the queue and the button becomes
*Repair N remaining* so you can pick up where you left off.

Resource use is deliberately bounded:

- The server repairs one file at a time behind a lock, so several browser tabs
  cannot gang up on it.
- Results are spooled to a temporary directory rather than held in RAM, capped
  by both file count and total bytes, and deleted when the server exits.
- Above five files the per-file 3D preview is skipped while the batch runs and
  only the final result is drawn, so a long queue does not spend its time
  parsing geometry it is about to discard.
- GPU buffers are explicitly disposed between files. They are not garbage
  collected with the objects that referenced them, so without this, memory would
  climb with every model.

**No WebGL?** If the browser cannot create a WebGL context, the 3D preview is
replaced by a short explanation and everything else keeps working. Repairing a
mesh does not need a GPU.

### Command line

```bash
./scripts/stl-repair model.stl
```

Writes `model_repaired.stl` beside the input.

```bash
# report without writing anything; exits 1 if repair is needed
./scripts/stl-repair model.stl --check-only

# whole folder into an output directory
./scripts/stl-repair ~/printer-files -o ./repaired

# a batch with a per-file report; .json also works
./scripts/stl-repair ~/printer-files -o ./repaired --report report.csv

# machine-readable, for scripting
./scripts/stl-repair model.stl --check-only --json
```

Colour switches itself off when piped and honours `NO_COLOR`. `--plain` gives
ASCII output for logs, `--color` forces colour back on.

### Batch reporting

More than one file prints a summary afterwards: a line per file with its tier,
triangle counts and outcome, then every problem found across the batch rolled up.

```
━━━ BATCH SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  file          tier                   triangles    time  result
  ─────────────────────────────────────────────────────────────────────────
  Ball.stl      conservative     28,872 → 28,818    1.7s  repaired
  Orb.stl       voxel           19,176 → 311,772   20.4s  repaired
  Wedge.stl     conservative     19,562 → 19,510    1.6s  repaired
  Zap.stl       conservative     14,532 → 14,512    1.0s  repaired
  ─────────────────────────────────────────────────────────────────────────
  4 files · 4 repaired · 24.6s total

  PROBLEMS ACROSS THE BATCH                                     before → after
    ✓ Non-manifold edges                                         176 →       0
    ✓ Self-intersections                                           3 →       0
    ✓ Degenerate faces                                           176 →       0
    ✓ Duplicate faces                                              1 →       0
    ● Disjoint shells                                              1 →       1

    • not checked on some files: self-intersections
```

`--report` writes one row per file with all nine checks before and after, the
tier used, timing, vertex and triangle counts, volume, bodies, and the steps
taken. Two details matter for auditing a bulk run: a check that could not be run
is recorded as `not checked` rather than `0`, so it can never be mistaken for a
pass, and a file that failed outright keeps its error message in its row rather
than vanishing from the report.

Outcomes are one of `repaired` (was broken, now clean), `unchanged` (was already
clean), `incomplete` (improved but problems remain) or `failed`.

### Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--mode` | `auto` | `conservative`, `auto`, or `force`. See tiers below. |
| `--check-only` | off | Report only. Exit code 1 when repair is needed. |
| `--json` | off | Emit the full report as JSON. |
| `-v`, `--verbose` | off | List every check, including passes. |
| `--ascii` | off | Write ASCII STL instead of binary. |
| `--weld-tol` | 1e-6 of bbox diagonal | Distance below which vertices merge. |
| `--no-fill-holes` | off | Leave boundaries open. |
| `--max-hole-edges` | 0 (no limit) | Skip holes larger than this many edges. |
| `--keep-non-manifold` | off | Do not trim surplus faces from non-manifold edges. |
| `--min-shell-fraction` | 0 (keep all) | Drop shells below this share of the largest, by surface area. |
| `--voxel-resolution` | 256 | Voxels across the longest axis in the voxel tier. |
| `--no-self-check` | off | Skip self-intersection detection. |

### As a library

```python
from stlrepair import load_stl, weld, diagnose, repair, RepairOptions

vertices, faces = weld(*load_stl("model.stl"))
print(diagnose(vertices, faces).to_dict())

result = repair(vertices, faces, RepairOptions(mode="auto"))
print(result.tier, result.after.is_watertight, result.steps)
```

## How repair works

Three tiers, each re-diagnosed afterwards. **A tier that fails to improve the mesh
is discarded rather than returned**, so escalating can never make things worse.

**1. Conservative** — welds coincident vertices, drops degenerate and duplicate
faces, trims surplus faces off non-manifold edges, makes winding consistent and
turns inside-out shells the right way, then caps holes by ear-clipping a fitted
plane. Never resamples geometry. Most files stop here.

**2. Manifold** — rebuilds through [manifold3d](https://github.com/elalish/manifold)'s
exact boolean kernel. This is what fuses overlapping bodies and resolves
self-intersections, because every output of that kernel is intersection-free by
construction.

**3. Voxel** — re-derives the surface from a filled voxel volume: rasterise, flood
fill from outside, extract the isosurface from a signed distance field. Always
produces a watertight solid, at the cost of resampling detail at the voxel size.
The morphological closing that seals openings escalates automatically until the
interior fill stops leaking, so it handles holes far wider than a fixed radius
would.

The surface is rasterised by sampling each triangle in proportion to its area,
in fixed-size batches, so peak memory does not depend on the mesh. Subdividing
every triangle down to the voxel pitch instead — the obvious approach, and what
the usual library call does — costs a power of four per level: a model mixing
15 mm faces with a 0.1 mm pitch produces tens of millions of sub-triangles and
exhausts memory long before it finishes. The grid size is capped too, and an
over-ambitious resolution is lowered rather than allocated.

`conservative` stops at tier 1. `auto` escalates only when the mesh is still
broken. `force` always runs the heavier path.

### Which triangles get trimmed

When an edge carries more than two faces, the surplus has to go, and *which* two
survive matters. Candidates are ranked by how many clean two-face edges they
already have, so well-attached surface wins and loose fins lose. Picking
arbitrarily deletes real geometry.

## Accuracy and performance

Timings on a 16-core desktop, single-threaded:

| Triangles | Diagnose | Full repair |
| --- | --- | --- |
| 14.5k | 42 ms | 957 ms |
| 20k | 100 ms | 337 ms |
| 82k | 414 ms | 1.5 s |
| 327k | 1.8 s | 7.0 s |

The voxel tier reconstructs a sphere to within 0.8% of true volume at resolution
96, and 0.5% at 128.

Memory is bounded and released between files. A badly damaged 43k-triangle model
that falls all the way through to the voxel tier peaks at about 1.3 GB at
resolution 256 and takes roughly 20 seconds; the server returns to its baseline
afterwards, so a long batch does not accumulate. Drop the resolution to 128 if
the machine is tight.

## Tests

```bash
./scripts/test
```

57 tests. Every check has a fixture that triggers it and nothing else, every
repair path is verified by geometry rather than by exit code (a repaired cube must
have volume exactly 1.0), and the report is tested for ordering, for collapsing,
and for the property that colour changes no character positions.

## Limitations

- Self-intersection detection skips coplanar overlaps. Adjacent flat surfaces
  produce them constantly and they are not what breaks a slicer.
- On very dense or pathological meshes the self-intersection search gives up
  rather than run for minutes. It says so instead of claiming a pass.
- The voxel tier resamples. It is a last resort, not a default, and the report
  always states which tier produced the output.
- STL only. No 3MF, OBJ or STEP.

## Licence

MIT, see [LICENSE](LICENSE). Vendored three.js under `src/stlrepair/web/vendor/`
is MIT, copyright the three.js authors.
