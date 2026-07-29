"""Command line front end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import mesh as mesh_io
from .diagnostics import diagnose
from .report import Theme, human_size, render_check, render_repair
from .repair import MODES, RepairOptions, repair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stl-repair",
        description="Diagnose and repair STL files locally.",
    )
    parser.add_argument("input", type=Path, nargs="+", help="STL file(s) or a folder")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file, or output folder when repairing several inputs",
    )
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report problems without writing a repaired file",
    )
    parser.add_argument("--ascii", action="store_true", help="write ASCII STL")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument(
        "--weld-tol",
        type=float,
        help="absolute distance below which vertices merge "
        "(default: 1e-6 of the bounding box diagonal)",
    )
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument(
        "--max-hole-edges",
        type=int,
        default=0,
        help="skip holes with more edges than this (0 means no limit)",
    )
    parser.add_argument(
        "--keep-non-manifold",
        action="store_true",
        help="do not trim surplus faces from non-manifold edges",
    )
    parser.add_argument(
        "--min-shell-fraction",
        type=float,
        default=0.0,
        help="discard shells smaller than this fraction of the largest, "
        "e.g. 0.01 to drop specks",
    )
    parser.add_argument("--voxel-resolution", type=int, default=256)
    parser.add_argument("--voxel-smoothing", type=float, default=0.8)
    parser.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip self-intersection detection (faster on dense meshes)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="list every check, including the ones that passed",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="ASCII output with no colour, for logs and dumb terminals",
    )
    parser.add_argument(
        "--color",
        dest="color",
        action="store_true",
        default=None,
        help="force colour even when piping",
    )
    parser.add_argument(
        "--no-color", dest="color", action="store_false", help="disable colour"
    )
    return parser


def collect_inputs(paths) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob("*.stl")))
        else:
            found.append(path)
    return found


def options_from_args(args) -> RepairOptions:
    return RepairOptions(
        mode=args.mode,
        weld_tol=args.weld_tol,
        fill_holes=not args.no_fill_holes,
        max_hole_edges=args.max_hole_edges,
        resolve_non_manifold=not args.keep_non_manifold,
        min_shell_fraction=args.min_shell_fraction,
        voxel_resolution=args.voxel_resolution,
        voxel_smoothing=args.voxel_smoothing,
        check_self_intersections=not args.no_self_check,
    )


def resolve_output(args, source: Path, many: bool) -> Path:
    if args.output is None:
        return source.with_name(f"{source.stem}_repaired.stl")
    if many or args.output.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        return args.output / f"{source.stem}_repaired.stl"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args.output


def friendly_path(path: Path) -> str:
    """Shorten a path for display: relative to the shell's folder, or via ~."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        pass
    try:
        return "~/" + str(path.resolve().relative_to(Path.home()))
    except ValueError:
        return str(path)


def describe(source: Path, report) -> list[str]:
    """The one-line facts shown under the filename."""
    facts = []
    try:
        facts.append(human_size(source.stat().st_size))
    except OSError:
        pass
    facts.append(f"{mesh_io.detect_format(source)} STL")
    facts.append(f"{report.triangle_count:,} triangles")
    if report.shell_count > 1:
        facts.append(f"{report.shell_count:,} bodies")
    return facts


def process(source: Path, args, options: RepairOptions, many: bool) -> dict:
    vertices, faces = mesh_io.load_stl(source)
    show = not args.quiet and not args.json
    theme = Theme.detect(force_plain=args.plain, force_colour=args.color)

    if args.check_only:
        welded_v, welded_f = mesh_io.weld(vertices, faces, options.weld_tol)
        report = diagnose(welded_v, welded_f, options.check_self_intersections)
        if show:
            print()
            print(
                render_check(
                    report, source.name, describe(source, report), theme, args.verbose
                )
            )
        return {"file": str(source), "check_only": True, **report.to_dict()}

    result = repair(vertices, faces, options)
    destination = resolve_output(args, source, many)
    mesh_io.save_stl(destination, result.vertices, result.faces, args.ascii)

    if show:
        print()
        print(
            render_repair(
                result,
                source.name,
                describe(source, result.before),
                theme,
                destination=friendly_path(destination),
                show_all=args.verbose,
            )
        )

    return {"file": str(source), "output": str(destination), **result.to_dict()}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    options = options_from_args(args)

    sources = collect_inputs(args.input)
    if not sources:
        print("no STL files found", file=sys.stderr)
        return 2

    reports = []
    failed = 0
    for source in sources:
        if not source.exists():
            print(f"missing: {source}", file=sys.stderr)
            failed += 1
            continue
        try:
            reports.append(process(source, args, options, len(sources) > 1))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
            print(f"failed: {source}: {exc}", file=sys.stderr)
            reports.append({"file": str(source), "error": str(exc)})
            failed += 1

    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))

    if failed:
        return 1
    if args.check_only and any(not r.get("is_clean", True) for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
