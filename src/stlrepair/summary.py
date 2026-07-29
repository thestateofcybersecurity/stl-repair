"""Per-file reporting for batches.

One implementation shared by the CLI and the web UI, so a batch summary says
the same thing wherever it is read. Rows are built from the dictionaries
``RepairResult.to_dict`` already produces, which keeps this decoupled from the
repair pipeline.
"""

from __future__ import annotations

import csv
import io
import json

from .report import CHECK_ORDER, Theme, human_size

# Columns in the flat per-file table, in reading order.
BASE_COLUMNS = (
    ("file", "file"),
    ("tier", "tier used"),
    ("outcome", "outcome"),
    ("elapsed_ms", "milliseconds"),
    ("size_bytes", "input bytes"),
    ("vertices_before", "vertices before"),
    ("vertices_after", "vertices after"),
    ("triangles_before", "triangles before"),
    ("triangles_after", "triangles after"),
    ("watertight_before", "watertight before"),
    ("watertight_after", "watertight after"),
    ("volume_after", "volume after"),
    ("shells_after", "bodies after"),
)


def outcome_of(entry: dict) -> str:
    """A single word for what happened to this file."""
    if entry.get("error"):
        return "failed"
    before, after = entry.get("before"), entry.get("after")
    if not before or not after:
        return "unknown"
    if not after.get("is_clean"):
        return "incomplete"
    return "unchanged" if before.get("is_clean") else "repaired"


def row_for(entry: dict) -> dict:
    """Flatten one file's report into a single record."""
    before = entry.get("before") or {}
    after = entry.get("after") or {}

    row = {
        "file": entry.get("file") or entry.get("filename") or "",
        "tier": entry.get("tier", ""),
        "outcome": outcome_of(entry),
        "elapsed_ms": round(entry.get("elapsed_ms", 0)),
        "size_bytes": entry.get("size_bytes", ""),
        "vertices_before": before.get("vertex_count", ""),
        "vertices_after": after.get("vertex_count", ""),
        "triangles_before": before.get("triangle_count", ""),
        "triangles_after": after.get("triangle_count", ""),
        "watertight_before": before.get("is_watertight", ""),
        "watertight_after": after.get("is_watertight", ""),
        "volume_after": (
            round(after.get("volume", 0.0), 4) if after.get("is_watertight") else ""
        ),
        "shells_after": after.get("shell_count", ""),
    }

    # Every check, before and after. A skipped check is recorded as such rather
    # than as a zero, so a reader cannot mistake it for a pass.
    for field, label in ((f, l) for f, l, _ in CHECK_ORDER):
        row[f"{field}_before"] = _cell(before.get(field))
        row[f"{field}_after"] = _cell(after.get(field))

    if entry.get("error"):
        row["error"] = entry["error"]
    row["steps"] = "; ".join(entry.get("steps", []))
    return row


def _cell(value):
    return "not checked" if value is None else value


def columns() -> list[str]:
    names = [name for name, _ in BASE_COLUMNS]
    for field, _, _ in CHECK_ORDER:
        names += [f"{field}_before", f"{field}_after"]
    return names + ["steps", "error"]


def to_csv(entries: list[dict]) -> str:
    rows = [row_for(entry) for entry in entries]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns(), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def to_json(entries: list[dict]) -> str:
    return json.dumps([row_for(entry) for entry in entries], indent=2)


def totals(entries: list[dict]) -> dict:
    """Aggregate each check across the batch, plus outcome counts."""
    found = {}
    for field, label, _ in CHECK_ORDER:
        before = sum(
            (e.get("before") or {}).get(field) or 0
            for e in entries
            if (e.get("before") or {}).get(field) is not None
        )
        after = sum(
            (e.get("after") or {}).get(field) or 0
            for e in entries
            if (e.get("after") or {}).get(field) is not None
        )
        if before or after:
            found[label] = (before, after)

    counts: dict[str, int] = {}
    for entry in entries:
        key = outcome_of(entry)
        counts[key] = counts.get(key, 0) + 1

    return {
        "checks": found,
        "outcomes": counts,
        "files": len(entries),
        "elapsed_ms": sum(e.get("elapsed_ms", 0) for e in entries),
        "unchecked": sorted(
            {
                label
                for e in entries
                for field, label, _ in CHECK_ORDER
                if (e.get("after") or {}).get(field, 0) is None
            }
        ),
    }


def render_batch(entries: list[dict], theme: Theme) -> str:
    """A terminal table: one line per file, then the batch rollup."""
    from .report import BOLD, DIM, GREEN, RED, YELLOW, ReportRenderer

    r = ReportRenderer(theme)
    r.banner("BATCH SUMMARY")

    width = theme.width
    arrow = theme.glyph("arrow")

    name_width = max(12, min(26, max((len(_short(e)) for e in entries), default=12)))
    header = (
        f"  {'file'.ljust(name_width)}  {'tier':<12} "
        f"{'triangles':>19} {'time':>7}  result"
    )
    r.lines.append(theme.paint(header[:width], DIM))
    r.rule()

    palette = {
        "repaired": GREEN,
        "unchanged": DIM,
        "incomplete": YELLOW,
        "failed": RED,
        "unknown": DIM,
    }

    for entry in entries:
        row = row_for(entry)
        counts = f"{row['triangles_before']:,} {arrow} {row['triangles_after']:,}" \
            if row["triangles_before"] != "" else "-"
        seconds = f"{row['elapsed_ms'] / 1000:.1f}s" if row["elapsed_ms"] else "-"
        line = (
            f"  {_short(entry)[:name_width].ljust(name_width)}  "
            f"{(row['tier'] or '-'):<12} {counts:>19} {seconds:>7}  "
        )
        r.lines.append(
            line[:width] + theme.paint(row["outcome"], palette[row["outcome"]])
        )

    r.rule()

    summary = totals(entries)
    order = ("repaired", "unchanged", "incomplete", "failed", "unknown")
    parts = [f"{summary['outcomes'][k]} {k}" for k in order if k in summary["outcomes"]]
    bullet = f" {theme.glyph('bullet')} "
    r.lines.append(
        theme.paint(
            f"  {summary['files']} files{bullet}{bullet.join(parts)}"
            f"{bullet}{summary['elapsed_ms'] / 1000:.1f}s total",
            BOLD,
        )
    )

    if summary["checks"]:
        r.blank()
        r.section("PROBLEMS ACROSS THE BATCH", f"before {arrow} after")
        blocking = {label: flag for _, label, flag in CHECK_ORDER}
        for label, (before, after) in summary["checks"].items():
            value = f"{before:>9,} {arrow} {after:>7,}"
            if after == 0:
                colour, mark = GREEN, theme.glyph("ok")
            else:
                # Separate bodies are legal, so they never read as damage.
                colour = RED if blocking[label] else YELLOW
                mark = theme.glyph("dot")
            r.row(label, value, colour, mark)

    if summary["unchecked"]:
        r.blank()
        r.text(
            f"{theme.glyph('info')} not checked on some files: "
            + ", ".join(s.lower() for s in summary["unchecked"]),
            colour=YELLOW,
        )

    r.footer()
    return r.render()


def _short(entry: dict) -> str:
    from pathlib import Path

    name = entry.get("file") or entry.get("filename") or ""
    return Path(name).name


def size_label(n) -> str:
    return human_size(n) if isinstance(n, (int, float)) else ""
