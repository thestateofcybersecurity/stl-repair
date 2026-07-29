"""Terminal rendering of the diagnosis and repair report.

The layout follows the order a person actually reads in: which file, what is
wrong with it, what was fine, what was done about it, and what came out. A flat
list of every check puts the two numbers that matter alongside seven zeroes, so
passing checks are collapsed into a single line unless asked for.
"""

from __future__ import annotations

import os
import shutil
import sys

MIN_WIDTH = 62
MAX_WIDTH = 78

# Ordered worst-first, so the most serious problem is always the first thing
# on screen. `blocking` marks the checks that stop a model printing; disjoint
# shells are legal, so they are reported without being called damage.
CHECK_ORDER = (
    ("non_manifold_edges", "Non-manifold edges", True),
    ("self_intersections", "Self-intersections", True),
    ("naked_edges", "Naked edges", True),
    ("non_planar_holes", "Non-planar holes", True),
    ("planar_holes", "Planar holes", True),
    ("inverted_normals", "Inverted normals", True),
    ("degenerate_faces", "Degenerate faces", True),
    ("duplicate_faces", "Duplicate faces", True),
    ("disjoint_shells", "Disjoint shells", False),
)

# What each tier actually did, in the user's terms rather than ours.
TIER_SUMMARY = {
    "conservative": "cleaned and closed in place, no geometry resampled",
    "manifold": "rebuilt through the exact boolean kernel",
    "voxel": "re-derived the surface from a filled voxel volume",
}

UNICODE_GLYPHS = {
    "rule": "─",
    "heavy": "━",
    "ok": "✓",
    "bad": "✗",
    "info": "•",
    "arrow": "→",
    "bullet": "·",
    "dot": "●",
}

# Deliberately one character each so columns line up exactly as they do in the
# unicode theme; only the arrow is allowed to be wider.
ASCII_GLYPHS = {
    "rule": "-",
    "heavy": "=",
    "ok": "+",
    "bad": "!",
    "info": "*",
    "arrow": "->",
    "bullet": "-",
    "dot": "*",
}


class Theme:
    """Colour and glyph choices, resolved once against the actual terminal."""

    def __init__(self, colour: bool, unicode_ok: bool, width: int):
        self.colour = colour
        self.glyphs = UNICODE_GLYPHS if unicode_ok else ASCII_GLYPHS
        self.width = width

    @classmethod
    def detect(cls, stream=None, force_plain: bool = False, force_colour=None):
        stream = stream or sys.stdout

        colour = bool(force_colour)
        if force_colour is None and not force_plain:
            # https://no-color.org plus the usual dumb-terminal and pipe checks.
            colour = (
                hasattr(stream, "isatty")
                and stream.isatty()
                and os.environ.get("NO_COLOR") is None
                and os.environ.get("TERM") != "dumb"
            )

        unicode_ok = not force_plain
        if unicode_ok:
            encoding = (getattr(stream, "encoding", None) or "").lower()
            unicode_ok = "utf" in encoding

        width = shutil.get_terminal_size((MAX_WIDTH, 24)).columns
        return cls(colour, unicode_ok, max(MIN_WIDTH, min(MAX_WIDTH, width)))

    def paint(self, text: str, *codes: str) -> str:
        if not self.colour or not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def glyph(self, name: str) -> str:
        return self.glyphs[name]


DIM = "2"
BOLD = "1"
RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "36"


def _fit(label: str, value: str, width: int, indent: int = 4) -> tuple[str, str]:
    """Split a row into left and right halves padded to the full width.

    Padding is computed on the uncoloured text, then colour is applied by the
    caller, so escape sequences never disturb the alignment.
    """
    left = " " * indent + label
    gap = max(1, width - len(left) - len(value))
    return left, " " * gap + value


def _wrap(items: list[str], width: int, indent: int, separator: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for item in items:
        candidate = item if not current else f"{current}{separator}{item}"
        if len(candidate) + indent > width:
            lines.append(" " * indent + current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(" " * indent + current)
    return lines


def _count(value) -> str:
    return "n/a" if value is None else f"{value:,}"


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


class ReportRenderer:
    def __init__(self, theme: Theme):
        self.t = theme
        self.lines: list[str] = []

    # -- building blocks ---------------------------------------------------

    def blank(self):
        self.lines.append("")

    def rule(self, heavy: bool = False):
        glyph = self.t.glyph("heavy" if heavy else "rule")
        self.lines.append(self.t.paint(glyph * self.t.width, DIM))

    def banner(self, title: str):
        glyph = self.t.glyph("heavy")
        tail = self.t.width - len(title) - 6
        bar = f"{glyph * 3} {title} {glyph * max(3, tail)}"
        self.lines.append(self.t.paint(bar[: self.t.width], BOLD, BLUE))

    def section(self, title: str, note: str = ""):
        if not note:
            self.lines.append(self.t.paint("  " + title, BOLD))
            return
        left, right = _fit(title, note, self.t.width, indent=2)
        self.lines.append(self.t.paint(left, BOLD) + self.t.paint(right, DIM))

    def row(self, label: str, value: str, colour: str = "", marker: str = " "):
        left, right = _fit(f"{marker} {label}", value, self.t.width)
        self.lines.append(left + self.t.paint(right, colour) if colour else left + right)

    def text(self, body: str, indent: int = 4, colour: str = DIM):
        self.lines.append(self.t.paint(" " * indent + body, colour))

    # -- composite sections ------------------------------------------------

    def header(self, name: str, facts: list[str]):
        self.banner("STL REPAIR")
        self.lines.append(self.t.paint("  " + name, BOLD))
        for line in _wrap(facts, self.t.width, 2, f" {self.t.glyph('bullet')} "):
            self.lines.append(self.t.paint(line, DIM))

    def findings(self, before, after=None, show_all: bool = False):
        """Problems first, worst first. Everything clean collapses to one line."""
        damage, notable, passed, skipped = [], [], [], []
        for field, label, blocking in CHECK_ORDER:
            value = getattr(before, field)
            # None means the check could not be run. Reporting that as a pass
            # would be a lie, so it gets its own category.
            if value is None:
                skipped.append(label)
            elif not value:
                passed.append(label.lower())
            elif blocking:
                damage.append((field, label, value))
            else:
                notable.append((field, label, value))

        note = f"before {self.t.glyph('arrow')} after" if after is not None else ""

        def emit(field, label, value, blocking):
            if after is None:
                self.row(
                    label,
                    _count(value),
                    RED if blocking else YELLOW,
                    self.t.glyph("dot"),
                )
                return
            remaining = getattr(after, field)
            fixed = not remaining
            marker = self.t.glyph("ok") if fixed else self.t.glyph("dot")
            colour = GREEN if fixed else (RED if blocking else YELLOW)
            arrow = self.t.glyph("arrow")
            self.row(
                label,
                f"{_count(value):>9} {arrow} {_count(remaining):>7}",
                colour,
                marker,
            )

        if damage:
            self.section("PROBLEMS FOUND", note)
            for field, label, value in damage:
                emit(field, label, value, True)
        else:
            self.section("PROBLEMS FOUND")
            self.text(
                f"{self.t.glyph('ok')} none, the mesh is structurally sound",
                colour=GREEN,
            )

        # Not damage, but worth stating plainly: a model may legitimately be
        # more than one body, and silently "fixing" that would be wrong.
        if notable or skipped:
            self.blank()
            self.section("WORTH KNOWING", note if not damage else "")
            for field, label, value in notable:
                emit(field, label, value, False)
            for label in skipped:
                self.row(label, "not checked", YELLOW, self.t.glyph("info"))

        if show_all and passed:
            self.blank()
            self.section("ALSO CHECKED")
            for field, label, _ in CHECK_ORDER:
                if label.lower() in passed:
                    self.row(label, "0", DIM, self.t.glyph("bullet"))
        elif passed and damage:
            self.blank()
            summary = f"{self.t.glyph('ok')} {len(passed)} other checks passed"
            self.lines.append(self.t.paint("  " + summary, GREEN))
            for line in _wrap(passed, self.t.width, 4, f" {self.t.glyph('bullet')} "):
                self.lines.append(self.t.paint(line, DIM))

    def actions(self, tier: str, elapsed_ms: float, steps: list[str]):
        self.section(
            "REPAIR", f"{tier} tier {self.t.glyph('bullet')} {elapsed_ms:.0f} ms"
        )
        self.text(TIER_SUMMARY.get(tier, tier), colour=DIM)

        # "<tier> tier accepted" is bookkeeping the header already states.
        # Everything else stays: a tier that ran and was turned down, or that
        # failed, explains why the result came from where it did.
        actions = [s for s in steps if not s.endswith("tier accepted")]
        if not actions:
            return
        self.blank()
        for step in actions:
            self.text(f"{self.t.glyph('bullet')} {step}")

    def result(self, before, after):
        self.section("RESULT")
        for label, was, now in (
            ("Triangles", before.triangle_count, after.triangle_count),
            ("Vertices", before.vertex_count, after.vertex_count),
        ):
            delta = now - was
            shown = f"{delta:+,}" if delta else ("+0" if delta == 0 else f"{delta:+,}")
            value = f"{was:>9,} {self.t.glyph('arrow')} {now:>7,}  {shown:>6}"
            self.row(label, value)

        self.row(
            "Watertight",
            "yes" if after.is_watertight else "no",
            GREEN if after.is_watertight else RED,
        )
        if after.is_watertight:
            self.row("Volume", f"{after.volume:,.3f}")
        if after.shell_count > 1:
            self.row("Separate bodies", f"{after.shell_count:,}", YELLOW)

    def verdict(self, clean: bool, repaired: bool = True, was_clean: bool = False,
                unverified: str = ""):
        """The single line a person reads if they read nothing else."""
        self.blank()
        if not clean:
            text = "PROBLEMS REMAIN" if repaired else "REPAIR NEEDED"
            detail = (
                "see the unresolved checks above"
                if repaired
                else "run without --check-only to fix"
            )
            self.lines.append(
                self.t.paint(f"  {self.t.glyph('bad')}  {text}", BOLD, RED)
                + self.t.paint(f"   {detail}", DIM)
            )
            return

        if not repaired:
            text, detail = "CLEAN", "nothing to fix"
        elif was_clean:
            text, detail = "ALREADY CLEAN", "no repair was needed"
        else:
            text, detail = "REPAIRED", "watertight and ready to print"

        if unverified:
            detail = f"{detail}, but {unverified} was not checked"

        self.lines.append(
            self.t.paint(f"  {self.t.glyph('ok')}  {text}", BOLD, GREEN)
            + self.t.paint(f"   {detail}", DIM)
        )

    def footer(self, destination: str = ""):
        self.rule(heavy=True)
        if destination:
            self.lines.append(
                self.t.paint("  saved to ", DIM) + self.t.paint(destination, BLUE)
            )

    def render(self) -> str:
        return "\n".join(self.lines)


def unverified_checks(report) -> str:
    """Names of checks that could not be run, for the verdict caveat."""
    names = [label.lower() for field, label, _ in CHECK_ORDER
             if getattr(report, field) is None]
    return " and ".join(names)


def render_check(report, name: str, facts: list[str], theme: Theme,
                 show_all: bool = False) -> str:
    r = ReportRenderer(theme)
    r.header(name, facts)
    r.blank()
    r.findings(report, None, show_all)
    r.blank()
    r.section("MESH")
    r.row("Triangles", f"{report.triangle_count:,}")
    r.row("Vertices", f"{report.vertex_count:,}")
    r.row("Watertight", "yes" if report.is_watertight else "no",
          GREEN if report.is_watertight else RED)
    if report.is_watertight:
        r.row("Volume", f"{report.volume:,.3f}")
    r.verdict(report.is_clean, repaired=False,
              unverified=unverified_checks(report))
    r.footer()
    return r.render()


def render_repair(result, name: str, facts: list[str], theme: Theme,
                  destination: str = "", show_all: bool = False) -> str:
    r = ReportRenderer(theme)
    r.header(name, facts)
    r.blank()
    r.findings(result.before, result.after, show_all)
    r.blank()
    r.actions(result.tier, result.elapsed_ms, result.steps)
    r.blank()
    r.result(result.before, result.after)
    r.verdict(result.after.is_clean, was_clean=result.before.is_clean,
              unverified=unverified_checks(result.after))
    r.footer(destination)
    return r.render()
