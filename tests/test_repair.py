"""End-to-end checks for detection, repair and file round-tripping.

Run with:  scripts/test
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "pylibs"), str(ROOT / "src"), str(Path(__file__).parent)]

import fixtures  # noqa: E402
from stlrepair import mesh as mesh_io  # noqa: E402
from stlrepair.diagnostics import diagnose  # noqa: E402
from stlrepair.geometry import count_self_intersections  # noqa: E402
from stlrepair.repair import RepairOptions, manifold_pass, repair  # noqa: E402
from stlrepair.report import Theme, render_check, render_repair  # noqa: E402
from stlrepair.topology import boundary_loops, shell_signed_volume  # noqa: E402
from stlrepair.voxel import voxel_remesh  # noqa: E402


class TestDetection(unittest.TestCase):
    """Every check must fire on its own defect and stay quiet on the others."""

    def test_clean_cube_reports_nothing(self):
        report = diagnose(*fixtures.good_cube())
        self.assertEqual(report.defect_total, 0)
        self.assertTrue(report.is_clean)
        self.assertTrue(report.is_watertight)
        self.assertAlmostEqual(report.volume, 1.0)

    def test_naked_edges_and_planar_hole(self):
        report = diagnose(*fixtures.cube_with_hole())
        self.assertEqual(report.naked_edges, 3)
        self.assertEqual(report.planar_holes, 1)
        self.assertEqual(report.non_planar_holes, 0)
        self.assertFalse(report.is_watertight)

    def test_non_planar_hole(self):
        report = diagnose(*fixtures.punctured_sphere(2, 6))
        self.assertEqual(report.planar_holes, 0)
        self.assertEqual(report.non_planar_holes, 1)

    def test_single_inverted_face(self):
        report = diagnose(*fixtures.cube_with_flipped_face())
        self.assertEqual(report.inverted_normals, 1)

    def test_wholly_inside_out_shell(self):
        vertices, faces = fixtures.inside_out_cube()
        report = diagnose(vertices, faces)
        self.assertEqual(report.inverted_normals, len(faces))
        self.assertLess(shell_signed_volume(vertices, faces), 0)

    def test_duplicate_face(self):
        self.assertEqual(diagnose(*fixtures.cube_with_duplicate_face()).duplicate_faces, 1)

    def test_degenerate_face(self):
        self.assertEqual(diagnose(*fixtures.cube_with_degenerate_face()).degenerate_faces, 1)

    def test_non_manifold_edge(self):
        self.assertEqual(
            diagnose(*fixtures.cube_with_non_manifold_edge()).non_manifold_edges, 1
        )

    def test_disjoint_shells_are_counted_but_not_damage(self):
        report = diagnose(*fixtures.two_disjoint_cubes())
        self.assertEqual(report.shell_count, 2)
        self.assertEqual(report.disjoint_shells, 1)
        self.assertTrue(report.is_clean, "separate bodies are legal, not a defect")

    def test_self_intersection(self):
        self.assertEqual(count_self_intersections(*fixtures.self_intersecting_pair()), 1)
        self.assertEqual(count_self_intersections(*fixtures.good_cube()), 0)
        self.assertEqual(count_self_intersections(*fixtures.sphere(2)), 0)

    def test_adjacent_faces_are_not_self_intersections(self):
        """Touching neighbours must never be mistaken for crossings."""
        self.assertEqual(count_self_intersections(*fixtures.two_disjoint_cubes()), 0)

    def test_overlapping_shells_intersect(self):
        self.assertGreater(count_self_intersections(*fixtures.overlapping_cubes()), 0)

    def test_boundary_loop_is_closed(self):
        loops = boundary_loops(fixtures.cube_with_hole()[1])
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 3)


class TestRepair(unittest.TestCase):
    OPTIONS = RepairOptions(mode="auto", voxel_resolution=96)

    def assert_repaired(self, vertices, faces, expected_volume=None, places=3):
        result = repair(vertices, faces, self.OPTIONS)
        self.assertTrue(
            result.after.is_clean,
            f"still broken via {result.tier}: {result.after.report_lines()}",
        )
        self.assertTrue(result.after.is_watertight)
        if expected_volume is not None:
            self.assertAlmostEqual(result.after.volume, expected_volume, places=places)
        return result

    def test_clean_input_is_left_alone(self):
        result = self.assert_repaired(*fixtures.good_cube(), expected_volume=1.0)
        self.assertEqual(result.after.triangle_count, 12)
        self.assertEqual(result.tier, "conservative")

    def test_welding_an_unwelded_file(self):
        result = self.assert_repaired(*fixtures.unwelded_cube(), expected_volume=1.0)
        self.assertEqual(result.after.vertex_count, 8)

    def test_hole_is_capped(self):
        self.assert_repaired(*fixtures.cube_with_hole(), expected_volume=1.0)

    def test_flipped_face_is_reversed(self):
        self.assert_repaired(*fixtures.cube_with_flipped_face(), expected_volume=1.0)

    def test_inside_out_shell_is_turned_the_right_way(self):
        self.assert_repaired(*fixtures.inside_out_cube(), expected_volume=1.0)

    def test_duplicate_is_dropped(self):
        result = self.assert_repaired(
            *fixtures.cube_with_duplicate_face(), expected_volume=1.0
        )
        self.assertEqual(result.after.triangle_count, 12)

    def test_degenerate_is_dropped(self):
        result = self.assert_repaired(
            *fixtures.cube_with_degenerate_face(), expected_volume=1.0
        )
        self.assertEqual(result.after.triangle_count, 12)

    def test_non_manifold_fin_is_trimmed_not_the_surface(self):
        """The stray fin must go, and the cube must survive intact."""
        result = self.assert_repaired(
            *fixtures.cube_with_non_manifold_edge(), expected_volume=1.0
        )
        self.assertEqual(result.after.triangle_count, 12)

    def test_separate_bodies_are_preserved(self):
        result = self.assert_repaired(*fixtures.two_disjoint_cubes(), expected_volume=2.0)
        self.assertEqual(result.after.shell_count, 2)

    def test_overlapping_bodies_are_fused(self):
        """Union of two unit cubes offset by half: 2 - 0.5**3."""
        result = self.assert_repaired(*fixtures.overlapping_cubes(), expected_volume=1.875)
        self.assertEqual(result.tier, "manifold")
        self.assertEqual(result.after.shell_count, 1)
        self.assertEqual(result.after.self_intersections, 0)

    def test_ragged_hole_in_a_sphere(self):
        self.assert_repaired(*fixtures.punctured_sphere(3, 6))

    def test_specks_can_be_discarded(self):
        options = RepairOptions(mode="auto", min_shell_fraction=0.1)
        result = repair(*fixtures.cube_with_speck(), options)
        self.assertEqual(result.after.shell_count, 1)
        self.assertAlmostEqual(result.after.volume, 1.0, places=3)

    def test_specks_are_kept_by_default(self):
        result = repair(*fixtures.cube_with_speck(), RepairOptions())
        self.assertEqual(result.after.shell_count, 2)

    def test_conservative_mode_never_escalates(self):
        result = repair(*fixtures.overlapping_cubes(), RepairOptions(mode="conservative"))
        self.assertEqual(result.tier, "conservative")

    def test_force_rebuilds_even_when_nothing_is_wrong(self):
        """Force means the rebuild was asked for, not that it must pay its way.

        Judging the rebuild only on whether it scored better meant an already
        sound mesh kept the conservative result and reported that tier, which
        contradicts what the caller selected.
        """
        result = repair(*fixtures.good_cube(), RepairOptions(mode="force"))
        self.assertEqual(result.tier, "manifold")
        self.assertTrue(result.after.is_clean)
        self.assertAlmostEqual(result.after.volume, 1.0, places=6)

    def test_force_still_reports_manifold_after_a_repair(self):
        result = repair(*fixtures.cube_with_hole(), RepairOptions(mode="force"))
        self.assertEqual(result.tier, "manifold")
        self.assertAlmostEqual(result.after.volume, 1.0, places=6)

    def test_auto_does_not_rebuild_a_sound_mesh(self):
        """The counterpart: auto must not do the expensive work for nothing."""
        result = repair(*fixtures.good_cube(), RepairOptions(mode="auto"))
        self.assertEqual(result.tier, "conservative")

    def test_force_never_accepts_a_worse_result(self):
        """Only an equal or better rebuild is taken, even under force."""
        result = repair(*fixtures.two_disjoint_cubes(), RepairOptions(mode="force"))
        self.assertTrue(result.after.is_clean)
        self.assertAlmostEqual(result.after.volume, 2.0, places=6)
        self.assertEqual(result.after.shell_count, 2)

    def test_repair_reports_honest_counts(self):
        result = repair(*fixtures.cube_with_hole(), self.OPTIONS)
        self.assertEqual(result.before.naked_edges, 3)
        self.assertEqual(result.after.naked_edges, 0)
        self.assertGreater(result.elapsed_ms, 0)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            repair(*fixtures.good_cube(), RepairOptions(mode="nonsense"))


class TestTiers(unittest.TestCase):
    def test_manifold_pass_unions_overlapping_shells(self):
        vertices, faces = manifold_pass(*fixtures.overlapping_cubes())
        self.assertAlmostEqual(shell_signed_volume(vertices, faces), 1.875, places=4)

    def test_manifold_pass_rejects_broken_input(self):
        with self.assertRaises(RuntimeError):
            manifold_pass(*fixtures.cube_with_hole())

    def test_voxel_remesh_is_watertight(self):
        vertices, faces = fixtures.sphere(3, 10.0)
        out_v, out_f = voxel_remesh(vertices, faces, resolution=64)
        report = diagnose(*mesh_io.weld(out_v, out_f), check_self_intersections=False)
        self.assertTrue(report.is_watertight)
        self.assertEqual(report.shell_count, 1)

    def test_voxel_remesh_approximates_the_volume(self):
        vertices, faces = fixtures.sphere(3, 10.0)
        out_v, out_f = voxel_remesh(vertices, faces, resolution=96)
        report = diagnose(*mesh_io.weld(out_v, out_f), check_self_intersections=False)
        true_volume = 4 / 3 * np.pi * 1000
        self.assertLess(abs(report.volume - true_volume) / true_volume, 0.03)

    def test_voxel_remesh_bridges_a_large_hole(self):
        """Adaptive closing must seal openings far wider than the default."""
        vertices, faces = fixtures.punctured_sphere(3, 40)  # unit radius
        notes: list[str] = []
        out_v, out_f = voxel_remesh(vertices, faces, resolution=96, notes=notes)
        report = diagnose(*mesh_io.weld(out_v, out_f), check_self_intersections=False)
        self.assertTrue(report.is_watertight)
        # The opening is closed rather than leaked through, so the solid should
        # come back close to a whole unit sphere.
        self.assertAlmostEqual(report.volume, 4 / 3 * np.pi, places=1)
        self.assertIn("sealed", notes[0])


class TestFileRoundTrip(unittest.TestCase):
    def round_trip(self, ascii_mode: bool):
        vertices, faces = fixtures.sphere(2, 5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.stl"
            mesh_io.save_stl(path, vertices, faces, ascii_mode=ascii_mode)
            back_v, back_f = mesh_io.load_stl(path)
            welded_v, welded_f = mesh_io.weld(back_v, back_f)
        self.assertEqual(len(welded_f), len(faces))
        self.assertAlmostEqual(
            shell_signed_volume(welded_v, welded_f),
            shell_signed_volume(vertices, faces),
            places=2,
        )

    def test_binary_round_trip(self):
        self.round_trip(ascii_mode=False)

    def test_ascii_round_trip(self):
        self.round_trip(ascii_mode=True)

    def test_stl_files_arrive_unwelded(self):
        """STL has no vertex sharing, so a loaded cube starts with 36 vertices."""
        vertices, faces = fixtures.good_cube()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.stl"
            mesh_io.save_stl(path, vertices, faces)
            raw_v, raw_f = mesh_io.load_stl(path)
        self.assertEqual(len(raw_v), 36)
        self.assertEqual(len(mesh_io.weld(raw_v, raw_f)[0]), 8)


class TestReport(unittest.TestCase):
    """The report is the product's face; its ordering and honesty are tested."""

    PLAIN = Theme(colour=False, unicode_ok=False, width=78)

    def check(self, vertices, faces, **kwargs):
        report = diagnose(vertices, faces)
        return render_check(report, "model.stl", ["fixture"], self.PLAIN, **kwargs)

    def repaired(self, vertices, faces, **kwargs):
        result = repair(vertices, faces, RepairOptions(mode="auto"))
        return render_repair(result, "model.stl", ["fixture"], self.PLAIN, **kwargs)

    def test_worst_problem_is_listed_first(self):
        """A non-manifold edge outranks the naked edges it leaves behind."""
        text = self.check(*fixtures.cube_with_non_manifold_edge())
        self.assertLess(
            text.index("Non-manifold edges"),
            text.index("Naked edges"),
            "checks must be ordered by severity, not by field order",
        )

    def test_passing_checks_are_collapsed(self):
        text = self.check(*fixtures.cube_with_hole())
        self.assertIn("7 other checks passed", text)
        self.assertNotRegex(text, r"Duplicate faces\s+0")

    def test_verbose_lists_every_check(self):
        text = self.check(*fixtures.cube_with_hole(), show_all=True)
        self.assertIn("ALSO CHECKED", text)
        self.assertRegex(text, r"Duplicate faces\s+0")

    def test_clean_mesh_says_so(self):
        text = self.check(*fixtures.good_cube())
        self.assertIn("none, the mesh is structurally sound", text)
        self.assertIn("CLEAN", text)
        self.assertNotIn("REPAIR NEEDED", text)

    def test_separate_bodies_are_not_called_damage(self):
        """A two-body model is legal; it must not be reported as broken."""
        text = self.repaired(*fixtures.two_disjoint_cubes())
        self.assertIn("WORTH KNOWING", text)
        self.assertIn("none, the mesh is structurally sound", text)
        self.assertLess(text.index("PROBLEMS FOUND"), text.index("Disjoint shells"))
        self.assertNotIn("PROBLEMS REMAIN", text)

    def test_untouched_mesh_is_not_claimed_as_repaired(self):
        text = self.repaired(*fixtures.good_cube())
        self.assertIn("ALREADY CLEAN", text)

    def test_repair_shows_before_and_after(self):
        text = self.repaired(*fixtures.cube_with_hole())
        self.assertRegex(text, r"Naked edges\s+3\s*->\s*0")
        self.assertIn("REPAIRED", text)

    def test_tier_bookkeeping_is_hidden_but_explained(self):
        """The user gets what the tier did, not our internal accept log."""
        text = self.repaired(*fixtures.overlapping_cubes())
        self.assertNotIn("tier accepted", text)
        self.assertIn("rebuilt through the exact boolean kernel", text)

    def test_a_tier_that_was_turned_down_is_still_reported(self):
        """Hiding it made a heavier mode look like it had done nothing."""
        result = repair(*fixtures.good_cube(), RepairOptions(mode="force"))
        result.tier = "conservative"          # as if the rebuild lost
        result.steps = ["manifold tier ran but did not improve the mesh, "
                        "so the conservative result was kept"]
        text = render_repair(result, "m.stl", ["f"], self.PLAIN)
        self.assertIn("did not improve the mesh", text)

    def test_skipped_check_is_never_reported_as_passed(self):
        """A check that could not run must not be counted among the passes."""
        report = diagnose(*fixtures.cube_with_hole())
        report.self_intersections = None  # what the broad-phase cap produces

        text = render_check(report, "m.stl", ["f"], self.PLAIN)
        self.assertRegex(text, r"Self-intersections\s+not checked")
        self.assertIn("6 other checks passed", text)
        self.assertNotIn("self-intersections ", text.split("other checks passed")[-1])

    def test_clean_verdict_admits_an_unrun_check(self):
        """"Ready to print" must not be claimed on the back of a skipped check."""
        report = diagnose(*fixtures.good_cube())
        report.self_intersections = None

        text = render_check(report, "m.stl", ["f"], self.PLAIN)
        self.assertIn("self-intersections was not checked", text)

    def test_plain_theme_emits_no_escape_codes(self):
        self.assertNotIn("\033[", self.repaired(*fixtures.cube_with_hole()))

    def test_lines_respect_the_terminal_width(self):
        narrow = Theme(colour=False, unicode_ok=False, width=62)
        result = repair(*fixtures.cube_with_hole(), RepairOptions())
        text = render_repair(result, "model.stl", ["fixture"], narrow)
        for line in text.splitlines():
            self.assertLessEqual(len(line), 62, f"line overflows: {line!r}")

    def test_colour_theme_wraps_values_not_layout(self):
        """Colour must not change where anything sits on the line."""
        coloured = Theme(colour=True, unicode_ok=False, width=78)
        result = repair(*fixtures.cube_with_hole(), RepairOptions())
        with_colour = render_repair(result, "m.stl", ["f"], coloured)
        without = render_repair(result, "m.stl", ["f"], self.PLAIN)
        stripped = re.sub(r"\033\[[0-9;]*m", "", with_colour)
        self.assertEqual(stripped, without)

    def test_singular_counts_read_naturally(self):
        result = repair(*fixtures.cube_with_hole(), RepairOptions())
        self.assertIn("capped 1 hole", result.steps)
        self.assertNotIn("capped 1 holes", result.steps)


class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [str(ROOT / "scripts" / "stl-repair"), *args],
            capture_output=True,
            text=True,
        )

    def test_check_only_flags_a_broken_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.stl"
            mesh_io.save_stl(path, *fixtures.cube_with_hole())
            result = self.run_cli(str(path), "--check-only")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("PROBLEMS FOUND", result.stdout)
        self.assertRegex(result.stdout, r"Naked edges\s+3")
        self.assertRegex(result.stdout, r"Planar holes\s+1")
        self.assertIn("REPAIR NEEDED", result.stdout)

    def test_repair_writes_a_clean_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "broken.stl"
            target = Path(tmp) / "fixed.stl"
            mesh_io.save_stl(source, *fixtures.cube_with_hole())

            result = self.run_cli(str(source), "-o", str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(target.exists())

            report = diagnose(*mesh_io.weld(*mesh_io.load_stl(target)))
        self.assertTrue(report.is_clean)
        self.assertIn("REPAIRED", result.stdout)
        # Piping disables colour but not unicode, so either arrow may appear.
        self.assertRegex(result.stdout, r"Naked edges\s+3\s*(->|→)\s*0")

    def test_output_parent_folder_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "broken.stl"
            mesh_io.save_stl(source, *fixtures.cube_with_hole())
            target = Path(tmp) / "nested" / "deeper" / "fixed.stl"
            result = self.run_cli(str(source), "-o", str(target))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_piped_output_carries_no_escape_codes(self):
        """subprocess is not a tty, so colour must switch itself off."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.stl"
            mesh_io.save_stl(path, *fixtures.cube_with_hole())
            result = self.run_cli(str(path), "--check-only")
        self.assertNotIn("\033[", result.stdout)

    def test_batch_mode_writes_one_file_per_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "in"
            folder.mkdir()
            mesh_io.save_stl(folder / "a.stl", *fixtures.cube_with_hole())
            mesh_io.save_stl(folder / "b.stl", *fixtures.cube_with_flipped_face())

            out = Path(tmp) / "out"
            result = self.run_cli(str(folder), "-o", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            written = sorted(p.name for p in out.glob("*.stl"))
        self.assertEqual(written, ["a_repaired.stl", "b_repaired.stl"])

    def test_json_output_is_machine_readable(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.stl"
            mesh_io.save_stl(path, *fixtures.cube_with_hole())
            result = self.run_cli(str(path), "--check-only", "--json")
            payload = json.loads(result.stdout)
        self.assertEqual(payload["naked_edges"], 3)
        self.assertFalse(payload["is_clean"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
