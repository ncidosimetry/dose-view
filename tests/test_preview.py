"""Preview-directory integration with generated DICOM, never clinical fixtures."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest

import test_triplanar as fixtures
from dose_view import plot_dose_preview
from dose_view.cli import main
from dose_view.preview import _source_path


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TriplanarTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root.resolve()
        self.preview = self.root / "previews"
        self.preview.mkdir()
        self.manifest = self.preview / "synthetic_geometry.json"
        self.data = {
            "sources": {"CT": "../ct", "RTDOSE": "../dose.dcm"},
            "ct_series_instance_uid": self.fixture.ct_series,
            "dose_max_Gy": 42,
            "checks": {"do_not_copy_this_historical_claim": True},
            "display": {"crosshair_zyx_mm": [12, -4, 2], "dose_threshold_Gy": 4.2,
                        "ct_window_level_HU": 50, "ct_window_width_HU": 350},
        }
        self.write()

    def write(self):
        self.manifest.write_text(json.dumps(self.data))

    def render(self, path=None, **kwargs):
        return plot_dose_preview(path or self.preview, workers=1, dpi=30, **kwargs)

    def test_legacy_report_renders_without_changing_input(self):
        before = self.manifest.read_bytes()
        report, = self.render()
        output = Path(report["figure"])
        self.assertEqual(output, self.preview / "dose_view" / "synthetic_triplanar.png")
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(report["display"]["center_xyz_mm"], [2, -4, 12])
        self.assertEqual(report["display"]["ct_window_level_HU"], 50)
        self.assertEqual(report["display"]["ct_window_width_HU"], 350)
        self.assertAlmostEqual(report["display"]["dose_threshold_fraction"], .1)
        self.assertNotIn("do_not_copy_this_historical_claim", report["checks"])
        self.assertEqual(report["preview_source"], str(self.manifest))
        self.assertEqual(report, json.loads(Path(report["geometry_report"]).read_text()))
        self.assertEqual(before, self.manifest.read_bytes())

    def test_new_report_round_trip_reuses_recorded_display_settings(self):
        original, = self.render(title="Synthetic title", subtitle="Test subtitle",
                                crop_mm={"x": [1, 3]}, alpha=.4, cmap="viridis")
        updated, = self.render(Path(original["geometry_report"]), output_dir=self.root / "rerender")
        for field in ("title", "subtitle", "center_xyz_mm", "crop_mm", "alpha", "cmap",
                      "ct_window_level_HU", "ct_window_width_HU", "dose_threshold_fraction"):
            self.assertEqual(original["display"][field], updated["display"][field])
        self.assertEqual(original["dose_max"], updated["dose_max"])

    def test_recursive_search_preserves_case_directories_and_excludes_output(self):
        for case in ("a", "b"):
            directory = self.preview / case
            directory.mkdir()
            data = dict(self.data, sources={"CT": str(self.fixture.ct_path),
                                            "RTDOSE": str(self.fixture.dose_path)})
            (directory / self.manifest.name).write_text(json.dumps(data))
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, "No geometry sidecars"):
            self.render()
        reports = self.render(recursive=True)
        self.assertEqual(len(reports), 2)
        self.assertEqual({Path(r["figure"]).parent.name for r in reports}, {"a", "b"})
        again = self.render(recursive=True, overwrite=True)
        self.assertEqual(len(again), 2)

    def test_existing_outputs_require_explicit_overwrite(self):
        report, = self.render()
        before = Path(report["figure"]).read_bytes()
        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            self.render()
        self.assertEqual(before, Path(report["figure"]).read_bytes())
        self.assertEqual(len(self.render(overwrite=True)), 1)

    def test_path_remapping_respects_directory_boundaries(self):
        self.data["sources"] = {"CT": "/old/mount/ct", "RTDOSE": "/old/mount/dose.dcm"}
        self.write()
        report, = self.render(path_map={"/old/mount": str(self.root)})
        self.assertEqual(report["dose"]["path"], str(self.fixture.dose_path.resolve()))
        with self.assertRaises(FileNotFoundError):
            _source_path("/old/mount2/dose.dcm", self.manifest, {"/old/mount": str(self.root)})

    def test_missing_source_rejects_before_writing(self):
        self.data["sources"]["RTDOSE"] = "missing.dcm"
        self.write()
        with self.assertRaisesRegex(FileNotFoundError, "Mount the source volume"):
            self.render()
        self.assertFalse((self.preview / "dose_view").exists())

    def test_filename_collisions_reject_before_writing(self):
        (self.preview / "synthetic.geometry.json").write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Conflicting output"):
            self.render()
        self.assertFalse((self.preview / "dose_view").exists())

    def test_pick_override_recomputes_saved_center(self):
        report, = self.render(pick="center")
        self.assertEqual(report["display"]["center_xyz_mm"], [2, -3.5, 12])
        self.assertEqual(report["display"]["selection"], "CT volume center")

    def test_cli_directory_command_and_actionable_error(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(["preview", str(self.preview), "--kwargs", '{"workers":1,"dpi":30}'])
        result = json.loads(stdout.getvalue())
        self.assertTrue(Path(result[0]["figure"]).is_file())
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(["preview", str(self.preview), "--kwargs", "[]"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("JSON object", stderr.getvalue())

    def test_input_reports_are_protected_even_with_overwrite(self):
        before = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "separate"):
            self.render(output_dir=self.preview, overwrite=True)
        renamed = self.preview / "synthetic_triplanar.geometry.json"
        self.manifest.rename(renamed)
        with self.assertRaisesRegex(ValueError, "Conflicting output"):
            self.render(renamed, output_dir=self.preview, overwrite=True)
        self.assertEqual(renamed.read_bytes(), before)

    def test_invalid_manifest_and_png_only_directory(self):
        for invalid in ([], {}, {"sources": {}}, {"sources": []}):
            self.manifest.write_text(json.dumps(invalid))
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.render()
        self.manifest.unlink()
        (self.preview / "only.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        with self.assertRaisesRegex(ValueError, "No geometry sidecars"):
            self.render()


if __name__ == "__main__":
    unittest.main()
