"""Local web UI. Binds to loopback; nothing leaves the machine."""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import shutil
import tempfile
import threading
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from . import mesh as mesh_io
from . import topology
from .repair import RepairOptions, repair
from .summary import to_csv, to_json

WEB_ROOT = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_HIGHLIGHT_EDGES = 40_000

# Results are spooled to disk rather than held in memory: a batch of large
# models would otherwise pin hundreds of megabytes of RAM for as long as the
# browser might still ask for them.
MAX_RESULT_FILES = 500
MAX_RESULT_BYTES = 2 * 1024 * 1024 * 1024

_SPOOL = Path(tempfile.mkdtemp(prefix="stl-repair-"))
_results: OrderedDict[str, dict] = OrderedDict()
_store_lock = threading.Lock()

# One repair at a time. Each one is CPU and memory hungry, so letting several
# run together on a desktop machine helps nobody; queueing keeps the box
# usable even if several tabs submit at once.
_repair_lock = threading.Lock()


@atexit.register
def _cleanup_spool():
    shutil.rmtree(_SPOOL, ignore_errors=True)


def _remember(name: str, payload: bytes, report: dict | None = None) -> str:
    token = uuid.uuid4().hex
    path = _SPOOL / f"{token}.stl"
    path.write_bytes(payload)

    with _store_lock:
        _results[token] = {
            "name": name,
            "path": path,
            "size": len(payload),
            # Kept so a batch report can be assembled server-side, using the
            # same code the CLI uses.
            "report": report or {},
        }
        total = sum(entry["size"] for entry in _results.values())
        while _results and (
            len(_results) > MAX_RESULT_FILES or total > MAX_RESULT_BYTES
        ):
            _, oldest = _results.popitem(last=False)
            total -= oldest["size"]
            oldest["path"].unlink(missing_ok=True)
    return token


def _naked_edge_segments(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Endpoints of every naked edge, as base64 float32, for the viewer."""
    if len(faces) == 0:
        return ""
    keys, inverse, counts = topology.edge_table(faces)
    naked = keys[counts == 1]
    if len(naked) == 0:
        return ""
    if len(naked) > MAX_HIGHLIGHT_EDGES:
        naked = naked[:MAX_HIGHLIGHT_EDGES]
    segments = vertices[naked.reshape(-1)].astype(np.float32)
    return base64.b64encode(segments.tobytes()).decode("ascii")


def _options_from_form(form) -> RepairOptions:
    def flag(name: str, default: bool) -> bool:
        raw = form.get(name)
        return default if raw is None else raw.lower() in ("1", "true", "on", "yes")

    def number(name, cast, default):
        raw = form.get(name)
        if raw in (None, ""):
            return default
        try:
            return cast(raw)
        except ValueError:
            return default

    mode = form.get("mode", "auto")
    return RepairOptions(
        mode=mode if mode in ("conservative", "auto", "force") else "auto",
        weld_tol=number("weld_tol", float, None),
        fill_holes=flag("fill_holes", True),
        max_hole_edges=number("max_hole_edges", int, 0),
        resolve_non_manifold=flag("resolve_non_manifold", True),
        min_shell_fraction=number("min_shell_fraction", float, 0.0),
        voxel_resolution=number("voxel_resolution", int, 256),
        check_self_intersections=flag("check_self_intersections", True),
    )


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    def _fresh(response):
        """Never let the browser reuse a stale asset.

        This runs on localhost where there is nothing to save by caching, and a
        half-updated page (new markup against an old script) fails in ways that
        look like the app is broken rather than out of date.
        """
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    @app.get("/")
    def index():
        return _fresh(send_from_directory(WEB_ROOT, "index.html"))

    @app.get("/<path:filename>")
    def assets(filename: str):
        target = (WEB_ROOT / filename).resolve()
        if not target.is_file() or WEB_ROOT.resolve() not in target.parents:
            abort(404)
        return _fresh(send_from_directory(WEB_ROOT, filename))

    @app.post("/api/repair")
    def api_repair():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"error": "no file uploaded"}), 400

        options = _options_from_form(request.form)
        ascii_out = request.form.get("ascii", "").lower() in ("1", "true", "on")

        try:
            raw = upload.read()
            vertices, faces = mesh_io.load_stl(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001 - surface parse errors to the UI
            return jsonify({"error": f"could not read STL: {exc}"}), 400

        if len(faces) == 0:
            return jsonify({"error": "file contains no triangles"}), 400

        try:
            # Serialised deliberately: see _repair_lock.
            with _repair_lock:
                result = repair(vertices, faces, options)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"repair failed: {exc}"}), 500

        welded_v, welded_f = mesh_io.weld(vertices, faces, options.weld_tol)
        stem = Path(upload.filename).stem or "model"
        payload = mesh_io.export_bytes(result.vertices, result.faces, ascii_out)

        report = {
            **result.to_dict(),
            "file": upload.filename,
            "size_bytes": len(raw),
        }
        token = _remember(f"{stem}_repaired.stl", payload, report)

        return jsonify(
            {
                **report,
                "filename": upload.filename,
                "download": f"/api/download/{token}",
                "naked_edges_b64": _naked_edge_segments(welded_v, welded_f),
            }
        )

    @app.get("/api/download/<token>")
    def api_download(token: str):
        with _store_lock:
            entry = _results.get(token)
        if entry is None or not entry["path"].is_file():
            abort(404)
        return send_file(
            entry["path"],
            mimetype="model/stl",
            as_attachment=request.args.get("attach") == "1",
            download_name=entry["name"],
        )

    def _lookup(tokens):
        with _store_lock:
            found = [_results.get(t) for t in tokens]
        return [e for e in found if e and e["path"].is_file()]

    @app.get("/api/report")
    def api_report():
        """The per-file report for a batch, as CSV or JSON."""
        tokens = [t for t in request.args.get("tokens", "").split(",") if t]
        entries = _lookup(tokens)
        if not entries:
            return jsonify({"error": "no results to report on"}), 404

        reports = [e["report"] for e in entries if e["report"]]
        as_json = request.args.get("format") == "json"
        body = to_json(reports) if as_json else to_csv(reports)

        return send_file(
            io.BytesIO(body.encode("utf-8")),
            mimetype="application/json" if as_json else "text/csv",
            as_attachment=True,
            download_name=f"stl_repair_report.{'json' if as_json else 'csv'}",
        )

    @app.get("/api/bundle")
    def api_bundle():
        """Zip several results together, so a batch is one download."""
        tokens = [t for t in request.args.get("tokens", "").split(",") if t]
        if not tokens:
            return jsonify({"error": "no tokens given"}), 400

        entries = _lookup(tokens)
        if not entries:
            return jsonify({"error": "nothing left to bundle"}), 404

        # Written to disk, not memory: a batch of large models zipped in RAM
        # would undo the point of spooling them in the first place.
        archive = tempfile.NamedTemporaryFile(
            suffix=".zip", dir=_SPOOL, delete=False
        )
        used: set[str] = set()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for entry in entries:
                name = entry["name"]
                stem, suffix = Path(name).stem, Path(name).suffix
                counter = 2
                while name in used:  # two inputs can share a filename
                    name = f"{stem}_{counter}{suffix}"
                    counter += 1
                used.add(name)
                bundle.write(entry["path"], arcname=name)

            # The report travels with the meshes, so a bulk job is one download
            # and the record of what was checked is not left behind.
            reports = [e["report"] for e in entries if e["report"]]
            if reports:
                bundle.writestr("report.csv", to_csv(reports))
                bundle.writestr("report.json", to_json(reports))
        archive.close()

        response = send_file(
            archive.name,
            mimetype="application/zip",
            as_attachment=True,
            download_name="repaired_stls.zip",
        )

        @response.call_on_close
        def _discard():
            Path(archive.name).unlink(missing_ok=True)

        return response

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="stl-repair-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    print(f"STL repair UI: http://{args.host}:{args.port}")
    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
