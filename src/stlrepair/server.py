"""Local web UI. Binds to loopback; nothing leaves the machine."""

from __future__ import annotations

import argparse
import base64
import io
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from . import mesh as mesh_io
from . import topology
from .repair import RepairOptions, repair

WEB_ROOT = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_RESULTS = 8
MAX_HIGHLIGHT_EDGES = 40_000

# Repaired files live in memory only; the browser fetches them straight back.
_results: OrderedDict[str, tuple[str, bytes]] = OrderedDict()


def _remember(name: str, payload: bytes) -> str:
    token = uuid.uuid4().hex
    _results[token] = (name, payload)
    while len(_results) > MAX_RESULTS:
        _results.popitem(last=False)
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

    @app.get("/")
    def index():
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/<path:filename>")
    def assets(filename: str):
        target = (WEB_ROOT / filename).resolve()
        if not target.is_file() or WEB_ROOT.resolve() not in target.parents:
            abort(404)
        return send_from_directory(WEB_ROOT, filename)

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
            result = repair(vertices, faces, options)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"repair failed: {exc}"}), 500

        welded_v, welded_f = mesh_io.weld(vertices, faces, options.weld_tol)
        stem = Path(upload.filename).stem or "model"
        payload = mesh_io.export_bytes(result.vertices, result.faces, ascii_out)
        token = _remember(f"{stem}_repaired.stl", payload)

        return jsonify(
            {
                **result.to_dict(),
                "filename": upload.filename,
                "download": f"/api/download/{token}",
                "size_bytes": len(payload),
                "naked_edges_b64": _naked_edge_segments(welded_v, welded_f),
            }
        )

    @app.get("/api/download/<token>")
    def api_download(token: str):
        entry = _results.get(token)
        if entry is None:
            abort(404)
        name, payload = entry
        return send_file(
            io.BytesIO(payload),
            mimetype="model/stl",
            as_attachment=request.args.get("attach") == "1",
            download_name=name,
        )

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
