#!/usr/bin/env python3
"""
Plasmid Design Agent — Web UI

HTTP entry point and request router. Stateful logic is split across:
  - sessions.py  — all in-memory state (sessions, batch jobs, live streams)
  - streaming.py — Anthropic streaming agent loop
  - batch_worker.py — background batch job workers
  - static/       — the single-page HTML/CSS/JS interface (served from disk)
  - bulk_planner.py — bulk design planning + cost estimation

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python app/app.py
    # Open http://localhost:8000 in your browser
"""

# ── Standard library ─────────────────────────────────────────────────────────
import csv
import io
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional
from urllib.parse import urlparse

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed; rely on environment variables

# ── Project root setup ────────────────────────────────────────────────────────
# Add the repo root so src/ is importable as a package from anywhere the
# server is launched (matches app/agent.py and evals/run_agent_evals.py).
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Project src/ imports ──────────────────────────────────────────────────────
from src.library import load_backbones, load_inserts
from src.plasmid_intake import parse_upload, run_plannotate, build_intake_message

# ── App-level imports ─────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css",
    ".js":   "application/javascript",
}

from bulk_planner import (
    generate_bulk_plan,
    generate_from_template,
    COST_WARN_THRESHOLD,
    COST_SPLIT_THRESHOLD,
)

# database.py is loaded via importlib rather than a direct import because the
# module name ("plasmid_database") would otherwise shadow stdlib names, and the
# module adds PROJECT_ROOT to sys.path itself on load. importlib gives us a
# scoped reference without polluting the import namespace.
_DB_MODULE_PATH = Path(__file__).parent / "database.py"
import importlib.util as _importlib_util
_db_spec = _importlib_util.spec_from_file_location("plasmid_database", _DB_MODULE_PATH)
_db_mod = _importlib_util.module_from_spec(_db_spec)
_db_spec.loader.exec_module(_db_mod)
_init_db              = _db_mod.init_db
_db_save_construct    = _db_mod.save_construct
_db_list_constructs   = _db_mod.list_constructs
_db_update_construct  = _db_mod.update_construct
_db_get_genbank       = _db_mod.get_construct_genbank
_db_get_graph         = _db_mod.get_graph_data
_db_get_by_local_path = _db_mod.get_construct_by_local_path
_db_delete_construct  = _db_mod.delete_construct
build_parts_from_library   = _db_mod.build_parts_from_library
run_validation_structured  = _db_mod.run_validation_structured

from sessions import (
    _sessions, _active_turns,
    _session_live_streams, _session_live_streams_lock,
    _batch_jobs, _bulk_plans,
    _get_row_gate, _get_pause_event,
    create_session, get_session, delete_session_by_id,
    list_sessions, cancel_session,
    _save_sessions,
)

from streaming import (
    MODEL,
    run_agent_turn_streaming,
    reset_client,
)

from batch_worker import start_batch_job, _continue_batch_row

# ── Server-level config ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
LIBRARY_PATH = PROJECT_ROOT / "library"

# ── Database init ─────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "constructs.db"
_init_db(DB_PATH)

# ── Settings (.env) helpers ───────────────────────────────────────────────────
ENV_FILE = Path(__file__).parent / ".env"

SETTINGS_FIELDS = [
    "ANTHROPIC_API_KEY",
    "ADDGENE_API_TOKEN",
    "NCBI_API_KEY",
    "NCBI_EMAIL",
    "UNPAYWALL_EMAIL",
    "BENCHLING_SUBDOMAIN",
    "PLASMID_USER_LIBRARY",
    "PLASMID_ENABLE_PUBMED",
    "PORT",
]

def _read_env_file() -> dict:
    env: dict = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env

def _write_env_file(settings: dict) -> None:
    lines = [f"{k}={v}" for k, v in settings.items() if v]
    ENV_FILE.write_text("\n".join(lines) + ("\n" if lines else ""))
    for k, v in settings.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


def _enrich_parts_from_references(parts: list[dict], references: list[dict]) -> None:
    """Fill in missing source fields on parts using the session's reference list."""
    for part in parts:
        if part.get("source_url") or part.get("addgene_id") or part.get("genbank_accession"):
            continue
        part_type = part.get("part_type")
        part_name = (part.get("part_name") or "").lower()
        for ref in references:
            if ref.get("component_type") != part_type:
                continue
            if (ref.get("name") or "").lower() not in part_name and part_name not in (ref.get("name") or "").lower():
                continue
            if ref.get("source") == "addgene":
                addgene_id = str(ref.get("identifier") or "")
                part["source_system"] = "Addgene"
                part["source_url"] = ref.get("url") or (f"https://www.addgene.org/{addgene_id}/" if addgene_id else None)
                part["addgene_id"] = addgene_id or None
            elif ref.get("source") == "ncbi":
                accession = ref.get("accession") or ref.get("identifier")
                part["source_system"] = "NCBI"
                part["source_url"] = ref.get("url") or (f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}" if accession else None)
                part["genbank_accession"] = accession
            elif ref.get("source") == "library":
                part["source_system"] = "local library"
            break


# ── HTTP Server ─────────────────────────────────────────────────────────

class AgentHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving the UI and API endpoints."""

    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            data = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(data)

        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            file_path = STATIC_DIR / rel
            if file_path.is_file():
                data = file_path.read_bytes()
                suffix = file_path.suffix
                mime = MIME_TYPES.get(suffix, "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/api/health":
            self._send_json({"status": "ok"})

        elif path == "/api/sessions":
            self._send_json(list_sessions())

        elif path.startswith("/api/sessions/") and path.endswith("/messages"):
            session_id = path.split("/")[3]
            session = get_session(session_id)
            if session:
                if session.get("batch_job_id"):
                    # Keep any prior chat history (e.g. the in-chat preview conversation)
                    # and append the batch marker, rather than discarding it.
                    self._send_json(list(session["display_messages"]) + [{
                        "type": "batch_session",
                        "batch_job_id": session["batch_job_id"],
                        "batch_filename": session.get("batch_filename", ""),
                        "batch_model": session.get("batch_model", ""),
                        "batch_row_count": session.get("batch_row_count", 0),
                    }])
                else:
                    self._send_json(session["display_messages"])
            else:
                self._send_json([], 404)

        elif path.startswith("/api/sessions/") and path.endswith("/status"):
            session_id = path.split("/")[3]
            session = get_session(session_id)
            if session:
                self._send_json({
                    "session_id": session_id,
                    "running": session_id in _active_turns,
                })
            else:
                self._send_json({"error": "Session not found"}, 404)

        elif path.startswith("/api/sessions/") and path.endswith("/stream"):
            session_id = path.split("/")[3]
            with _session_live_streams_lock:
                entry = _session_live_streams.get(session_id)
            if not entry:
                self._send_json({"error": "Session not running"}, 404)
                return

            live_log, live_cond = entry

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def _write_sse(data: dict):
                line = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            offset = 0
            stream_done = False
            while not stream_done:
                with live_cond:
                    if offset >= len(live_log):
                        live_cond.wait(timeout=30)
                    new_events = live_log[offset:]
                    offset += len(new_events)

                if not new_events:
                    # Keepalive on timeout
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    continue

                for evt in new_events:
                    if evt is None:
                        stream_done = True
                        break
                    try:
                        _write_sse(evt)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return

        elif path == "/api/user-library":
            from src.user_library import load_user_designed_constructs
            all_bb = load_backbones()["backbones"]
            bb = [b for b in all_bb if b.get("source") == "user_library"]
            vendor_bb = [b for b in all_bb if b.get("source") == "vendor"]
            ins = [i for i in load_inserts()["inserts"] if i.get("source") == "user_library"]
            designed = load_user_designed_constructs()
            self._send_json({
                "configured": bool(os.environ.get("PLASMID_USER_LIBRARY")),
                "vendor_backbones": [
                    {k: v for k, v in {
                        "id": b["id"],
                        "name": b.get("name"),
                        "description": b.get("description"),
                        "company": b.get("company"),
                        "assembly_enzyme": b.get("assembly_enzyme"),
                        "size_bp": b.get("size_bp"),
                    }.items() if v is not None}
                    for b in vendor_bb
                ],
                "backbones": [
                    {k: v for k, v in {
                        "id": b["id"],
                        "name": b.get("name"),
                        "description": b.get("description"),
                        "assembly_enzyme": b.get("assembly_enzyme"),
                        "bacterial_resistance": b.get("bacterial_resistance"),
                        "mammalian_selection": b.get("mammalian_selection"),
                        "ecoli_strain": b.get("ecoli_strain"),
                        "next_step_enzyme": b.get("next_step_enzyme"),
                        "overhang_left": b.get("overhang_left"),
                        "overhang_right": b.get("overhang_right"),
                        "overhang_left_2": b.get("overhang_left_2"),
                        "overhang_right_2": b.get("overhang_right_2"),
                        "size_bp": b.get("size_bp"),
                    }.items() if v is not None}
                    for b in bb
                ],
                "inserts": [
                    {k: v for k, v in {
                        "id": i["id"],
                        "name": i.get("name"),
                        "description": i.get("description"),
                        "category": i.get("category"),
                        "assembly_enzyme": i.get("assembly_enzyme"),
                        "overhang_l": i.get("overhang_l"),
                        "overhang_r": i.get("overhang_r"),
                        "insert_size_bp": i.get("insert_size_bp"),
                        "size_bp": i.get("size_bp"),
                        "bacterial_resistance": i.get("bacterial_resistance"),
                    }.items() if v is not None}
                    for i in ins
                ],
                "designed_constructs": [
                    {k: v for k, v in {
                        "id": c["id"],
                        "name": c.get("name"),
                        "size_bp": c.get("size_bp"),
                        "description": c.get("description"),
                    }.items() if v is not None}
                    for c in designed
                ],
            })

        elif path.startswith("/api/batch/") and path.endswith("/download-all"):
            # GET /api/batch/{job_id}/download-all — ZIP of all exports
            import zipfile as _zipfile
            job_id = path.split("/")[3]
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            buf = io.BytesIO()
            with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
                # Preview construct (#1) comes first
                for exp in job.get("preview_exports", []):
                    zf.writestr(exp["filename"], exp["content"])
                for row in job["rows"]:
                    for exp in row.get("exports", []):
                        zf.writestr(exp["filename"], exp["content"])
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="batch_designs.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif path.startswith("/api/batch/") and "/rows/" in path and "/plot/" in path:
            # GET /api/batch/{job_id}/rows/{row_idx}/plot/{export_idx}
            parts = path.split("/")
            try:
                job_id = parts[3]
                row_idx = int(parts[5])
                export_idx = int(parts[7]) if len(parts) > 7 else 0
                export = _batch_jobs[job_id]["rows"][row_idx]["exports"][export_idx]
                plot_json = export.get("plot_json")
                if not plot_json:
                    self._send_json({"error": "No plot available"}, 404)
                    return
                self._send_json(plot_json)
            except (KeyError, IndexError, ValueError):
                self.send_error(404)

        elif path.startswith("/api/batch/") and "/download/" in path:
            # GET /api/batch/{job_id}/download/{row_idx}/{export_idx}
            parts = path.split("/")
            try:
                job_id = parts[3]
                row_idx = int(parts[5])
                export_idx = int(parts[6]) if len(parts) > 6 else 0
                export = _batch_jobs[job_id]["rows"][row_idx]["exports"][export_idx]
                filename = export["filename"]
                content = export["content"]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except (KeyError, IndexError, ValueError):
                self.send_error(404)

        elif path.startswith("/api/batch/"):
            # GET /api/batch/{job_id} — return job status (no full file content)
            job_id = path.split("/")[3]
            job = _batch_jobs.get(job_id)
            if job:
                rows_summary = [
                    {
                        "description": r["description"],
                        "name": r["name"],
                        "status": r["status"],
                        "paused": r.get("paused", False),
                        "error": r["error"],
                        "exports": [
                            {"filename": e["filename"], "has_plot": bool(e.get("plot_json"))}
                            for e in r["exports"]
                        ],
                        "log": r.get("log", []),
                    }
                    for r in job["rows"]
                ]
                self._send_json({"status": job["status"], "rows": rows_summary})
            else:
                self._send_json({"error": "Job not found"}, 404)

        elif path == "/api/config/user-library":
            user_lib = os.environ.get("PLASMID_USER_LIBRARY")
            self._send_json({
                "available": bool(user_lib and Path(user_lib).expanduser().is_dir()),
                "path": user_lib or None,
            })

        elif path == "/api/settings":
            env_vals = _read_env_file()
            result = {f: env_vals.get(f, os.environ.get(f, "")) for f in SETTINGS_FIELDS}
            self._send_json(result)

        elif path == "/api/settings/pick-folder":
            import platform
            import subprocess as _sp
            if platform.system() == "Darwin":
                try:
                    r = _sp.run(
                        ["osascript", "-e",
                         'POSIX path of (choose folder with prompt "Select plasmid library folder")'],
                        capture_output=True, text=True, timeout=60,
                    )
                    if r.returncode == 0:
                        self._send_json({"path": r.stdout.strip().rstrip("/")})
                    else:
                        self._send_json({"cancelled": True})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)
            else:
                self._send_json({"error": "Folder picker only supported on macOS"}, 400)

        elif path == "/api/db/user-library-preview":
            user_lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
            if not user_lib_dir or not Path(user_lib_dir).expanduser().is_dir():
                self._send_json({"error": "PLASMID_USER_LIBRARY not set"}, 400)
                return
            from src.user_library import load_user_backbones, load_user_inserts, GENBANK_EXTENSIONS
            items = []
            for bb in load_user_backbones():
                lp = bb.get("local_path")
                items.append({
                    "local_path": lp,
                    "name": bb.get("name") or bb.get("id", ""),
                    "part_type": "backbone",
                    "size_bp": bb.get("size_bp"),
                    "description": bb.get("description", ""),
                    "bacterial_resistance": bb.get("bacterial_resistance"),
                    "assembly_enzyme": bb.get("assembly_enzyme"),
                    "already_imported": bool(lp and _db_get_by_local_path(DB_PATH, lp)),
                })
            for ins in load_user_inserts():
                lp = ins.get("local_path")
                items.append({
                    "local_path": lp,
                    "name": ins.get("name") or ins.get("id", ""),
                    "part_type": "insert",
                    "size_bp": ins.get("insert_size_bp") or ins.get("size_bp"),
                    "description": ins.get("description", ""),
                    "category": ins.get("category"),
                    "already_imported": bool(lp and _db_get_by_local_path(DB_PATH, lp)),
                })
            ann_dir = Path(user_lib_dir).expanduser() / "annotations"
            if ann_dir.is_dir():
                for f in sorted(ann_dir.iterdir()):
                    if f.suffix.lower() in GENBANK_EXTENSIONS:
                        lp = str(f)
                        items.append({
                            "local_path": lp,
                            "name": f.stem,
                            "part_type": "annotation",
                            "size_bp": None,
                            "description": "",
                            "already_imported": bool(_db_get_by_local_path(DB_PATH, lp)),
                        })
            self._send_json(items)

        # ── Plasmid library DB ────────────────────────────────────────────
        elif path == "/api/db/constructs":
            self._send_json(_db_list_constructs(DB_PATH))

        elif path == "/api/db/graph":
            self._send_json(_db_get_graph(DB_PATH))

        elif path.startswith("/api/db/constructs/") and path.endswith("/genbank"):
            parts_path = path.split("/")
            try:
                construct_id = int(parts_path[4])
            except (IndexError, ValueError):
                self.send_error(400)
                return
            result = _db_get_genbank(DB_PATH, construct_id)
            if result is None:
                self.send_error(404)
                return
            name, content = result
            filename = name.replace(" ", "_") + ".gb"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            user_message = body.get("message", "")
            request_model = body.get("model", MODEL)

            if not user_message.strip():
                self._send_json({"error": "Empty message"}, 400)
                return

            # Get or create session.
            # If a session_id was provided but doesn't exist, that's an error
            # (stale client state) — don't silently create a fresh one, or the
            # user thinks they're continuing a conversation when they're not.
            session_id = body.get("session_id")
            if session_id and not get_session(session_id):
                self._send_json({
                    "error": (
                        "Session not found. It may have expired or been "
                        "cleared. Please start a new conversation."
                    )
                }, 404)
                return
            if not session_id:
                session_id = create_session()

            # SSE streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def _write_sse(data: dict):
                line = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            # Send session_id synchronously before handing off to the thread
            try:
                _write_sse({"type": "session_id", "session_id": session_id})
            except (BrokenPipeError, ConnectionResetError):
                return

            # Run the agent in a background thread so the run survives if the
            # client navigates away. Events are queued; this handler drains the
            # queue and forwards to the SSE client until it disconnects or the
            # agent finishes.
            import queue as _q
            event_queue: _q.Queue = _q.Queue()

            def _agent_thread():
                try:
                    run_agent_turn_streaming(
                        user_message, session_id,
                        write_event=event_queue.put,
                        model=request_model,
                    )
                except Exception as e:
                    logger.exception("Agent error")
                    event_queue.put({"type": "error", "content": str(e)})
                finally:
                    event_queue.put(None)  # sentinel — agent done

            threading.Thread(target=_agent_thread, daemon=True).start()

            # Forward events to the SSE client until it disconnects or agent finishes
            while True:
                try:
                    item = event_queue.get(timeout=30)
                except _q.Empty:
                    # Send a keepalive comment to detect dead connections
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break  # client gone; agent thread keeps running
                    continue
                if item is None:
                    break  # sentinel — agent finished
                try:
                    _write_sse(item)
                except (BrokenPipeError, ConnectionResetError):
                    break  # client gone; agent thread keeps running

        elif path.startswith("/api/sessions/") and path.endswith("/cancel"):
            session_id = path.split("/")[3]
            cancel_session(session_id)
            self._send_json({"status": "ok"})

        elif path.startswith("/api/sessions/") and path.endswith("/outcome"):
            # POST /api/sessions/{id}/outcome — record experimental result
            session_id = path.split("/")[3]
            session = get_session(session_id)
            if not session:
                self._send_json({"error": "Session not found"}, 404)
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            status = body.get("status")
            observation = body.get("observation")
            if status not in ("success", "failed", "partial"):
                self._send_json({"error": "status must be 'success', 'failed', or 'partial'"}, 400)
                return
            if not observation:
                self._send_json({"error": "observation is required"}, 400)
                return
            session.setdefault("experimental_outcomes", []).append({
                "status": status,
                "observation": observation,
                "construct_name": body.get("construct_name", ""),
                "timestamp": time.time(),
            })
            if body.get("project_name"):
                session["project_name"] = body["project_name"]
            _save_sessions()
            self._send_json({
                "status": "ok",
                "outcomes_count": len(session["experimental_outcomes"]),
            })

        elif path.startswith("/api/batch/") and "/rows/" in path and path.endswith("/continue"):
            # POST /api/batch/{job_id}/rows/{row_idx}/continue
            parts = path.split("/")
            try:
                job_id = parts[3]
                row_idx = int(parts[5])
            except (IndexError, ValueError):
                self._send_json({"error": "Bad request"}, 400)
                return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            row = job["rows"][row_idx]
            if row["status"] == "running":
                self._send_json({"error": "Row is still running"}, 409)
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            message = body.get("message", "").strip()
            if not message:
                self._send_json({"error": "Empty message"}, 400)
                return
            threading.Thread(
                target=_continue_batch_row,
                args=(job_id, row_idx, message),
                daemon=True,
            ).start()
            self._send_json({"status": "ok"})

        elif path.startswith("/api/batch/") and "/rows/" in path and path.endswith("/pause"):
            # POST /api/batch/{job_id}/rows/{row_idx}/pause
            parts_p = path.split("/")
            try:
                job_id = parts_p[3]; row_idx = int(parts_p[5])
            except (IndexError, ValueError):
                self._send_json({"error": "Bad request"}, 400); return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404); return
            if row_idx < len(job["rows"]) and job["rows"][row_idx]["status"] == "running":
                _get_pause_event(job_id, row_idx).clear()
                job["rows"][row_idx]["paused"] = True
            self._send_json({"status": "ok"})

        elif path.startswith("/api/batch/") and "/rows/" in path and path.endswith("/resume"):
            # POST /api/batch/{job_id}/rows/{row_idx}/resume
            parts_p = path.split("/")
            try:
                job_id = parts_p[3]; row_idx = int(parts_p[5])
            except (IndexError, ValueError):
                self._send_json({"error": "Bad request"}, 400); return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404); return
            if row_idx < len(job["rows"]):
                _get_pause_event(job_id, row_idx).set()
                job["rows"][row_idx]["paused"] = False
            self._send_json({"status": "ok"})

        elif path.startswith("/api/batch/") and path.endswith("/pause-all"):
            # POST /api/batch/{job_id}/pause-all
            job_id = path.split("/")[3]
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404); return
            for idx, row in enumerate(job["rows"]):
                if row["status"] == "running":
                    _get_pause_event(job_id, idx).clear()
                    row["paused"] = True
            self._send_json({"status": "ok"})

        elif path.startswith("/api/batch/") and path.endswith("/resume-all"):
            # POST /api/batch/{job_id}/resume-all
            job_id = path.split("/")[3]
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404); return
            for idx, row in enumerate(job["rows"]):
                if row.get("paused"):
                    _get_pause_event(job_id, idx).set()
                    row["paused"] = False
            self._send_json({"status": "ok"})

        elif __import__('re').search(r"^/api/batch/[^/]+/proceed/\d+$", path):
            # POST /api/batch/{job_id}/proceed/{row_idx}
            parts = path.split("/")
            try:
                job_id  = parts[3]
                row_idx = int(parts[5])
            except (IndexError, ValueError):
                self._send_json({"error": "Bad request"}, 400); return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404); return
            _get_row_gate(job_id, row_idx).set()
            self._send_json({"status": "ok"})

        elif path == "/api/upload-plasmid":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            file_content = body.get("content", "")
            filename = body.get("filename", "plasmid.gb")
            if not file_content.strip():
                self._send_json({"error": "Empty file content"}, 400)
                return
            try:
                parsed = parse_upload(file_content, filename)
                features = run_plannotate(parsed["sequence"])
                message = build_intake_message(filename, parsed, features)
                self._send_json({
                    "message": message,
                    "size_bp": parsed["size_bp"],
                    "topology": parsed["topology"],
                    "feature_count": len(features),
                })
            except ValueError as e:
                self._send_json({"error": str(e)}, 400)
            except Exception as e:
                logger.exception("Error processing uploaded plasmid")
                self._send_json({"error": f"Failed to process file: {e}"}, 500)

        # ── Template + CSV sequential run ────────────────────────────────
        elif path == "/api/bulk/template-run":
            content_length = int(self.headers.get("Content-Length", 0))
            body     = json.loads(self.rfile.read(content_length)) if content_length else {}
            template = body.get("template", "").strip()
            csv_rows = body.get("csv_rows", [])
            model    = body.get("model", MODEL)

            if not template or not csv_rows:
                self._send_json({"error": "template and csv_rows required"}, 400)
                return

            try:
                plan = generate_from_template(template, csv_rows, run_model=model)
            except Exception as e:
                logger.exception("generate_from_template failed")
                self._send_json({"error": f"Merge failed: {e}"}, 500)
                return

            batch_rows = [
                {
                    "description": r["enriched_prompt"],
                    "name":        r["name"],
                    "output_format": r["output_format"],
                }
                for r in plan.enriched_rows
            ]
            job_id = start_batch_job(batch_rows, model, approval_required=True)

            session_id = str(uuid.uuid4())
            _sessions[session_id] = {
                "history": [],
                "display_messages": [],
                "created_at":    time.time(),
                "first_message": f"Bulk design ({len(batch_rows)} constructs)",
                "project_name":  None,
                "experimental_outcomes": [],
                "batch_job_id":    job_id,
                "batch_filename":  "bulk_design.csv",
                "batch_model":     model,
                "batch_row_count": len(batch_rows),
            }
            _save_sessions()
            self._send_json({
                "job_id":    job_id,
                "session_id": session_id,
                "row_count": len(batch_rows),
            })

        # ── Bulk design planning endpoints ───────────────────────────────
        elif path == "/api/bulk/plan":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            csv_text     = body.get("csv_content", "")
            user_context = body.get("user_context", "")
            run_model    = body.get("model", "claude-sonnet-4-6")
            filename     = body.get("filename", "bulk_design.csv")
            # Direct rows list (from submit_bulk_designs tool) takes priority over CSV
            direct_rows  = body.get("rows")

            rows: list[dict] = []
            if direct_rows:
                rows = [r for r in direct_rows if r.get("description", "").strip()]
            elif csv_text.strip():
                reader = csv.DictReader(io.StringIO(csv_text))
                rows = [r for r in reader if r.get("description", "").strip()]

            if not rows:
                self._send_json({"error": "No rows found"}, 400)
                return

            try:
                plan = generate_bulk_plan(rows, user_context, run_model)
            except Exception as e:
                logger.exception("bulk_plan generation failed")
                self._send_json({"error": f"Planning failed: {e}"}, 500)
                return

            _bulk_plans[plan.plan_id] = {
                "plan_id":            plan.plan_id,
                "summary":            plan.summary,
                "enriched_rows":      plan.enriched_rows,
                "job_groups":         plan.job_groups,
                "model_suggestion":   plan.model_suggestion,
                "estimated_cost_usd": plan.estimated_cost_usd,
                "complexity":         plan.complexity,
                "batch_eligible":     plan.batch_eligible,
                "batch_prompt":       plan.batch_prompt,
                "shared_context":     plan.shared_context,
                "filename":           filename,
                "original_rows":      rows,
            }

            warning = None
            if plan.estimated_cost_usd > COST_SPLIT_THRESHOLD:
                warning = "orange"
            elif plan.estimated_cost_usd > COST_WARN_THRESHOLD:
                warning = "yellow"

            self._send_json({
                "plan_id":            plan.plan_id,
                "summary":            plan.summary,
                "rows":               [
                    {"name": r["name"], "description": r["description"]}
                    for r in plan.enriched_rows
                ],
                "estimated_cost_usd": plan.estimated_cost_usd,
                "model_suggestion":   plan.model_suggestion,
                "job_groups":         plan.job_groups,
                "warning":            warning,
                "filename":           filename,
                "batch_eligible":     plan.batch_eligible,
                "shared_context":     plan.shared_context,
            })

        elif path == "/api/bulk/sample":
            content_length = int(self.headers.get("Content-Length", 0))
            body     = json.loads(self.rfile.read(content_length)) if content_length else {}
            plan_id  = body.get("plan_id", "")
            model    = body.get("model", "claude-sonnet-4-6")

            plan_data = _bulk_plans.get(plan_id)
            if not plan_data:
                self._send_json({"error": "Plan not found"}, 404)
                return

            row_0 = plan_data["enriched_rows"][0]
            sample_rows = [{
                "description": row_0["enriched_prompt"],
                "name":        row_0["name"],
                "output_format": row_0["output_format"],
            }]
            job_id = start_batch_job(sample_rows, model)

            session_id = str(uuid.uuid4())
            _sessions[session_id] = {
                "history": [],
                "display_messages": [],
                "created_at":   time.time(),
                "first_message": f"Sample: {row_0['name']}",
                "project_name": None,
                "experimental_outcomes": [],
                "batch_job_id":    job_id,
                "batch_filename":  f"sample: {row_0['name']}",
                "batch_model":     model,
                "batch_row_count": 1,
                "is_bulk_sample":  True,
                "bulk_plan_id":    plan_id,
            }
            _save_sessions()
            self._send_json({"job_id": job_id, "session_id": session_id, "row_count": 1})

        elif path == "/api/bulk/run":
            content_length   = int(self.headers.get("Content-Length", 0))
            body             = json.loads(self.rfile.read(content_length)) if content_length else {}
            plan_id          = body.get("plan_id", "")
            model            = body.get("model", "claude-sonnet-4-6")
            sample_job_id    = body.get("sample_job_id")
            # selected_indices: list of row indices the user wants to run (1-based from UI
            # but 0-based here). None or empty means run all.
            selected_indices = body.get("selected_indices")  # list[int] or None
            # direct_rows: enriched rows from complete_bulk_preview approval (bypass plan lookup)
            direct_rows      = body.get("enriched_rows")     # list[{description, name, output_format}] or None

            pre_seeded: dict[int, dict] = {}
            sample_history: Optional[list] = None
            batch_grps: Optional[list[dict]] = None

            if direct_rows is not None:
                # New path: enriched rows sent directly from the in-chat approval card.
                # No plan lookup needed; rows already have enriched prompts embedded.
                all_batch_rows = [
                    {
                        "description":  r.get("description", ""),
                        "name":         r.get("name", ""),
                        "output_format": r.get("output_format", "genbank"),
                    }
                    for r in direct_rows
                ]
                batch_rows    = all_batch_rows
                filename      = body.get("filename", "bulk_design.csv")
                preview_expts = body.get("preview_exports", [])
                # Seed history from the preview run's chat session so batch rows
                # don't re-fetch the backbone/insertion site the agent already found.
                preview_sid = body.get("session_id")
                if preview_sid:
                    preview_sess = _sessions.get(preview_sid)
                    if preview_sess:
                        h = preview_sess.get("history", [])
                        start = preview_sess.get("_preview_history_start", 0)
                        if h and 0 < start < len(h):
                            sample_history = list(h[start:])
                        else:
                            sample_history = list(h) if h else None
            else:
                # Legacy path: plan_id lookup (CSV upload flow and old chat flow)
                preview_expts = []
                plan_data = _bulk_plans.get(plan_id)
                if not plan_data:
                    self._send_json({"error": "Plan not found"}, 404)
                    return

                enriched_rows = plan_data["enriched_rows"]
                all_batch_rows = [
                    {
                        "description": r["enriched_prompt"],
                        "name":        r["name"],
                        "output_format": r["output_format"],
                    }
                    for r in enriched_rows
                ]

                # Extract sample history for context seeding
                if sample_job_id:
                    sample_job = _batch_jobs.get(sample_job_id)
                    if sample_job and sample_job.get("rows"):
                        sr = sample_job["rows"][0]
                        pre_seeded[0] = {
                            "description":   enriched_rows[0]["description"],
                            "name":          enriched_rows[0]["name"],
                            "output_format": enriched_rows[0]["output_format"],
                            "status":        sr.get("status", "done"),
                            "paused":        False,
                            "exports":       sr.get("exports", []),
                            "error":         sr.get("error"),
                            "log":           sr.get("log", []),
                            "history":       sr.get("history", []),
                        }
                        sample_history = sr.get("history") or None

                # Filter to only user-selected rows; always include row 0
                if selected_indices is not None and len(selected_indices) > 0:
                    selected_set = set(selected_indices)
                    selected_set.add(0)
                    batch_rows = [r for i, r in enumerate(all_batch_rows) if i in selected_set]
                else:
                    batch_rows = all_batch_rows

                # Build batch_groups when the plan is batch-eligible
                if plan_data.get("batch_eligible") and plan_data.get("batch_prompt"):
                    remaining = [i for i in range(len(batch_rows)) if i not in pre_seeded]
                    if remaining:
                        batch_grps = [{"prompt": plan_data["batch_prompt"], "indices": remaining}]

                filename = plan_data.get("filename", "bulk_design.csv")

            job_id = start_batch_job(
                batch_rows, model,
                pre_seeded_rows=pre_seeded,
                batch_groups=batch_grps,
                seed_history=sample_history,
                preview_exports=preview_expts,
            )

            # For the new in-chat path, attach the batch job to the existing chat session
            # so no new sidebar entry is created.  For the legacy CSV/plan_id path, create
            # a dedicated background session as before (that flow intentionally navigates away).
            preview_sid_for_attach = body.get("session_id") if direct_rows is not None else None
            if preview_sid_for_attach and preview_sid_for_attach in _sessions:
                _sessions[preview_sid_for_attach]["batch_job_id"]    = job_id
                _sessions[preview_sid_for_attach]["batch_filename"]   = filename
                _sessions[preview_sid_for_attach]["batch_model"]      = model
                _sessions[preview_sid_for_attach]["batch_row_count"]  = len(batch_rows)
                _save_sessions()
            else:
                bg_session_id = str(uuid.uuid4())
                _sessions[bg_session_id] = {
                    "history": [],
                    "display_messages": [],
                    "created_at":    time.time(),
                    "first_message": f"Bulk design: {filename}",
                    "project_name":  None,
                    "experimental_outcomes": [],
                    "batch_job_id":    job_id,
                    "batch_filename":  filename,
                    "batch_model":     model,
                    "batch_row_count": len(batch_rows),
                }
                _save_sessions()

            self._send_json({
                "job_id":     job_id,
                "row_count":  len(batch_rows),
                "filename":   filename,
                "batch_mode": batch_grps is not None,
            })

        elif path == "/api/batch":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            csv_text = body.get("csv_content", "")
            request_model = body.get("model", MODEL)
            batch_filename = body.get("filename", "batch.csv")

            if not csv_text.strip():
                self._send_json({"error": "No CSV content provided"}, 400)
                return

            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)

            if not rows or "description" not in rows[0]:
                self._send_json({"error": "CSV must have a 'description' column"}, 400)
                return

            rows = [r for r in rows if r.get("description", "").strip()]
            if not rows:
                self._send_json({"error": "No non-empty rows found"}, 400)
                return

            job_id = start_batch_job(rows, request_model)

            # Create a dedicated session for this batch job so it persists in the
            # sessions pane and survives the user navigating to another chat.
            batch_session_id = str(uuid.uuid4())
            _sessions[batch_session_id] = {
                "history": [],
                "display_messages": [],
                "created_at": time.time(),
                "first_message": f"Bulk design: {batch_filename}",
                "project_name": None,
                "experimental_outcomes": [],
                "batch_job_id": job_id,
                "batch_filename": batch_filename,
                "batch_model": request_model,
                "batch_row_count": len(rows),
            }
            _save_sessions()

            self._send_json({"job_id": job_id, "row_count": len(rows), "session_id": batch_session_id})

        elif path.startswith("/api/batch/") and "/rows/" in path and "/save-construct/" in path:
            # POST /api/batch/{job_id}/rows/{row_idx}/save-construct/{exp_idx}
            parts_path = path.split("/")
            try:
                job_id = parts_path[3]
                row_idx = int(parts_path[5])
                exp_idx = int(parts_path[7])
            except (IndexError, ValueError):
                self._send_json({"error": "Bad request"}, 400)
                return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            try:
                export = job["rows"][row_idx]["exports"][exp_idx]
            except (IndexError, KeyError):
                self._send_json({"error": "Export not found"}, 404)
                return
            genbank_content = export.get("content", "")
            filename = export.get("filename", "construct.gb")
            construct_name = Path(filename).stem.replace("_", " ")
            total_size_bp = None
            try:
                from Bio import SeqIO as _sio
                record = next(_sio.parse(io.StringIO(genbank_content), "genbank"))
                total_size_bp = len(record.seq)
                if record.name and record.name not in (".", "unknown"):
                    construct_name = record.name
            except Exception:
                pass
            construct_id = _db_save_construct(
                DB_PATH,
                construct_name=construct_name,
                genbank_content=genbank_content,
                total_size_bp=total_size_bp,
                session_id=None,
                backbone_name="",
                insert_names=[],
                parts=[],
                validations=[],
            )
            self._send_json({"id": construct_id, "status": "saved"})

        elif path.startswith("/api/batch/") and "/rows/" in path and "/save-local/" in path:
            # POST /api/batch/{job_id}/rows/{row_idx}/save-local/{exp_idx}
            parts_path = path.split("/")
            try:
                job_id = parts_path[3]
                row_idx = int(parts_path[5])
                exp_idx = int(parts_path[7])
            except (IndexError, ValueError):
                self._send_json({"error": "Bad request"}, 400)
                return
            user_lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
            if not user_lib_dir or not Path(user_lib_dir).expanduser().is_dir():
                self._send_json({"error": "PLASMID_USER_LIBRARY not set"}, 400)
                return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            try:
                export = job["rows"][row_idx]["exports"][exp_idx]
            except (IndexError, KeyError):
                self._send_json({"error": "Export not found"}, 404)
                return
            filename = export.get("filename", "construct.gb")
            content = export.get("content", "")
            constructs_dir = Path(user_lib_dir).expanduser() / "constructs"
            constructs_dir.mkdir(exist_ok=True)
            safe_name = re.sub(r'[^\w\-. ]', '_', Path(filename).stem).strip().replace(' ', '_')
            out_path = constructs_dir / f"{safe_name}.gb"
            out_path.write_text(content)
            self._send_json({"saved_to": str(out_path)})

        elif path.startswith("/api/batch/") and path.endswith("/save-all-constructs"):
            # POST /api/batch/{job_id}/save-all-constructs
            job_id = path.split("/")[3]
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            saved = 0
            for row in job["rows"]:
                for export in row.get("exports", []):
                    if not export.get("filename", "").lower().endswith((".gb", ".gbk", ".genbank")):
                        continue
                    genbank_content = export.get("content", "")
                    filename = export.get("filename", "construct.gb")
                    construct_name = Path(filename).stem.replace("_", " ")
                    total_size_bp = None
                    try:
                        from Bio import SeqIO as _sio2
                        record = next(_sio2.parse(io.StringIO(genbank_content), "genbank"))
                        total_size_bp = len(record.seq)
                        if record.name and record.name not in (".", "unknown"):
                            construct_name = record.name
                    except Exception:
                        pass
                    _db_save_construct(
                        DB_PATH,
                        construct_name=construct_name,
                        genbank_content=genbank_content,
                        total_size_bp=total_size_bp,
                        session_id=None,
                        backbone_name="",
                        insert_names=[],
                        parts=[],
                        validations=[],
                    )
                    saved += 1
            self._send_json({"saved": saved})

        elif path.startswith("/api/batch/") and path.endswith("/save-all-local"):
            # POST /api/batch/{job_id}/save-all-local
            job_id = path.split("/")[3]
            user_lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
            if not user_lib_dir or not Path(user_lib_dir).expanduser().is_dir():
                self._send_json({"error": "PLASMID_USER_LIBRARY not set"}, 400)
                return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            constructs_dir = Path(user_lib_dir).expanduser() / "constructs"
            constructs_dir.mkdir(exist_ok=True)
            saved = 0
            for row in job["rows"]:
                for export in row.get("exports", []):
                    if not export.get("filename", "").lower().endswith((".gb", ".gbk", ".genbank")):
                        continue
                    filename = export.get("filename", "construct.gb")
                    content = export.get("content", "")
                    safe_name = re.sub(r'[^\w\-. ]', '_', Path(filename).stem).strip().replace(' ', '_')
                    out_path = constructs_dir / f"{safe_name}.gb"
                    out_path.write_text(content)
                    saved += 1
            self._send_json({"saved": saved})

        elif path == "/api/reset":
            # Legacy endpoint — clear all sessions
            _sessions.clear()
            _save_sessions()
            self._send_json({"status": "ok"})

        elif path == "/api/settings":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            env_vals = _read_env_file()
            for field in SETTINGS_FIELDS:
                if field in body:
                    val = str(body[field]).strip()
                    if val:
                        env_vals[field] = val
                    elif field in env_vals:
                        del env_vals[field]
            _write_env_file(env_vals)
            reset_client()  # re-create on next call with the updated key
            self._send_json({"ok": True})

        # ── Plasmid library DB ────────────────────────────────────────────
        elif path == "/api/db/constructs":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            construct_name = body.get("construct_name", "construct")
            genbank_content = body.get("genbank_content", "")
            session_id = body.get("session_id")
            backbone_name = body.get("backbone_name", "")
            raw_insert_name = body.get("insert_name", "")
            total_size_bp = body.get("total_size_bp")
            sequence_cache_key = body.get("sequence_cache_key", "")

            # Parse fusion inserts (e.g. "EGFP-mCherry" → ["EGFP", "mCherry"])
            insert_names = [n.strip() for n in raw_insert_name.split("-") if n.strip()]

            # Extract Addgene ID from cache key (e.g. "addgene:41393" → "41393")
            backbone_addgene_id = None
            if sequence_cache_key and sequence_cache_key.startswith("addgene:"):
                backbone_addgene_id = sequence_cache_key[len("addgene:"):]

            parts = build_parts_from_library(backbone_name, insert_names,
                                             backbone_addgene_id=backbone_addgene_id)

            # Enrich parts with tracker data captured during the agent turn
            if session_id:
                sess = get_session(session_id)
                if sess and sess.get("last_export_references"):
                    _enrich_parts_from_references(parts, sess["last_export_references"])

            validations = run_validation_structured(genbank_content, backbone_name,
                                                    raw_insert_name)

            # Derive total_size_bp from GenBank if not provided
            if not total_size_bp and genbank_content:
                try:
                    import io as _io
                    from Bio import SeqIO as _SeqIO
                    record = next(_SeqIO.parse(_io.StringIO(genbank_content), "genbank"))
                    total_size_bp = len(record.seq)
                except Exception:
                    pass

            construct_id = _db_save_construct(
                DB_PATH,
                construct_name=construct_name,
                genbank_content=genbank_content,
                total_size_bp=total_size_bp,
                session_id=session_id,
                backbone_name=backbone_name,
                insert_names=insert_names,
                parts=parts,
                validations=validations,
            )
            self._send_json({"id": construct_id, "status": "saved"})

        elif path == "/api/constructs/save-local":
            # POST /api/constructs/save-local — save GenBank content from main chat to user library dir
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            genbank_content = body.get("genbank_content", "")
            filename = body.get("filename", "construct.gb")
            user_lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
            if not user_lib_dir or not Path(user_lib_dir).expanduser().is_dir():
                self._send_json({"error": "PLASMID_USER_LIBRARY not set"}, 400)
                return
            constructs_dir = Path(user_lib_dir).expanduser() / "constructs"
            constructs_dir.mkdir(exist_ok=True)
            safe_name = re.sub(r'[^\w\-. ]', '_', Path(filename).stem).strip().replace(' ', '_')
            out_path = constructs_dir / f"{safe_name}.gb"
            out_path.write_text(genbank_content)
            self._send_json({"saved_to": str(out_path)})

        elif path == "/api/db/import-user-library":
            user_lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
            if not user_lib_dir or not Path(user_lib_dir).expanduser().is_dir():
                self._send_json({"error": "PLASMID_USER_LIBRARY not set or not a directory"}, 400)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            filter_paths = set(body.get("local_paths") or [])

            from src.user_library import load_user_backbones, load_user_inserts, GENBANK_EXTENSIONS

            imported = 0
            skipped = 0

            _META_KEYS = [
                "description", "category", "assembly_enzyme", "next_step_enzyme",
                "overhang_l", "overhang_r", "overhang_left", "overhang_right",
                "overhang_left_2", "overhang_right_2", "insert_size_bp",
                "bacterial_resistance", "mammalian_selection", "ecoli_strain",
            ]

            entries: list[tuple[str, dict]] = []
            for bb in load_user_backbones():
                entries.append(("backbone", bb))
            for ins in load_user_inserts():
                entries.append(("insert", ins))

            ann_dir = Path(user_lib_dir).expanduser() / "annotations"
            if ann_dir.is_dir():
                for f in sorted(ann_dir.iterdir()):
                    if f.suffix.lower() in GENBANK_EXTENSIONS:
                        entries.append(("annotation", {
                            "local_path": str(f),
                            "name": f.stem,
                            "size_bp": None,
                            "id": f.stem,
                        }))

            for part_type, entry in entries:
                local_path = entry.get("local_path")
                if not local_path:
                    skipped += 1
                    continue
                if filter_paths and local_path not in filter_paths:
                    skipped += 1
                    continue
                if _db_get_by_local_path(DB_PATH, local_path):
                    skipped += 1
                    continue
                try:
                    genbank_content = Path(local_path).read_text(errors="replace")
                except Exception:
                    skipped += 1
                    continue

                origin = "annotation" if part_type == "annotation" else "user_library"
                bb_name = entry.get("id", "") if part_type == "backbone" else ""
                ins_names = [entry.get("id", "")] if part_type == "insert" else []
                meta = {k: entry[k] for k in _META_KEYS if entry.get(k) is not None}
                # Use insert_size_bp for size display on inserts; size_bp is the carrier vector
                display_size = entry.get("insert_size_bp") or entry.get("size_bp")

                _db_save_construct(
                    DB_PATH,
                    construct_name=entry.get("name", Path(local_path).stem),
                    genbank_content=genbank_content,
                    total_size_bp=display_size,
                    session_id=None,
                    backbone_name=bb_name,
                    insert_names=ins_names,
                    parts=[],
                    validations=[],
                    origin=origin,
                    local_path=local_path,
                    part_type=part_type,
                    metadata=meta or None,
                )
                imported += 1

            self._send_json({"imported": imported, "skipped": skipped})

        elif path == "/api/local-library/save":
            user_lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
            if not user_lib_dir or not Path(user_lib_dir).expanduser().is_dir():
                self._send_json({"error": "PLASMID_USER_LIBRARY not set or not a directory"}, 400)
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            name = body.get("name", "construct")
            content = body.get("content", "")
            overwrite = bool(body.get("overwrite", False))
            if not content:
                self._send_json({"error": "No content provided"}, 400)
                return
            constructs_dir = Path(user_lib_dir).expanduser() / "designed_constructs"
            constructs_dir.mkdir(exist_ok=True)
            safe_name = re.sub(r'[^\w\-. ]', '_', name).strip().replace(' ', '_')
            out_path = constructs_dir / f"{safe_name}.gb"
            if out_path.exists() and not overwrite:
                # Find the next free numbered suffix
                n = 1
                while (constructs_dir / f"{safe_name}_{n}.gb").exists():
                    n += 1
                self._send_json({"exists": True, "suggested_name": f"{safe_name}_{n}"})
                return
            out_path.write_text(content)
            self._send_json({"saved_to": str(out_path)})

        elif path.startswith("/api/db/constructs/") and path.endswith("/save-to-library"):
            m2 = re.match(r"^/api/db/constructs/(\d+)/save-to-library$", path)
            if not m2:
                self.send_error(400)
                return
            construct_id = int(m2.group(1))
            user_lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
            if not user_lib_dir or not Path(user_lib_dir).expanduser().is_dir():
                self._send_json({"error": "PLASMID_USER_LIBRARY not set or not a directory"}, 400)
                return
            result = _db_get_genbank(DB_PATH, construct_id)
            if result is None:
                self.send_error(404)
                return
            db_name, content = result
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            name = body.get("name") or db_name
            overwrite = bool(body.get("overwrite", False))
            constructs_dir = Path(user_lib_dir).expanduser() / "designed_constructs"
            constructs_dir.mkdir(exist_ok=True)
            safe_name = re.sub(r'[^\w\-. ]', '_', name).strip().replace(' ', '_')
            out_path = constructs_dir / f"{safe_name}.gb"
            if out_path.exists() and not overwrite:
                n = 1
                while (constructs_dir / f"{safe_name}_{n}.gb").exists():
                    n += 1
                self._send_json({"exists": True, "suggested_name": f"{safe_name}_{n}"})
                return
            out_path.write_text(content)
            _db_update_construct(DB_PATH, construct_id, {"local_path": str(out_path)})
            self._send_json({"saved_to": str(out_path)})

        else:
            self.send_error(404)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        path = parsed.path
        m = re.match(r"^/api/db/constructs/(\d+)$", path)
        if m:
            construct_id = int(m.group(1))
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            ok = _db_update_construct(DB_PATH, construct_id, body)
            self._send_json({"ok": ok})
        else:
            self.send_error(404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/sessions/"):
            session_id = path.split("/")[3]
            deleted = delete_session_by_id(session_id)
            self._send_json({"deleted": deleted})
        elif path.startswith("/api/db/constructs/"):
            m = re.match(r"^/api/db/constructs/(\d+)$", path)
            if m:
                construct_id = int(m.group(1))
                deleted = _db_delete_construct(DB_PATH, construct_id)
                self._send_json({"deleted": deleted})
            else:
                self.send_error(400)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def _run_server(port: int):
    """Run the HTTP server."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("=" * 60)
        print("WARNING: ANTHROPIC_API_KEY not set.")
        print("Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("The UI will load but chat will fail without it.")
        print("=" * 60)
        print()

    # ThreadingMixIn: each incoming connection spawns a new thread. Required
    # because SSE streams hold connections open for the full duration of an
    # agent turn (~10s+). Without threading, any second request (poll,
    # session list, etc.) would block until the stream finished.
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadingHTTPServer(("0.0.0.0", port), AgentHandler)
    print(f"Plasmid Designer running at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


def _run_with_reload(port: int):
    """Watch for file changes and restart the server automatically."""
    import subprocess

    watch_paths = [Path(__file__).parent, PROJECT_ROOT / "src"]

    def get_mtimes() -> dict[str, float]:
        mtimes = {}
        for d in watch_paths:
            if not d.exists():
                continue
            for f in d.rglob("*.py"):
                try:
                    mtimes[str(f)] = f.stat().st_mtime
                except OSError:
                    pass
        return mtimes

    print(f"Plasmid Designer running at http://localhost:{port} (auto-reload enabled)")
    print("Watching for file changes in app/ and src/...")
    print("Press Ctrl+C to stop.\n")

    while True:
        mtimes = get_mtimes()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--port", str(port)]
        proc = subprocess.Popen(cmd)

        try:
            while True:
                time.sleep(1)
                new_mtimes = get_mtimes()
                if new_mtimes != mtimes:
                    changed = set()
                    for f in set(list(mtimes.keys()) + list(new_mtimes.keys())):
                        if mtimes.get(f) != new_mtimes.get(f):
                            changed.add(Path(f).name)
                    print(f"\nFile changes detected: {', '.join(sorted(changed))}")
                    print("Restarting server...\n")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

                if proc.poll() is not None:
                    print("\nServer process exited.")
                    return
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("\nShutting down.")
            return


def _cmd_list_library():
    """Print user library entries loaded from $PLASMID_USER_LIBRARY."""
    lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
    if not lib_dir:
        print("PLASMID_USER_LIBRARY is not set.")
        return
    backbones = [b for b in load_backbones()["backbones"] if b.get("source") == "user_library"]
    inserts = [i for i in load_inserts()["inserts"] if i.get("source") == "user_library"]
    print(f"User library: {lib_dir}")
    print(f"  {len(backbones)} backbone(s), {len(inserts)} insert(s)\n")
    if backbones:
        print("Backbones:")
        for b in backbones:
            meta = " | ".join(filter(None, [
                b.get("assembly_enzyme"),
                b.get("bacterial_resistance"),
                b.get("mammalian_selection"),
            ]))
            print(f"  {b['id']:<40} {b.get('name', '')}")
            if meta:
                print(f"    {meta}")
    if inserts:
        print("\nInserts:")
        for i in inserts:
            size = f"{i['insert_size_bp']} bp" if i.get("insert_size_bp") else (f"{i['size_bp']} bp" if i.get("size_bp") else "")
            meta = " | ".join(filter(None, [
                i.get("category"),
                i.get("assembly_enzyme"),
                size,
            ]))
            print(f"  {i['id']:<40} {i.get('name', '')}")
            if meta:
                print(f"    {meta}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plasmid Designer Web UI")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--reload", action="store_true", help="Auto-reload on file changes")
    parser.add_argument("--list-library", action="store_true", help="Print user library entries and exit")
    args = parser.parse_args()

    if args.list_library:
        _cmd_list_library()
        return

    if args.reload:
        _run_with_reload(args.port)
    else:
        _run_server(args.port)


if __name__ == "__main__":
    main()
