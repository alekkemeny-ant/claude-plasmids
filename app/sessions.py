"""
Session and batch-job state for the Plasmid Designer server.

This module is the single source of truth for all mutable server state:
  - _sessions: in-memory chat session store (persisted to .sessions.json)
  - _batch_jobs: batch job store (persisted to .batch_jobs.json)
  - _bulk_plans: transient bulk plan store (in-memory only, consumed by /api/bulk/run)
  - _cancelled_sessions, _active_turns: turn-lifecycle bookkeeping
  - _session_live_streams: per-session SSE replay log for reconnecting clients
  - _row_gate_events, _batch_pause_events: threading.Event gates for batch control

streaming.py and batch_worker.py import state dicts and functions from here
directly. sessions.py has no dependency on those modules — keeping the import
graph acyclic.
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"  # lives in app/
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""


# ── Session management ──────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_cancelled_sessions: set[str] = set()
_active_turns: set[str] = set()   # sessions with a turn currently in flight
_sessions_lock = threading.Lock()

# Live event log for SSE reconnect: session_id → (event_list, Condition)
# threading.Condition (not a list) so reconnect consumers can wait() for new
# events without polling — notify_all() wakes them when events arrive.
_session_live_streams: dict = {}
_session_live_streams_lock = threading.Lock()


# ── Batch job state ─────────────────────────────────────────────────────
_batch_jobs: dict[str, dict] = {}
_batch_pause_events: dict[str, threading.Event] = {}


# threading.Event is used here instead of a Lock because batch workers need
# to *block waiting* for a signal (pause.wait(), gate.wait()), not to guard a
# shared resource. event.wait() yields the thread with zero CPU cost.

# ── Bulk plan store (in-memory, lives until /api/bulk/run consumes it) ──
_bulk_plans: dict[str, dict] = {}


# ── Row-gate events — cleared by default; set by /api/batch/{id}/proceed/{idx} ──
_row_gate_events: dict[str, threading.Event] = {}

def _get_row_gate(job_id: str, row_idx: int) -> threading.Event:
    key = f"{job_id}:gate:{row_idx}"
    if key not in _row_gate_events:
        ev = threading.Event()
        # Starts cleared — worker blocks until user approves
        _row_gate_events[key] = ev
    return _row_gate_events[key]

def _get_pause_event(job_id: str, row_idx: int) -> threading.Event:
    key = f"{job_id}:{row_idx}"
    if key not in _batch_pause_events:
        ev = threading.Event()
        ev.set()  # starts unpaused (set = allowed to run)
        _batch_pause_events[key] = ev
    return _batch_pause_events[key]


SESSIONS_FILE = Path(__file__).parent / ".sessions.json"
BATCH_JOBS_FILE = Path(__file__).parent / ".batch_jobs.json"
_batch_jobs_lock = threading.Lock()


def _serialize_content(content):
    """Convert Anthropic SDK content blocks to JSON-serializable format.

    Filters out thinking blocks and non-API-compatible fields so the
    serialized history can be safely replayed to the Anthropic API.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        serialized = []
        for b in content:
            if hasattr(b, "model_dump"):
                d = b.model_dump()
            elif isinstance(b, dict):
                d = b
            else:
                continue
            # Preserve thinking blocks that carry a signature (Opus 4.7
            # adaptive thinking) — they must be replayed in multi-turn history.
            # Strip unsigned thinking blocks (Opus 4.6 and older) which cause
            # 400 errors on replay.
            if isinstance(d, dict) and d.get("type") == "thinking":
                if not d.get("signature"):
                    continue
            serialized.append(d)
        return serialized
    return content


def _save_sessions():
    """Persist sessions to disk so they survive server restarts.

    Uses atomic write (write tmp -> copy backup -> replace) to avoid
    race conditions where the sessions file disappears mid-write.
    Thread-safe via _sessions_lock.
    """
    import shutil

    with _sessions_lock:
        try:
            serializable = {}
            for sid, data in _sessions.items():
                # Serialize history message-by-message so one bad message
                # doesn't drop the entire session (which is what caused
                # users to see their chat history vanish on reload).
                safe_history = []
                for m in data.get("history", []):
                    try:
                        sm = {"role": m["role"], "content": _serialize_content(m["content"])}
                        json.dumps(sm)
                        safe_history.append(sm)
                    except (TypeError, ValueError) as e:
                        logger.warning(
                            f"Dropping unserializable message in session "
                            f"{sid[:8]} (role={m.get('role','?')}): {e}"
                        )
                        # Preserve turn structure so replay doesn't break
                        safe_history.append({
                            "role": m.get("role", "user"),
                            "content": "[message serialization failed]",
                        })
                # Base fields (always serializable — primitive types only)
                base_fields = {
                    "created_at": data.get("created_at", time.time()),
                    "first_message": data.get("first_message"),
                    "history": safe_history,
                    # Phase-2 troubleshooting/project-memory fields — default
                    # to empty for sessions created before these were added.
                    "project_name": data.get("project_name"),
                    "experimental_outcomes": data.get("experimental_outcomes", []),
                    # Batch session fields (None for regular chats)
                    "batch_job_id": data.get("batch_job_id"),
                    "batch_filename": data.get("batch_filename"),
                    "batch_model": data.get("batch_model"),
                    "batch_row_count": data.get("batch_row_count"),
                }
                try:
                    s = {"display_messages": data.get("display_messages", []), **base_fields}
                    json.dumps(s)
                    serializable[sid] = s
                except (TypeError, ValueError) as e:
                    # Fall back to saving session metadata + history only
                    # (display_messages may contain the bad block)
                    logger.warning(
                        f"Session {sid[:8]} display_messages unserializable, "
                        f"saving with empty display: {e}"
                    )
                    serializable[sid] = {"display_messages": [], **base_fields}

            tmp_file = SESSIONS_FILE.with_suffix(".json.tmp")
            with open(tmp_file, "w") as f:
                json.dump(serializable, f)

            if SESSIONS_FILE.exists():
                bak_file = SESSIONS_FILE.with_suffix(".json.bak")
                try:
                    shutil.copy2(str(SESSIONS_FILE), str(bak_file))
                except OSError:
                    pass

            os.replace(str(tmp_file), str(SESSIONS_FILE))
        except Exception as e:
            logger.debug(f"Failed to save sessions: {e}")
            bak_file = SESSIONS_FILE.with_suffix(".json.bak")
            if not SESSIONS_FILE.exists() and bak_file.exists():
                try:
                    shutil.copy2(str(bak_file), str(SESSIONS_FILE))
                except OSError:
                    pass


def _load_sessions():
    """Load sessions from disk on startup. Falls back to .bak if main file is corrupt."""
    global _sessions
    for filepath in [SESSIONS_FILE, SESSIONS_FILE.with_suffix(".json.bak")]:
        try:
            if filepath.exists():
                with open(filepath) as f:
                    _sessions = json.load(f)
                if _sessions:
                    return
        except Exception as e:
            logger.debug(f"Failed to load sessions from {filepath}: {e}")
    _sessions = {}


# Load persisted sessions at import time
_load_sessions()


def _save_batch_jobs():
    """Persist completed batch job data to disk so it survives server restarts."""
    import shutil as _shutil
    with _batch_jobs_lock:
        try:
            serializable = {}
            for job_id, job in _batch_jobs.items():
                rows = []
                for row in job.get("rows", []):
                    rows.append({
                        "description": row.get("description", ""),
                        "name": row.get("name", ""),
                        "output_format": row.get("output_format", "genbank"),
                        "status": row.get("status", "pending"),
                        "paused": False,
                        "exports": [
                            {"filename": e.get("filename", ""), "content": e.get("content", ""),
                             "plot_json": e.get("plot_json")}
                            for e in row.get("exports", [])
                        ],
                        "error": row.get("error"),
                        "log": row.get("log", []),
                    })
                serializable[job_id] = {
                    "status": job.get("status", "done"),
                    "model": job.get("model", ""),
                    "rows": rows,
                }
            tmp = BATCH_JOBS_FILE.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(serializable, f)
            if BATCH_JOBS_FILE.exists():
                bak = BATCH_JOBS_FILE.with_suffix(".json.bak")
                try:
                    _shutil.copy2(str(BATCH_JOBS_FILE), str(bak))
                except OSError:
                    pass
            os.replace(str(tmp), str(BATCH_JOBS_FILE))
        except Exception as e:
            logger.debug(f"Failed to save batch jobs: {e}")


def _load_batch_jobs():
    """Load batch jobs from disk on startup, marking any mid-run rows as interrupted."""
    global _batch_jobs
    for filepath in [BATCH_JOBS_FILE, BATCH_JOBS_FILE.with_suffix(".json.bak")]:
        try:
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                # Fix up any rows that were still running when the server stopped
                for job in data.values():
                    for row in job.get("rows", []):
                        if row.get("status") in ("running", "pending"):
                            row["status"] = "error"
                            row["error"] = "Interrupted: server was restarted."
                    # Mark the whole job done so it doesn't appear stuck
                    job["status"] = "done"
                _batch_jobs = data
                if _batch_jobs:
                    return
        except Exception as e:
            logger.debug(f"Failed to load batch jobs from {filepath}: {e}")
    _batch_jobs = {}


_load_batch_jobs()


def create_session() -> str:
    """Create a new conversation session.

    The session is registered in memory immediately. Persistence to disk is
    deferred to the end-of-turn _save_sessions() call in run_agent_turn_streaming
    so the HTTP handler is not blocked by a large sessions-file write before
    it can send the SSE response headers.
    """
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        # API message history — replayed each turn for multi-turn context.
        "history": [],
        "display_messages": [],
        "created_at": time.time(),
        "first_message": None,
        # Troubleshooting / project-memory fields (Phase 2)
        "project_name": None,            # user-assigned project label (optional)
        "experimental_outcomes": [],     # list of {status, observation, construct_name, timestamp}
    }
    return sid


def _build_system_prompt(session: dict) -> str:
    """Build the system prompt for a turn, injecting per-session context.

    Starts with the static SYSTEM_PROMPT and appends troubleshooting
    context if the session has prior experimental outcomes. This enables
    "project memory" — the agent can see what the user already tried.
    """
    prompt = SYSTEM_PROMPT
    outcomes = session.get("experimental_outcomes") or []
    if outcomes:
        prompt += "\n\n---\n\n## Troubleshooting Context — Prior Experimental Outcomes\n\n"
        prompt += (
            "This session has recorded wet-lab outcomes for constructs the "
            "user previously tried. Use this history to diagnose failures "
            "and propose revised designs (see Troubleshooting Mode section "
            "above).\n\n"
        )
        for i, o in enumerate(outcomes, 1):
            cname = o.get("construct_name") or "unnamed construct"
            prompt += (
                f"**Prior attempt {i}** ({cname}):\n"
                f"  Status: {o.get('status', '?')}\n"
                f"  Observation: {o.get('observation', '?')}\n\n"
            )
    return prompt


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def delete_session_by_id(session_id: str) -> bool:
    deleted = _sessions.pop(session_id, None) is not None
    if deleted:
        _save_sessions()
    return deleted


def list_sessions() -> list[dict]:
    result = []
    for sid, data in sorted(
        _sessions.items(), key=lambda x: x[1]["created_at"], reverse=True
    ):
        result.append({
            "session_id": sid,
            "first_message": data["first_message"],
            "created_at": data["created_at"],
            "project_name": data.get("project_name"),
            "outcomes_count": len(data.get("experimental_outcomes") or []),
            "batch_job_id": data.get("batch_job_id"),
        })
    return result


def cancel_session(session_id: str):
    _cancelled_sessions.add(session_id)

