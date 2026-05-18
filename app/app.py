#!/usr/bin/env python3
"""
Plasmid Design Agent — Web UI

A chat interface for the Claude-powered plasmid design agent.
Streams via the Anthropic API directly for low TTFT, dispatching tool
calls in-process to the same handlers (src/tools.py:ALL_TOOLS) that
back the Agent SDK MCP server used by app/agent.py and the eval harness.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python app.py
    # Open http://localhost:8000 in your browser
"""

import asyncio
import csv
import io
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
import threading
import uuid
import time

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed; rely on environment variables

# Add project root to path so src/ is importable as a package (matches
# app/agent.py and evals/run_agent_evals.py).
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic

from src.tools import (
    get_anthropic_tool_schemas,
    get_tool_dispatch,
    set_tracker,
    get_last_plot_json,
    clear_last_plot_json,
)
from src.references import ReferenceTracker
from src.library import load_backbones, load_inserts
from src.plasmid_intake import parse_upload, run_plannotate, build_intake_message
_DB_MODULE_PATH = Path(__file__).parent / "database.py"
import importlib.util as _importlib_util
_db_spec = _importlib_util.spec_from_file_location("plasmid_database", _DB_MODULE_PATH)
_db_mod = _importlib_util.module_from_spec(_db_spec)
_db_spec.loader.exec_module(_db_mod)
_init_db = _db_mod.init_db
_db_save_construct = _db_mod.save_construct
_db_list_constructs = _db_mod.list_constructs
_db_update_construct = _db_mod.update_construct
_db_get_genbank = _db_mod.get_construct_genbank
_db_get_graph = _db_mod.get_graph_data
_db_get_by_local_path = _db_mod.get_construct_by_local_path
_db_delete_construct = _db_mod.delete_construct
build_parts_from_library = _db_mod.build_parts_from_library
run_validation_structured = _db_mod.run_validation_structured

logger = logging.getLogger(__name__)

LIBRARY_PATH = PROJECT_ROOT / "library"

# ── Load system prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"  # lives in app/
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""

# ── Tool schemas + dispatch ─────────────────────────────────────────────
# Tool definitions live in src/tools.py:ALL_TOOLS — the same list the
# Agent SDK MCP server (app/agent.py, evals) is built from. We project
# them into Anthropic API format and dispatch in-process so the web UI
# gets ~1s direct-API TTFT instead of the SDK subprocess's ~5s, while
# tool implementations stay single-sourced.

TOOLS = get_anthropic_tool_schemas()
_TOOL_HANDLERS = get_tool_dispatch()


def _tool_result_text(content) -> str:
    """Flatten an MCP-shaped tool result content into a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return "" if content is None else str(content)


def _dispatch_tool(name: str, args: dict) -> str:
    """Run a tool handler in-process and return its text result.

    Handlers are async (defined for the SDK MCP server), so we drive them
    with a short-lived event loop. They return MCP-shaped
    ``{"content": [{"type": "text", "text": ...}]}`` which we flatten.
    """
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        result = asyncio.run(handler(args))
    except Exception as e:  # noqa: BLE001 — surface tool errors to the model
        return f"Tool error ({name}): {e}"
    if isinstance(result, dict) and "content" in result:
        return _tool_result_text(result["content"])
    return _tool_result_text(result)


# ── Session management ──────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_cancelled_sessions: set[str] = set()
_active_turns: set[str] = set()   # sessions with a turn currently in flight
_sessions_lock = threading.Lock()

# Live event log for reconnect streaming: session_id → (event_list, Condition)
_session_live_streams: dict = {}
_session_live_streams_lock = threading.Lock()

# ── Batch job state ─────────────────────────────────────────────────────
_batch_jobs: dict[str, dict] = {}
_batch_pause_events: dict[str, threading.Event] = {}

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

MODEL = "claude-opus-4-7"

# Context window sizes by model (tokens)
CONTEXT_WINDOW = {
    "claude-opus-4-7":          1_000_000,
    "claude-opus-4-6":          1_000_000,
    "claude-sonnet-4-6":        1_000_000,
    "claude-haiku-4-5-20251001":  200_000,
}



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

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "constructs.db"
_init_db(DB_PATH)


def create_session() -> str:
    """Create a new conversation session."""
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
    _save_sessions()
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


# ── Agent loop ──────────────────────────────────────────────────────────
# Streams via the Anthropic API directly (~1s TTFT). Tool calls dispatch
# to src/tools.py:ALL_TOOLS handlers in-process — same implementations the
# Agent SDK MCP server (app/agent.py, evals) uses.

_anthropic_client: anthropic.Anthropic | None = None


def _client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _emit_tool_result(
    tool_name: str,
    tool_input: dict,
    result_str: str,
    *,
    safe_write,
    session: dict,
    assistant_blocks: list,
) -> None:
    """Emit SSE event(s) for a completed tool call and apply side effects.

    Handles export_construct download/plot, log_experimental_outcome
    session persistence, and display-message recording. Shared between
    the streaming loop and any future callers so behaviour stays in sync.
    """
    if (
        tool_name == "log_experimental_outcome"
        and result_str.startswith("[OUTCOME_LOGGED]")
    ):
        session.setdefault("experimental_outcomes", []).append({
            "status": tool_input.get("status"),
            "observation": tool_input.get("observation"),
            "construct_name": tool_input.get("construct_name", ""),
            "timestamp": time.time(),
        })
        _save_sessions()
    if tool_name == "export_construct":
        fmt = tool_input.get("output_format", "raw")
        cname = tool_input.get("construct_name", "construct")
        ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
        filename = cname + ext
        display_result = f"Exported: {filename}"
        event_data = {
            "type": "tool_result",
            "tool": tool_name,
            "input": tool_input,
            "content": display_result,
            "download_content": result_str,
            "download_filename": filename,
        }
    else:
        display_result = result_str[:2000] + "..." if len(result_str) > 2000 else result_str
        event_data = {
            "type": "tool_result",
            "tool": tool_name,
            "input": tool_input,
            "content": display_result,
        }
    safe_write(event_data)
    plot = get_last_plot_json()
    if tool_name == "export_construct" and plot:
        safe_write({"type": "plot_data", "plot_json": json.loads(plot)})
        clear_last_plot_json()
    assistant_blocks.append({
        "type": "tool_use",
        "name": tool_name,
        "input": tool_input,
        "result": display_result,
        "download_content": event_data.get("download_content"),
        "download_filename": event_data.get("download_filename"),
    })


def run_agent_turn_streaming(user_message: str, session_id: str, write_event, model: str = MODEL):
    """Run one agent turn with streaming, scoped to a session."""
    _cancelled_sessions.discard(session_id)

    session = get_session(session_id)
    if not session:
        write_event({"type": "error", "content": "Session not found"})
        return

    # Guard against concurrent turns on the same session. ThreadingMixIn means
    # two HTTP requests can race: the old turn's history.append(assistant+tool_use)
    # and the new turn's history.append(user_message) interleave, leaving an
    # orphaned tool_use block that causes API 400 errors on the next request.
    if session_id in _active_turns:
        write_event({
            "type": "error",
            "content": (
                "A previous response is still being generated for this session. "
                "Please wait for it to finish or click Stop first."
            ),
        })
        write_event({"type": "done"})
        return
    _active_turns.add(session_id)

    # Per-turn live event log so clients can reconnect and replay the stream
    _live_log: list = []
    _live_cond = threading.Condition()
    with _session_live_streams_lock:
        _session_live_streams[session_id] = (_live_log, _live_cond)

    _orig_write_event = write_event
    def write_event(data):  # type: ignore[assignment]
        _orig_write_event(data)
        with _live_cond:
            _live_log.append(data)
            _live_cond.notify_all()

    tracker = ReferenceTracker()
    set_tracker(tracker)
    clear_last_plot_json()
    export_called = False
    # Build the system prompt once per turn (not per retry) so that
    # prompt caching works. The prompt is dynamic because it includes
    # per-session troubleshooting context (experimental_outcomes).
    turn_system_prompt = _build_system_prompt(session)
    history = session["history"]
    history.append({"role": "user", "content": user_message})
    session["display_messages"].append({"role": "user", "content": user_message, "timestamp": time.time()})

    if session["first_message"] is None:
        session["first_message"] = user_message[:80]

    disconnected = False
    is_cancelled = lambda: session_id in _cancelled_sessions

    def safe_write(data: dict):
        nonlocal disconnected
        if disconnected or is_cancelled():
            return
        try:
            write_event(data)
        except (BrokenPipeError, ConnectionResetError):
            disconnected = True

    max_iterations = 15
    max_retries = 3
    assistant_text = ""
    assistant_blocks: list[dict] = []
    current_thinking_text = ""
    current_text_content = ""

    try:
        for _ in range(max_iterations):
            if is_cancelled():
                break

            stop_reason = None
            final_message = None
            tool_results: list = []

            for retry_attempt in range(max_retries + 1):
                # Reset per-API-call state on each retry. If a stream partially
                # succeeded before rate-limiting, any tool_results accumulated
                # reference tool_use_ids from the aborted stream — replaying
                # them alongside the retry's fresh tool_use_ids causes a 400.
                current_block_type = None
                current_tool_name = None
                current_tool_id = None
                current_tool_input_json = ""
                thinking_block_emitted = False
                tool_results = []
                try:
                    thinking_config = (
                        {"type": "adaptive"}
                        if model.startswith("claude-opus-4-7")
                        else {"type": "enabled", "budget_tokens": 5000}
                    )
                    with _client().messages.stream(
                        model=model,
                        max_tokens=16000,
                        system=turn_system_prompt,
                        tools=TOOLS,
                        messages=history,
                        thinking=thinking_config,
                    ) as stream:
                        for event in stream:
                            if is_cancelled():
                                stream.close()
                                break

                            if event.type == "content_block_start":
                                block = event.content_block
                                if block.type == "thinking":
                                    current_block_type = "thinking"
                                    current_thinking_text = ""
                                    thinking_block_emitted = False
                                elif block.type == "text":
                                    current_block_type = "text"
                                    current_text_content = ""
                                    safe_write({"type": "text_start"})
                                elif block.type == "tool_use":
                                    current_block_type = "tool_use"
                                    current_tool_name = block.name
                                    current_tool_id = block.id
                                    current_tool_input_json = ""
                                    safe_write({"type": "tool_use_start", "tool": block.name})

                            elif event.type == "content_block_delta":
                                delta = event.delta
                                if delta.type == "thinking_delta":
                                    current_thinking_text += delta.thinking
                                    if not thinking_block_emitted:
                                        safe_write({"type": "thinking_start"})
                                        thinking_block_emitted = True
                                    safe_write({"type": "thinking_delta", "content": delta.thinking})
                                elif delta.type == "text_delta":
                                    assistant_text += delta.text
                                    current_text_content += delta.text
                                    safe_write({"type": "text_delta", "content": delta.text})
                                elif delta.type == "input_json_delta":
                                    current_tool_input_json += delta.partial_json

                            elif event.type == "content_block_stop":
                                if current_block_type == "thinking":
                                    assistant_blocks.append({"type": "thinking", "content": current_thinking_text})
                                    if thinking_block_emitted:
                                        safe_write({"type": "thinking_end"})
                                elif current_block_type == "text":
                                    assistant_blocks.append({"type": "text", "content": current_text_content})
                                    safe_write({"type": "text_end"})
                                elif current_block_type == "tool_use":
                                    if is_cancelled():
                                        break
                                    tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                                    result_str = _dispatch_tool(current_tool_name, tool_input)
                                    if current_tool_name == "export_construct":
                                        export_called = True
                                    _emit_tool_result(
                                        current_tool_name, tool_input, result_str,
                                        safe_write=safe_write, session=session,
                                        assistant_blocks=assistant_blocks,
                                    )
                                    tool_results.append({
                                        "type": "tool_result",
                                        "tool_use_id": current_tool_id,
                                        "content": result_str,
                                    })
                                current_block_type = None

                            elif event.type == "message_delta":
                                stop_reason = event.delta.stop_reason

                        if is_cancelled():
                            break

                        final_message = stream.get_final_message()
                        if final_message and hasattr(final_message, "usage"):
                            safe_write({
                                "type": "token_usage",
                                "input_tokens": final_message.usage.input_tokens,
                                "context_window": CONTEXT_WINDOW.get(model, 1_000_000),
                            })
                    break  # stream succeeded, leave retry loop

                except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                    if retry_attempt < max_retries:
                        wait_time = 2 ** retry_attempt
                        kind = "Rate limited" if isinstance(e, anthropic.RateLimitError) else "Server error"
                        safe_write({"type": "text_delta", "content": f"\n[{kind}, retrying in {wait_time}s...]\n"})
                        time.sleep(wait_time)
                        continue
                    safe_write({"type": "error", "content": f"{type(e).__name__} after retries. Please try again."})
                    break
                except Exception:
                    if is_cancelled():
                        break
                    raise

            if is_cancelled() or final_message is None:
                break

            # Convert content blocks to plain dicts to strip extra SDK fields
            # (e.g. parsed_output) that cause 400 errors on replay. Unknown
            # block types are dropped — passing them through can fail when
            # the SDK emits a new type we don't handle.
            filtered_content = []
            for b in final_message.content:
                btype = getattr(b, "type", None)
                if btype == "thinking":
                    # Opus 4.7 adaptive thinking requires thinking blocks with
                    # signatures to be preserved in multi-turn history.
                    # Stripping them causes the model to lose reasoning context
                    # and produce inconsistent tool_use/tool_result sequences.
                    if model.startswith("claude-opus-4-7"):
                        filtered_content.append({
                            "type": "thinking",
                            "thinking": b.thinking,
                            "signature": b.signature,
                        })
                    continue
                if btype == "text":
                    filtered_content.append({"type": "text", "text": b.text})
                elif btype == "tool_use":
                    filtered_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                else:
                    logger.warning("Dropping unknown content block type from history: %s", btype or type(b).__name__)
            history.append({"role": "assistant", "content": filtered_content})

            if tool_results:
                history.append({"role": "user", "content": tool_results})
            else:
                break

            if stop_reason == "end_turn":
                break
    finally:
        # All post-loop work is here so it runs whether the loop exited normally
        # or via an exception, and _active_turns.discard happens last — after the
        # session is saved — so the polling indicator sees a consistent state.

        # Flush any in-progress block that was interrupted mid-stream
        if current_text_content and not any(
            b.get("type") == "text" and b.get("content") == current_text_content
            for b in assistant_blocks
        ):
            assistant_blocks.append({"type": "text", "content": current_text_content})
        if current_thinking_text and not any(
            b.get("type") == "thinking" and b.get("content") == current_thinking_text
            for b in assistant_blocks
        ):
            assistant_blocks.append({"type": "thinking", "content": current_thinking_text})

        # Append formatted references only when a sequence file was exported this turn
        if export_called and not is_cancelled():
            refs_text = tracker.format_references()
            if refs_text:
                ref_block = f"\n\n{refs_text}"
                assistant_text += ref_block
                assistant_blocks.append({"type": "text", "content": ref_block})
                safe_write({"type": "text_start"})
                safe_write({"type": "text_delta", "content": ref_block})
                safe_write({"type": "text_end"})
            session["last_export_references"] = tracker.to_list()

        if assistant_text or assistant_blocks:
            session["display_messages"].append({
                "role": "assistant",
                "content": assistant_text,
                "blocks": assistant_blocks,
            })
        elif is_cancelled():
            # Remove dangling user message if the run was explicitly cancelled
            if history and history[-1]["role"] == "user" and isinstance(history[-1].get("content"), str):
                history.pop()
                if session["display_messages"] and session["display_messages"][-1]["role"] == "user":
                    session["display_messages"].pop()

        _save_sessions()

        if not disconnected:
            try:
                write_event({"type": "done"})
            except (BrokenPipeError, ConnectionResetError):
                pass

        set_tracker(None)
        _active_turns.discard(session_id)  # Last: poll / status endpoint now sees saved state

        # Signal the sentinel so reconnect-stream consumers know we're done
        with _live_cond:
            _live_log.append(None)
            _live_cond.notify_all()
        with _session_live_streams_lock:
            _session_live_streams.pop(session_id, None)


# ── HTML UI ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plasmid Designer</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23D97757' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5'/%3E%3C/svg%3E">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.1.min.js"></script>
<link href="https://unpkg.com/tabulator-tables@6.3.0/dist/css/tabulator_bootstrap5.min.css" rel="stylesheet">
<script src="https://unpkg.com/tabulator-tables@6.3.0/dist/js/tabulator.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
<script src="https://unpkg.com/klayjs@0.4.1/klay.js"></script>
<script src="https://unpkg.com/cytoscape-klay@3.1.4/cytoscape-klay.js"></script>
<style>
  :root {
    --brand-fig: #D97757;
    --brand-fig-hover: #B5603F;
    --brand-fig-10: rgba(217,119,87,0.1);
    --brand-fig-30: rgba(217,119,87,0.3);
    --brand-aqua: #24B283;
    --brand-aqua-dark: #0E6B54;
    --brand-aqua-10: rgba(36,178,131,0.1);
    --brand-aqua-20: rgba(36,178,131,0.2);
    --brand-orange: #E86235;
    --brand-orange-100: #FAEFEB;
    --brand-coral: #F5E0D8;
    --brand-coral-30: rgba(245,224,216,0.3);
    --sand-50: #FAF9F7;
    --sand-100: #F5F3EF;
    --sand-200: #E8E6DC;
    --sand-300: #D4D1C7;
    --sand-400: #ADAAA0;
    --sand-500: #87867F;
    --sand-600: #5C5B56;
    --sand-700: #3D3D3A;
    --sand-800: #2A2A28;
    --sand-900: #1A1A19;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: white;
    color: var(--sand-900);
    height: 100vh;
    display: flex;
    flex-direction: column;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Header ── */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    border-bottom: 1px solid var(--sand-200);
    background: white;
    flex-shrink: 0;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .header-logo {
    width: 32px; height: 32px; border-radius: 8px;
    background: var(--brand-fig-10);
    display: flex; align-items: center; justify-content: center;
  }
  .header-logo svg { width: 20px; height: 20px; stroke: var(--brand-fig); fill: none; }
  .header-title h1 { font-size: 16px; font-weight: 600; color: var(--sand-800); line-height: 1.2; }
  .header-title p { font-size: 12px; color: var(--sand-400); line-height: 1.2; }
  .health-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 500; border: 1px solid; transition: all 0.2s;
  }
  .health-badge.online {
    background: var(--brand-aqua-10); color: var(--brand-aqua-dark); border-color: var(--brand-aqua-20);
  }
  .health-badge.offline {
    background: var(--sand-100); color: var(--sand-500); border-color: var(--sand-200);
  }
  .health-dot { width: 6px; height: 6px; border-radius: 50%; }
  .health-badge.online .health-dot { background: var(--brand-aqua); }
  .health-badge.offline .health-dot { background: var(--sand-400); }

  /* ── Layout ── */
  .main { flex: 1; display: flex; overflow: hidden; }

  /* ── Sidebar ── */
  .sidebar {
    width: 240px; background: var(--sand-50);
    border-right: 1px solid var(--sand-200);
    display: flex; flex-direction: column; flex-shrink: 0;
    transition: width 0.3s ease; overflow: hidden;
  }
  .sidebar.collapsed { width: 0; border-right: none; }
  .sidebar-toolbar {
    padding: 12px; display: flex; align-items: center; gap: 8px;
  }
  .sidebar-toggle-btn {
    width: 32px; height: 32px; border: none; background: none;
    border-radius: 8px; color: var(--sand-400); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: all 0.15s;
  }
  .sidebar-toggle-btn:hover { color: var(--sand-600); background: var(--sand-100); }
  .new-chat-btn {
    flex: 1; display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border: 1px solid var(--sand-200);
    background: none; border-radius: 8px; color: var(--sand-600);
    font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.15s;
    font-family: inherit;
  }
  .new-chat-btn:hover { background: var(--sand-100); }
  .sessions-list {
    flex: 1; overflow-y: auto; padding: 0 8px 12px;
    scrollbar-width: thin; scrollbar-color: transparent transparent;
  }
  .sessions-list:hover { scrollbar-color: var(--sand-300) transparent; }
  .session-item {
    width: 100%; text-align: left; padding: 10px 12px;
    border-radius: 8px; border: none; background: none;
    color: var(--sand-500); font-size: 14px; cursor: pointer;
    transition: all 0.15s; display: flex; align-items: center;
    justify-content: space-between; gap: 8px; margin-bottom: 2px;
    font-family: inherit;
  }
  .session-item:hover { background: var(--sand-100); color: var(--sand-700); }
  .session-item.active { background: var(--sand-200); color: var(--sand-800); }
  .session-name {
    overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; font-weight: 500; flex: 1;
  }
  .delete-btn {
    opacity: 0; border: none; background: none;
    color: var(--sand-300); cursor: pointer; padding: 2px;
    flex-shrink: 0; transition: opacity 0.15s, color 0.15s;
  }
  .session-item:hover .delete-btn { opacity: 1; }
  .delete-btn:hover { color: var(--brand-orange); }
  .no-sessions { text-align: center; font-size: 12px; color: var(--sand-400); margin-top: 16px; }
  .user-library-panel {
    border-top: 1px solid var(--sand-200); padding: 8px;
    flex-shrink: 0;
  }
  .user-library-toggle {
    width: 100%; text-align: left; padding: 8px 12px;
    border: none; background: none; border-radius: 8px;
    color: var(--sand-500); font-size: 12px; font-weight: 600;
    cursor: pointer; display: flex; align-items: center;
    justify-content: space-between; gap: 6px;
    font-family: inherit; transition: background 0.15s;
  }
  .user-library-toggle:hover { background: var(--sand-100); color: var(--sand-700); }
  .user-library-toggle .chevron { transition: transform 0.2s; font-style: normal; }
  .user-library-toggle.open .chevron { transform: rotate(180deg); }
  .user-library-refresh-btn {
    border: none; background: none; color: var(--sand-400); cursor: pointer;
    padding: 2px 4px; border-radius: 4px; font-size: 14px; line-height: 1;
    flex-shrink: 0; transition: color 0.15s, background 0.15s;
  }
  .user-library-refresh-btn:hover { color: var(--sand-700); background: var(--sand-200); }
  .user-library-body { display: none; padding: 0 4px 4px; }
  .user-library-body.open {
    display: block; max-height: 240px; overflow-y: auto;
    scrollbar-width: thin; scrollbar-color: transparent transparent;
  }
  .user-library-body.open:hover { scrollbar-color: var(--sand-300) transparent; }
  .user-library-section { margin-bottom: 8px; }
  .user-library-section-title {
    font-size: 11px; color: var(--sand-400); font-weight: 600;
    padding: 4px 8px 2px; text-transform: uppercase; letter-spacing: 0.04em;
  }
  .user-library-entry {
    padding: 4px 8px; border-radius: 6px; font-size: 12px;
    color: var(--sand-600); line-height: 1.4; cursor: pointer;
    transition: background 0.12s;
  }
  .user-library-entry:hover { background: var(--sand-100); }
  .user-library-entry.expanded { background: var(--sand-100); }
  .user-library-entry .entry-header { display: flex; align-items: center; justify-content: space-between; gap: 4px; }
  .user-library-entry .entry-name { font-weight: 500; color: var(--sand-700); }
  .user-library-entry .entry-meta { color: var(--sand-400); font-size: 11px; }
  .user-library-entry .entry-chevron {
    color: var(--sand-300); font-size: 10px; flex-shrink: 0;
    transition: transform 0.15s; font-style: normal;
  }
  .user-library-entry.expanded .entry-chevron { transform: rotate(180deg); }
  .entry-detail {
    display: none; margin-top: 4px; padding: 6px 0 2px;
    border-top: 1px solid var(--sand-200);
  }
  .entry-detail.open { display: block; }
  .entry-detail-row {
    display: flex; gap: 6px; font-size: 11px; line-height: 1.5;
    color: var(--sand-500);
  }
  .entry-detail-row .dk { color: var(--sand-400); flex-shrink: 0; min-width: 72px; }
  .entry-detail-row .dv { color: var(--sand-700); word-break: break-all; }
  .user-library-empty { font-size: 12px; color: var(--sand-400); padding: 4px 8px; }
  .sidebar-reopen-btn {
    position: absolute; top: 68px; left: 12px; z-index: 10;
    width: 32px; height: 32px; border: none; background: none;
    border-radius: 8px; color: var(--sand-400); cursor: pointer;
    display: none; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  .sidebar-reopen-btn:hover { color: var(--sand-600); background: var(--sand-100); }
  .sidebar-reopen-btn.visible { display: flex; }

  /* ── Chat Panel ── */
  .chat-panel { flex: 1; display: flex; flex-direction: column; background: white; min-width: 0; position: relative; }
  .messages {
    flex: 1; overflow-y: auto; padding: 24px;
    scrollbar-width: thin; scrollbar-color: transparent transparent;
  }
  .messages:hover { scrollbar-color: var(--sand-300) transparent; }
  .messages::-webkit-scrollbar { width: 6px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb { background: transparent; border-radius: 3px; }
  .messages:hover::-webkit-scrollbar-thumb { background: var(--sand-300); }
  .messages-inner { max-width: 768px; margin: 0 auto; }

  /* ── Welcome ── */
  .welcome {
    display: flex; align-items: center; justify-content: center;
    padding-top: 128px; text-align: center;
  }
  .welcome-icon {
    width: 48px; height: 48px; border-radius: 50%;
    background: var(--brand-coral-30);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 12px;
  }
  .welcome-icon svg { width: 24px; height: 24px; stroke: var(--brand-fig); fill: none; }
  .welcome h2 {
    font-size: 16px; font-weight: 500; color: var(--sand-600); margin-bottom: 8px;
  }
  .welcome p { font-size: 13px; color: var(--sand-400); line-height: 1.5; margin-bottom: 4px; }
  .examples {
    display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px;
  }
  .examples button {
    background: white; border: 1px solid var(--sand-200);
    border-radius: 12px; padding: 8px 14px; color: var(--sand-600);
    font-size: 13px; cursor: pointer; transition: border-color 0.15s, color 0.15s;
    font-family: inherit;
  }
  .examples button:hover { border-color: var(--brand-fig); color: var(--sand-800); }

  /* ── Messages ── */
  .msg { margin-bottom: 24px; display: flex; }
  .msg.user { justify-content: flex-end; }
  .msg.assistant { justify-content: flex-start; }
  .msg-bubble-user {
    background: var(--sand-100); color: var(--sand-800);
    border-radius: 16px 16px 4px 16px;
    padding: 10px 16px; max-width: 80%;
    font-size: 14px; line-height: 1.6;
    white-space: pre-wrap; word-wrap: break-word;
  }
  .msg-date {
    font-size: 11px; color: var(--sand-400);
    text-align: right; margin-top: 4px;
  }
  .msg-bubble-assistant {
    color: var(--sand-800); max-width: 80%;
    font-size: 14px; line-height: 1.6;
  }
  /* Batch cards fill the full message column width */
  .msg.assistant:has(.batch-card) { width: 100%; }
  .msg.assistant:has(.batch-card) > .batch-card { max-width: none; }

  /* ── Streaming cursor ── */
  .streaming-cursor::after {
    content: '\25CF';
    animation: blink 1s step-end infinite;
    color: var(--brand-fig);
    font-size: 0.5em; vertical-align: middle; margin-left: 2px;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── Working indicator ── */
  .working-indicator {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0; color: var(--sand-500); font-size: 13px;
  }
  .working-dots { display: flex; align-items: center; gap: 3px; }
  .working-dots span {
    display: inline-block; width: 5px; height: 5px;
    background: var(--brand-fig); border-radius: 50%;
    animation: working-bounce 1.1s ease-in-out infinite;
    opacity: 0.35;
  }
  .working-dots span:nth-child(2) { animation-delay: 0.18s; }
  .working-dots span:nth-child(3) { animation-delay: 0.36s; }
  @keyframes working-bounce {
    0%, 100% { transform: translateY(0); opacity: 0.35; }
    40% { transform: translateY(-4px); opacity: 1; }
  }
  .slow-note {
    font-size: 12px; color: var(--sand-400);
    margin-top: 4px; font-style: italic;
    animation: fadein 0.6s ease;
  }
  @keyframes fadein { from { opacity: 0; } to { opacity: 1; } }

  /* ── Collapsible blocks (thinking + tool) ── */
  .thinking-block, .tool-block { margin: 6px 0; }
  .block-card {
    border: 1px solid var(--sand-200); border-radius: 8px; overflow: hidden;
  }
  .block-header {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px; background: var(--sand-50);
    cursor: pointer; user-select: none; transition: background 0.15s;
  }
  .block-header:hover { background: var(--sand-100); }
  .block-header svg { width: 14px; height: 14px; flex-shrink: 0; }
  .block-label { font-size: 12px; font-weight: 500; color: var(--sand-600); }
  .block-meta { margin-left: auto; font-size: 11px; color: var(--sand-400); }
  .block-chevron {
    width: 14px; height: 14px; color: var(--sand-400);
    transition: transform 0.2s; flex-shrink: 0;
  }
  .block-chevron.open { transform: rotate(90deg); }
  .block-body {
    display: none; padding: 8px 12px;
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
    font-size: 12px; color: var(--sand-600);
    white-space: pre-wrap; word-break: break-word;
    max-height: 256px; overflow-y: auto; line-height: 1.5;
    border-top: 1px solid var(--sand-100); background: var(--sand-50);
  }
  .block-body.open { display: block; }
  .block-body .section { margin-bottom: 8px; }
  .block-body .label {
    font-size: 11px; color: var(--sand-400);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
  }
  .pulse-dot {
    display: inline-block; width: 6px; height: 6px;
    background: var(--brand-fig); border-radius: 50%;
    animation: pulse-tool 1.5s ease-in-out infinite;
    margin-left: 4px;
  }
  @keyframes pulse-tool { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }

  /* ── Code blocks & tables ── */
  .msg-bubble-assistant pre, .msg-bubble-assistant code {
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    background: var(--sand-50); border-radius: 4px;
  }
  .msg-bubble-assistant code { padding: 2px 5px; font-size: 13px; }
  .msg-bubble-assistant pre {
    padding: 12px; overflow-x: auto; margin: 8px 0;
    border: 1px solid var(--sand-200); font-size: 12px; line-height: 1.5;
  }
  .msg-bubble-assistant pre code { padding: 0; background: none; }
  .msg-bubble-assistant table {
    width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12px;
  }
  .msg-bubble-assistant th, .msg-bubble-assistant td {
    border: 1px solid var(--sand-200); padding: 6px 12px; text-align: left;
  }
  .msg-bubble-assistant th { background: var(--sand-50); font-weight: 600; }
  .msg-bubble-assistant tr:nth-child(even) { background: var(--sand-50); }

  /* ── DNA sequence display ── */
  .seq-block { position: relative; }
  .seq-block pre { padding-right: 40px !important; }
  .seq-copy-btn {
    background: none; border: 1px solid transparent; cursor: pointer; padding: 3px 4px;
    color: var(--sand-500); border-radius: 3px; line-height: 1;
    display: inline-flex; align-items: center; justify-content: center;
    vertical-align: middle; margin-left: 4px; transition: color 0.1s, background 0.1s, border-color 0.1s;
  }
  .seq-copy-btn:hover { color: var(--sand-800); background: var(--sand-100); border-color: var(--sand-300); }
  .seq-copy-btn.copied { color: #16a34a; }
  .seq-copy-btn.block-btn { position: absolute; top: 6px; right: 6px; margin-left: 0; }
  .seq-table { margin: 8px 0; overflow-x: auto; }
  .seq-table table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .seq-table th, .seq-table td { border: 1px solid var(--sand-200); padding: 5px 10px; text-align: left; }
  .seq-table th { background: var(--sand-50); font-weight: 600; position: relative; }
  .seq-table tr:nth-child(even) { background: var(--sand-50); }
  .seq-table td:first-child, .seq-table th:first-child { width: 1%; white-space: nowrap; }
  .seq-table td:last-child { width: 32px; text-align: center; padding: 3px; }
  .col-resizer {
    position: absolute; right: 0; top: 0; bottom: 0; width: 4px;
    cursor: col-resize; user-select: none; z-index: 1;
  }
  .col-resizer:hover, .col-resizer:active { background: var(--sand-300); }
  code.dna-seq { letter-spacing: 0.03em; }
  .tbl-copy-row { display: flex; justify-content: flex-end; margin-top: 4px; }
  .code-block-wrap { position: relative; }
  .code-block-wrap pre { padding-right: 40px !important; }
  .code-block-wrap .seq-copy-btn { position: absolute; top: 6px; right: 6px; margin-left: 0; }

  /* ── Input area ── */
  .input-area { padding: 8px 24px 24px; flex-shrink: 0; }
  .input-wrapper {
    max-width: 768px; margin: 0 auto; position: relative;
  }
  .input-wrapper textarea {
    width: 100%; resize: none;
    border: 1px solid var(--sand-200); border-radius: 16px;
    padding: 12px 16px 48px; font-family: inherit; font-size: 14px;
    color: var(--sand-900); outline: none;
    min-height: 52px; max-height: 200px; line-height: 1.4;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .input-wrapper textarea:focus {
    border-color: rgba(217,119,87,0.5);
    box-shadow: 0 0 0 3px rgba(217,119,87,0.1);
  }
  .input-wrapper textarea::placeholder { color: var(--sand-400); }
  .input-wrapper textarea:disabled { background: var(--sand-50); color: var(--sand-400); }
  .input-meta { position: absolute; left: 12px; bottom: 12px; display: flex; align-items: center; }
  .model-select {
    appearance: none; -webkit-appearance: none;
    background: var(--sand-50); border: 1px solid var(--sand-200); border-radius: 8px;
    padding: 4px 24px 4px 8px; font-size: 11px; font-family: inherit;
    color: var(--sand-500); cursor: pointer; outline: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23ADAAA0'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 6px center;
    transition: border-color 0.15s;
  }
  .model-select:hover { border-color: var(--sand-300); }
  .model-select:focus { border-color: var(--brand-fig); }
  .token-indicator {
    display: none; align-items: center; gap: 5px;
    margin-left: 8px; font-size: 11px; color: var(--sand-400);
    white-space: nowrap;
  }
  .token-indicator.visible { display: flex; }
  .token-bar-track {
    width: 48px; height: 4px; border-radius: 2px;
    background: var(--sand-200); overflow: hidden;
  }
  .token-bar-fill {
    height: 100%; border-radius: 2px;
    background: var(--brand-aqua);
    transition: width 0.4s ease, background 0.4s ease;
  }
  .token-bar-fill.warn  { background: var(--brand-fig); }
  .token-bar-fill.alert { background: var(--brand-orange); }
  .input-buttons { position: absolute; right: 12px; bottom: 12px; }
  .send-btn, .stop-btn {
    width: 36px; height: 36px; border: none; border-radius: 12px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  .send-btn {
    background: var(--brand-fig); color: white;
  }
  .send-btn:hover { background: var(--brand-fig-hover); }
  .send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .send-btn svg, .stop-btn svg { width: 16px; height: 16px; }
  .stop-btn {
    background: white; border: 1px solid var(--sand-300); color: var(--sand-600);
  }
  .stop-btn:hover { background: var(--sand-50); }

  /* ── Download button ── */
  .download-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border: 1px solid var(--brand-aqua-20);
    background: var(--brand-aqua-10); border-radius: 8px;
    color: var(--brand-aqua-dark); font-size: 12px; font-weight: 500;
    cursor: pointer; transition: all 0.15s; font-family: inherit;
  }
  .download-btn:hover { background: var(--brand-aqua-20); border-color: var(--brand-aqua); }
  .download-btn svg { flex-shrink: 0; }
  /* ── Download split button ── */
  .dl-split-wrap { position: relative; display: inline-flex; align-items: stretch; }
  .dl-split-wrap .download-btn { border-radius: 8px 0 0 8px; border-right: none; }
  .dl-chevron-btn {
    display: inline-flex; align-items: center; justify-content: center; padding: 0 9px;
    border: 1px solid var(--brand-aqua-20); border-left: 1px solid var(--brand-aqua-30, rgba(62,169,159,0.3));
    background: var(--brand-aqua-10); border-radius: 0 8px 8px 0;
    color: var(--brand-aqua-dark); cursor: pointer; transition: all 0.15s; font-family: inherit;
  }
  .dl-chevron-btn:hover { background: var(--brand-aqua-20); border-color: var(--brand-aqua); }
  .dl-menu {
    display: none; position: absolute; top: calc(100% + 4px); left: 0; z-index: 300;
    background: #FAFAF8; border: 1px solid var(--sand-200); border-radius: 8px;
    padding: 4px; min-width: 200px; box-shadow: 0 4px 14px rgba(0,0,0,0.10);
  }
  .dl-menu.open { display: block; }
  .dl-menu-item {
    display: flex; align-items: center; gap: 8px; width: 100%; text-align: left;
    padding: 7px 10px; border: none; background: none; cursor: pointer;
    font-size: 12px; font-family: inherit; border-radius: 5px; color: var(--sand-700);
  }
  .dl-menu-item:hover { background: var(--sand-100); }
  .dl-rename-row {
    display: flex; align-items: center; gap: 6px; margin-top: 6px; flex-wrap: wrap;
  }
  .dl-rename-row input {
    flex: 1; min-width: 120px; padding: 4px 8px; border: 1px solid var(--sand-300);
    border-radius: 6px; font-size: 12px; font-family: inherit; background: #fff;
    color: var(--sand-800);
  }
  .dl-rename-row input:focus { outline: none; border-color: var(--brand-aqua); }
  .dl-rename-confirm { padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 500;
    font-family: inherit; cursor: pointer; border: none;
    background: var(--brand-aqua, #3ea99f); color: #fff; }
  .dl-rename-confirm:hover { opacity: 0.85; }
  .dl-rename-cancel { padding: 4px 8px; border-radius: 6px; font-size: 12px;
    font-family: inherit; cursor: pointer; border: 1px solid var(--sand-200);
    background: none; color: var(--sand-600); }
  .dl-rename-cancel:hover { background: var(--sand-100); }
  /* ── Batch confirmation card ── */
  .batch-confirm-card {
    background: white; border: 1px solid var(--sand-200); border-radius: 10px;
    padding: 16px 20px; max-width: 480px;
  }
  .batch-confirm-rows {
    max-height: 180px; overflow-y: auto; border: 1px solid var(--sand-200);
    border-radius: 6px; margin-bottom: 12px;
  }
  .batch-confirm-row {
    display: flex; align-items: center; gap: 8px; padding: 6px 10px;
    border-bottom: 1px solid var(--sand-100); cursor: pointer; transition: background 0.1s;
  }
  .batch-confirm-row:last-child { border-bottom: none; }
  .batch-confirm-row:hover { background: var(--sand-50); }
  .batch-confirm-row input[type=checkbox] { flex-shrink: 0; accent-color: var(--brand-fig); cursor: pointer; }
  .batch-confirm-row-num { font-size: 11px; color: var(--sand-400); flex-shrink: 0; width: 20px; text-align: right; }
  .batch-confirm-row-desc { font-size: 12px; color: var(--sand-700); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .batch-confirm-row-name { font-size: 11px; color: var(--sand-400); flex-shrink: 0; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .batch-confirm-select-all {
    display: flex; align-items: center; gap: 8px; padding: 6px 10px;
    background: var(--sand-50); border: 1px solid var(--sand-200); border-radius: 6px;
    margin-bottom: 6px; font-size: 12px; color: var(--sand-600); cursor: pointer;
  }
  .batch-confirm-select-all input[type=checkbox] { accent-color: var(--brand-fig); cursor: pointer; }
  .batch-advisory {
    background: var(--brand-orange-100, #fff4ee); border: 1px solid rgba(232,98,53,0.18);
    border-radius: 7px; padding: 10px 13px; margin-bottom: 12px; font-size: 12px;
    color: var(--sand-700); line-height: 1.55;
  }
  .batch-advisory strong { color: var(--sand-800); }
  .batch-advisory ul { margin: 6px 0 0 0; padding-left: 16px; }
  .batch-advisory ul li { margin-bottom: 4px; }
  /* ── Per-row pause/resume button ── */
  .batch-row-pause-btn {
    flex-shrink: 0; width: 22px; height: 22px; border-radius: 6px; border: none;
    background: transparent; color: var(--sand-400); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.12s, color 0.12s; margin-top: 1px;
  }
  .batch-row-pause-btn:hover { background: var(--sand-100); color: var(--sand-700); }

  /* ── Error ── */
  .error-banner {
    background: var(--brand-orange-100); border: 1px solid rgba(232,98,53,0.2);
    color: var(--brand-orange); border-radius: 8px; padding: 12px 16px;
    font-size: 13px; margin-bottom: 24px;
  }

  /* ── Drop overlay (shown when a file is dragged over the chat area) ── */
  .drop-overlay {
    display: none; position: absolute; inset: 0; z-index: 50;
    background: rgba(217,119,87,0.06); border: 3px dashed var(--brand-fig);
    border-radius: 0; align-items: center; justify-content: flex-end;
    flex-direction: column; gap: 10px; pointer-events: none; padding-bottom: 144px;
  }
  .drop-overlay.active { display: flex; }
  .drop-overlay-label { font-size: 16px; font-weight: 600; color: var(--brand-fig); }
  .drop-overlay-sub { font-size: 13px; color: var(--brand-fig-hover); text-align: center; }
  /* ── Attach button (plasmid file picker) ── */
  .attach-btn {
    width: 34px; height: 34px; border-radius: 8px; border: none;
    background: transparent; color: var(--sand-400); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s, color 0.15s;
  }
  .attach-btn:hover { background: var(--sand-100); color: var(--sand-700); }

  /* ── Batch cards (rendered inline in the chat) ── */
  .batch-card {
    border: 1px solid var(--sand-200); border-radius: 10px;
    overflow: hidden; background: white; width: 100%;
  }
  .batch-plot-wrapper { overflow: visible; }
  .batch-row-header {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px; cursor: pointer; user-select: none;
    transition: background 0.12s;
  }
  .batch-row-header:hover { background: var(--sand-50); }
  .batch-row-status { flex-shrink: 0; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; margin-top: 1px; }
  .batch-row-body { flex: 1; min-width: 0; }
  .batch-row-desc { font-size: 13px; color: var(--sand-700); font-weight: 500; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .batch-row-meta { font-size: 12px; color: var(--sand-400); }
  .batch-row-downloads { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
  .batch-row-chevron { flex-shrink: 0; color: var(--sand-300); margin-top: 3px; transition: transform 0.2s; }
  .batch-row-chevron.open { transform: rotate(90deg); }
  .batch-row-log {
    display: none; border-top: 1px solid var(--sand-100);
    padding: 12px 16px; background: var(--sand-50);
  }
  .batch-row-log.open { display: block; }
  .batch-log-entry { margin-bottom: 8px; font-size: 12px; }
  .batch-log-tool {
    border: 1px solid var(--sand-200); border-radius: 6px; overflow: hidden;
  }
  .batch-log-tool-header {
    padding: 4px 8px; background: var(--sand-100);
    font-weight: 600; color: var(--sand-700);
    display: flex; align-items: center; gap: 6px;
  }
  .batch-log-tool-result {
    padding: 6px 8px; color: var(--sand-600);
    white-space: pre-wrap; word-break: break-word;
    max-height: 140px; overflow-y: auto; line-height: 1.5;
  }
  .batch-log-text { color: var(--sand-600); line-height: 1.5; padding: 2px 0; }
  .batch-log-user {
    background: var(--sand-100); border-radius: 8px; padding: 6px 10px;
    color: var(--sand-700); line-height: 1.5;
  }
  .batch-log-error { color: var(--brand-orange); line-height: 1.5; }
  /* Follow-up input inside expanded batch card */
  .batch-followup {
    display: flex; gap: 8px; padding: 10px 14px;
    border-top: 1px solid var(--sand-200); align-items: flex-end;
  }
  .batch-followup-input {
    flex: 1; resize: none; border: 1px solid var(--sand-200); border-radius: 8px;
    padding: 7px 10px; font-size: 13px; font-family: inherit; outline: none;
    line-height: 1.4; min-height: 34px; max-height: 100px; overflow-y: auto;
    background: white;
  }
  .batch-followup-input:focus { border-color: var(--brand-fig); }
  .batch-followup-send {
    width: 32px; height: 32px; flex-shrink: 0; border-radius: 8px;
    background: var(--brand-fig); border: none; cursor: pointer; color: white;
    display: flex; align-items: center; justify-content: center; transition: background 0.15s;
  }
  .batch-followup-send:hover { background: var(--brand-fig-hover); }
  .batch-followup-send:disabled { opacity: 0.35; cursor: not-allowed; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spin { animation: spin 1s linear infinite; transform-origin: center; }

  #global-tip {
    position: fixed; z-index: 99999; pointer-events: none;
    opacity: 0; transition: opacity 0.15s;
    background: #2D2C28; color: #F5F3ED;
    font-size: 11px; font-family: Inter, sans-serif; font-weight: 400;
    line-height: 1.55; padding: 9px 12px; border-radius: 7px;
    width: 230px; white-space: normal;
    box-shadow: 0 4px 18px rgba(0,0,0,0.22);
  }

  /* ── Saved Constructs button in header ──────────────────────────────── */
  .saved-constructs-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 500;
    cursor: pointer; border: 1.5px solid var(--sand-300); background: var(--sand-50);
    color: var(--text-primary); font-family: Inter, sans-serif;
    transition: background 0.15s, border-color 0.15s;
  }
  .saved-constructs-btn:hover { background: var(--sand-100); border-color: var(--sand-400); }
  .saved-constructs-btn.active {
    background: var(--brand-fig); border-color: var(--brand-fig); color: white;
  }
  .saved-constructs-btn.active svg { stroke: white; }

  /* ── Saved Constructs full-screen overlay ───────────────────────────── */
  .library-overlay {
    position: fixed; inset: 0; z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.45);
  }
  .library-card {
    background: #FAFAF8; border-radius: 12px;
    width: min(1200px, 96vw); height: 90vh;
    display: flex; flex-direction: column;
    box-shadow: 0 8px 40px rgba(0,0,0,0.22); overflow: hidden;
  }
  .library-panel-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 20px; border-bottom: 1px solid var(--sand-200);
    background: #FAFAF8; flex-shrink: 0;
    justify-content: space-between;
  }
  .library-panel-header > div > span {
    font-weight: 600; font-size: 14px; color: var(--text-primary);
  }
  .library-panel-header > div > svg { stroke: var(--brand-fig); }
  .lib-tab {
    padding: 4px 10px; font-size: 12px; border-radius: 4px; cursor: pointer;
    border: 1px solid var(--sand-300); background: var(--sand-50);
    color: var(--text-secondary); font-family: Inter, sans-serif;
  }
  .lib-tab:hover { background: var(--sand-200); }
  .lib-tab.active { background: var(--brand-fig); color: #fff; border-color: var(--brand-fig); }
  .lib-close {
    padding: 5px; border: none; background: none; cursor: pointer;
    color: var(--text-secondary); border-radius: 6px; display: flex; align-items: center;
    flex-shrink: 0;
  }
  .lib-close:hover { background: var(--sand-200); color: var(--text-primary); }
  .lib-tab-bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  #lib-table-pane { flex: 1; overflow: auto; padding: 0; min-height: 0; }
  #lib-graph-pane { flex: 1; overflow: hidden; position: relative; min-height: 0; }
  #constructs-graph { width: 100%; height: 100%; min-height: 400px; }
  .save-btn {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 500;
    cursor: pointer; border: none; font-family: Inter, sans-serif;
    background: var(--brand-fig); color: #fff; margin-left: 8px;
  }
  .save-btn:hover { background: var(--brand-fig-hover); }
  .save-btn:disabled { opacity: 0.6; cursor: default; }
  /* Tabulator theme overrides */
  .tabulator { font-family: Inter, sans-serif; font-size: 12px;
    border: none; background: var(--sand-50); }
  .tabulator .tabulator-header { background: var(--sand-100);
    border-bottom: 1px solid var(--sand-300); }
  .tabulator .tabulator-header .tabulator-col { background: var(--sand-100);
    border-right: 1px solid var(--sand-200); }
  .tabulator .tabulator-row { background: var(--sand-50); border-bottom: 1px solid var(--sand-100); }
  .tabulator .tabulator-row:hover { background: var(--sand-100); }
  .tabulator .tabulator-row.tabulator-selected { background: var(--brand-fig-10); }
  .parts-sub-table { width: 100%; border-collapse: collapse;
    font-size: 11px; margin: 6px 0; }
  .parts-sub-table th { background: var(--sand-200); padding: 4px 8px;
    text-align: left; font-weight: 500; color: var(--text-secondary); }
  .parts-sub-table td { padding: 4px 8px; border-bottom: 1px solid var(--sand-100);
    color: var(--text-primary); }
  .parts-sub-table tr:last-child td { border-bottom: none; }
  .parts-sub-table a { color: var(--brand-fig); text-decoration: none; }
  .parts-sub-table a:hover { text-decoration: underline; }
  .row-detail-wrap { padding: 8px 12px; background: var(--sand-100); }
  .row-detail-wrap h4 { font-size: 11px; font-weight: 600; color: var(--text-secondary);
    margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.05em; }
  .upload-verified-btn { font-size: 11px; margin-top: 6px; padding: 3px 8px;
    background: var(--sand-200); border: 1px solid var(--sand-300); border-radius: 4px;
    cursor: pointer; font-family: Inter, sans-serif; color: var(--text-secondary); }
  .upload-verified-btn:hover { background: var(--sand-300); }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="header-logo">
      <svg viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5"/>
      </svg>
    </div>
    <div class="header-title">
      <h1>Plasmid Designer</h1>
      <p>Allen Institute - OCTO AI</p>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <button class="saved-constructs-btn" id="lib-panel-btn" onclick="toggleLibraryPanel()"
      data-tooltip="Constructs designed here are saved to a local database on your machine — not committed to GitHub or shared with anyone.">
      <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
      </svg>
      Saved Constructs
    </button>
    <span class="health-badge offline" id="health-badge">
      <span class="health-dot"></span>
      <span id="health-text">Agent Offline</span>
    </span>
  </div>
</div>

<!-- Main layout -->
<div class="main">
  <!-- Sessions sidebar -->
  <div class="sidebar" id="sidebar">
    <div class="sidebar-toolbar">
      <button class="sidebar-toggle-btn" onclick="toggleSidebar()" title="Hide sidebar">
        <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
          <path d="M3 6h10M3 12h18M3 18h10"/>
        </svg>
      </button>
      <button class="new-chat-btn" onclick="newChat()">
        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
          <path d="M12 4v16m8-8H4"/>
        </svg>
        New Chat
      </button>
    </div>
    <div class="sessions-list" id="sessions-list">
      <p class="no-sessions">No conversations yet</p>
    </div>
    <div class="user-library-panel" id="user-library-panel" style="display:none">
      <div style="display:flex;align-items:center;gap:2px;">
        <button class="user-library-toggle" id="user-library-toggle" onclick="toggleUserLibrary()"
          style="flex:1;"
          data-tooltip="Parts available for design — backbones, inserts, and designed constructs loaded from your local library folder.">
          <span>Your Library</span>
          <em class="chevron">&#8964;</em>
        </button>
        <button class="user-library-refresh-btn" onclick="loadUserLibrary()" title="Refresh library">&#8635;</button>
      </div>
      <div class="user-library-body" id="user-library-body"></div>
    </div>
  </div>

  <!-- Sidebar reopen button -->
  <button class="sidebar-reopen-btn" id="sidebar-reopen-btn" onclick="toggleSidebar()" title="Show sidebar">
    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
      <path d="M3 6h10M3 12h18M3 18h10"/>
    </svg>
  </button>

  <!-- Chat panel -->
  <div class="chat-panel" id="chat-panel">
    <!-- Drop overlay: shown when any file is dragged over the chat area -->
    <div class="drop-overlay" id="drop-overlay">
      <svg width="36" height="36" fill="none" stroke="var(--brand-fig)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <div class="drop-overlay-label" id="drop-overlay-label">Drop a file</div>
      <div class="drop-overlay-sub" id="drop-overlay-sub">.gb / .gbk / .fasta — add plasmid to library &nbsp;·&nbsp; .csv — batch design</div>
    </div>
    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <div>
          <div class="welcome-icon">
            <svg viewBox="0 0 24 24" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5"/>
            </svg>
          </div>
          <h2>Design an expression plasmid</h2>
          <p>Describe what you want to build. Claude will retrieve verified sequences,<br>
          assemble your construct, validate it, and export the result.</p>
          <p style="font-size:12px;color:var(--sand-300);margin-top:4px;">
            Drag &amp; drop a <strong>.gb</strong> / <strong>.fasta</strong> file to add a plasmid to your library &nbsp;·&nbsp; drop a <strong>.csv</strong> to batch design.
          </p>
          <div class="examples">
            <button onclick="sendExample(this)">Design an EGFP expression plasmid using pcDNA3.1(+)</button>
            <button onclick="sendExample(this)">Put mCherry into a mammalian expression vector</button>
            <button onclick="sendExample(this)">What backbones are available?</button>
            <button onclick="sendExample(this)">Assemble tdTomato in pcDNA3.1(+) and export as GenBank</button>
          </div>
        </div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-wrapper">
        <textarea id="input" placeholder="Describe the plasmid you want to design…" rows="1"
          oninput="autoResize(this)"></textarea>
        <div class="input-meta">
          <select id="model-select" class="model-select">
            <option value="claude-opus-4-7">Opus 4.7</option>
            <option value="claude-opus-4-6">Opus 4.6</option>
            <option value="claude-sonnet-4-6">Sonnet 4.6</option>
            <option value="claude-haiku-4-5-20251001">Haiku 4.5</option>
          </select>
          <div class="token-indicator" id="token-indicator">
            <div class="token-bar-track"><div class="token-bar-fill" id="token-bar"></div></div>
            <span id="token-label"></span>
          </div>
        </div>
        <div id="plasmid-badge" style="display:none;align-items:center;gap:6px;padding:4px 8px;background:var(--sand-100);border:1px solid var(--sand-200);border-radius:6px;font-size:12px;color:var(--sand-700);margin-bottom:4px;">
          <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
          <span id="plasmid-badge-name"></span>
          <span id="plasmid-badge-status" style="color:var(--sand-400);">analyzing…</span>
        </div>
        <div class="input-buttons">
          <button class="attach-btn" id="attach-btn" title="Attach plasmid file (.gb, .gbk, .fasta)" onclick="document.getElementById('plasmid-file-input').click()">
            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
          </button>
          <button class="send-btn" id="send-btn" onclick="sendMessage()">
            <svg fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
              <path d="M12 19V5M5 12l7-7 7 7"/>
            </svg>
          </button>
          <button class="stop-btn" id="stop-btn" onclick="stopGeneration()" style="display:none">
            <svg fill="currentColor" viewBox="0 0 16 16">
              <rect x="3" y="3" width="10" height="10" rx="1.5"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>

  <input type="file" id="batch-csv-input" accept=".csv" style="display:none" onchange="onBatchFileChosen(this)">
  <input type="file" id="plasmid-file-input" accept=".gb,.gbk,.genbank,.fasta,.fa,.seq" style="display:none" onchange="onPlasmidFileChosen(this)">

</div>

<!-- Saved Constructs full-screen overlay -->
<div class="library-overlay" id="library-panel" style="display:none">
  <div class="library-card">
    <div class="library-panel-header">
      <div style="display:flex;align-items:center;gap:10px">
        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
          <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
        </svg>
        <span>Library</span>
      </div>
      <div class="lib-tab-bar">
        <button class="lib-tab active" id="lib-tab-table" onclick="showLibraryTab('table')">&#9776; Table</button>
        <button class="lib-tab" id="lib-tab-graph" onclick="showLibraryTab('graph')">&#9672; Graph</button>
        <button class="lib-tab" onclick="refreshLibraryData()">&#8635; Refresh</button>
        <button class="lib-tab" id="import-lib-btn" style="display:none" onclick="importUserLibrary()">&#8679; Import from Library</button>
      </div>
      <button id="lib-remove-btn" onclick="_removeSelected()" style="display:none;
        padding:5px 14px;border:none;border-radius:6px;cursor:pointer;font-size:12px;
        font-family:Inter,sans-serif;font-weight:500;background:#E86235;color:#fff;
        white-space:nowrap;flex-shrink:0">
        Remove <span id="lib-remove-count">0</span> selected
      </button>
      <button class="lib-close" onclick="toggleLibraryPanel()" title="Close">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" viewBox="0 0 24 24"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
    </div>
    <div id="lib-table-pane" style="flex:1;overflow:auto;min-height:0">
      <div id="constructs-table"></div>
    </div>
    <div id="lib-graph-pane" style="display:none;flex:1;overflow:hidden;min-height:0;position:relative">
      <div id="constructs-graph" style="width:100%;height:100%"></div>
      <div id="cy-tooltip" style="display:none;position:absolute;z-index:100;pointer-events:none;
        background:white;border:1px solid var(--sand-200);border-radius:8px;
        padding:10px 13px;font-size:12px;font-family:Inter,sans-serif;
        box-shadow:0 4px 16px rgba(0,0,0,0.12);max-width:240px;line-height:1.5"></div>
    </div>
  </div>
</div>

<!-- Import library modal -->
<div id="import-modal" style="display:none;position:fixed;inset:0;z-index:2000;
  background:rgba(0,0,0,0.35);align-items:center;justify-content:center">
  <div style="background:#FAFAF8;border-radius:12px;width:min(760px,88vw);max-height:78vh;
    display:flex;flex-direction:column;box-shadow:0 12px 48px rgba(0,0,0,0.28);overflow:hidden">
    <div style="padding:16px 20px;border-bottom:1px solid var(--sand-200);display:flex;align-items:center;justify-content:space-between">
      <strong style="font-size:14px;font-family:Inter,sans-serif">Import from Your Library</strong>
      <div style="display:flex;gap:8px;align-items:center">
        <button onclick="toggleImportSelectAll()" id="import-select-all-btn"
          style="font-size:12px;padding:4px 10px;border:1px solid var(--sand-300);border-radius:6px;
          background:var(--sand-100);cursor:pointer;font-family:Inter,sans-serif">Select all</button>
        <button onclick="closeImportModal()"
          style="background:none;border:none;cursor:pointer;color:var(--text-secondary);font-size:18px;line-height:1">&#10005;</button>
      </div>
    </div>
    <div style="overflow-y:auto;flex:1">
      <table id="import-preview-table" style="width:100%;border-collapse:collapse;font-size:12px;font-family:Inter,sans-serif">
        <thead style="position:sticky;top:0;background:#F5F3ED;z-index:1">
          <tr>
            <th style="width:32px;padding:8px 10px;text-align:center;border-bottom:1px solid var(--sand-200)">
              <input type="checkbox" id="import-check-all" onchange="onCheckAllChange(this)"></th>
            <th style="padding:8px 10px;text-align:left;border-bottom:1px solid var(--sand-200)">Name</th>
            <th style="padding:8px 10px;text-align:left;border-bottom:1px solid var(--sand-200)">Type</th>
            <th style="padding:8px 10px;text-align:left;border-bottom:1px solid var(--sand-200)">Size</th>
            <th style="padding:8px 10px;text-align:left;border-bottom:1px solid var(--sand-200)">Category / Enzyme</th>
            <th style="padding:8px 10px;text-align:left;border-bottom:1px solid var(--sand-200)">Resistance</th>
            <th style="padding:8px 10px;text-align:left;border-bottom:1px solid var(--sand-200)">Status</th>
          </tr>
        </thead>
        <tbody id="import-preview-body"></tbody>
      </table>
    </div>
    <div style="padding:12px 20px;border-top:1px solid var(--sand-200);display:flex;align-items:center;gap:10px;justify-content:flex-end">
      <span id="import-selected-count" style="font-size:12px;color:var(--text-secondary);font-family:Inter,sans-serif;margin-right:auto">0 selected</span>
      <button onclick="closeImportModal()"
        style="padding:7px 16px;border:1px solid var(--sand-300);border-radius:8px;background:var(--sand-100);
        cursor:pointer;font-size:13px;font-family:Inter,sans-serif">Cancel</button>
      <button id="import-confirm-btn" onclick="confirmImport()"
        style="padding:7px 16px;border:none;border-radius:8px;background:var(--brand-fig);color:#fff;
        cursor:pointer;font-size:13px;font-family:Inter,sans-serif;font-weight:500">Import Selected</button>
    </div>
  </div>
</div>

<script>
// ── Global tooltip ──
(function() {
  const tip = document.createElement('div');
  tip.id = 'global-tip';
  document.body.appendChild(tip);
  document.addEventListener('mouseover', function(e) {
    const el = e.target.closest('[data-tooltip]');
    if (!el) return;
    tip.textContent = el.getAttribute('data-tooltip');
    const r = el.getBoundingClientRect();
    let left = r.right - 230;
    let top = r.bottom + 8;
    left = Math.max(8, Math.min(left, window.innerWidth - 238));
    top  = Math.max(8, Math.min(top,  window.innerHeight - 120));
    tip.style.left = left + 'px';
    tip.style.top  = top  + 'px';
    tip.style.opacity = '1';
  });
  document.addEventListener('mouseout', function(e) {
    if (e.target.closest('[data-tooltip]')) tip.style.opacity = '0';
  });
})();

// ── State ──
let currentSessionId = localStorage.getItem('plasmid_session_id') || null;
let sessions = [];
let isStreaming = false;
let abortController = null;
let _userLibraryAvailable = false;

async function _checkUserLibrary() {
  try {
    const r = await fetch('/api/config/user-library');
    const d = await r.json();
    _userLibraryAvailable = d.available || false;
    const btn = document.getElementById('import-lib-btn');
    if (btn) btn.style.display = _userLibraryAvailable ? '' : 'none';
  } catch(e) { _userLibraryAvailable = false; }
}

// ── Token indicator ──
function updateTokenIndicator(inputTokens, contextWindow) {
  const indicator = document.getElementById('token-indicator');
  const bar = document.getElementById('token-bar');
  const label = document.getElementById('token-label');
  if (!indicator || !bar || !label) return;
  const pct = Math.min(inputTokens / contextWindow, 1);
  const remaining = contextWindow - inputTokens;
  const remainingK = remaining >= 1000
    ? (remaining / 1000).toFixed(0) + 'k'
    : remaining.toString();
  bar.style.width = (pct * 100).toFixed(1) + '%';
  bar.className = 'token-bar-fill' + (pct >= 0.9 ? ' alert' : pct >= 0.7 ? ' warn' : '');
  label.textContent = remainingK + ' context window tokens left';
  indicator.classList.add('visible');
}

function saveSessionId(id) {
  currentSessionId = id;
  if (id) {
    localStorage.setItem('plasmid_session_id', id);
  } else {
    localStorage.removeItem('plasmid_session_id');
  }
}

// ── DOM refs ──
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const modelSelect = document.getElementById('model-select');
const sidebarEl = document.getElementById('sidebar');
const sessionsListEl = document.getElementById('sessions-list');
const reopenBtn = document.getElementById('sidebar-reopen-btn');
const healthBadge = document.getElementById('health-badge');
const healthText = document.getElementById('health-text');

// ── Helpers ──
function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function scrollToBottom() {
  // Only auto-scroll if we're viewing the session that's streaming
  if (streamingSessionId && currentSessionId !== streamingSessionId) return;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Health check ──
async function checkHealth() {
  try {
    const r = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
    const ok = r.ok;
    healthBadge.className = 'health-badge ' + (ok ? 'online' : 'offline');
    healthText.textContent = ok ? 'Agent Online' : 'Agent Offline';
  } catch {
    healthBadge.className = 'health-badge offline';
    healthText.textContent = 'Agent Offline';
  }
}

// ── Sessions ──
async function loadSessions() {
  try {
    const r = await fetch('/api/sessions');
    sessions = await r.json();
    renderSessions();
  } catch {}
}

function renderSessions() {
  if (sessions.length === 0) {
    sessionsListEl.innerHTML = '<p class="no-sessions">No conversations yet</p>';
    return;
  }
  sessionsListEl.innerHTML = sessions.map(function(s) {
    const active = s.session_id === currentSessionId ? ' active' : '';
    const name = escapeHtml((s.first_message || 'New conversation').slice(0, 40));
    const batchIcon = s.batch_job_id
      ? '<svg width="11" height="11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" style="flex-shrink:0;opacity:0.6"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>'
      : '';
    return '<div class="session-item' + active + '" onclick="selectSession(\'' + s.session_id + '\')">' +
      '<span class="session-name" style="display:flex;align-items:center;gap:5px;">' + batchIcon + name + '</span>' +
      '<button class="delete-btn" onclick="event.stopPropagation(); deleteSessionById(\'' + s.session_id + '\')" title="Delete">' +
        '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
          '<path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>' +
        '</svg>' +
      '</button>' +
    '</div>';
  }).join('');
}

async function selectSession(sessionId) {
  // If streaming, detach from the SSE connection but let the backend keep running
  if (isStreaming) {
    _detachStream();
    // Reset streaming UI state
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
  }

  // Pause DOM-update polling for the session we're leaving (batch keeps running backend)
  if (currentSessionId && _batchPollTimers[currentSessionId]) {
    clearInterval(_batchPollTimers[currentSessionId]);
    delete _batchPollTimers[currentSessionId];
  }

  saveSessionId(sessionId);
  renderSessions();

  try {
    const r = await fetch('/api/sessions/' + sessionId + '/messages');
    const msgs = await r.json();
    // Guard: if user switched to another session while fetch was in flight, discard
    if (currentSessionId !== sessionId) return;
    renderStoredMessages(msgs);
    // If the agent is still running in the background, reconnect to the live
    // stream so the user sees the response streaming in real time.
    _reconnectToStream(sessionId);
  } catch {
    // Don't clear messages on fetch failure (e.g., during server reload)
    // — leave the current display intact rather than showing empty state
  }
}

async function _reconnectToStream(sessionId) {
  if (isStreaming) return;

  // Try to open the live replay stream. Returns 404 if the run already ended.
  abortController = new AbortController();
  let resp;
  try {
    resp = await fetch('/api/sessions/' + sessionId + '/stream', { signal: abortController.signal });
  } catch (err) {
    abortController = null;
    if (err.name !== 'AbortError') {
      // Network error — fall back to a one-shot message reload
      try {
        const msgs = await fetch('/api/sessions/' + sessionId + '/messages').then(function(r) { return r.json(); });
        if (currentSessionId === sessionId) renderStoredMessages(msgs);
      } catch {}
    }
    return;
  }

  if (!resp.ok) {
    abortController = null;
    // Run already finished — just reload stored messages
    try {
      const msgs = await fetch('/api/sessions/' + sessionId + '/messages').then(function(r) { return r.json(); });
      if (currentSessionId === sessionId) renderStoredMessages(msgs);
    } catch {}
    return;
  }

  // Run is in progress — set up streaming UI and replay/continue the event stream
  isStreaming = true;
  streamingSessionId = sessionId;
  streamingInner = messagesEl.querySelector('.messages-inner');
  if (!streamingInner) {
    streamingInner = document.createElement('div');
    streamingInner.className = 'messages-inner';
    messagesEl.innerHTML = '';
    messagesEl.appendChild(streamingInner);
  }
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  inputEl.disabled = true;
  showPendingCursor();

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      let streamDone = false;
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const jsonStr = trimmed.slice(6);
        if (!jsonStr) continue;
        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }
        switch (event.type) {
          case 'thinking_start': clearPendingCursor(); startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': clearPendingCursor(); flushTextBuffer(); startTextBlock(); break;
          case 'text_delta': bufferTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': clearPendingCursor(); startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'token_usage': updateTokenIndicator(event.input_tokens, event.context_window); break;
          case 'error': clearPendingCursor(); startTextBlock(); appendTextDelta('Error: ' + event.content); endTextBlock(); break;
          case 'done': streamDone = true; break;
        }
        if (streamDone) break;
      }
      if (streamDone) break;
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      clearPendingCursor(); startTextBlock(); appendTextDelta('Connection error: ' + err.message); endTextBlock();
    }
  }

  clearPendingCursor();
  isStreaming = false;
  abortController = null;
  streamingInner = null;
  streamingSessionId = null;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  inputEl.disabled = false;
  const cursor = messagesEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
  loadUserLibrary();

  // Reload stored messages so the view is stable (not dependent on DOM built during streaming)
  if (currentSessionId === sessionId) {
    try {
      const msgs = await fetch('/api/sessions/' + sessionId + '/messages').then(function(r) { return r.json(); });
      if (currentSessionId === sessionId) renderStoredMessages(msgs);
    } catch {}
  }
}

function renderStoredBlock(block, container) {
  const uid = 'stored-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  if (block.type === 'thinking') {
    const wc = (block.content || '').trim().split(/\s+/).length;
    const div = document.createElement('div');
    div.className = 'thinking-block';
    div.innerHTML = '<div class="block-card">' +
      '<div class="block-header" onclick="toggleBlock(\'' + uid + '\')">' +
        '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a1 1 0 01-1 1h-6a1 1 0 01-1-1v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7zM9 21h6M10 21v-1h4v1"/>' +
        '</svg>' +
        '<span class="block-label">Thought process</span>' +
        '<span class="block-meta">' + wc + ' words</span>' +
        '<svg class="block-chevron" id="' + uid + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
      '</div>' +
      '<div class="block-body" id="' + uid + '-body">' + escapeHtml(block.content || '') + '</div>' +
    '</div>';
    container.appendChild(div);
  } else if (block.type === 'tool_use') {
    const div = document.createElement('div');
    div.className = 'tool-block';
    const inputStr = JSON.stringify(block.input || {}, null, 2);
    const bodyHtml = '<div class="section"><div class="label">Input</div>' + escapeHtml(inputStr) + '</div>' +
      '<div class="section"><div class="label">Result</div>' + escapeHtml(block.result || '') + '</div>';
    div.innerHTML = '<div class="block-card">' +
      '<div class="block-header" onclick="toggleBlock(\'' + uid + '\')">' +
        '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>' +
        '</svg>' +
        '<span class="block-label">' + escapeHtml(block.name || 'tool') + '</span>' +
        '<svg class="block-chevron" id="' + uid + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
      '</div>' +
      '<div class="block-body" id="' + uid + '-body">' + bodyHtml + '</div>' +
    '</div>';
    container.appendChild(div);
    if (block.download_content && block.download_filename) {
      const isGb = block.name === 'export_construct' &&
          ['genbank', 'gb'].includes((block.input && block.input.output_format || '').toLowerCase());
      if (isGb) {
        addExportButtons(container, block.input || {}, block.download_content, block.download_filename);
      } else {
        addDownloadButton(container, block.download_content, block.download_filename);
      }
    }
  } else if (block.type === 'text') {
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(block.content || '') + '</div>';
    makeTablesResizable(div);
    container.appendChild(div);
  }
}

function renderStoredMessages(msgs) {
  if (msgs.length === 0) {
    showWelcome();
    return;
  }
  if (msgs.length === 1 && msgs[0].type === 'batch_session') {
    restoreBatchSession(msgs[0]);
    return;
  }
  hideWelcome();
  const inner = document.createElement('div');
  inner.className = 'messages-inner';
  msgs.forEach(function(m) {
    if (m.role === 'user') {
      const div = document.createElement('div');
      div.className = 'msg user';
      const dateStr = m.timestamp ? new Date(m.timestamp * 1000).toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'}) : '';
      div.innerHTML = '<div><div class="msg-bubble-user">' + escapeHtml(m.content) + '</div>' + (dateStr ? '<div class="msg-date">' + dateStr + '</div>' : '') + '</div>';
      inner.appendChild(div);
    } else if (m.blocks && m.blocks.length > 0) {
      m.blocks.forEach(function(block) { renderStoredBlock(block, inner); });
    } else {
      const div = document.createElement('div');
      div.className = 'msg assistant';
      div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(m.content || '') + '</div>';
      makeTablesResizable(div);
      inner.appendChild(div);
    }
  });
  messagesEl.innerHTML = '';
  messagesEl.appendChild(inner);
  scrollToBottom();
}

async function restoreBatchSession(meta) {
  var jobId = meta.batch_job_id;
  var filename = meta.batch_filename || '';
  var model = meta.batch_model || '';
  var rowCount = meta.batch_row_count || 0;
  var sessionId = currentSessionId;

  // Clear stale content and set up a fresh container
  messagesEl.innerHTML = '';

  try {
    const r = await fetch('/api/batch/' + jobId);
    const data = await r.json();
    if (currentSessionId !== sessionId) return;
    if (data.error) {
      messagesEl.innerHTML = '<div class="messages-inner"><div class="msg assistant"><div class="msg-bubble-assistant" style="color:var(--sand-400);font-size:13px;">Could not load batch results.</div></div></div>';
      return;
    }

    // Render the batch label + placeholder cards, then immediately update with real state
    initBatchCards(jobId, rowCount, filename, model);
    updateBatchCards(jobId, data.rows);

    var anyRunning = data.rows && data.rows.some(function(r) {
      return r.status === 'running' || r.status === 'pending';
    });

    _batchSessions[sessionId] = jobId;

    if (data.status !== 'done' || anyRunning) {
      // Batch is still in progress — resume polling
      if (_batchPollTimers[sessionId]) clearInterval(_batchPollTimers[sessionId]);
      _batchPollTimers[sessionId] = setInterval(function() { pollBatchForSession(sessionId); }, 2000);
    } else {
      // Batch finished — show the download-all button if not already there
      var ctrlEl = document.getElementById('batch-ctrl-' + jobId);
      if (ctrlEl) ctrlEl.style.display = 'none';
      var labelEl = document.getElementById('batch-label-' + jobId);
      if (labelEl && !labelEl.querySelector('.batch-dl-all-btn')) {
        var bubble = labelEl.querySelector('.msg-bubble-assistant');
        if (bubble) {
          var wrap = document.createElement('div');
          wrap.className = 'dl-split-wrap batch-dl-all-btn';
          wrap.style.cssText = 'margin-top:10px;';
          var allMenuId = 'dlmenu-all-' + jobId;
          wrap.innerHTML =
            '<button class="download-btn" onclick="downloadAllBatch(\'' + jobId + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
            '<button class="dl-chevron-btn" onclick="toggleDlMenu(event,\'' + allMenuId + '\')" title="More options">' + _CHEV_DOWN_SVG + '</button>' +
            '<div class="dl-menu" id="' + allMenuId + '">' +
              '<button class="dl-menu-item" onclick="downloadAllBatch(\'' + jobId + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
              (_userLibraryAvailable ? '<button class="dl-menu-item" id="savall-local-' + jobId + '" onclick="event.stopPropagation();saveAllBatchToLocal(\'' + jobId + '\',document.getElementById(\'savall-local-' + jobId + '\'))">' +
                '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 002-2v-4M17 8l-5-5-5 5M12 3v12"/></svg> Save All to Local Library</button>' : '') +
              '<button class="dl-menu-item" id="savall-con-' + jobId + '" onclick="event.stopPropagation();saveAllBatchConstructs(\'' + jobId + '\',document.getElementById(\'savall-con-' + jobId + '\'))">' + _SAVE_SVG + ' Save All Constructs</button>' +
            '</div>';
          bubble.appendChild(document.createElement('br'));
          bubble.appendChild(wrap);
        }
      }
    }
  } catch(e) {
    messagesEl.innerHTML = '<div class="messages-inner"><div class="msg assistant"><div class="msg-bubble-assistant" style="color:var(--sand-400);font-size:13px;">Could not reach the server to load batch status.</div></div></div>';
  }
}

async function deleteSessionById(sessionId) {
  try {
    await fetch('/api/sessions/' + sessionId, { method: 'DELETE' });
    if (currentSessionId === sessionId) {
      saveSessionId(null);
      showWelcome();
    }
    loadSessions();
  } catch {}
}

function newChat() {
  if (isStreaming) {
    _detachStream();
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
  }
  // Pause DOM polling for the session being left
  if (currentSessionId && _batchPollTimers[currentSessionId]) {
    clearInterval(_batchPollTimers[currentSessionId]);
    delete _batchPollTimers[currentSessionId];
  }
  saveSessionId(null);
  renderSessions();
  showWelcome();
  inputEl.focus();
}

function showWelcome() {
  messagesEl.innerHTML = '';
  const w = document.createElement('div');
  w.className = 'welcome';
  w.id = 'welcome';
  w.innerHTML = '<div>' +
    '<div class="welcome-icon">' +
      '<svg viewBox="0 0 24 24" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5"/>' +
      '</svg>' +
    '</div>' +
    '<h2>Design an expression plasmid</h2>' +
    '<p>Describe what you want to build. Claude will retrieve verified sequences,<br>' +
    'assemble your construct, validate it, and export the result.</p>' +
    '<p style="font-size:12px;color:var(--sand-300);margin-top:4px;">Drag &amp; drop a CSV file here to batch design multiple plasmids at once.</p>' +
    '<div class="examples">' +
      '<button onclick="sendExample(this)">Design an EGFP expression plasmid using pcDNA3.1(+)</button>' +
      '<button onclick="sendExample(this)">Put mCherry into a mammalian expression vector</button>' +
      '<button onclick="sendExample(this)">What backbones are available?</button>' +
      '<button onclick="sendExample(this)">Assemble tdTomato in pcDNA3.1(+) and export as GenBank</button>' +
    '</div>' +
  '</div>';
  messagesEl.appendChild(w);
}

function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}

// ── Sidebar toggle ──
function toggleSidebar() {
  sidebarEl.classList.toggle('collapsed');
  reopenBtn.classList.toggle('visible', sidebarEl.classList.contains('collapsed'));
}

// ── User Library Panel ──
const _ulEntries = {};   // id → full entry object

function _ulDetailRows(entry, isInsert) {
  const fields = isInsert ? [
    ['ID', entry.id],
    ['Category', entry.category],
    ['Enzyme', entry.assembly_enzyme],
    ['Overhangs', entry.overhang_l && entry.overhang_r ? entry.overhang_l + ' / ' + entry.overhang_r : null],
    ['Insert size', entry.insert_size_bp ? entry.insert_size_bp + ' bp' : null],
    ['Vector size', entry.size_bp ? entry.size_bp + ' bp' : null],
    ['Resistance', entry.bacterial_resistance],
    ['Description', entry.description],
  ] : [
    ['ID', entry.id],
    ['Company', entry.company],
    ['Enzyme', entry.assembly_enzyme],
    ['Overhangs 1', entry.overhang_left && entry.overhang_right ? entry.overhang_left + ' / ' + entry.overhang_right : null],
    ['Overhangs 2', entry.overhang_left_2 && entry.overhang_right_2 ? entry.overhang_left_2 + ' / ' + entry.overhang_right_2 : null],
    ['Next enzyme', entry.next_step_enzyme],
    ['E. coli', entry.ecoli_strain],
    ['Resistance', entry.bacterial_resistance],
    ['Mammalian', entry.mammalian_selection],
    ['Size', entry.size_bp ? entry.size_bp + ' bp' : null],
    ['Description', entry.description],
  ];
  return fields.filter(function(f) { return f[1]; }).map(function(f) {
    return '<div class="entry-detail-row"><span class="dk">' + escapeHtml(f[0]) + '</span><span class="dv">' + escapeHtml(String(f[1])) + '</span></div>';
  }).join('');
}

function _ulBuildEntries(items, isInsert) {
  return items.map(function(entry) {
    const eid = entry.id.replace(/[^a-zA-Z0-9_-]/g, '_');
    _ulEntries[eid] = {entry: entry, isInsert: isInsert};
    const meta = isInsert
      ? [entry.category, entry.assembly_enzyme, entry.insert_size_bp ? entry.insert_size_bp + ' bp' : null].filter(Boolean).join(' · ')
      : [entry.company, entry.assembly_enzyme, entry.bacterial_resistance].filter(Boolean).join(' · ');
    return '<div class="user-library-entry" id="ule-' + eid + '" onclick="toggleULEntry(\'' + eid + '\')">' +
      '<div class="entry-header">' +
        '<div><div class="entry-name">' + escapeHtml(entry.name || entry.id) + '</div>' +
        (meta ? '<div class="entry-meta">' + escapeHtml(meta) + '</div>' : '') + '</div>' +
        '<em class="entry-chevron">&#8964;</em>' +
      '</div>' +
      '<div class="entry-detail" id="uld-' + eid + '"></div>' +
    '</div>';
  }).join('');
}

async function loadUserLibrary() {
  try {
    const r = await fetch('/api/user-library');
    const data = await r.json();
    const panel = document.getElementById('user-library-panel');
    const hasContent = data.configured || (data.vendor_backbones && data.vendor_backbones.length);
    if (!hasContent) return;
    panel.style.display = '';
    const body = document.getElementById('user-library-body');
    let html = '';
    if (data.backbones && data.backbones.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Backbones</div>' +
        _ulBuildEntries(data.backbones, false) + '</div>';
    }
    if (data.inserts && data.inserts.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Inserts</div>' +
        _ulBuildEntries(data.inserts, true) + '</div>';
    }
    if (data.vendor_backbones && data.vendor_backbones.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Vendor Backbones</div>' +
        _ulBuildEntries(data.vendor_backbones, false) + '</div>';
    }
    if (data.designed_constructs && data.designed_constructs.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Designed Constructs</div>' +
        _ulBuildEntries(data.designed_constructs, false) + '</div>';
    }
    if (!html) html = '<div class="user-library-empty">No entries loaded.</div>';
    body.innerHTML = html;
  } catch {}
}

function toggleULEntry(eid) {
  const row = document.getElementById('ule-' + eid);
  const detail = document.getElementById('uld-' + eid);
  const isOpen = detail.classList.contains('open');
  if (!isOpen && !detail.innerHTML) {
    const rec = _ulEntries[eid];
    detail.innerHTML = _ulDetailRows(rec.entry, rec.isInsert);
  }
  detail.classList.toggle('open', !isOpen);
  row.classList.toggle('expanded', !isOpen);
}

function toggleUserLibrary() {
  const btn = document.getElementById('user-library-toggle');
  const body = document.getElementById('user-library-body');
  btn.classList.toggle('open');
  body.classList.toggle('open');
}

// ── Markdown rendering ──
// ── DNA sequence helpers ─────────────────────────────────────────────────────
const _CLIP_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
const _CHECK_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

function isDnaSeq(s) {
  // Strip whitespace, digits, hyphens, prime/apostrophe, brackets, parens, slashes, punctuation
  const clean = s.replace(/[\s\d\-'()\[\]\/\\.,;:*>]/g, '').toUpperCase();
  return clean.length >= 10 && /^[ACGTURYSWKMBDHVN]+$/.test(clean);
}

function mkCopyBtn(seq, extraClass) {
  return '<button class="seq-copy-btn' + (extraClass ? ' ' + extraClass : '') +
    '" data-seq="' + escapeHtml(seq) + '" onclick="copySeq(this.dataset.seq,this)" title="Copy sequence">' +
    _CLIP_SVG + '</button>';
}

function copySeq(seq, btn) {
  seq = seq.replace(/\s/g, '');
  navigator.clipboard.writeText(seq).then(function() {
    var orig = btn.innerHTML;
    btn.innerHTML = _CHECK_SVG;
    btn.classList.add('copied');
    setTimeout(function() { btn.innerHTML = orig; btn.classList.remove('copied'); }, 1500);
  });
}

function copyRaw(text, btn) {
  navigator.clipboard.writeText(text).then(function() {
    var orig = btn.innerHTML;
    btn.innerHTML = _CHECK_SVG;
    btn.classList.add('copied');
    setTimeout(function() { btn.innerHTML = orig; btn.classList.remove('copied'); }, 1500);
  });
}

function mkRawCopyBtn(text, extraClass) {
  return '<button class="seq-copy-btn' + (extraClass ? ' ' + extraClass : '') +
    '" data-raw="' + escapeHtml(text) + '" onclick="copyRaw(this.dataset.raw,this)" title="Copy">' +
    _CLIP_SVG + '</button>';
}

function stripMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#{1,6}\s+/, '');
}

function makeTablesResizable(root) {
  if (!root) return;
  root.querySelectorAll('table:not([data-resizable])').forEach(function(table) {
    table.setAttribute('data-resizable', '1');
    table.querySelectorAll('th').forEach(function(th) {
      var resizer = document.createElement('div');
      resizer.className = 'col-resizer';
      th.appendChild(resizer);
      var startX, startW;
      resizer.addEventListener('mousedown', function(e) {
        var allThs = table.querySelectorAll('th');
        allThs.forEach(function(t) { t.style.width = t.offsetWidth + 'px'; });
        table.style.tableLayout = 'fixed';
        table.style.width = table.offsetWidth + 'px';
        startX = e.pageX;
        startW = th.offsetWidth;
        function onMove(e) { th.style.width = Math.max(40, startW + e.pageX - startX) + 'px'; }
        function onUp() { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        e.preventDefault();
      });
    });
  });
}

function renderSeqCodeBlock(rawCode) {
  var langMatch = rawCode.match(/^([a-zA-Z][a-zA-Z0-9_]*)\n/);
  var code = langMatch ? rawCode.slice(langMatch[0].length) : rawCode;
  var trimmed = code.trim();
  var lines = trimmed.split('\n').filter(function(l) { return l.trim(); });

  // FASTA format (lines starting with >)
  if (lines.length >= 1 && lines[0].charAt(0) === '>') {
    var entries = [], curName = null, curSeq = '';
    lines.forEach(function(l) {
      if (l.charAt(0) === '>') {
        if (curName !== null) entries.push({name: curName, seq: curSeq});
        curName = l.slice(1).trim(); curSeq = '';
      } else { curSeq += l.trim(); }
    });
    if (curName !== null) entries.push({name: curName, seq: curSeq});
    if (entries.length > 0 && entries.every(function(e) { return isDnaSeq(e.seq); })) {
      if (entries.length === 1) {
        return '<div class="seq-block"><pre><code>&gt;' + escapeHtml(entries[0].name) + '\n' +
          escapeHtml(entries[0].seq) + '</code></pre>' + mkCopyBtn(entries[0].seq, 'block-btn') + '</div>';
      }
      var allTsv = 'Name\tSequence\n' + entries.map(function(e) { return e.name + '\t' + e.seq; }).join('\n');
      var t = '<div class="seq-table"><table><thead><tr><th>Name</th><th>Sequence</th><th></th></tr></thead><tbody>';
      entries.forEach(function(e) {
        t += '<tr><td>' + escapeHtml(e.name) + '</td><td><code class="dna-seq">' +
          escapeHtml(e.seq) + '</code></td><td>' + mkCopyBtn(e.seq) + '</td></tr>';
      });
      return t + '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(allTsv) + '</div></div>';
    }
  }

  // Named sequences: "Label: SEQUENCE" or "Label = SEQUENCE"
  var namedRe = /^(.{1,50}?)\s*[:=]\s*([ACGTUacgtuRYSWKMBDHVNryswkmbdhvn]{10,})\s*$/;
  var namedMs = lines.map(function(l) { return l.match(namedRe); });
  if (lines.length >= 2 && namedMs.every(function(m) { return m; })) {
    var allTsv = 'Name\tSequence\n' + namedMs.map(function(m) { return m[1].trim() + '\t' + m[2]; }).join('\n');
    var t = '<div class="seq-table"><table><thead><tr><th>Name</th><th>Sequence</th><th></th></tr></thead><tbody>';
    namedMs.forEach(function(m) {
      t += '<tr><td>' + escapeHtml(m[1].trim()) + '</td><td><code class="dna-seq">' +
        escapeHtml(m[2]) + '</code></td><td>' + mkCopyBtn(m[2]) + '</td></tr>';
    });
    return t + '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(allTsv) + '</div></div>';
  }

  // Multiple bare sequences (one per line)
  if (lines.length >= 2 && lines.every(function(l) { return isDnaSeq(l.trim()); })) {
    var seqs = lines.map(function(l) { return l.trim(); });
    var allTsv = seqs.join('\n');
    var t = '<div class="seq-table"><table><thead><tr><th>#</th><th>Sequence</th><th></th></tr></thead><tbody>';
    seqs.forEach(function(s, i) {
      t += '<tr><td>' + (i+1) + '</td><td><code class="dna-seq">' +
        escapeHtml(s) + '</code></td><td>' + mkCopyBtn(s) + '</td></tr>';
    });
    return t + '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(allTsv) + '</div></div>';
  }

  // Single DNA sequence
  if (isDnaSeq(trimmed)) {
    return '<div class="seq-block"><pre><code>' + escapeHtml(trimmed) + '</code></pre>' +
      mkCopyBtn(trimmed.replace(/\s/g, ''), 'block-btn') + '</div>';
  }

  // Regular code block
  return '<div class="code-block-wrap"><pre><code>' + escapeHtml(code) + '</code></pre>' +
    mkRawCopyBtn(code, 'block-btn') + '</div>';
}

// Wrap bare DNA sequences in plain text (skips content already inside backticks)
function applyBareDnaInLine(h) {
  var parts = h.split(/(`[^`]*`)/);
  return parts.map(function(part, i) {
    if (i % 2 === 1) return part;
    return part.replace(/\b([ACGTUacgtu][ACGTUacgtuRYSWKMBDHVNryswkmbdhvn]{14,})\b/g, function(m) {
      return isDnaSeq(m) ? ('<code class="dna-seq">' + m + '</code>' + mkCopyBtn(m)) : m;
    });
  }).join('');
}

function inlineMarkdown(text) {
  let h = escapeHtml(text);
  h = applyBareDnaInLine(h);
  h = h.replace(/`([^`]+)`/g, function(match, code) {
    if (isDnaSeq(code)) return '<code class="dna-seq">' + code + '</code>' + mkCopyBtn(code);
    return '<code>' + code + '</code>';
  });
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  return h;
}

function renderContent(text) {
  const codeBlocks = [];
  text = text.replace(/```([\s\S]*?)```/g, function(match, code) {
    codeBlocks.push(code);
    return '%%CODEBLOCK' + (codeBlocks.length - 1) + '%%';
  });

  const lines = text.split('\n');
  const outputParts = [];
  let i = 0;
  while (i < lines.length) {
    if (i + 1 < lines.length &&
        lines[i].trim().startsWith('|') &&
        /^\|[\s:]*-+[\s:]*/.test(lines[i + 1].trim())) {
      const headerCells = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
      i += 2;
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        const cells = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
        bodyRows.push(cells);
        i++;
      }
      var tsvRows = [headerCells.map(stripMarkdown).join('\t')].concat(bodyRows.map(function(row) { return row.map(stripMarkdown).join('\t'); }));
      var tblTsv = tsvRows.join('\n');
      let t = '<div class="seq-table"><table><thead><tr>';
      headerCells.forEach(function(c) { t += '<th>' + inlineMarkdown(c) + '</th>'; });
      t += '</tr></thead><tbody>';
      bodyRows.forEach(function(row) {
        t += '<tr>';
        row.forEach(function(c) {
          const stripped = c.trim();
          if (isDnaSeq(stripped)) {
            t += '<td><code class="dna-seq">' + escapeHtml(stripped) + '</code>' + mkCopyBtn(stripped) + '</td>';
          } else {
            t += '<td>' + inlineMarkdown(c) + '</td>';
          }
        });
        t += '</tr>';
      });
      t += '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(tblTsv) + '</div></div>';
      outputParts.push(t);
    } else {
      const trimmed = lines[i].trim();
      // Horizontal rule
      if (/^-{3,}$/.test(trimmed) || /^\*{3,}$/.test(trimmed)) {
        outputParts.push('<hr style="border:none;border-top:1px solid var(--sand-200);margin:12px 0">');
        i++;
        continue;
      }
      let h = escapeHtml(lines[i]);
      h = applyBareDnaInLine(h);
      h = h.replace(/`([^`]+)`/g, function(match, code) {
        if (isDnaSeq(code)) return '<code class="dna-seq">' + code + '</code>' + mkCopyBtn(code);
        return '<code>' + code + '</code>';
      });
      h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      h = h.replace(/^### (.+)$/, '<strong style="font-size:14px">$1</strong>');
      h = h.replace(/^## (.+)$/, '<strong style="font-size:15px">$1</strong>');
      h = h.replace(/^# (.+)$/, '<strong style="font-size:16px">$1</strong>');
      outputParts.push(h);
      i++;
    }
  }

  let html = outputParts.join('<br>\n');
  codeBlocks.forEach(function(code, idx) {
    html = html.replace('%%CODEBLOCK' + idx + '%%', renderSeqCodeBlock(code));
  });
  return html;
}

// ── Streaming blocks ──
let currentTextDiv = null;
let currentTextRaw = '';
let currentThinkingId = null;
let currentThinkingBody = null;
let currentToolId = null;
// Pinned reference to the .messages-inner container for the active stream.
// Ensures streaming writes go to the correct session even if the user
// clicks a different session in the sidebar mid-stream.
let streamingInner = null;
let streamingSessionId = null;

function getInner() {
  // While streaming, always write to the pinned container
  if (streamingInner) return streamingInner;
  let inner = messagesEl.querySelector('.messages-inner');
  if (!inner) {
    inner = document.createElement('div');
    inner.className = 'messages-inner';
    messagesEl.innerHTML = '';
    messagesEl.appendChild(inner);
  }
  return inner;
}

function toggleBlock(id) {
  const body = document.getElementById(id + '-body');
  const chevron = document.getElementById(id + '-chevron');
  if (body && chevron) {
    body.classList.toggle('open');
    chevron.classList.toggle('open');
  }
}

function startThinkingBlock() {
  currentThinkingId = 'think-' + Date.now();
  const div = document.createElement('div');
  div.className = 'thinking-block';
  div.innerHTML = '<div class="block-card">' +
    '<div class="block-header" onclick="toggleBlock(\'' + currentThinkingId + '\')">' +
      '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a1 1 0 01-1 1h-6a1 1 0 01-1-1v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7zM9 21h6M10 21v-1h4v1"/>' +
      '</svg>' +
      '<span class="block-label">Thinking...</span>' +
      '<span class="block-meta" id="' + currentThinkingId + '-meta"></span>' +
      '<svg class="block-chevron" id="' + currentThinkingId + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
    '</div>' +
    '<div class="block-body" id="' + currentThinkingId + '-body"></div>' +
  '</div>';
  getInner().appendChild(div);
  currentThinkingBody = document.getElementById(currentThinkingId + '-body');
  scrollToBottom();
}

function appendThinkingDelta(text) {
  if (currentThinkingBody) {
    currentThinkingBody.textContent += text;
    if (currentThinkingBody.classList.contains('open')) {
      currentThinkingBody.scrollTop = currentThinkingBody.scrollHeight;
    }
    scrollToBottom();
  }
}

function endThinkingBlock() {
  if (currentThinkingId) {
    const card = currentThinkingBody.closest('.block-card');
    const label = card.querySelector('.block-label');
    if (label) label.textContent = 'Thought process';
    const meta = document.getElementById(currentThinkingId + '-meta');
    if (meta && currentThinkingBody) {
      const wc = currentThinkingBody.textContent.trim().split(/\s+/).length;
      meta.textContent = wc + ' words';
    }
  }
  currentThinkingBody = null;
  currentThinkingId = null;
}

function startTextBlock() {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant"><span class="text-content"></span></div>';
  getInner().appendChild(div);
  currentTextDiv = div.querySelector('.text-content');
  currentTextRaw = '';
  scrollToBottom();
}

function appendTextDelta(text) {
  if (currentTextDiv) {
    currentTextRaw += text;
    currentTextDiv.innerHTML = renderContent(currentTextRaw);
    // Add streaming cursor
    let cursor = currentTextDiv.querySelector('.streaming-cursor');
    if (!cursor) {
      cursor = document.createElement('span');
      cursor.className = 'streaming-cursor';
      currentTextDiv.appendChild(cursor);
    }
    scrollToBottom();
  }
}

// ── Smooth streaming ──────────────────────────────────────────────────
// API text_delta events arrive in ~40-100 char bursts every ~400-500ms.
// Dumping each burst at once looks choppy. Buffer incoming chars and
// drain via requestAnimationFrame so text "types" smoothly. The drain
// rate adapts (1/8 of buffer per frame, clamped [2,12] chars) so it
// never lags far behind the model (~180 ch/s) but stays smooth.
let textBuffer = '';
let drainHandle = null;

function bufferTextDelta(text) {
  textBuffer += text;
  if (drainHandle === null) drainHandle = requestAnimationFrame(drainText);
}

function drainText() {
  drainHandle = null;
  if (textBuffer.length === 0) return;
  const n = Math.max(2, Math.min(12, Math.ceil(textBuffer.length / 8)));
  appendTextDelta(textBuffer.slice(0, n));
  textBuffer = textBuffer.slice(n);
  if (textBuffer.length > 0) drainHandle = requestAnimationFrame(drainText);
}

function flushTextBuffer() {
  if (drainHandle !== null) { cancelAnimationFrame(drainHandle); drainHandle = null; }
  if (textBuffer) { appendTextDelta(textBuffer); textBuffer = ''; }
}

// Show a blinking cursor immediately on send so the user sees activity
// during TTFT, before any text/thinking/tool event arrives.
let pendingCursorEl = null;
let pendingCursorTimer = null;
const SLOW_THRESHOLD_MS = 7000;
const SLOW_NOTE = 'Complex designs can take a minute or two — hang tight.';

const WORKING_LABELS = [
  'Pipetting…',
  'Running a gel…',
  'Miniprepping…',
  'Consulting the literature…',
  'Transforming bacteria…',
  'Checking the freezer…',
  'Growing colonies…',
  'Spinning down…',
  'Thawing reagents…',
  'Asking a grad student…',
  'Reading the manual…',
  'Autoclaving…',
  'Counting clones…',
  'Labeling tubes…',
  'Calibrating the pipette…',
  'Staring at the gel…',
  'Refilling tip boxes…',
  'Making competent cells…',
  'Waiting for the PCR…',
  'Pouring a plate…',
  'Streaking for singles…',
  'Checking the incubator…',
  'Ordering reagents…',
  'Waiting for the centrifuge…',
  'Defrosting the -20°C…',
  'Wiping down the bench…',
  'Preparing the buffer…',
  'Changing gloves…',
  'Signing the safety form…',
  'Finding the right tube…',
  'Checking the OD600…',
  'Setting up the water bath…',
  'Asking the PI…',
  'Asking a postdoc…',
  'Reprinting the label…',
  'Hunting for the protocol binder…',
  'Waiting for the autoclave…',
  'Mixing by pipetting up and down…',
  'Pipetting…',
  'Vortexing…',
  'Flash freezing…',
  'Filling out the order form…',
  'Checking if the kit expired…',
  'Making 10× buffer…',
  'Weighing out the powder…',
  'pH-ing the solution…',
  'Waiting for the gel to set…',
  'Realizing the gel ran backwards…',
  'Borrowing tips from the next lab…',
  'Searching for the marker…',
  'Checking the thermocycler program…',
  'Waiting for the overnight culture…',
  'Diluting the sample…',
  'Aliquoting…',
  'Topping off the liquid nitrogen…',
  'De-icing the freezer…',
  'Wiping down the hood…',
  'Running the Western…',
  'Blocking the membrane…',
  'Developing the blot…',
  'Staining with EtBr…',
  'Destaining the gel…',
  'Imaging the blot…',
  'Spinning the columns…',
  'Eluting the DNA…',
  'Measuring the absorbance…',
  'Plating the cells…',
  'Trypsinizing…',
  'Counting cells…',
  'Checking confluency…',
  'Changing the media…',
  'Spinning down the pellet…',
  'Resuspending in buffer…',
  'Snap freezing…',
  'Running the SDS-PAGE…',
  'Loading the samples…',
  'Casting the gel…',
  'Transferring to membrane…',
  'Probing with antibody…',
  'Washing the membrane…',
  'Exposing the film…',
  'Scraping the cells…',
  'Lysing the cells…',
  'Sonicating…',
  'Clarifying the lysate…',
  'Checking the protein concentration…',
  'Setting up the ligation…',
  'Running the digest…',
  'Gel extracting…',
  'Incubating on ice…',
  'Heat shocking…',
  'Recovering in SOC…',
  'Spreading the plates…',
  'Picking colonies…',
  'Inoculating the culture…',
  'Doing a colony PCR…',
  'Checking the growth curve…',
  'Inducing expression…',
  'Harvesting cells…',
  'Resuspending the pellet…',
  'Filtering the solution…',
  'Running the FPLC…',
  'Collecting fractions…',
  'Pooling the peaks…',
  'Concentrating the sample…',
  'Running a Bradford…',
  'Preparing the cryovials…',
  'Labeling the boxes…',
  'Checking the balance…',
];

function randomWorkingLabel() {
  return WORKING_LABELS[Math.floor(Math.random() * WORKING_LABELS.length)];
}

function showPendingCursor(label) {
  clearPendingCursor();
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML =
    '<div class="msg-bubble-assistant">' +
      '<div class="working-indicator">' +
        '<span class="working-dots"><span></span><span></span><span></span></span>' +
        '<span class="working-label">' + (label || randomWorkingLabel()) + '</span>' +
      '</div>' +
      '<div class="slow-note" style="display:none"></div>' +
    '</div>';
  getInner().appendChild(div);
  pendingCursorEl = div;
  pendingCursorTimer = setTimeout(function() {
    if (!pendingCursorEl) return;
    const note = pendingCursorEl.querySelector('.slow-note');
    if (note) { note.textContent = SLOW_NOTE; note.style.display = ''; }
  }, SLOW_THRESHOLD_MS);
  scrollToBottom();
}

function clearPendingCursor() {
  if (pendingCursorTimer) { clearTimeout(pendingCursorTimer); pendingCursorTimer = null; }
  if (pendingCursorEl) { pendingCursorEl.remove(); pendingCursorEl = null; }
}

function endTextBlock() {
  flushTextBuffer();
  if (currentTextDiv) {
    const cursor = currentTextDiv.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
    makeTablesResizable(currentTextDiv);
  }
  currentTextDiv = null;
  currentTextRaw = '';
}

function startToolBlock(toolName) {
  currentToolId = 'tool-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'tool-block';
  div.innerHTML = '<div class="block-card">' +
    '<div class="block-header" onclick="toggleBlock(\'' + currentToolId + '\')">' +
      '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>' +
      '</svg>' +
      '<span class="block-label">' + escapeHtml(toolName) + '</span>' +
      '<span class="pulse-dot" id="' + currentToolId + '-pulse"></span>' +
      '<svg class="block-chevron" id="' + currentToolId + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
    '</div>' +
    '<div class="block-body" id="' + currentToolId + '-body"><div class="section"><div class="label">Running...</div></div></div>' +
  '</div>';
  getInner().appendChild(div);
  scrollToBottom();
}

function addPlasmidPlot(plotJson) {
  var bokehItem = plotJson.plot !== undefined ? plotJson.plot : plotJson;
  var isLinear = plotJson.linear === true;
  var label = isLinear ? 'Linear DNA Map' : 'Plasmid Map';
  const plotId = 'plot-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant" style="margin-top:8px;padding:12px;width:100%;max-width:640px;">' +
    '<div style="font-size:11px;font-weight:600;color:var(--sand-500);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">' + label + '</div>' +
    '<div id="' + plotId + '" style="width:100%;"></div>' +
  '</div>';
  getInner().appendChild(div);
  Bokeh.embed.embed_item(bokehItem, plotId);
  scrollToBottom();
}

function addDownloadButton(container, content, filename) {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant" style="margin-top:8px">' +
    '<button class="download-btn">' +
      '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
        '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>' +
      '</svg>' +
      ' Download ' + escapeHtml(filename) +
    '</button></div>';
  container.appendChild(div);
  div.querySelector('.download-btn').addEventListener('click', function() {
    const blob = new Blob([content], {type: 'application/octet-stream'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

function finishToolBlock(toolName, toolInput, toolResult, downloadContent, downloadFilename) {
  if (currentToolId) {
    const pulse = document.getElementById(currentToolId + '-pulse');
    if (pulse) pulse.remove();
    const body = document.getElementById(currentToolId + '-body');
    if (body) {
      const inputStr = JSON.stringify(toolInput, null, 2);
      let html = '<div class="section"><div class="label">Input</div>' + escapeHtml(inputStr) + '</div>' +
        '<div class="section"><div class="label">Result</div>' + escapeHtml(toolResult) + '</div>';
      body.innerHTML = html;
    }
  }
  // Surface action buttons in the main chat (not just inside the collapsed tool block)
  if (downloadContent && downloadFilename) {
    const isGenbank = toolName === 'export_construct' &&
        ['genbank', 'gb'].includes((toolInput.output_format || '').toLowerCase());
    if (isGenbank) {
      addExportButtons(getInner(), toolInput, downloadContent, downloadFilename);
    } else {
      addDownloadButton(getInner(), downloadContent, downloadFilename);
    }
  }
  currentToolId = null;
  showPendingCursor();
  scrollToBottom();
}

// ── Send / Stop ──
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isStreaming) return;

  isStreaming = true;
  streamingSessionId = currentSessionId;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  inputEl.value = '';
  inputEl.disabled = true;
  autoResize(inputEl);
  hideWelcome();

  const inner = getInner();
  // Pin this container so stream events write here even if user switches sessions
  streamingInner = inner;
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  const nowStr = new Date().toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'});
  userDiv.innerHTML = '<div><div class="msg-bubble-user">' + escapeHtml(text) + '</div><div class="msg-date">' + nowStr + '</div></div>';
  inner.appendChild(userDiv);
  scrollToBottom();
  showPendingCursor();

  abortController = new AbortController();

  try {
    const reqBody = { message: text, model: modelSelect.value };
    if (currentSessionId) reqBody.session_id = currentSessionId;

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(reqBody),
      signal: abortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {stream: true});
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      let streamDone = false;
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const jsonStr = trimmed.slice(6);
        if (!jsonStr) continue;

        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }

        switch (event.type) {
          case 'session_id':
            saveSessionId(event.session_id);
            loadSessions();
            break;
          case 'thinking_start': clearPendingCursor(); startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': clearPendingCursor(); flushTextBuffer(); startTextBlock(); break;
          case 'text_delta': bufferTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': clearPendingCursor(); startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'token_usage': updateTokenIndicator(event.input_tokens, event.context_window); break;
          case 'error':
            clearPendingCursor();
            startTextBlock();
            appendTextDelta('Error: ' + event.content);
            endTextBlock();
            break;
          case 'done': streamDone = true; break;
        }
        if (streamDone) break;
      }
      if (streamDone) break;
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      clearPendingCursor();
      startTextBlock();
      appendTextDelta('Connection error: ' + err.message);
      endTextBlock();
    }
  }

  clearPendingCursor();
  isStreaming = false;
  abortController = null;
  streamingInner = null;
  streamingSessionId = null;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  inputEl.disabled = false;
  inputEl.focus();
  // Remove any leftover streaming cursor
  const cursor = messagesEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
  // Refresh library panel in case the agent saved a new backbone or construct
  loadUserLibrary();
}

function _detachStream() {
  // Drop the SSE connection without cancelling the server-side run.
  // Use this when navigating away — the agent keeps going in the background.
  if (abortController) abortController.abort();
}

function stopGeneration() {
  // Explicitly cancel the run (Stop button). Aborts client AND tells server to stop.
  if (abortController) abortController.abort();
  if (currentSessionId) {
    fetch('/api/sessions/' + currentSessionId + '/cancel', { method: 'POST' }).catch(function(){});
  }
}

function sendExample(btn) {
  inputEl.value = btn.textContent;
  sendMessage();
}

// ── Keyboard ──
inputEl.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Batch state (must be declared before init so selectSession can reference them) ──
var _batchSessions = {};    // sessionId → jobId
var _batchPollTimers = {};  // sessionId → interval timer
var _batchConfirmData = {};
const chatPanelEl = document.getElementById('chat-panel');
const dropOverlayEl = document.getElementById('drop-overlay');

// ── Init ──
checkHealth();
setInterval(checkHealth, 5000);
loadSessions();
loadUserLibrary();
_checkUserLibrary();
setInterval(loadSessions, 5000);
// Restore active session on page load
if (currentSessionId) {
  selectSession(currentSessionId);
}
inputEl.focus();

// ── Drag & drop onto the chat area (CSV batch or plasmid file) ──
var dragCounter = 0;

var PLASMID_EXTS = ['.gb', '.gbk', '.genbank', '.fasta', '.fa', '.seq'];

function isFileDrag(e) {
  var types = e.dataTransfer && e.dataTransfer.types;
  return types && Array.from(types).indexOf('Files') !== -1;
}

function isPlasmidFile(file) {
  var name = (file.name || '').toLowerCase();
  return PLASMID_EXTS.some(function(ext) { return name.endsWith(ext); });
}

chatPanelEl.addEventListener('dragenter', function(e) {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragCounter++;
  dropOverlayEl.classList.add('active');
});

chatPanelEl.addEventListener('dragleave', function(e) {
  if (!isFileDrag(e)) return;
  dragCounter--;
  if (dragCounter <= 0) { dragCounter = 0; dropOverlayEl.classList.remove('active'); }
});

chatPanelEl.addEventListener('dragover', function(e) {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

chatPanelEl.addEventListener('drop', function(e) {
  e.preventDefault();
  dragCounter = 0;
  dropOverlayEl.classList.remove('active');
  var file = e.dataTransfer.files[0];
  if (!file) return;
  var reader = new FileReader();
  if (isPlasmidFile(file)) {
    reader.onload = function(ev) { uploadPlasmidFile(ev.target.result, file.name); };
    reader.readAsText(file);
  } else if (file.name.endsWith('.csv') || file.type === 'text/csv') {
    reader.onload = function(ev) { showBatchConfirm(ev.target.result, file.name); };
    reader.readAsText(file);
  } else {
    alert('Supported file types: .gb, .gbk, .fasta (plasmid library) or .csv (batch design).');
  }
});

function onBatchFileChosen(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) { showBatchConfirm(e.target.result, file.name); };
  reader.readAsText(file);
  input.value = '';
}

function onPlasmidFileChosen(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) { uploadPlasmidFile(e.target.result, file.name); };
  reader.readAsText(file);
  input.value = '';
}

// ── Plasmid file upload ──
var plasmidBadgeEl = document.getElementById('plasmid-badge');
var plasmidBadgeNameEl = document.getElementById('plasmid-badge-name');
var plasmidBadgeStatusEl = document.getElementById('plasmid-badge-status');

function uploadPlasmidFile(text, filename) {
  // Show "analyzing" badge in the input area
  plasmidBadgeNameEl.textContent = filename;
  plasmidBadgeStatusEl.textContent = 'analyzing with plannotate…';
  plasmidBadgeEl.style.display = 'flex';

  var model = modelSelect.value;
  fetch('/api/upload-plasmid', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({content: text, filename: filename}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    plasmidBadgeEl.style.display = 'none';
    if (data.error) {
      alert('Could not process plasmid file: ' + data.error);
      return;
    }
    // Auto-send the intake message as if the user typed it
    hideWelcome();
    sendPlasmidIntakeMessage(data.message, model, filename, data.size_bp, data.feature_count);
  })
  .catch(function(e) {
    plasmidBadgeEl.style.display = 'none';
    alert('Upload failed: ' + e);
  });
}

async function sendPlasmidIntakeMessage(apiMessage, model, filename, sizeBp, featureCount) {
  if (isStreaming) return;
  isStreaming = true;
  streamingSessionId = currentSessionId;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  inputEl.disabled = true;
  hideWelcome();

  const inner = getInner();
  streamingInner = inner;

  // Show short summary as user bubble (not the full sequence)
  var summary = '📎 ' + filename;
  if (sizeBp) summary += ' — ' + sizeBp.toLocaleString() + ' bp';
  if (featureCount) summary += ', ' + featureCount + ' plannotate feature(s)';
  const nowStr = new Date().toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'});
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.innerHTML = '<div><div class="msg-bubble-user">' + escapeHtml(summary) + '</div><div class="msg-date">' + nowStr + '</div></div>';
  inner.appendChild(userDiv);
  scrollToBottom();
  showPendingCursor();

  abortController = new AbortController();
  try {
    const reqBody = {message: apiMessage, model: model};
    if (currentSessionId) reqBody.session_id = currentSessionId;
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(reqBody),
      signal: abortController.signal,
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      let streamDone = false;
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed.startsWith('data: ')) continue;
        let event;
        try { event = JSON.parse(trimmed.slice(6)); } catch { continue; }
        switch (event.type) {
          case 'session_id': saveSessionId(event.session_id); loadSessions(); break;
          case 'thinking_start': clearPendingCursor(); startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': clearPendingCursor(); flushTextBuffer(); startTextBlock(); break;
          case 'text_delta': bufferTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': clearPendingCursor(); startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'token_usage': updateTokenIndicator(event.input_tokens, event.context_window); break;
          case 'error': clearPendingCursor(); startTextBlock(); appendTextDelta('Error: ' + event.content); endTextBlock(); break;
          case 'done': streamDone = true; break;
        }
        if (streamDone) break;
      }
      if (streamDone) break;
    }
  } catch(err) {
    if (err.name !== 'AbortError') {
      clearPendingCursor(); startTextBlock(); appendTextDelta('Connection error: ' + err.message); endTextBlock();
    }
  }
  clearPendingCursor();
  isStreaming = false; abortController = null; streamingInner = null; streamingSessionId = null;
  sendBtn.style.display = 'flex'; stopBtn.style.display = 'none';
  inputEl.disabled = false; inputEl.focus();
  const cursor = messagesEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
  // Refresh library panel — agent may have saved a new backbone during this conversation turn
  loadUserLibrary();
}

function _splitCSVLine(line) {
  var result = [], field = '', inQuote = false;
  for (var i = 0; i < line.length; i++) {
    var c = line[i];
    if (c === '"') { inQuote = !inQuote; }
    else if (c === ',' && !inQuote) { result.push(field.trim()); field = ''; }
    else { field += c; }
  }
  result.push(field.trim());
  return result;
}

function _parseCSVRows(csvText) {
  var rawLines = csvText.split('\n');
  var headerLine = '';
  var headerFields = [];
  for (var i = 0; i < rawLines.length; i++) {
    if (rawLines[i].trim()) { headerLine = rawLines[i]; headerFields = _splitCSVLine(headerLine); break; }
  }
  var descIdx = headerFields.findIndex(function(h) { return h.trim().toLowerCase() === 'description'; });
  var nameIdx = headerFields.findIndex(function(h) { return h.trim().toLowerCase() === 'name'; });
  if (descIdx < 0) return {header: headerLine, rows: []};
  var rows = [];
  for (var j = i + 1; j < rawLines.length; j++) {
    var line = rawLines[j];
    if (!line.trim()) continue;
    var fields = _splitCSVLine(line);
    var desc = fields[descIdx] || '';
    if (!desc.trim()) continue;
    rows.push({
      description: desc,
      name: nameIdx >= 0 ? (fields[nameIdx] || '') : '',
      originalLine: line,
    });
  }
  return {header: headerLine, rows: rows};
}

function showBatchConfirm(csvText, filename) {
  var parsed = _parseCSVRows(csvText);
  var rows = parsed.rows;
  var confirmId = 'batch-confirm-' + Date.now();
  _batchConfirmData[confirmId] = {csvText: csvText, filename: filename, header: parsed.header, rows: rows};
  hideWelcome();
  var inner = getInner();
  var card = document.createElement('div');
  card.className = 'msg assistant';
  card.id = confirmId;
  var curModel = 'claude-sonnet-4-6';
  var modelOpts = [
    ['claude-opus-4-7', 'Opus 4.7 — most capable'],
    ['claude-opus-4-6', 'Opus 4.6'],
    ['claude-sonnet-4-6', 'Sonnet 4.6 — recommended for bulk assembly'],
    ['claude-haiku-4-5-20251001', 'Haiku 4.5 — fastest'],
  ].map(function(o) {
    return '<option value="' + o[0] + '"' + (curModel === o[0] ? ' selected' : '') + '>' + o[1] + '</option>';
  }).join('');

  var rowsHtml = rows.map(function(r, i) {
    var nameHtml = r.name ? '<span class="batch-confirm-row-name">' + escapeHtml(r.name) + '</span>' : '';
    return '<label class="batch-confirm-row">' +
      '<input type="checkbox" id="' + confirmId + '-row-' + i + '" checked onchange="updateBatchConfirmCount(\'' + confirmId + '\')">' +
      '<span class="batch-confirm-row-num">' + (i + 1) + '</span>' +
      '<span class="batch-confirm-row-desc">' + escapeHtml(r.description) + '</span>' +
      nameHtml +
    '</label>';
  }).join('');

  card.innerHTML = '<div class="msg-bubble-assistant"><div class="batch-confirm-card">' +
    '<div style="font-size:14px;font-weight:600;color:var(--sand-800);margin-bottom:10px">' +
      escapeHtml(filename) +
    '</div>' +
    '<div class="batch-advisory">' +
      '<strong>Before you run a bulk design</strong>' +
      '<ul>' +
        '<li><strong>Test first.</strong> Run a few representative prompts as individual chats and confirm they succeed before scaling up. Failures in bulk are harder to debug and still cost tokens.</li>' +
        '<li><strong>Design by parts works best.</strong> Bulk mode is optimized for assembly-style designs (backbone + insert combinations). Bespoke or highly custom designs often need back-and-forth that bulk mode can\'t do.</li>' +
        '<li><strong>Use a cheaper model.</strong> Sonnet 4.6 handles most bulk assembly tasks well at a fraction of the cost of Opus. Switch the model below before starting.</li>' +
      '</ul>' +
    '</div>' +
    '<label class="batch-confirm-select-all">' +
      '<input type="checkbox" id="' + confirmId + '-selectall" checked onchange="toggleBatchSelectAll(\'' + confirmId + '\',' + rows.length + ')">' +
      '<span>Select All — <span id="' + confirmId + '-selcount">' + rows.length + '</span> of ' + rows.length + ' selected</span>' +
    '</label>' +
    '<div class="batch-confirm-rows">' + rowsHtml + '</div>' +
    '<div style="margin-bottom:14px">' +
      '<label style="font-size:12px;font-weight:500;color:var(--sand-500);display:block;margin-bottom:5px">Model</label>' +
      '<select id="' + confirmId + '-model" class="model-select" style="font-size:12px;max-width:100%">' + modelOpts + '</select>' +
      '<div style="font-size:11px;color:var(--sand-400);margin-top:5px">Tip: Sonnet 4.6 handles bulk assembly by parts well at lower cost than Opus.</div>' +
    '</div>' +
    '<div style="display:flex;gap:8px;align-items:center">' +
      '<button id="' + confirmId + '-startbtn" class="send-btn" style="width:auto;padding:0 18px;height:32px;font-size:13px;border-radius:10px" ' +
        'onclick="startBatchFromConfirm(\'' + confirmId + '\',' + rows.length + ')">' +
        'Start <span id="' + confirmId + '-btncount">' + rows.length + '</span> design' + (rows.length === 1 ? '' : 's') +
      '</button>' +
      '<button onclick="cancelBatchConfirm(\'' + confirmId + '\')" ' +
        'style="padding:0 14px;height:32px;font-size:13px;background:transparent;border:1px solid var(--sand-200);border-radius:10px;cursor:pointer;color:var(--sand-600);font-family:inherit">Cancel</button>' +
    '</div>' +
  '</div></div>';
  inner.appendChild(card);
  scrollToBottom();
}

function updateBatchConfirmCount(confirmId) {
  var data = _batchConfirmData[confirmId];
  if (!data) return;
  var n = data.rows.length;
  var checked = 0;
  for (var i = 0; i < n; i++) {
    var cb = document.getElementById(confirmId + '-row-' + i);
    if (cb && cb.checked) checked++;
  }
  var selCount = document.getElementById(confirmId + '-selcount');
  var btnCount = document.getElementById(confirmId + '-btncount');
  var startBtn = document.getElementById(confirmId + '-startbtn');
  var selectAll = document.getElementById(confirmId + '-selectall');
  if (selCount) selCount.textContent = checked;
  if (btnCount) btnCount.textContent = checked;
  if (startBtn) startBtn.disabled = checked === 0;
  if (selectAll) selectAll.indeterminate = (checked > 0 && checked < n);
  if (selectAll && !selectAll.indeterminate) selectAll.checked = (checked === n);
}

function toggleBatchSelectAll(confirmId, total) {
  var selectAllEl = document.getElementById(confirmId + '-selectall');
  if (!selectAllEl) return;
  var checked = selectAllEl.checked;
  for (var i = 0; i < total; i++) {
    var cb = document.getElementById(confirmId + '-row-' + i);
    if (cb) cb.checked = checked;
  }
  updateBatchConfirmCount(confirmId);
}

function startBatchFromConfirm(confirmId, total) {
  var data = _batchConfirmData[confirmId];
  if (!data) return;
  delete _batchConfirmData[confirmId];
  var modelEl = document.getElementById(confirmId + '-model');
  var model = modelEl ? modelEl.value : modelSelect.value;
  // Build filtered CSV from checked rows
  var selectedLines = [];
  for (var i = 0; i < (data.rows || []).length; i++) {
    var cb = document.getElementById(confirmId + '-row-' + i);
    if (cb && cb.checked) selectedLines.push(data.rows[i].originalLine);
  }
  var card = document.getElementById(confirmId);
  if (card) card.remove();
  if (!selectedLines.length) return;
  var filteredCSV = data.header + '\n' + selectedLines.join('\n');
  uploadBatchCSV(filteredCSV, data.filename, model);
}

function cancelBatchConfirm(confirmId) {
  delete _batchConfirmData[confirmId];
  var card = document.getElementById(confirmId);
  if (card) card.remove();
}

function uploadBatchCSV(csvText, filename, model) {
  model = model || modelSelect.value;
  fetch('/api/batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({csv_content: csvText, model: model, filename: filename}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    var sid = data.session_id;
    var jobId = data.job_id;
    // Switch to the dedicated batch session
    saveSessionId(sid);
    loadSessions();
    // Clear any stale DOM content from the previous session
    messagesEl.innerHTML = '';
    // Render batch cards into the new session's container
    initBatchCards(jobId, data.row_count, filename, model);
    // Track and start polling per-session
    _batchSessions[sid] = jobId;
    if (_batchPollTimers[sid]) clearInterval(_batchPollTimers[sid]);
    _batchPollTimers[sid] = setInterval(function() { pollBatchForSession(sid); }, 2000);
    pollBatchForSession(sid);
  })
  .catch(function(e) { alert('Upload failed: ' + e); });
}

var _MODEL_LABELS = {
  'claude-opus-4-7': 'Opus 4.7',
  'claude-opus-4-6': 'Opus 4.6',
  'claude-sonnet-4-6': 'Sonnet 4.6',
  'claude-haiku-4-5-20251001': 'Haiku 4.5',
};

function initBatchCards(jobId, count, filename, model) {
  hideWelcome();
  var inner = getInner();
  var modelLabel = _MODEL_LABELS[model] || model || '';
  // Label with Pause All / Resume All controls
  var label = document.createElement('div');
  label.className = 'msg assistant';
  label.id = 'batch-label-' + jobId;
  label.innerHTML = '<div class="msg-bubble-assistant" style="color:var(--sand-500);font-size:13px;">' +
    'Batch designing <strong>' + count + ' plasmid' + (count === 1 ? '' : 's') + '</strong> from <em>' + escapeHtml(filename) + '</em>' +
    (modelLabel ? ' \u00b7 <span style="color:var(--sand-400)">' + escapeHtml(modelLabel) + '</span>' : '') + '. ' +
    'Click any row to expand and see what\u2019s happening.' +
    '<div style="display:flex;gap:8px;margin-top:10px;align-items:center" id="batch-ctrl-' + jobId + '">' +
      '<button class="batch-row-pause-btn" id="batch-pause-all-' + jobId + '" title="Pause all" style="width:auto;padding:0 10px;height:26px;font-size:12px;border-radius:6px;border:1px solid var(--sand-200);gap:5px;color:var(--sand-600)" onclick="pauseAllBatch(\'' + jobId + '\')">' + PAUSE_SVG + ' Pause all</button>' +
      '<button class="batch-row-pause-btn" id="batch-resume-all-' + jobId + '" title="Resume all" style="width:auto;padding:0 10px;height:26px;font-size:12px;border-radius:6px;border:1px solid var(--sand-200);gap:5px;color:var(--sand-600);display:none" onclick="resumeAllBatch(\'' + jobId + '\')">' + RESUME_SVG + ' Resume all</button>' +
    '</div>' +
    '</div>';
  inner.appendChild(label);
  // Placeholder cards
  for (var i = 0; i < count; i++) {
    var card = document.createElement('div');
    card.className = 'msg assistant';
    card.id = 'batch-card-' + jobId + '-' + i;
    card.innerHTML = buildBatchCardHtml(jobId, i, {
      status: 'pending', description: '\u2026', exports: [], error: null, log: [], paused: false
    }, false);
    inner.appendChild(card);
  }
  scrollToBottom();
}

function pollBatchForSession(sessionId) {
  var jobId = _batchSessions[sessionId];
  if (!jobId) return;
  fetch('/api/batch/' + jobId)
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) return;
    // Only update the DOM if this session is still active
    if (currentSessionId === sessionId) {
      updateBatchCards(jobId, data.rows);
    }
    var anyRunning = data.rows && data.rows.some(function(r) { return r.status === 'running' || r.status === 'pending'; });
    if (data.status === 'done' && !anyRunning) {
      clearInterval(_batchPollTimers[sessionId]);
      delete _batchPollTimers[sessionId];
      if (currentSessionId !== sessionId) return;
      // Hide pause controls when batch is fully done
      var ctrlEl = document.getElementById('batch-ctrl-' + jobId);
      if (ctrlEl) ctrlEl.style.display = 'none';
      // Add Download All split button to label message
      var labelEl = document.getElementById('batch-label-' + jobId);
      if (labelEl && !labelEl.querySelector('.batch-dl-all-btn')) {
        var bubble = labelEl.querySelector('.msg-bubble-assistant');
        if (bubble) {
          var wrap = document.createElement('div');
          wrap.className = 'dl-split-wrap batch-dl-all-btn';
          wrap.style.cssText = 'margin-top:10px;';
          var jid = jobId;
          var allMenuId = 'dlmenu-all-' + jid;
          wrap.innerHTML =
            '<button class="download-btn" onclick="downloadAllBatch(\'' + jid + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
            '<button class="dl-chevron-btn" onclick="toggleDlMenu(event,\'' + allMenuId + '\')" title="More options">' + _CHEV_DOWN_SVG + '</button>' +
            '<div class="dl-menu" id="' + allMenuId + '">' +
              '<button class="dl-menu-item" onclick="downloadAllBatch(\'' + jid + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
              (_userLibraryAvailable ? '<button class="dl-menu-item" id="savall-local-' + jid + '" onclick="event.stopPropagation();saveAllBatchToLocal(\'' + jid + '\',document.getElementById(\'savall-local-' + jid + '\'))">' +
                '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 002-2v-4M17 8l-5-5-5 5M12 3v12"/></svg> Save All to Local Library</button>' : '') +
              '<button class="dl-menu-item" id="savall-con-' + jid + '" onclick="event.stopPropagation();saveAllBatchConstructs(\'' + jid + '\',document.getElementById(\'savall-con-' + jid + '\'))">' + _SAVE_SVG + ' Save All Constructs</button>' +
            '</div>';
          bubble.appendChild(document.createElement('br'));
          bubble.appendChild(wrap);
        }
      }
    }
  })
  .catch(function() {});
}

var STATUS_ICONS = {
  pending: '<svg width="18" height="18" fill="none" stroke="var(--sand-300)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>',
  running: '<svg width="18" height="18" fill="none" stroke="var(--brand-fig)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" class="spin"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>',
  done: '<svg width="18" height="18" fill="none" stroke="var(--brand-aqua)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
  no_export: '<svg width="18" height="18" fill="none" stroke="var(--sand-400)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>',
  error: '<svg width="18" height="18" fill="none" stroke="var(--brand-orange)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg>',
  paused: '<svg width="18" height="18" fill="none" stroke="var(--sand-400)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>',
};
var STATUS_LABELS = {pending: 'Pending', running: 'Running\u2026', done: 'Done', no_export: 'No export produced', error: 'Error', paused: 'Paused'};
var CHEV_SVG = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>';
var PAUSE_SVG = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
var RESUME_SVG = '<svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>';

function renderBatchLog(log) {
  if (!log || !log.length) return '<div style="font-size:12px;color:var(--sand-400);padding:4px 0;">No activity yet.</div>';
  return log.map(function(entry) {
    if (entry.type === 'tool') {
      return '<div class="batch-log-entry batch-log-tool">' +
        '<div class="batch-log-tool-header">' +
          '<svg width="11" height="11" fill="none" stroke="var(--brand-fig)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3-3a1 1 0 000-1.4l-1.6-1.6a1 1 0 00-1.4 0l-3 3z"/><path d="M20.26 2.26L9 13.5l-5 1 1-5L16.5 3.74"/></svg>' +
          escapeHtml(entry.name) +
        '</div>' +
        '<div class="batch-log-tool-result">' + escapeHtml(entry.result || '') + '</div>' +
      '</div>';
    } else if (entry.type === 'text') {
      return '<div class="batch-log-entry batch-log-text">' + renderContent(entry.content || '') + '</div>';
    } else if (entry.type === 'user') {
      return '<div class="batch-log-entry batch-log-user">' + escapeHtml(entry.content || '') + '</div>';
    } else if (entry.type === 'error') {
      return '<div class="batch-log-entry batch-log-error">\u26a0 ' + escapeHtml(entry.content || '') + '</div>';
    }
    return '';
  }).join('');
}

var _DL_SVG = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
var _CHEV_DOWN_SVG = '<svg width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>';
var _SAVE_SVG = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>';

function toggleDlMenu(event, menuId) {
  event.stopPropagation();
  var menu = document.getElementById(menuId);
  if (!menu) return;
  var isOpen = menu.classList.toggle('open');
  if (isOpen) {
    document.querySelectorAll('.dl-menu.open').forEach(function(m) {
      if (m.id !== menuId) m.classList.remove('open');
    });
    function closeMenu(e) {
      if (!menu.contains(e.target)) {
        menu.classList.remove('open');
        document.removeEventListener('click', closeMenu, true);
      }
    }
    document.addEventListener('click', closeMenu, true);
  }
}

function buildDownloadsHtml(jobId, idx, exports) {
  if (!exports || !exports.length) return '';
  var html = '<div class="batch-row-downloads">';
  exports.forEach(function(exp, eidx) {
    var fname = escapeHtml(exp.filename);
    var isGbk = /\.(gb|gbk|genbank)$/i.test(exp.filename);
    var menuId = 'dlmenu-' + jobId + '-' + idx + '-' + eidx;
    var dlCall = 'event.stopPropagation();downloadBatchFile(\'' + jobId + '\',' + idx + ',' + eidx + ',\'' + fname + '\')';
    if (isGbk) {
      // Split button: download primary + chevron dropdown
      html += '<div class="dl-split-wrap" onclick="event.stopPropagation()">' +
        '<button class="download-btn" onclick="' + dlCall + '">' + _DL_SVG + ' ' + fname + '</button>' +
        '<button class="dl-chevron-btn" onclick="toggleDlMenu(event,\'' + menuId + '\')" title="More options">' + _CHEV_DOWN_SVG + '</button>' +
        '<div class="dl-menu" id="' + menuId + '">' +
          '<button class="dl-menu-item" onclick="event.stopPropagation();downloadBatchFile(\'' + jobId + '\',' + idx + ',' + eidx + ',\'' + fname + '\')">' + _DL_SVG + ' Download to computer</button>' +
          (_userLibraryAvailable ? '<button class="dl-menu-item" id="savlocal-' + menuId + '" onclick="event.stopPropagation();saveBatchToLocal(\'' + jobId + '\',' + idx + ',' + eidx + ',document.getElementById(\'savlocal-' + menuId + '\'))">' +
            '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 002-2v-4M17 8l-5-5-5 5M12 3v12"/></svg> Save to Local Library</button>' : '') +
        '</div>' +
      '</div>';
      // Save Construct button (to DB)
      html += '<button class="save-btn" id="savcon-' + menuId + '" style="margin-left:4px" onclick="event.stopPropagation();saveBatchConstruct(\'' + jobId + '\',' + idx + ',' + eidx + ',document.getElementById(\'savcon-' + menuId + '\'))">' +
        _SAVE_SVG + ' Save Construct</button>';
    } else {
      html += '<button class="download-btn" onclick="' + dlCall + '">' + _DL_SVG + ' ' + fname + '</button>';
    }
    if (exp.has_plot) {
      html += '<button class="download-btn" style="border-color:var(--brand-fig-30);color:var(--brand-fig);background:var(--brand-fig-10);margin-left:4px" ' +
        'onclick="event.stopPropagation();openBatchPlot(\'' + jobId + '\',' + idx + ',' + eidx + ')">' +
        '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>' +
        'View Map</button>';
    }
  });
  return html + '</div>';
}

function buildFollowupHtml(jobId, idx, status) {
  if (status === 'running' || status === 'pending') return '';
  var fid = 'batch-finput-' + jobId + '-' + idx;
  return '<div class="batch-followup">' +
    '<textarea class="batch-followup-input" id="' + fid + '" rows="1" ' +
      'placeholder="Follow up with the agent about this design\u2026" ' +
      'onkeydown="batchFollowupKey(event,\'' + jobId + '\',' + idx + ')" ' +
      'oninput="this.style.height=\'auto\';this.style.height=Math.min(this.scrollHeight,100)+\'px\'"></textarea>' +
    '<button class="batch-followup-send" onclick="sendBatchFollowup(\'' + jobId + '\',' + idx + ')" title="Send">' +
      '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>' +
    '</button>' +
  '</div>';
}

function buildBatchCardHtml(jobId, idx, row, isOpen) {
  var isPaused = row.paused && row.status === 'running';
  var icon = isPaused ? STATUS_ICONS.paused : (STATUS_ICONS[row.status] || STATUS_ICONS.pending);
  var label = isPaused ? STATUS_LABELS.paused : (STATUS_LABELS[row.status] || row.status);
  var desc = escapeHtml((row.description || '').slice(0, 120) + ((row.description || '').length > 120 ? '\u2026' : ''));
  var downloads = buildDownloadsHtml(jobId, idx, row.exports);
  var logId = 'batch-log-' + jobId + '-' + idx;
  var chevId = 'batch-chev-' + jobId + '-' + idx;
  var pauseBtn = '';
  if (row.status === 'running') {
    if (isPaused) {
      pauseBtn = '<button class="batch-row-pause-btn" title="Resume" onclick="event.stopPropagation();resumeBatchRow(\'' + jobId + '\',' + idx + ')">' + RESUME_SVG + '</button>';
    } else {
      pauseBtn = '<button class="batch-row-pause-btn" title="Pause" onclick="event.stopPropagation();pauseBatchRow(\'' + jobId + '\',' + idx + ')">' + PAUSE_SVG + '</button>';
    }
  }
  return '<div class="batch-card">' +
    '<div class="batch-row-header" onclick="toggleBatchCard(\'' + jobId + '\',' + idx + ')">' +
      '<div class="batch-row-status">' + icon + '</div>' +
      '<div class="batch-row-body">' +
        '<div class="batch-row-desc">' + desc + '</div>' +
        '<div class="batch-row-meta">' + (idx + 1) + ' \xb7 ' + label + '</div>' +
        downloads +
      '</div>' +
      pauseBtn +
      '<span id="' + chevId + '" class="batch-row-chevron' + (isOpen ? ' open' : '') + '">' + CHEV_SVG + '</span>' +
    '</div>' +
    '<div id="' + logId + '" class="batch-row-log' + (isOpen ? ' open' : '') + '">' +
      renderBatchLog(row.log) +
      buildFollowupHtml(jobId, idx, row.status) +
    '</div>' +
  '</div>';
}

function updateBatchCards(jobId, rows) {
  rows.forEach(function(row, idx) {
    var cardEl = document.getElementById('batch-card-' + jobId + '-' + idx);
    if (!cardEl) return;
    // Preserve expanded state
    var logEl = document.getElementById('batch-log-' + jobId + '-' + idx);
    var isOpen = logEl ? logEl.classList.contains('open') : false;
    cardEl.innerHTML = buildBatchCardHtml(jobId, idx, row, isOpen);
  });
}

function toggleBatchCard(jobId, idx) {
  var log = document.getElementById('batch-log-' + jobId + '-' + idx);
  var chev = document.getElementById('batch-chev-' + jobId + '-' + idx);
  if (!log) return;
  var open = log.classList.toggle('open');
  if (chev) chev.classList.toggle('open', open);
}

function batchFollowupKey(e, jobId, rowIdx) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBatchFollowup(jobId, rowIdx); }
}

function sendBatchFollowup(jobId, rowIdx) {
  var inputEl = document.getElementById('batch-finput-' + jobId + '-' + rowIdx);
  if (!inputEl) return;
  var message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  // Optimistically show the user message in the log
  var logEl = document.getElementById('batch-log-' + jobId + '-' + rowIdx);
  if (logEl) {
    var followup = logEl.querySelector('.batch-followup');
    var userDiv = document.createElement('div');
    userDiv.className = 'batch-log-entry batch-log-user';
    userDiv.textContent = message;
    if (followup) logEl.insertBefore(userDiv, followup);
    else logEl.appendChild(userDiv);
    // Disable input while running
    if (followup) {
      var btn = followup.querySelector('.batch-followup-send');
      if (inputEl) inputEl.disabled = true;
      if (btn) btn.disabled = true;
    }
  }
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/continue', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: message}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    // Restart polling for whichever session owns this job
    var ownerSid = Object.keys(_batchSessions).find(function(k) { return _batchSessions[k] === jobId; });
    if (ownerSid && !_batchPollTimers[ownerSid]) {
      _batchPollTimers[ownerSid] = setInterval(function() { pollBatchForSession(ownerSid); }, 2000);
    }
  })
  .catch(function(e) { alert('Failed to send: ' + e); });
}

function openBatchPlot(jobId, rowIdx, expIdx) {
  // Expand the card if collapsed
  var log = document.getElementById('batch-log-' + jobId + '-' + rowIdx);
  var chev = document.getElementById('batch-chev-' + jobId + '-' + rowIdx);
  if (log && !log.classList.contains('open')) {
    log.classList.add('open');
    if (chev) chev.classList.add('open');
  }
  // Don't render twice
  var plotWrapperId = 'bplotwrap-' + jobId + '-' + rowIdx + '-' + expIdx;
  if (document.getElementById(plotWrapperId)) return;
  var plotId = 'bplot-' + jobId + '-' + rowIdx + '-' + expIdx;
  // Insert plot container before the follow-up input
  var wrapper = document.createElement('div');
  wrapper.id = plotWrapperId;
  wrapper.className = 'batch-plot-wrapper';
  wrapper.style.cssText = 'padding:12px 16px;border-top:1px solid var(--sand-100);max-width:640px;';
  wrapper.innerHTML =
    '<div style="font-size:11px;font-weight:600;color:var(--sand-500);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;">Plasmid Map</div>' +
    '<div id="' + plotId + '" style="width:600px;height:600px;">Loading\u2026</div>';
  if (log) {
    var followup = log.querySelector('.batch-followup');
    if (followup) log.insertBefore(wrapper, followup);
    else log.appendChild(wrapper);
  }
  // Fetch the plot JSON then wait one animation frame so the browser has
  // laid out the container before Bokeh reads its dimensions.
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/plot/' + expIdx)
  .then(function(r) { return r.json(); })
  .then(function(data) {
    var el = document.getElementById(plotId);
    if (!el) return;
    if (data.error) { el.textContent = 'No map available.'; el.style.minHeight = ''; return; }
    el.innerHTML = '';
    // Double rAF ensures the element is fully painted before Bokeh measures it
    requestAnimationFrame(function() {
      requestAnimationFrame(function() {
        Bokeh.embed.embed_item(data, plotId);
      });
    });
  })
  .catch(function() {
    var el = document.getElementById(plotId);
    if (el) { el.textContent = 'Failed to load map.'; el.style.minHeight = ''; }
  });
}

function downloadAllBatch(jobId) {
  var a = document.createElement('a');
  a.href = '/api/batch/' + jobId + '/download-all';
  a.download = 'batch_designs.zip';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function pauseBatchRow(jobId, rowIdx) {
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/pause', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .catch(function() {});
}

function resumeBatchRow(jobId, rowIdx) {
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/resume', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .catch(function() {});
}

function pauseAllBatch(jobId) {
  fetch('/api/batch/' + jobId + '/pause-all', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function() {
    // Swap button visibility
    var p = document.getElementById('batch-pause-all-' + jobId);
    var r = document.getElementById('batch-resume-all-' + jobId);
    if (p) p.style.display = 'none';
    if (r) r.style.display = '';
  })
  .catch(function() {});
}

function resumeAllBatch(jobId) {
  fetch('/api/batch/' + jobId + '/resume-all', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function() {
    var p = document.getElementById('batch-pause-all-' + jobId);
    var r = document.getElementById('batch-resume-all-' + jobId);
    if (p) p.style.display = '';
    if (r) r.style.display = 'none';
  })
  .catch(function() {});
}

// ── Saved Constructs ─────────────────────────────────────────────────────────

const _SVG_CHECK = '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>';
const _SVG_DL = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>';
const _SVG_FOLDER = '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>';

function _triggerDownload(content, filename) {
  const blob = new Blob([content], {type: 'application/octet-stream'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function addExportButtons(container, toolInput, genbankContent, filename) {
  const outer = document.createElement('div');
  outer.className = 'msg assistant';

  const _SVG_CHOOSE = '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>';
  const localLibItem = _userLibraryAvailable
    ? '<button class="dl-menu-item" data-role="dl-library">' + _SVG_FOLDER + ' Save to Local Library</button>'
    : '';

  outer.innerHTML =
    '<div class="msg-bubble-assistant" style="margin-top:8px">' +
      '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">' +
        '<div class="dl-split-wrap">' +
          '<button class="download-btn" data-role="dl" data-tooltip="Download this file to your computer">' + _SVG_DL + ' Download</button>' +
          '<button class="dl-chevron-btn" data-role="dl-chevron" aria-label="More save options">' +
            '<svg width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>' +
          '</button>' +
          '<div class="dl-menu" style="display:none">' +
            '<button class="dl-menu-item" data-role="dl-computer">' + _SVG_DL + ' Download to computer</button>' +
            '<button class="dl-menu-item" data-role="dl-choosepath">' + _SVG_CHOOSE + ' Save to…</button>' +
            localLibItem +
          '</div>' +
        '</div>' +
        '<button class="save-btn" data-role="save" data-tooltip="Save to local database">' +
          '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>' +
          ' Save Construct' +
        '</button>' +
      '</div>' +
    '</div>';
  container.appendChild(outer);

  // Download main button — direct download to computer
  outer.querySelector('[data-role="dl"]').addEventListener('click', function() {
    _triggerDownload(genbankContent, filename);
  });

  // Chevron toggles dropdown
  const chevronBtn = outer.querySelector('[data-role="dl-chevron"]');
  const menu = outer.querySelector('.dl-menu');
  chevronBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    menu.style.display = menu.style.display === 'none' ? '' : 'none';
  });
  document.addEventListener('click', function() { menu.style.display = 'none'; });

  outer.querySelector('[data-role="dl-computer"]').addEventListener('click', function() {
    menu.style.display = 'none';
    _triggerDownload(genbankContent, filename);
  });

  outer.querySelector('[data-role="dl-choosepath"]').addEventListener('click', async function() {
    menu.style.display = 'none';
    if (window.showSaveFilePicker) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: filename,
          types: [{ description: 'GenBank file', accept: {'text/plain': ['.gb', '.gbk']} }],
        });
        const writable = await handle.createWritable();
        await writable.write(genbankContent);
        await writable.close();
      } catch(e) {
        if (e.name !== 'AbortError') _triggerDownload(genbankContent, filename);
      }
    }
  });

  const libItem = outer.querySelector('[data-role="dl-library"]');
  if (libItem) {
    async function _saveToLib(name, overwrite) {
      const dlBtn = outer.querySelector('[data-role="dl"]');
      const origHtml = dlBtn.innerHTML;
      dlBtn.disabled = true; dlBtn.textContent = 'Saving…';
      try {
        const r = await fetch('/api/local-library/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({name, content: genbankContent, overwrite}),
        });
        const data = await r.json();
        if (data.saved_to) {
          dlBtn.innerHTML = _SVG_CHECK + ' Saved: ' + escapeHtml(data.saved_to.split('/').pop());
          dlBtn.style.opacity = '0.75';
          // Remove any rename row
          const rr = outer.querySelector('.dl-rename-row');
          if (rr) rr.remove();
        } else if (data.exists) {
          dlBtn.innerHTML = origHtml; dlBtn.disabled = false;
          // Show inline rename form
          let rr = outer.querySelector('.dl-rename-row');
          if (!rr) {
            rr = document.createElement('div');
            rr.className = 'dl-rename-row';
            outer.querySelector('.msg-bubble-assistant > div').after(rr);
          }
          rr.innerHTML =
            '<span style="font-size:11px;color:var(--sand-500)">A file with that name already exists. Save as:</span>' +
            '<input type="text" value="' + escapeHtml(data.suggested_name) + '">.gb' +
            '<button class="dl-rename-confirm">Save</button>' +
            '<button class="dl-rename-cancel">Cancel</button>';
          const inp = rr.querySelector('input');
          inp.focus(); inp.select();
          rr.querySelector('.dl-rename-confirm').addEventListener('click', function() {
            _saveToLib(inp.value.trim() || data.suggested_name, true);
          });
          inp.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') _saveToLib(inp.value.trim() || data.suggested_name, true);
            if (e.key === 'Escape') rr.remove();
          });
          rr.querySelector('.dl-rename-cancel').addEventListener('click', function() { rr.remove(); });
        } else { dlBtn.innerHTML = origHtml; dlBtn.disabled = false; }
      } catch(e) { dlBtn.innerHTML = origHtml; dlBtn.disabled = false; }
    }
    libItem.addEventListener('click', function() {
      menu.style.display = 'none';
      _saveToLib(toolInput.construct_name || 'construct', false);
    });
  }

  // Save Construct → DB
  outer.querySelector('[data-role="save"]').addEventListener('click', async function() {
    const btn = this;
    btn.disabled = true; btn.textContent = 'Saving…';
    const body = {
      construct_name: toolInput.construct_name || 'construct',
      genbank_content: genbankContent,
      total_size_bp: null,
      session_id: currentSessionId,
      backbone_name: toolInput.backbone_name || '',
      insert_name: toolInput.insert_name || '',
      sequence_cache_key: toolInput.sequence_cache_key || '',
    };
    try {
      const r = await fetch('/api/db/constructs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      const data = await r.json();
      if (data.id) { btn.innerHTML = _SVG_CHECK + ' Saved'; btn.style.opacity = '0.75'; refreshLibraryData(); }
      else { btn.textContent = 'Save failed'; btn.disabled = false; }
    } catch(e) { btn.textContent = 'Save failed'; btn.disabled = false; }
  });
}

// ── Library panel toggle ─────────────────────────────────────────────────────

let _libraryPanelOpen = false;

function toggleLibraryPanel() {
  const panel = document.getElementById('library-panel');
  _libraryPanelOpen = !_libraryPanelOpen;
  panel.style.display = _libraryPanelOpen ? 'flex' : 'none';
  const btn = document.getElementById('lib-panel-btn');
  if (btn) btn.classList.toggle('active', _libraryPanelOpen);
  if (_libraryPanelOpen) {
    _checkUserLibrary();
    // Defer init until after the browser has painted the panel so Tabulator
    // measures the real container width, not 0.
    requestAnimationFrame(function() {
      if (!_constructsTable) {
        _initTabulator();
      } else {
        _constructsTable.setData('/api/db/constructs');
      }
      if (!_cy) _initCytoscape();
    });
  }
}

function showLibraryTab(tab) {
  document.getElementById('lib-table-pane').style.display = tab === 'table' ? '' : 'none';
  document.getElementById('lib-graph-pane').style.display = tab === 'graph' ? 'flex' : 'none';
  document.getElementById('lib-tab-table').classList.toggle('active', tab === 'table');
  document.getElementById('lib-tab-graph').classList.toggle('active', tab === 'graph');
  if (tab === 'graph') {
    _loadGraphData();
    if (_cy) { setTimeout(function() { _cy.resize(); _cy.fit(); }, 50); }
  }
}

function refreshLibraryData() {
  if (_constructsTable) _constructsTable.setData('/api/db/constructs');
  if (_cy && document.getElementById('lib-graph-pane').style.display !== 'none') _loadGraphData();
}

// ── Tabulator ────────────────────────────────────────────────────────────────

let _constructsTable = null;

function _initTabulator() {
  _constructsTable = new Tabulator('#constructs-table', {
    ajaxURL: '/api/db/constructs',
    layout: 'fitColumns',
    height: 'calc(90vh - 58px)',
    placeholder: 'No constructs saved yet. Export a construct as GenBank and click "Save Construct".',
    columns: [
      {formatter: 'rowSelection', titleFormatter: 'rowSelection', width: 42,
       hozAlign: 'center', headerSort: false, frozen: true,
       cellClick: function(e) { e.stopPropagation(); }},
      {title: 'ID', field: 'accession', frozen: true, width: 100,
       sorter: 'string', hozAlign: 'center',
       formatter: function(cell) {
         return '<code style="font-size:11px;color:var(--brand-fig);font-weight:600">' + escapeHtml(cell.getValue() || '') + '</code>';
       }},
      {title: 'Construct Name', field: 'construct_name', frozen: true, width: 190,
       sorter: 'string', tooltip: true},
      {title: 'Source', field: 'origin', width: 110, headerSort: false,
       formatter: function(cell) {
         const v = cell.getValue() || 'designer';
         const cfg = {
           'designer':     {bg:'#3B82F6', label:'Designed'},
           'user_library': {bg:'#10B981', label:'Your Library'},
           'annotation':   {bg:'#8B5CF6', label:'Annotation'},
         };
         const c = cfg[v] || cfg['designer'];
         return '<span style="background:' + c.bg + ';color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;font-family:Inter,sans-serif;white-space:nowrap">' + c.label + '</span>';
       }},
      {title: 'Type', field: 'part_type', width: 80, headerSort: false,
       formatter: function(cell) {
         const v = cell.getValue();
         if (!v) return '';
         return '<span style="font-size:11px;color:var(--text-secondary);font-family:Inter,sans-serif">' + escapeHtml(v) + '</span>';
       }},
      {title: 'User Label', field: 'user_name', editor: 'input', width: 130,
       cellEdited: _onCellEdited, placeholder: 'Add label…'},
      {title: 'bp', field: 'total_size_bp', sorter: 'number', width: 70, hozAlign: 'right'},
      {title: 'Created', field: 'created_at', sorter: 'datetime', width: 140,
       formatter: function(cell) {
         const v = cell.getValue();
         return v ? v.slice(0, 16).replace('T', ' ') : '';
       }},
      {title: '&#10003;', field: 'sequence_verified', formatter: 'tickCross',
       editor: true, width: 42, hozAlign: 'center', cellEdited: _onCellEdited,
       headerTooltip: 'Sequence verified'},
      {title: 'File', width: 52, formatter: function(cell) {
         const id = cell.getRow().getData().id;
         return '<a class="download-btn" style="font-size:11px;padding:3px 7px" href="/api/db/constructs/' + id + '/genbank">GBK</a>';
       }, hozAlign: 'center', headerSort: false, cellClick: function(e) { e.stopPropagation(); }},
      {title: 'Notes', field: 'notes', editor: 'textarea', widthGrow: 1,
       cellEdited: _onCellEdited, formatter: 'plaintext', tooltip: true},
    ],
    rowFormatter: function(row) {
      row.getElement().addEventListener('click', function(e) {
        if (e.target.tagName === 'A' || e.target.tagName === 'INPUT') return;
        _toggleRowDetail(row);
      });
    },
  });
  _constructsTable.on('rowSelectionChanged', function(data, rows) {
    const n = rows.length;
    const btn = document.getElementById('lib-remove-btn');
    const cnt = document.getElementById('lib-remove-count');
    if (btn) btn.style.display = n > 0 ? '' : 'none';
    if (cnt) cnt.textContent = n;
  });
}

async function _onCellEdited(cell) {
  const id = cell.getRow().getData().id;
  const field = cell.getField();
  const value = cell.getValue();
  await fetch('/api/db/constructs/' + id, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({[field]: value}),
  });
}

function _toggleRowDetail(row) {
  const el = row.getElement();
  const existing = el.nextElementSibling;
  if (existing && existing.classList.contains('row-detail-wrap')) {
    existing.remove();
    return;
  }
  const data = row.getData();
  const wrap = document.createElement('div');
  wrap.className = 'row-detail-wrap';

  // Metadata grid (for imported library items)
  let partsHtml = '';
  const meta = data.metadata || {};
  const metaFields = [
    ['Description', meta.description],
    ['Category', meta.category],
    ['Assembly enzyme', meta.assembly_enzyme],
    ['Next step enzyme', meta.next_step_enzyme],
    ['Overhang L', meta.overhang_l],
    ['Overhang R', meta.overhang_r],
    ['Overhang pair 1', (meta.overhang_left && meta.overhang_right) ? meta.overhang_left + ' / ' + meta.overhang_right : null],
    ['Overhang pair 2', (meta.overhang_left_2 && meta.overhang_right_2) ? meta.overhang_left_2 + ' / ' + meta.overhang_right_2 : null],
    ['Insert size', meta.insert_size_bp ? meta.insert_size_bp + ' bp' : null],
    ['Bacterial resistance', meta.bacterial_resistance],
    ['Mammalian selection', meta.mammalian_selection],
    ['E. coli strain', meta.ecoli_strain],
  ].filter(function(r) { return r[1]; });
  if (metaFields.length) {
    partsHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px 16px;padding:10px 0 6px;border-bottom:1px solid var(--sand-200);margin-bottom:8px">';
    metaFields.forEach(function(r) {
      partsHtml += '<div style="font-size:12px;font-family:Inter,sans-serif">' +
        '<span style="color:var(--text-secondary)">' + escapeHtml(r[0]) + ':</span> ' +
        '<strong>' + escapeHtml(String(r[1])) + '</strong></div>';
    });
    partsHtml += '</div>';
  }

  // Parts table
  partsHtml += '<h4>Parts &amp; Provenance</h4>';
  if (data.parts && data.parts.length) {
    partsHtml += '<table class="parts-sub-table"><thead><tr>' +
      '<th>Part</th><th>Type</th><th>Region</th><th>Source</th><th>DOI / Accession</th>' +
      '</tr></thead><tbody>';
    data.parts.forEach(function(p) {
      const srcLink = p.source_url
        ? '<a href="' + escapeHtml(p.source_url) + '" target="_blank" rel="noopener">' + escapeHtml(p.source_system || p.source_url) + '</a>'
        : escapeHtml(p.source_system || '—');
      let ref = '—';
      if (p.source_doi) ref = '<a href="https://doi.org/' + escapeHtml(p.source_doi) + '" target="_blank" rel="noopener">' + escapeHtml(p.source_doi) + '</a>';
      else if (p.genbank_accession) ref = '<a href="https://www.ncbi.nlm.nih.gov/nuccore/' + escapeHtml(p.genbank_accession) + '" target="_blank" rel="noopener">' + escapeHtml(p.genbank_accession) + '</a>';
      else if (p.addgene_id) ref = '<a href="https://www.addgene.org/' + escapeHtml(p.addgene_id) + '/" target="_blank" rel="noopener">Addgene #' + escapeHtml(p.addgene_id) + '</a>';
      partsHtml += '<tr><td>' + escapeHtml(p.part_name) + '</td><td>' + escapeHtml(p.part_type) +
        '</td><td>' + escapeHtml(p.part_region || '—') + '</td><td>' + srcLink + '</td><td>' + ref + '</td></tr>';
    });
    partsHtml += '</tbody></table>';
  } else {
    partsHtml += '<p style="font-size:11px;color:var(--text-secondary)">No part details captured.</p>';
  }

  // Upload verified sequence
  const uploadId = 'upload-seq-' + data.id;
  partsHtml += '<label class="upload-verified-btn" for="' + uploadId + '">' +
    '&#8679; Upload verified sequence</label>' +
    '<input type="file" id="' + uploadId + '" accept=".gb,.fasta,.fa,.txt" style="display:none">';

  // Save to local library (only for designer constructs when user library is configured)
  if ((data.origin || 'designer') === 'designer' && _userLibraryAvailable) {
    partsHtml += '<button id="save-to-lib-' + data.id + '" class="upload-verified-btn" style="cursor:pointer;border:none;background:var(--sand-200)" onclick="saveToLocalLibrary(' + data.id + ', this)">&#8681; Save to Local Library</button>';
    if (data.local_path) {
      partsHtml += '<span style="font-size:11px;color:var(--text-secondary);margin-left:8px">Saved: ' + escapeHtml(data.local_path) + '</span>';
    }
  }

  wrap.innerHTML = partsHtml;
  el.after(wrap);

  // Wire up upload
  const fileInput = wrap.querySelector('#' + uploadId);
  fileInput.addEventListener('change', async function() {
    if (!fileInput.files.length) return;
    const text = await fileInput.files[0].text();
    await fetch('/api/db/constructs/' + data.id, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({verified_sequence: text, sequence_verified: true}),
    });
    // Update the cell in the table
    if (_constructsTable) {
      const r = _constructsTable.getRow(data.id);
      if (r) r.update({sequence_verified: true});
    }
    const lbl = wrap.querySelector('label.upload-verified-btn');
    if (lbl) lbl.textContent = '✓ Verified sequence uploaded';
  });
}

async function saveToLocalLibrary(constructId, btn, name, overwrite) {
  const origText = btn.dataset.origText || btn.textContent;
  btn.dataset.origText = origText;
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const body = {};
    if (name) body.name = name;
    if (overwrite) body.overwrite = true;
    const r = await fetch('/api/db/constructs/' + constructId + '/save-to-library', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.saved_to) {
      btn.textContent = '✓ Saved';
      // Remove any rename row that may be present
      const existing = btn.parentElement && btn.parentElement.querySelector('.lib-rename-row');
      if (existing) existing.remove();
      const span = document.createElement('span');
      span.style.cssText = 'font-size:11px;color:var(--text-secondary);margin-left:8px';
      span.textContent = data.saved_to;
      btn.after(span);
    } else if (data.exists) {
      btn.textContent = origText;
      btn.disabled = false;
      // Remove any stale rename row first
      const stale = btn.parentElement && btn.parentElement.querySelector('.lib-rename-row');
      if (stale) stale.remove();
      // Inline rename form
      const row = document.createElement('div');
      row.className = 'dl-rename-row lib-rename-row';
      row.style.marginTop = '6px';
      row.innerHTML =
        '<span style="font-size:11px;color:var(--sand-500)">File exists. Save as:</span>' +
        '<input type="text" value="' + escapeHtml(data.suggested_name) + '">.gb' +
        '<button class="dl-rename-confirm">Save</button>' +
        '<button class="dl-rename-cancel">Cancel</button>';
      btn.after(row);
      const inp = row.querySelector('input');
      inp.focus(); inp.select();
      row.querySelector('.dl-rename-confirm').addEventListener('click', function() {
        row.remove();
        saveToLocalLibrary(constructId, btn, inp.value.trim() || data.suggested_name, true);
      });
      inp.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
          row.remove();
          saveToLocalLibrary(constructId, btn, inp.value.trim() || data.suggested_name, true);
        }
        if (e.key === 'Escape') row.remove();
      });
      row.querySelector('.dl-rename-cancel').addEventListener('click', function() { row.remove(); });
    } else {
      btn.textContent = 'Save failed: ' + (data.error || 'unknown error');
      btn.disabled = false;
    }
  } catch(e) {
    btn.textContent = 'Save failed';
    btn.disabled = false;
  }
}

async function _removeSelected() {
  if (!_constructsTable) return;
  const rows = _constructsTable.getSelectedRows();
  if (!rows.length) return;
  const n = rows.length;
  if (!confirm('Remove ' + n + ' item' + (n > 1 ? 's' : '') + ' from the library?\nSource files on disk are not deleted.')) return;
  for (const row of rows) {
    const id = row.getData().id;
    try {
      await fetch('/api/db/constructs/' + id, {method: 'DELETE'});
      row.delete();
    } catch(e) { console.warn('Failed to delete', id, e); }
  }
}

// ── Import modal ─────────────────────────────────────────────────────────────

let _importItems = [];

async function importUserLibrary() {
  const modal = document.getElementById('import-modal');
  if (!modal) return;
  modal.style.display = 'flex';
  const tbody = document.getElementById('import-preview-body');
  tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:var(--text-secondary);font-family:Inter,sans-serif;font-size:12px">Loading…</td></tr>';
  _updateImportCount();
  try {
    const r = await fetch('/api/db/user-library-preview');
    if (!r.ok) { tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:#E86235">Failed to load library</td></tr>'; return; }
    _importItems = await r.json();
    _renderImportTable();
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:#E86235">' + escapeHtml(String(e)) + '</td></tr>';
  }
}

function _renderImportTable() {
  const tbody = document.getElementById('import-preview-body');
  if (!_importItems.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:var(--text-secondary);font-family:Inter,sans-serif;font-size:12px">No items found in library directory.</td></tr>';
    return;
  }
  const typeBadge = {'backbone':'#D97757','insert':'#24B283','annotation':'#8B5CF6'};
  tbody.innerHTML = _importItems.map(function(item, i) {
    const already = item.already_imported;
    const badge = typeBadge[item.part_type] || '#888';
    const sizeStr = item.size_bp ? item.size_bp.toLocaleString() + ' bp' : '—';
    const catEn = [item.category, item.assembly_enzyme].filter(Boolean).join(' / ') || '—';
    const res = item.bacterial_resistance || '—';
    return '<tr style="border-bottom:1px solid var(--sand-200);' + (already ? 'opacity:0.55' : '') + '">' +
      '<td style="padding:7px 10px;text-align:center">' +
        '<input type="checkbox" class="import-item-check" data-idx="' + i + '"' +
        (already ? ' disabled' : '') + ' onchange="_updateImportCount()"></td>' +
      '<td style="padding:7px 10px;font-family:Inter,sans-serif;font-size:12px">' + escapeHtml(item.name) + '</td>' +
      '<td style="padding:7px 10px"><span style="background:' + badge + ';color:#fff;padding:1px 7px;border-radius:10px;font-size:10px;font-family:Inter,sans-serif">' + escapeHtml(item.part_type) + '</span></td>' +
      '<td style="padding:7px 10px;font-size:12px;font-family:Inter,sans-serif;color:var(--text-secondary)">' + escapeHtml(sizeStr) + '</td>' +
      '<td style="padding:7px 10px;font-size:12px;font-family:Inter,sans-serif">' + escapeHtml(catEn) + '</td>' +
      '<td style="padding:7px 10px;font-size:12px;font-family:Inter,sans-serif">' + escapeHtml(res) + '</td>' +
      '<td style="padding:7px 10px;font-size:11px;color:var(--text-secondary);font-family:Inter,sans-serif">' +
        (already ? '&#10003; already imported' : '') + '</td>' +
      '</tr>';
  }).join('');
  _updateImportCount();
}

function _updateImportCount() {
  const checks = document.querySelectorAll('.import-item-check:checked');
  const lbl = document.getElementById('import-selected-count');
  if (lbl) lbl.textContent = checks.length + ' selected';
}

function onCheckAllChange(master) {
  document.querySelectorAll('.import-item-check:not(:disabled)').forEach(function(cb) {
    cb.checked = master.checked;
  });
  _updateImportCount();
}

function toggleImportSelectAll() {
  const checks = document.querySelectorAll('.import-item-check:not(:disabled)');
  const allChecked = Array.from(checks).every(function(c) { return c.checked; });
  checks.forEach(function(c) { c.checked = !allChecked; });
  const master = document.getElementById('import-check-all');
  if (master) master.checked = !allChecked;
  _updateImportCount();
}

function closeImportModal() {
  const modal = document.getElementById('import-modal');
  if (modal) modal.style.display = 'none';
}

async function confirmImport() {
  const checks = document.querySelectorAll('.import-item-check:checked');
  if (!checks.length) { alert('No items selected.'); return; }
  const selected = Array.from(checks).map(function(cb) {
    return _importItems[parseInt(cb.dataset.idx)].local_path;
  }).filter(Boolean);
  const btn = document.getElementById('import-confirm-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Importing…'; }
  try {
    const r = await fetch('/api/db/import-user-library', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({local_paths: selected}),
    });
    const data = await r.json();
    if (data.error) {
      alert('Import failed: ' + data.error);
    } else {
      closeImportModal();
      refreshLibraryData();
    }
  } catch(e) {
    alert('Import failed: ' + e);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Import Selected'; }
  }
}

// ── Cytoscape ────────────────────────────────────────────────────────────────

let _cy = null;

function _buildTooltipHtml(node) {
  const d = node.data();
  const type = d.nodeType;
  let rows = '';
  const row = (label, val) => val
    ? '<tr><td style="color:#87867F;padding-right:10px;white-space:nowrap">' + label + '</td>' +
      '<td style="font-weight:500">' + val + '</td></tr>'
    : '';

  if (type === 'construct') {
    rows += row('Accession', '<code style="color:#D97757;font-weight:600">' + escapeHtml(d.accession) + '</code>');
    rows += row('Name', escapeHtml(d.label));
    if (d.user_name) rows += row('Label', escapeHtml(d.user_name));
    if (d.size_bp) rows += row('Size', d.size_bp.toLocaleString() + ' bp');
    if (d.backbone_name) rows += row('Backbone', escapeHtml(d.backbone_name));
    if (d.insert_names && d.insert_names.length)
      rows += row('Inserts', escapeHtml(d.insert_names.join(', ')));
    if (d.created_at) rows += row('Created', escapeHtml(d.created_at));
    const originLabels = {designer:'Designed', user_library:'Your Library', annotation:'Annotation'};
    if (d.origin && d.origin !== 'designer') rows += row('Source', escapeHtml(originLabels[d.origin] || d.origin));
    if (d.sequence_verified)
      rows += row('', '<span style="color:#24B283;font-weight:600">&#10003; Sequence verified</span>');
  } else if (type === 'backbone') {
    rows += row('Backbone', '<strong>' + escapeHtml(d.label) + '</strong>');
    if (d.source_system) rows += row('Source', escapeHtml(d.source_system));
    if (d.addgene_id) rows += row('Addgene', '#' + escapeHtml(d.addgene_id));
    if (d.source_doi) rows += row('DOI', '<a href="https://doi.org/' + escapeHtml(d.source_doi) + '" target="_blank" style="color:#D97757">' + escapeHtml(d.source_doi) + '</a>');
    if (d.usage_count > 1) rows += row('Used in', d.usage_count + ' constructs');
  } else if (type === 'insert') {
    rows += row('Insert', '<strong>' + escapeHtml(d.label) + '</strong>');
    if (d.source_system) rows += row('Source', escapeHtml(d.source_system));
    if (d.genbank_accession) rows += row('Accession', escapeHtml(d.genbank_accession));
    if (d.usage_count > 1) rows += row('Used in', d.usage_count + ' constructs');
  }
  return '<table style="border-collapse:collapse;font-size:12px">' + rows + '</table>';
}

function _showCyTooltip(evt) {
  const tip = document.getElementById('cy-tooltip');
  if (!tip) return;
  tip.innerHTML = _buildTooltipHtml(evt.target);
  tip.style.display = 'block';
  _positionCyTooltip(evt);
}

function _positionCyTooltip(evt) {
  const tip = document.getElementById('cy-tooltip');
  if (!tip || tip.style.display === 'none') return;
  const container = document.getElementById('constructs-graph');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  const pos = evt.renderedPosition || evt.target.renderedPosition();
  let x = pos.x + 14;
  let y = pos.y + 14;
  // Clamp to container bounds
  const tw = tip.offsetWidth || 220;
  const th = tip.offsetHeight || 100;
  if (x + tw > rect.width - 10) x = pos.x - tw - 10;
  if (y + th > rect.height - 10) y = pos.y - th - 10;
  tip.style.left = Math.max(4, x) + 'px';
  tip.style.top = Math.max(4, y) + 'px';
}

function _hideCyTooltip() {
  const tip = document.getElementById('cy-tooltip');
  if (tip) tip.style.display = 'none';
}

function _initCytoscape() {
  const container = document.getElementById('constructs-graph');
  if (!container || typeof cytoscape === 'undefined') return;
  _cy = cytoscape({
    container: container,
    elements: [],
    style: [
      {selector: 'node[nodeType="construct"]', style: {
        'background-color': '#3B82F6',
        'label': 'data(label)',
        'font-size': '10px',
        'color': '#3D3D3A',
        'text-wrap': 'wrap',
        'text-max-width': '70px',
        'width': '65px', 'height': '65px',
        'border-width': '2px', 'border-color': '#E8E6DC',
        'font-family': 'Inter, sans-serif',
        'cursor': 'pointer',
      }},
      {selector: 'node[nodeType="construct"][origin="user_library"]', style: {
        'background-color': '#10B981',
      }},
      {selector: 'node[nodeType="construct"][origin="annotation"]', style: {
        'background-color': '#8B5CF6',
      }},
      {selector: 'node[nodeType="backbone"]', style: {
        'background-color': '#D97757',
        'shape': 'diamond',
        'label': 'data(label)',
        'font-size': '10px',
        'color': '#3D3D3A',
        'text-wrap': 'wrap',
        'text-max-width': '70px',
        'width': '64px', 'height': '64px',
        'font-family': 'Inter, sans-serif',
        'cursor': 'pointer',
      }},
      {selector: 'node[nodeType="insert"]', style: {
        'background-color': '#5C5B56',
        'shape': 'round-rectangle',
        'label': 'data(label)',
        'font-size': '10px',
        'color': '#3D3D3A',
        'text-wrap': 'wrap',
        'text-max-width': '65px',
        'width': '65px', 'height': '40px',
        'font-family': 'Inter, sans-serif',
        'cursor': 'pointer',
      }},
      {selector: 'node:selected', style: {
        'border-color': '#E86235', 'border-width': '3px',
      }},
      {selector: 'node:active', style: {
        'overlay-opacity': 0.1,
      }},
      {selector: 'edge', style: {
        'width': 1.5,
        'line-color': '#ADAAA0',
        'curve-style': 'bezier',
        'opacity': 0.7,
      }},
      {selector: 'edge:selected', style: {
        'line-color': '#D97757', 'opacity': 1, 'width': 2.5,
      }},
    ],
    layout: {name: 'klay', klay: {spacing: 55, direction: 'RIGHT'}},
  });

  _cy.on('tap', 'node[nodeType="construct"]', function(evt) {
    _hideCyTooltip();
    const rawId = evt.target.id().replace('c_', '');
    const numId = parseInt(rawId, 10);
    if (_constructsTable && !isNaN(numId)) {
      showLibraryTab('table');
      const r = _constructsTable.getRow(numId);
      if (r) { r.select(); r.scrollTo(); }
    }
  });

  _cy.on('mouseover', 'node', function(evt) { _showCyTooltip(evt); });
  _cy.on('mousemove', 'node', function(evt) { _positionCyTooltip(evt); });
  _cy.on('mouseout', 'node', function() { _hideCyTooltip(); });
  _cy.on('tap', 'node[nodeType="backbone"], node[nodeType="insert"]', function() {
    _hideCyTooltip();
  });
  _cy.on('tap', function(evt) {
    if (evt.target === _cy) _hideCyTooltip();
  });
}

async function _loadGraphData() {
  if (!_cy) return;
  try {
    const r = await fetch('/api/db/graph');
    const data = await r.json();
    _cy.elements().remove();
    _cy.add(data.nodes || []);
    _cy.add(data.edges || []);
    if (data.nodes && data.nodes.length) {
      _cy.layout({name: 'klay', klay: {spacing: 50, direction: 'RIGHT'}}).run();
    }
  } catch(e) {
    console.warn('Graph load failed', e);
  }
}

function downloadBatchFile(jobId, rowIdx, expIdx, filename) {
  fetch('/api/batch/' + jobId + '/download/' + rowIdx + '/' + expIdx)
  .then(function(r) { return r.blob(); })
  .then(function(blob) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  })
  .catch(function(e) { alert('Download failed: ' + e); });
}

function saveBatchConstruct(jobId, rowIdx, expIdx, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.innerHTML = _SAVE_SVG + ' Saving…';
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/save-construct/' + expIdx, {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.id) {
      btn.innerHTML = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg> Saved';
      btn.style.opacity = '0.7';
    } else {
      btn.innerHTML = _SAVE_SVG + ' Save failed';
      btn.disabled = false;
    }
  })
  .catch(function() {
    btn.innerHTML = _SAVE_SVG + ' Save failed';
    btn.disabled = false;
  });
}

function saveBatchToLocal(jobId, rowIdx, expIdx, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  var origHtml = btn.innerHTML;
  btn.textContent = 'Saving…';
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/save-local/' + expIdx, {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.saved_to) {
      btn.textContent = '✓ Saved to library';
    } else {
      btn.innerHTML = origHtml;
      btn.disabled = false;
      alert('Save failed: ' + (data.error || 'unknown error'));
    }
  })
  .catch(function() {
    btn.innerHTML = origHtml;
    btn.disabled = false;
    alert('Save failed');
  });
}

function saveAllBatchConstructs(jobId, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.innerHTML = _SAVE_SVG + ' Saving…';
  fetch('/api/batch/' + jobId + '/save-all-constructs', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.saved !== undefined) {
      btn.innerHTML = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg> Saved ' + data.saved + ' construct' + (data.saved === 1 ? '' : 's');
      btn.style.opacity = '0.7';
    } else {
      btn.innerHTML = _SAVE_SVG + ' Save failed';
      btn.disabled = false;
    }
  })
  .catch(function() {
    btn.innerHTML = _SAVE_SVG + ' Save failed';
    btn.disabled = false;
  });
}

function saveAllBatchToLocal(jobId, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.textContent = 'Saving…';
  fetch('/api/batch/' + jobId + '/save-all-local', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.saved !== undefined) {
      btn.textContent = '✓ Saved ' + data.saved + ' to library';
    } else {
      btn.textContent = 'Save failed';
      btn.disabled = false;
    }
  })
  .catch(function() {
    btn.textContent = 'Save failed';
    btn.disabled = false;
  });
}
</script>
</body>
</html>
"""


# ── Batch job runner ────────────────────────────────────────────────────

def _run_batch_agent(prompt: str, model: str, append_log, exports: list, *,
                     history: list,
                     row_name: Optional[str] = None,
                     pause_event: Optional[threading.Event] = None) -> None:
    """Shared agent runner for batch rows. Mutates ``history`` in place so
    follow-up messages (``_continue_batch_row``) replay the same context."""
    tracker = ReferenceTracker()
    set_tracker(tracker)
    clear_last_plot_json()
    history.append({"role": "user", "content": prompt})

    try:
        for _ in range(15):
            # Block here if this row has been paused
            if pause_event:
                pause_event.wait()
            response = _client().messages.create(
                model=model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=history,
            )
            tool_results: list[dict] = []
            filtered_content: list[dict] = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    if block.text.strip():
                        append_log({"type": "text", "content": block.text})
                    filtered_content.append({"type": "text", "text": block.text})
                elif btype == "tool_use":
                    result = _dispatch_tool(block.name, block.input)
                    result_preview = result[:600] + ("\u2026" if len(result) > 600 else "")
                    append_log({
                        "type": "tool",
                        "name": block.name,
                        "input": block.input,
                        "result": result_preview,
                    })
                    if block.name == "export_construct":
                        fmt = block.input.get("output_format", "genbank")
                        ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
                        plot = get_last_plot_json()
                        exports.append({
                            "filename": (row_name or "construct") + ext,
                            "content": result,
                            "plot_json": json.loads(plot) if plot else None,
                        })
                        clear_last_plot_json()
                    filtered_content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            history.append({"role": "assistant", "content": filtered_content})
            if tool_results:
                history.append({"role": "user", "content": tool_results})
            if response.stop_reason == "end_turn" or not tool_results:
                break
    finally:
        set_tracker(None)


def _run_batch_row(job_id: str, row_idx: int, row: dict, model: str) -> None:
    """Worker for a single CSV row — runs the agent and stores exports + log in _batch_jobs."""
    job = _batch_jobs.get(job_id)
    if not job:
        return

    row_state = job["rows"][row_idx]
    description = row.get("description", "").strip()
    output_format = (row.get("output_format") or "genbank").strip().lower()

    if output_format == "both":
        prompt = description + "\nPlease export the final construct in both GenBank and FASTA formats."
    elif output_format == "fasta":
        prompt = description + "\nPlease export the final construct in FASTA format."
    else:
        prompt = description + "\nPlease export the final construct in GenBank format."

    row_state["status"] = "running"
    row_state["paused"] = False
    row_state["log"] = []
    name = row.get("name", "").strip() or f"plasmid_{row_idx + 1:03d}"
    pause_event = _get_pause_event(job_id, row_idx)

    try:
        exports: list[dict] = []
        history: list[dict] = []
        _run_batch_agent(
            prompt, model,
            append_log=row_state["log"].append,
            exports=exports,
            history=history,
            row_name=name,
            pause_event=pause_event,
        )
        row_state["exports"] = exports
        row_state["history"] = history
        row_state["status"] = "done" if exports else "no_export"
        row_state["paused"] = False
    except Exception as e:
        row_state["status"] = "error"
        row_state["error"] = str(e)
        row_state["log"].append({"type": "error", "content": str(e)})
    finally:
        _save_batch_jobs()


def _continue_batch_row(job_id: str, row_idx: int, user_message: str) -> None:
    """Continue a finished batch row with a follow-up user message."""
    job = _batch_jobs.get(job_id)
    if not job:
        return
    row_state = job["rows"][row_idx]
    model = job["model"]

    row_state["status"] = "running"
    row_state["paused"] = False
    row_state["log"].append({"type": "user", "content": user_message})
    name = row_state.get("name", "").strip() or f"plasmid_{row_idx + 1:03d}"
    pause_event = _get_pause_event(job_id, row_idx)
    pause_event.set()  # ensure unpaused for follow-up

    try:
        _run_batch_agent(
            user_message, model,
            append_log=row_state["log"].append,
            exports=row_state["exports"],
            history=row_state.setdefault("history", []),
            row_name=name,
            pause_event=pause_event,
        )
        row_state["status"] = "done" if row_state["exports"] else "no_export"
        row_state["paused"] = False
    except Exception as e:
        row_state["status"] = "error"
        row_state["error"] = str(e)
        row_state["log"].append({"type": "error", "content": str(e)})
    finally:
        _save_batch_jobs()


def start_batch_job(rows: list, model: str) -> str:
    """Create a batch job, launch a background thread, return job_id."""
    job_id = str(uuid.uuid4())
    job: dict = {
        "status": "running",
        "model": model,
        "rows": [
            {
                "description": r.get("description", ""),
                "name": r.get("name", ""),
                "output_format": r.get("output_format", "genbank"),
                "status": "pending",
                "paused": False,
                "exports": [],
                "error": None,
            }
            for r in rows
        ],
    }
    _batch_jobs[job_id] = job

    # Run rows sequentially in one daemon thread to avoid hammering the API
    def worker():
        for idx, row in enumerate(rows):
            _run_batch_row(job_id, idx, row, model)
        job["status"] = "done"
        _save_batch_jobs()

    threading.Thread(target=worker, daemon=True).start()
    return job_id


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
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/health":
            self._send_json({"status": "ok"})

        elif path == "/api/sessions":
            self._send_json(list_sessions())

        elif path.startswith("/api/sessions/") and path.endswith("/messages"):
            session_id = path.split("/")[3]
            session = get_session(session_id)
            if session:
                if session.get("batch_job_id"):
                    self._send_json([{
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
            import re as _re3
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
            safe_name = _re3.sub(r'[^\w\-. ]', '_', Path(filename).stem).strip().replace(' ', '_')
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
            import re as _re4
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
                    safe_name = _re4.sub(r'[^\w\-. ]', '_', Path(filename).stem).strip().replace(' ', '_')
                    out_path = constructs_dir / f"{safe_name}.gb"
                    out_path.write_text(content)
                    saved += 1
            self._send_json({"saved": saved})

        elif path == "/api/reset":
            # Legacy endpoint — clear all sessions
            _sessions.clear()
            _save_sessions()
            self._send_json({"status": "ok"})

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
            import re as _re_local
            constructs_dir = Path(user_lib_dir).expanduser() / "constructs"
            constructs_dir.mkdir(exist_ok=True)
            safe_name = _re_local.sub(r'[^\w\-. ]', '_', Path(filename).stem).strip().replace(' ', '_')
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
            import re as _re_ll
            constructs_dir = Path(user_lib_dir).expanduser() / "designed_constructs"
            constructs_dir.mkdir(exist_ok=True)
            safe_name = _re_ll.sub(r'[^\w\-. ]', '_', name).strip().replace(' ', '_')
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
            import re as _re2
            m2 = _re2.match(r"^/api/db/constructs/(\d+)/save-to-library$", path)
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
            safe_name = _re2.sub(r'[^\w\-. ]', '_', name).strip().replace(' ', '_')
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

        import re as _re
        m = _re.match(r"^/api/db/constructs/(\d+)$", path)
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
            import re as _re_del
            m = _re_del.match(r"^/api/db/constructs/(\d+)$", path)
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
