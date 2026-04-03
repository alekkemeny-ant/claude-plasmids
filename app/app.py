#!/usr/bin/env python3
"""
Plasmid Design Agent — Web UI

A chat interface for the Claude-powered plasmid design agent.
Uses the Claude Agent SDK with the same MCP tool server as the CLI agent
(app/agent.py) and the eval harness (evals/run_agent_evals.py).

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

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    PermissionResultAllow,
)
from claude_agent_sdk.types import StreamEvent

from src.tools import (
    build_mcp_servers,
    set_tracker,
    get_last_plot_json,
    clear_last_plot_json,
)
from src.references import ReferenceTracker
from src.library import load_backbones, load_inserts

logger = logging.getLogger(__name__)

LIBRARY_PATH = PROJECT_ROOT / "library"

# ── Load system prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"  # lives in app/
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""


# ── MCP tool name helpers ───────────────────────────────────────────────
# SDK MCP tools surface as "mcp__plasmid-library__<name>". The frontend
# expects bare tool names in SSE events for display.

def _strip_mcp_prefix(name: str) -> str:
    if name.startswith("mcp__"):
        return name.rsplit("__", 1)[-1]
    return name


def _tool_result_text(content) -> str:
    """Extract a flat string from a ToolResultBlock.content (str or list[dict])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return "" if content is None else str(content)


async def _auto_approve(tool_name, tool_input, context):
    """Auto-approve all tool calls (in-process MCP tools are safe)."""
    return PermissionResultAllow()


# ── Warm-client pool ────────────────────────────────────────────────────
# The Agent SDK spawns a `claude` CLI subprocess per ClaudeSDKClient. That
# subprocess loads settings, hooks, and MCP servers before the API call,
# adding ~2-3s to TTFT on every turn. We pre-spawn a client per session on
# a background event loop and reuse it across turns so the user only pays
# the spawn cost once (on textarea focus / page load via /api/warmup).

import atexit
import concurrent.futures

_bg_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True, name="sdk-loop")
_bg_thread.start()

_warm_clients: dict[str, ClaudeSDKClient] = {}
_warming: dict[str, concurrent.futures.Future] = {}
_warm_lock = threading.Lock()


def _run_on_loop(coro) -> concurrent.futures.Future:
    """Schedule a coroutine on the background loop and return its Future."""
    return asyncio.run_coroutine_threadsafe(coro, _bg_loop)


async def _do_warmup(session_id: str, session: dict, model: str) -> None:
    options = _build_agent_options(session, model)
    client = ClaudeSDKClient(options=options)
    await client.__aenter__()
    with _warm_lock:
        _warm_clients[session_id] = client
        _warming.pop(session_id, None)


def warmup_session(session_id: str, model: str | None = None) -> None:
    """Pre-spawn the SDK subprocess for a session. Idempotent and non-blocking."""
    with _warm_lock:
        if session_id in _warm_clients or session_id in _warming:
            return
        session = _sessions.get(session_id)
        if session is None:
            return
        fut = _run_on_loop(_do_warmup(session_id, session, model or MODEL))
        _warming[session_id] = fut


def _take_warm_client(session_id: str) -> ClaudeSDKClient | None:
    """Pop a warm client, blocking briefly if warmup is in-flight."""
    with _warm_lock:
        client = _warm_clients.pop(session_id, None)
        fut = _warming.pop(session_id, None)
    if client is not None:
        return client
    if fut is not None:
        try:
            fut.result()  # warmup was in-flight; wait for it
        except Exception:
            return None
        with _warm_lock:
            return _warm_clients.pop(session_id, None)
    return None


def close_warm_client(session_id: str) -> None:
    """Tear down a session's warm subprocess (on session delete / shutdown)."""
    with _warm_lock:
        client = _warm_clients.pop(session_id, None)
        _warming.pop(session_id, None)
    if client is not None:
        _run_on_loop(client.__aexit__(None, None, None))


def _shutdown_warm_clients() -> None:
    for sid in list(_warm_clients):
        close_warm_client(sid)


atexit.register(_shutdown_warm_clients)


# ── Session management ──────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_cancelled_sessions: set[str] = set()
_sessions_lock = threading.Lock()

# ── Batch job state ─────────────────────────────────────────────────────
_batch_jobs: dict[str, dict] = {}
SESSIONS_FILE = Path(__file__).parent / ".sessions.json"

MODEL = "claude-opus-4-6"

# Context window sizes by model (tokens)
CONTEXT_WINDOW = {
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
            # Skip thinking blocks — they cause Error 400 on replay
            if isinstance(d, dict) and d.get("type") == "thinking":
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
                    "sdk_session_id": data.get("sdk_session_id"),
                    # Phase-2 troubleshooting/project-memory fields — default
                    # to empty for sessions created before these were added.
                    "project_name": data.get("project_name"),
                    "experimental_outcomes": data.get("experimental_outcomes", []),
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


def create_session() -> str:
    """Create a new conversation session."""
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        # The Agent SDK manages conversation history via its own session
        # store; we keep "history" for backward compat with old .sessions.json.
        "history": [],
        "sdk_session_id": None,
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
        close_warm_client(session_id)
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
        })
    return result


def cancel_session(session_id: str):
    _cancelled_sessions.add(session_id)


# ── Agent loop ──────────────────────────────────────────────────────────


def _build_agent_options(session: dict, model: str) -> ClaudeAgentOptions:
    """Build SDK options for a session turn.

    The system prompt is built per-turn so that newly logged
    experimental_outcomes are visible. The Agent SDK persists
    conversation history via resume=, so we only manage
    display_messages for UI rendering.
    """
    return ClaudeAgentOptions(
        system_prompt=_build_system_prompt(session),
        mcp_servers=build_mcp_servers(),
        permission_mode="acceptEdits",
        model=model,
        max_turns=15,
        cwd=str(PROJECT_ROOT),
        can_use_tool=_auto_approve,
        include_partial_messages=True,
        resume=session.get("sdk_session_id"),
    )


def run_agent_turn_streaming(user_message: str, session_id: str, write_event, model: str = MODEL):
    """Run one agent turn with streaming, scoped to a session.

    Bridges the async Agent SDK to the sync HTTP handler via asyncio.run().
    Translates SDK message types to the SSE event contract the frontend expects.
    """
    _cancelled_sessions.discard(session_id)

    session = get_session(session_id)
    if not session:
        write_event({"type": "error", "content": "Session not found"})
        return

    session["display_messages"].append({"role": "user", "content": user_message})
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

    tracker = ReferenceTracker()
    set_tracker(tracker)
    clear_last_plot_json()

    assistant_text = ""
    assistant_blocks: list[dict] = []
    export_called = False
    # Map tool_use_id -> {name, input} so we can pair tool results with calls.
    pending_tools: dict[str, dict] = {}

    # Acquire warm client on the HTTP thread (safe to block here; the bg
    # loop must not block on _warming futures or it deadlocks).
    warm_client = _take_warm_client(session_id)

    async def _run():
        nonlocal assistant_text, export_called
        client = warm_client
        if client is None:
            options = _build_agent_options(session, model)
            client = ClaudeSDKClient(options=options)
            await client.__aenter__()
        try:
            await client.query(user_message)
            async for message in client.receive_response():
                if is_cancelled() or disconnected:
                    try:
                        await client.interrupt()
                    except Exception:
                        pass
                    break

                if isinstance(message, StreamEvent):
                    ev = message.event
                    et = ev.get("type")
                    if et == "content_block_start":
                        block = ev.get("content_block", {})
                        bt = block.get("type")
                        if bt == "thinking":
                            safe_write({"type": "thinking_start"})
                        elif bt == "text":
                            safe_write({"type": "text_start"})
                        elif bt == "tool_use":
                            safe_write({
                                "type": "tool_use_start",
                                "tool": _strip_mcp_prefix(block.get("name", "")),
                            })
                    elif et == "content_block_delta":
                        delta = ev.get("delta", {})
                        dt = delta.get("type")
                        if dt == "thinking_delta":
                            safe_write({"type": "thinking_delta", "content": delta.get("thinking", "")})
                        elif dt == "text_delta":
                            text = delta.get("text", "")
                            assistant_text += text
                            safe_write({"type": "text_delta", "content": text})
                    elif et == "content_block_stop":
                        # We close text/thinking blocks here. Tool blocks are
                        # closed when the result arrives (UserMessage below)
                        # so the UI shows call+result as one unit.
                        idx = ev.get("index")
                        # We don't track block-type-by-index here; AssistantMessage
                        # below records the complete blocks for display history.
                        pass

                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            assistant_blocks.append({"type": "text", "content": block.text})
                            safe_write({"type": "text_end"})
                        elif isinstance(block, ThinkingBlock):
                            assistant_blocks.append({"type": "thinking", "content": block.thinking})
                            safe_write({"type": "thinking_end"})
                        elif isinstance(block, ToolUseBlock):
                            pending_tools[block.id] = {
                                "name": _strip_mcp_prefix(block.name),
                                "input": block.input,
                            }

                elif isinstance(message, UserMessage):
                    if not isinstance(message.content, list):
                        continue
                    for block in message.content:
                        if not isinstance(block, ToolResultBlock):
                            continue
                        call = pending_tools.pop(block.tool_use_id, {"name": "?", "input": {}})
                        tool_name = call["name"]
                        tool_input = call["input"]
                        result_str = _tool_result_text(block.content)
                        if tool_name == "export_construct":
                            export_called = True
                        # Intercept outcome-log marker and persist to session
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
                        display_result = result_str[:2000] + "..." if len(result_str) > 2000 else result_str
                        event_data = {
                            "type": "tool_result",
                            "tool": tool_name,
                            "input": tool_input,
                            "content": display_result,
                        }
                        if tool_name == "export_construct":
                            event_data["download_content"] = result_str
                            fmt = tool_input.get("output_format", "raw")
                            cname = tool_input.get("construct_name", "construct")
                            ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
                            event_data["download_filename"] = cname + ext
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

                elif isinstance(message, SystemMessage):
                    if message.subtype == "init":
                        sid = message.data.get("session_id")
                        if sid:
                            session["sdk_session_id"] = sid

                elif isinstance(message, ResultMessage):
                    if message.is_error and not assistant_text:
                        safe_write({"type": "error", "content": message.result or "Agent error"})
                    # Emit token usage for the context-window indicator (PR #15).
                    usage = getattr(message, "usage", None) or {}
                    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else getattr(usage, "input_tokens", None)
                    if input_tokens is not None:
                        safe_write({
                            "type": "token_usage",
                            "input_tokens": input_tokens,
                            "context_window": CONTEXT_WINDOW.get(model, 1_000_000),
                        })
                    break
        finally:
            # Keep the subprocess warm for the next turn unless the
            # request was aborted (cancel/disconnect → state is unknown,
            # close it to avoid leaking a half-consumed stream).
            if is_cancelled() or disconnected:
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass
            else:
                with _warm_lock:
                    _warm_clients[session_id] = client

    try:
        _run_on_loop(_run()).result()
    finally:
        set_tracker(None)

    # Append formatted references only when a sequence file was exported this turn
    if export_called and not (is_cancelled() or disconnected):
        refs_text = tracker.format_references()
        if refs_text:
            ref_block = f"\n\n{refs_text}"
            assistant_text += ref_block
            assistant_blocks.append({"type": "text", "content": ref_block})
            safe_write({"type": "text_start"})
            safe_write({"type": "text_delta", "content": ref_block})
            safe_write({"type": "text_end"})

    # Save assistant text and structured blocks to display messages
    if assistant_text or assistant_blocks:
        session["display_messages"].append({
            "role": "assistant",
            "content": assistant_text,
            "blocks": assistant_blocks,
        })
    elif is_cancelled() or disconnected:
        if session["display_messages"] and session["display_messages"][-1]["role"] == "user":
            session["display_messages"].pop()

    _save_sessions()

    if not disconnected:
        try:
            write_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            pass


# ── HTML UI ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plasmid Designer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.1.min.js"></script>
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
    text-transform: uppercase; letter-spacing: 0.05em;
    font-family: inherit; transition: background 0.15s;
  }
  .user-library-toggle:hover { background: var(--sand-100); color: var(--sand-700); }
  .user-library-toggle .chevron { transition: transform 0.2s; font-style: normal; }
  .user-library-toggle.open .chevron { transform: rotate(180deg); }
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

  /* ── Error ── */
  .error-banner {
    background: var(--brand-orange-100); border: 1px solid rgba(232,98,53,0.2);
    color: var(--brand-orange); border-radius: 8px; padding: 12px 16px;
    font-size: 13px; margin-bottom: 24px;
  }

  /* ── Drop overlay (shown when a CSV is dragged over the chat area) ── */
  .drop-overlay {
    display: none; position: absolute; inset: 0; z-index: 50;
    background: rgba(217,119,87,0.06); border: 3px dashed var(--brand-fig);
    border-radius: 0; align-items: center; justify-content: flex-end;
    flex-direction: column; gap: 10px; pointer-events: none; padding-bottom: 144px;
  }
  .drop-overlay.active { display: flex; }
  .drop-overlay-label { font-size: 16px; font-weight: 600; color: var(--brand-fig); }
  .drop-overlay-sub { font-size: 13px; color: var(--brand-fig-hover); }

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
  <div>
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
      <button class="user-library-toggle" id="user-library-toggle" onclick="toggleUserLibrary()">
        <span>Your Library</span>
        <em class="chevron">&#8964;</em>
      </button>
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
    <!-- Drop overlay: shown when a CSV is dragged over the chat area -->
    <div class="drop-overlay" id="drop-overlay">
      <svg width="36" height="36" fill="none" stroke="var(--brand-fig)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <div class="drop-overlay-label">Drop CSV to batch design</div>
      <div class="drop-overlay-sub">Required column: description &nbsp;·&nbsp; Optional: name, output_format</div>
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
            Drag &amp; drop a CSV file here to batch design multiple plasmids at once.
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
            <option value="claude-opus-4-6">Opus 4.6</option>
            <option value="claude-sonnet-4-6">Sonnet 4.6</option>
            <option value="claude-haiku-4-5-20251001">Haiku 4.5</option>
          </select>
          <div class="token-indicator" id="token-indicator">
            <div class="token-bar-track"><div class="token-bar-fill" id="token-bar"></div></div>
            <span id="token-label"></span>
          </div>
        </div>
        <div class="input-buttons">
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
</div>

<script>
// ── State ──
let currentSessionId = sessionStorage.getItem('plasmid_session_id') || null;
let sessions = [];
let isStreaming = false;
let abortController = null;

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
    sessionStorage.setItem('plasmid_session_id', id);
  } else {
    sessionStorage.removeItem('plasmid_session_id');
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

// Pre-spawn the agent subprocess so the first message doesn't pay the
// ~2-3s SDK startup cost. Fires on page load and on textarea focus
// (covers the "switch session, then type" path). Idempotent server-side.
async function warmup() {
  try {
    const r = await fetch('/api/warmup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: currentSessionId, model: modelSelect.value}),
    });
    const j = await r.json();
    if (j.session_id && !currentSessionId) {
      currentSessionId = j.session_id;
      sessionStorage.setItem('plasmid_session_id', currentSessionId);
    }
  } catch (e) { /* warmup is best-effort */ }
}
inputEl.addEventListener('focus', warmup);

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
    return '<div class="session-item' + active + '" onclick="selectSession(\'' + s.session_id + '\')">' +
      '<span class="session-name">' + name + '</span>' +
      '<button class="delete-btn" onclick="event.stopPropagation(); deleteSessionById(\'' + s.session_id + '\')" title="Delete">' +
        '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
          '<path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>' +
        '</svg>' +
      '</button>' +
    '</div>';
  }).join('');
}

async function selectSession(sessionId) {
  // If streaming, stop the current generation before switching
  if (isStreaming) {
    stopGeneration();
    // Reset streaming UI state
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
  }

  saveSessionId(sessionId);
  renderSessions();

  try {
    const r = await fetch('/api/sessions/' + sessionId + '/messages');
    const msgs = await r.json();
    // Guard: if user switched to another session while fetch was in flight, discard
    if (currentSessionId !== sessionId) return;
    renderStoredMessages(msgs);
  } catch {
    // Don't clear messages on fetch failure (e.g., during server reload)
    // — leave the current display intact rather than showing empty state
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
      addDownloadButton(container, block.download_content, block.download_filename);
    }
  } else if (block.type === 'text') {
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(block.content || '') + '</div>';
    container.appendChild(div);
  }
}

function renderStoredMessages(msgs) {
  if (msgs.length === 0) {
    showWelcome();
    return;
  }
  hideWelcome();
  const inner = document.createElement('div');
  inner.className = 'messages-inner';
  msgs.forEach(function(m) {
    if (m.role === 'user') {
      const div = document.createElement('div');
      div.className = 'msg user';
      div.innerHTML = '<div class="msg-bubble-user">' + escapeHtml(m.content) + '</div>';
      inner.appendChild(div);
    } else if (m.blocks && m.blocks.length > 0) {
      m.blocks.forEach(function(block) { renderStoredBlock(block, inner); });
    } else {
      const div = document.createElement('div');
      div.className = 'msg assistant';
      div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(m.content || '') + '</div>';
      inner.appendChild(div);
    }
  });
  messagesEl.innerHTML = '';
  messagesEl.appendChild(inner);
  scrollToBottom();
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
    stopGeneration();
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
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
      : [entry.assembly_enzyme, entry.bacterial_resistance].filter(Boolean).join(' · ');
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
    if (!data.configured) return;
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
function inlineMarkdown(text) {
  let h = escapeHtml(text);
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
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
      let t = '<table><thead><tr>';
      headerCells.forEach(function(c) { t += '<th>' + inlineMarkdown(c) + '</th>'; });
      t += '</tr></thead><tbody>';
      bodyRows.forEach(function(row) {
        t += '<tr>';
        row.forEach(function(c) { t += '<td>' + inlineMarkdown(c) + '</td>'; });
        t += '</tr>';
      });
      t += '</tbody></table>';
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
      h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
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
    html = html.replace('%%CODEBLOCK' + idx + '%%', '<pre><code>' + escapeHtml(code) + '</code></pre>');
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

function endTextBlock() {
  if (currentTextDiv) {
    const cursor = currentTextDiv.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
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
  const dlId = 'dl-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant" style="margin-top:8px">' +
    '<button class="download-btn" id="' + dlId + '">' +
      '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
        '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>' +
      '</svg>' +
      ' Download ' + escapeHtml(filename) +
    '</button></div>';
  container.appendChild(div);
  document.getElementById(dlId).addEventListener('click', function() {
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
  // Surface download button in the main chat (not just inside the collapsed tool block)
  if (downloadContent && downloadFilename) {
    addDownloadButton(getInner(), downloadContent, downloadFilename);
  }
  currentToolId = null;
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
  userDiv.innerHTML = '<div class="msg-bubble-user">' + escapeHtml(text) + '</div>';
  inner.appendChild(userDiv);
  scrollToBottom();

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
          case 'thinking_start': startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': startTextBlock(); break;
          case 'text_delta': appendTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'token_usage': updateTokenIndicator(event.input_tokens, event.context_window); break;
          case 'error':
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
      startTextBlock();
      appendTextDelta('Connection error: ' + err.message);
      endTextBlock();
    }
  }

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
}

function stopGeneration() {
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

// ── Init ──
checkHealth();
setInterval(checkHealth, 5000);
loadSessions();
loadUserLibrary();
setInterval(loadSessions, 5000);
// Restore active session on page load
if (currentSessionId) {
  selectSession(currentSessionId);
}
inputEl.focus();

// ── Batch ──
let batchJobId = null;
let batchPollTimer = null;
const chatPanelEl = document.getElementById('chat-panel');
const dropOverlayEl = document.getElementById('drop-overlay');

// ── Drag & drop CSV onto the chat area ──
var dragCounter = 0;

function isCsvDrag(e) {
  var types = e.dataTransfer && e.dataTransfer.types;
  return types && (Array.from(types).indexOf('Files') !== -1);
}

chatPanelEl.addEventListener('dragenter', function(e) {
  if (!isCsvDrag(e)) return;
  e.preventDefault();
  dragCounter++;
  dropOverlayEl.classList.add('active');
});

chatPanelEl.addEventListener('dragleave', function(e) {
  if (!isCsvDrag(e)) return;
  dragCounter--;
  if (dragCounter <= 0) { dragCounter = 0; dropOverlayEl.classList.remove('active'); }
});

chatPanelEl.addEventListener('dragover', function(e) {
  if (!isCsvDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

chatPanelEl.addEventListener('drop', function(e) {
  e.preventDefault();
  dragCounter = 0;
  dropOverlayEl.classList.remove('active');
  var file = e.dataTransfer.files[0];
  if (!file) return;
  if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
    alert('Please drop a .csv file.');
    return;
  }
  var reader = new FileReader();
  reader.onload = function(ev) { uploadBatchCSV(ev.target.result, file.name); };
  reader.readAsText(file);
});

function onBatchFileChosen(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) { uploadBatchCSV(e.target.result, file.name); };
  reader.readAsText(file);
  input.value = '';
}

function uploadBatchCSV(csvText, filename) {
  var model = modelSelect.value;
  fetch('/api/batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({csv_content: csvText, model: model}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    batchJobId = data.job_id;
    initBatchCards(data.job_id, data.row_count, filename);
    if (batchPollTimer) clearInterval(batchPollTimer);
    batchPollTimer = setInterval(pollBatchStatus, 2000);
    pollBatchStatus();
  })
  .catch(function(e) { alert('Upload failed: ' + e); });
}

function initBatchCards(jobId, count, filename) {
  hideWelcome();
  var inner = getInner();
  // Label
  var label = document.createElement('div');
  label.className = 'msg assistant';
  label.id = 'batch-label-' + jobId;
  label.innerHTML = '<div class="msg-bubble-assistant" style="color:var(--sand-500);font-size:13px;">' +
    'Batch designing <strong>' + count + ' plasmid' + (count === 1 ? '' : 's') + '</strong> from <em>' + escapeHtml(filename) + '</em>. ' +
    'Click any row to expand and see what\u2019s happening, or send a follow-up once it finishes.' +
    '</div>';
  inner.appendChild(label);
  // Placeholder cards
  for (var i = 0; i < count; i++) {
    var card = document.createElement('div');
    card.className = 'msg assistant';
    card.id = 'batch-card-' + jobId + '-' + i;
    card.innerHTML = buildBatchCardHtml(jobId, i, {
      status: 'pending', description: '\u2026', exports: [], error: null, log: []
    }, false);
    inner.appendChild(card);
  }
  scrollToBottom();
}

function pollBatchStatus() {
  if (!batchJobId) return;
  fetch('/api/batch/' + batchJobId)
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) return;
    updateBatchCards(batchJobId, data.rows);
    var anyRunning = data.rows && data.rows.some(function(r) { return r.status === 'running' || r.status === 'pending'; });
    if (data.status === 'done' && !anyRunning) {
      clearInterval(batchPollTimer);
      batchPollTimer = null;
      // Add Download All button to label message
      var labelEl = document.getElementById('batch-label-' + batchJobId);
      if (labelEl && !labelEl.querySelector('.batch-dl-all-btn')) {
        var bubble = labelEl.querySelector('.msg-bubble-assistant');
        if (bubble) {
          var btn = document.createElement('button');
          btn.className = 'download-btn batch-dl-all-btn';
          btn.style.cssText = 'margin-top:8px;display:inline-flex;';
          btn.innerHTML = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download All (.zip)';
          btn.onclick = function() { downloadAllBatch(batchJobId); };
          bubble.appendChild(document.createElement('br'));
          bubble.appendChild(btn);
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
};
var STATUS_LABELS = {pending: 'Pending', running: 'Running\u2026', done: 'Done', no_export: 'No export produced', error: 'Error'};
var CHEV_SVG = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>';

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

function buildDownloadsHtml(jobId, idx, exports) {
  if (!exports || !exports.length) return '';
  var html = '<div class="batch-row-downloads">';
  exports.forEach(function(exp, eidx) {
    html += '<button class="download-btn" onclick="event.stopPropagation();downloadBatchFile(\'' + jobId + '\',' + idx + ',' + eidx + ',\'' + escapeHtml(exp.filename) + '\')">' +
      '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
      escapeHtml(exp.filename) + '</button>';
    if (exp.has_plot) {
      html += '<button class="download-btn" style="border-color:var(--brand-fig-30);color:var(--brand-fig);background:var(--brand-fig-10);" ' +
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
  var icon = STATUS_ICONS[row.status] || STATUS_ICONS.pending;
  var label = STATUS_LABELS[row.status] || row.status;
  var desc = escapeHtml((row.description || '').slice(0, 120) + ((row.description || '').length > 120 ? '\u2026' : ''));
  var downloads = buildDownloadsHtml(jobId, idx, row.exports);
  var logId = 'batch-log-' + jobId + '-' + idx;
  var chevId = 'batch-chev-' + jobId + '-' + idx;
  return '<div class="batch-card">' +
    '<div class="batch-row-header" onclick="toggleBatchCard(\'' + jobId + '\',' + idx + ')">' +
      '<div class="batch-row-status">' + icon + '</div>' +
      '<div class="batch-row-body">' +
        '<div class="batch-row-desc">' + desc + '</div>' +
        '<div class="batch-row-meta">' + (idx + 1) + ' \xb7 ' + label + '</div>' +
        downloads +
      '</div>' +
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
    if (!batchPollTimer) batchPollTimer = setInterval(pollBatchStatus, 2000);
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
</script>
</body>
</html>
"""


# ── Batch job runner ────────────────────────────────────────────────────

def _run_batch_agent(prompt: str, model: str, append_log, exports: list, *,
                     resume: Optional[str] = None,
                     row_name: Optional[str] = None) -> Optional[str]:
    """Shared SDK runner for batch rows. Returns the SDK session_id for resume."""
    tracker = ReferenceTracker()
    set_tracker(tracker)
    clear_last_plot_json()
    pending_tools: dict[str, dict] = {}
    sdk_session_id = resume

    async def _run():
        nonlocal sdk_session_id
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers=build_mcp_servers(),
            permission_mode="acceptEdits",
            model=model,
            max_turns=15,
            cwd=str(PROJECT_ROOT),
            can_use_tool=_auto_approve,
            resume=resume,
        )
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            append_log({"type": "text", "content": block.text})
                        elif isinstance(block, ToolUseBlock):
                            pending_tools[block.id] = {
                                "name": _strip_mcp_prefix(block.name),
                                "input": block.input,
                            }
                elif isinstance(message, UserMessage) and isinstance(message.content, list):
                    for block in message.content:
                        if not isinstance(block, ToolResultBlock):
                            continue
                        call = pending_tools.pop(block.tool_use_id, {"name": "?", "input": {}})
                        result = _tool_result_text(block.content)
                        result_preview = result[:600] + ("\u2026" if len(result) > 600 else "")
                        append_log({
                            "type": "tool",
                            "name": call["name"],
                            "input": call["input"],
                            "result": result_preview,
                        })
                        if call["name"] == "export_construct":
                            fmt = call["input"].get("output_format", "genbank")
                            ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
                            plot = get_last_plot_json()
                            exports.append({
                                "filename": (row_name or "construct") + ext,
                                "content": result,
                                "plot_json": json.loads(plot) if plot else None,
                            })
                            clear_last_plot_json()
                elif isinstance(message, SystemMessage) and message.subtype == "init":
                    sid = message.data.get("session_id")
                    if sid:
                        sdk_session_id = sid
                elif isinstance(message, ResultMessage):
                    break

    try:
        asyncio.run(_run())
    finally:
        set_tracker(None)
    return sdk_session_id


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
    row_state["log"] = []
    name = row.get("name", "").strip() or f"plasmid_{row_idx + 1:03d}"

    try:
        exports: list[dict] = []
        sdk_sid = _run_batch_agent(
            prompt, model,
            append_log=row_state["log"].append,
            exports=exports,
            row_name=name,
        )
        row_state["exports"] = exports
        row_state["sdk_session_id"] = sdk_sid
        row_state["status"] = "done" if exports else "no_export"
    except Exception as e:
        row_state["status"] = "error"
        row_state["error"] = str(e)
        row_state["log"].append({"type": "error", "content": str(e)})


def _continue_batch_row(job_id: str, row_idx: int, user_message: str) -> None:
    """Continue a finished batch row with a follow-up user message."""
    job = _batch_jobs.get(job_id)
    if not job:
        return
    row_state = job["rows"][row_idx]
    model = job["model"]

    row_state["status"] = "running"
    row_state["log"].append({"type": "user", "content": user_message})
    name = row_state.get("name", "").strip() or f"plasmid_{row_idx + 1:03d}"

    try:
        _run_batch_agent(
            user_message, model,
            append_log=row_state["log"].append,
            exports=row_state["exports"],
            resume=row_state.get("sdk_session_id"),
            row_name=name,
        )
        row_state["status"] = "done" if row_state["exports"] else "no_export"
    except Exception as e:
        row_state["status"] = "error"
        row_state["error"] = str(e)
        row_state["log"].append({"type": "error", "content": str(e)})


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

    threading.Thread(target=worker, daemon=True).start()
    return job_id


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
                self._send_json(session["display_messages"])
            else:
                self._send_json([], 404)

        elif path == "/api/user-library":
            bb = [b for b in load_backbones()["backbones"] if b.get("source") == "user_library"]
            ins = [i for i in load_inserts()["inserts"] if i.get("source") == "user_library"]
            self._send_json({
                "configured": bool(os.environ.get("PLASMID_USER_LIBRARY")),
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

            def write_event(data: dict):
                try:
                    line = f"data: {json.dumps(data)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            # Send session_id to client
            write_event({"type": "session_id", "session_id": session_id})

            try:
                run_agent_turn_streaming(user_message, session_id, write_event, model=request_model)
            except Exception as e:
                logger.exception("Agent error")
                write_event({"type": "error", "content": str(e)})

        elif path == "/api/warmup":
            # Pre-spawn the SDK subprocess so the next /api/chat skips
            # the ~2-3s startup overhead. Idempotent. Creates a session
            # if one isn't provided so the frontend can warm before the
            # first message is composed.
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            session_id = body.get("session_id")
            if not session_id or not get_session(session_id):
                session_id = create_session()
            warmup_session(session_id, body.get("model"))
            self._send_json({"status": "warming", "session_id": session_id})

        elif path.startswith("/api/sessions/") and path.endswith("/cancel"):
            session_id = path.split("/")[3]
            cancel_session(session_id)
            close_warm_client(session_id)
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

        elif path == "/api/batch":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            csv_text = body.get("csv_content", "")
            request_model = body.get("model", MODEL)

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
            self._send_json({"job_id": job_id, "row_count": len(rows)})

        elif path == "/api/reset":
            # Legacy endpoint — clear all sessions
            _sessions.clear()
            _save_sessions()
            self._send_json({"status": "ok"})

        else:
            self.send_error(404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/sessions/"):
            session_id = path.split("/")[3]
            deleted = delete_session_by_id(session_id)
            self._send_json({"deleted": deleted})
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
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
