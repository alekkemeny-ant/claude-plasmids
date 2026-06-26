"""
Streaming agent loop for the Plasmid Designer web UI.

Drives one agent turn using the Anthropic streaming API directly (~1s TTFT),
dispatching tool calls in-process via the same handlers as app/agent.py and
the eval harness (src/tools.py:ALL_TOOLS).

Key design decisions:
  - One asyncio.run() per tool call: handlers are defined as async for the
    MCP server but we drive them synchronously here to avoid running an event
    loop across SSE stream boundaries.
  - Retry logic (max 3 attempts, exponential backoff) is scoped to the API
    call, not the tool call — tool errors are returned to the model, not retried.
  - tool_results accumulated before a rate-limit retry are discarded and rebuilt
    from scratch on retry because their tool_use_ids reference the aborted stream.
  - Context window tracking (token_usage event) uses per-model sizes from
    CONTEXT_WINDOW so the frontend can show a usage bar without an extra API call.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time

import anthropic

from src.tools import (
    get_anthropic_tool_schemas,
    get_tool_dispatch,
    set_tracker,
    get_last_plot_json,
    clear_last_plot_json,
)
from src.references import ReferenceTracker

from sessions import (
    _cancelled_sessions,
    _active_turns,
    _session_live_streams,
    _session_live_streams_lock,
    _save_sessions,
    get_session,
    _build_system_prompt,
)

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────

MODEL = "claude-opus-4-7"

# Context window sizes by model (tokens). Used to render the token-usage bar
# in the UI — the bar needs the ceiling to compute a percentage.
CONTEXT_WINDOW = {
    "claude-opus-4-7":          1_000_000,
    "claude-opus-4-6":          1_000_000,
    "claude-sonnet-4-6":        1_000_000,
    "claude-haiku-4-5-20251001":  200_000,
}

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


def reset_client():
    """Reset the cached Anthropic client (call after API key changes)."""
    global _anthropic_client
    _anthropic_client = None


def _emit_tool_result(
    tool_name: str,
    tool_input: dict,
    result_str: str,
    *,
    safe_write,
    session: dict,
    assistant_blocks: list,
    preview_state: dict | None = None,
    current_model: str = "claude-sonnet-4-6",
) -> None:
    """Emit SSE event(s) for a completed tool call and apply side effects.

    Handles export_construct download/plot, log_experimental_outcome
    session persistence, and display-message recording. Shared between
    the streaming loop and any future callers so behaviour stays in sync.
    """
    if tool_name == "submit_bulk_designs" and result_str.startswith("[BULK_DESIGNS_READY]"):
        # Legacy path: CSV-upload agent still emits this. Keep handling it.
        try:
            rows_json = result_str[len("[BULK_DESIGNS_READY] "):]
            rows_data = json.loads(rows_json)
        except Exception:
            rows_data = []
        safe_write({"type": "bulk_design_rows", "rows": rows_data})
        assistant_blocks.append({
            "type": "tool_use",
            "name": tool_name,
            "input": tool_input,
            "result": f"Handed {len(rows_data)} construct(s) to the bulk design planner.",
        })
        return  # don't emit a regular tool_result card

    if tool_name == "submit_bulk_designs" and result_str.startswith("[BULK_DESIGNS_REGISTERED]"):
        # New path: agent handles design directly; start preview token tracking and
        # emit event so the frontend can show the model picker inline.
        # `rows` here is the FULL list submitted (system prompt: "the full list"),
        # but construct #1 becomes the in-chat preview, so n_constructs must reflect
        # only what's left — showBulkPreviewModelCard adds 1 back for display.
        n_total = len(tool_input.get("rows", []))
        n_remaining = max(n_total - 1, 0)
        if preview_state is not None:
            preview_state["tracking"] = True
            preview_state["in"] = 0
            preview_state["out"] = 0
            preview_state["exports"] = []
            session["_preview_state"] = dict(preview_state)
            # Mark where the bulk session begins in the history so batch rows
            # can be seeded with only the preview-relevant turns rather than
            # the full session. history hasn't been extended with the current
            # assistant turn yet, so this index points to that upcoming turn.
            session["_preview_history_start"] = len(session.get("history", []))
        safe_write({
            "type": "bulk_designs_registered",
            "n_constructs": n_remaining,
            "preview_model": current_model,
        })
        # Fall through — let the result render as a normal tool_result card so
        # the agent receives its workflow instructions.

    if tool_name == "complete_bulk_preview" and result_str.startswith("[BULK_PREVIEW_READY]"):
        try:
            payload_json = result_str[len("[BULK_PREVIEW_READY] "):]
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
        n_remaining = len(payload.get("remaining_rows", []))
        # Attach actual preview token counts so the frontend can show real estimates.
        if preview_state is not None:
            payload["preview_tokens_in"]  = preview_state.get("in", 0)
            payload["preview_tokens_out"] = preview_state.get("out", 0)
            payload["preview_model"]      = current_model
            payload["preview_exports"]    = preview_state.get("exports", [])
            preview_state["tracking"] = False
            preview_state["exports"] = []
            session.pop("_preview_state", None)
        safe_write({"type": "bulk_preview_complete", **payload})
        assistant_blocks.append({
            "type": "tool_use",
            "name": tool_name,
            "input": tool_input,
            "result": f"Preview complete. {n_remaining} construct(s) queued for user approval.",
        })
        return  # don't emit a regular tool_result card

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
        # Capture exports during the preview run; suppress the in-chat download
        # button so the file only appears in the approval card (not twice).
        is_preview_export = preview_state is not None and preview_state.get("tracking")
        if is_preview_export:
            preview_state.setdefault("exports", []).append({
                "filename": filename,
                "content": result_str,
            })
            safe_write({"type": "bulk_preview_export", "filename": filename, "content": result_str})
            event_data = {
                "type": "tool_result",
                "tool": tool_name,
                "input": tool_input,
                "content": display_result,
                # download_content/download_filename omitted — shown in the approval card
            }
        else:
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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        write_event({"type": "no_api_key"})
        write_event({"type": "done"})
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
    # Accumulates tokens for the bulk preview construct (#1) so the approval
    # card can show real per-construct cost estimates for the remaining runs.
    # Persisted in session so tracking survives across the two turns (plan turn
    # where submit_bulk_designs fires, and preview turn where the build happens).
    _ps = session.get("_preview_state", {})
    preview_state: dict = {
        "tracking": bool(_ps.get("tracking")),
        "in":       int(_ps.get("in", 0)),
        "out":      int(_ps.get("out", 0)),
        "exports":  list(_ps.get("exports", [])),
    }
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
                        max_tokens=32000,
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
                                        preview_state=preview_state,
                                        current_model=model,
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
                            if preview_state["tracking"]:
                                preview_state["in"]  += final_message.usage.input_tokens
                                preview_state["out"] += getattr(final_message.usage, "output_tokens", 0)
                                session["_preview_state"] = dict(preview_state)
                            safe_write({
                                "type": "token_usage",
                                "input_tokens": final_message.usage.input_tokens,
                                "context_window": CONTEXT_WINDOW.get(model, 1_000_000),
                            })
                    break  # stream succeeded, leave retry loop

                except anthropic.AuthenticationError:
                    safe_write({"type": "no_api_key"})
                    break
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

        # If the model produced only thinking (no text, no tool calls) — typically
        # because max_tokens was hit during the thinking phase — emit a visible
        # explanation so the user doesn't see an empty response.
        only_thinking = (
            not assistant_text
            and assistant_blocks
            and all(b.get("type") == "thinking" for b in assistant_blocks)
            and not is_cancelled()
            and sys.exc_info()[1] is None
        )
        if only_thinking:
            fallback = (
                "I ran out of output tokens while working through this request. "
                "The request may be too complex or require a very detailed analysis. "
                "Try breaking it into smaller steps, or ask me a more focused question."
            )
            assistant_text = fallback
            assistant_blocks.append({"type": "text", "content": fallback})
            safe_write({"type": "text_start"})
            safe_write({"type": "text_delta", "content": fallback})
            safe_write({"type": "text_end"})

        if assistant_text or assistant_blocks:
            session["display_messages"].append({
                "role": "assistant",
                "content": assistant_text,
                "blocks": assistant_blocks,
            })
        else:
            # Remove dangling user message — covers both explicit cancel and error cases.
            # When a turn produces no response (cancelled or API error), the user message
            # appended at turn-start has no matching assistant response in history. Leaving
            # it causes the next follow-up to produce two consecutive user messages, which
            # triggers a 400 on the next API call and silently swallows that turn too.
            if history and history[-1]["role"] == "user" and isinstance(history[-1].get("content"), str):
                history.pop()
                if session["display_messages"] and session["display_messages"][-1]["role"] == "user":
                    session["display_messages"].pop()

        _save_sessions()

        if not disconnected:
            # If an exception is propagating (e.g. re-raised BadRequestError from a
            # consecutive-user-message 400), emit an error event before done so the
            # client shows feedback instead of a silent empty turn.
            exc = sys.exc_info()[1]
            if exc is not None and not is_cancelled():
                try:
                    write_event({"type": "error", "content": f"Request failed: {exc}"})
                except (BrokenPipeError, ConnectionResetError):
                    pass
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





