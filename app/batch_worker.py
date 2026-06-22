"""
Batch job workers for the Plasmid Designer.

Runs multiple construct designs in background threads, one per CSV row.
Each row gets its own agent loop (_run_batch_agent) and stores exports +
a structured log in the shared _batch_jobs dict (owned by sessions.py).

Concurrency model:
  - Each row runs in a daemon thread launched by start_batch_job().
  - Pause/resume is controlled via threading.Event gates from sessions.py:
      _get_pause_event() — pause.wait() blocks the worker thread until
      the user resumes via POST /api/batch/{id}/rows/{idx}/pause.
  - Per-row approval gates (_get_row_gate) let the user inspect row N-1
    before row N starts (approval_required=True mode).
  - batch_groups collapse multiple similar rows into a single agent call,
    then distribute exports back by filename slug matching.
"""

import json
import logging
import re
import threading
import uuid
from typing import Optional

from src.tools import set_tracker, get_last_plot_json, clear_last_plot_json
from src.references import ReferenceTracker

from sessions import (
    SYSTEM_PROMPT,
    _batch_jobs,
    _save_batch_jobs,
    _get_row_gate,
    _get_pause_event,
)
from streaming import (
    TOOLS,
    _client,
    _dispatch_tool,
)

logger = logging.getLogger(__name__)

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


def _run_batch_row(job_id: str, row_idx: int, row: dict, model: str,
                    seed_history: Optional[list] = None) -> None:
    """Worker for a single CSV row — runs the agent and stores exports + log in _batch_jobs.

    seed_history: optional agent history from a prior run (e.g. the in-chat preview)
    to pre-populate this row's conversation, so it doesn't re-fetch the backbone/
    insertion site already resolved there. Without this, each row starts cold and
    only sees a textual reminder, which the model isn't reliably guaranteed to obey.
    """
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
        history: list[dict] = list(seed_history) if seed_history else []
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


def _run_batch_group(job_id: str, row_indices: list, combined_prompt: str, model: str,
                     seed_history: Optional[list] = None) -> None:
    """Run ONE agent call for multiple similar rows, then distribute exports by name."""
    job = _batch_jobs.get(job_id)
    if not job:
        return

    for idx in row_indices:
        job["rows"][idx]["status"] = "running"
        job["rows"][idx]["paused"] = False
        job["rows"][idx]["log"] = []

    combined_exports: list[dict] = []
    combined_log: list[dict] = []
    # Pre-populate history with sample context so backbone lookups aren't repeated
    history: list[dict] = list(seed_history) if seed_history else []

    try:
        _run_batch_agent(
            combined_prompt, model,
            append_log=combined_log.append,
            exports=combined_exports,
            history=history,
        )
    except Exception as e:
        for idx in row_indices:
            job["rows"][idx]["status"] = "error"
            job["rows"][idx]["error"] = str(e)
            job["rows"][idx]["log"] = combined_log
        _save_batch_jobs()
        return

    # Distribute exports back to individual rows by matching filename vs row name.
    # Fuzzy slug comparison is needed because the agent names exports after the
    # construct (e.g. "pAAV_EGFP") while the row name may differ slightly
    # (e.g. "pAAV-EGFP-v2") — exact matching would leave rows with no export.
    assigned: set[int] = set()
    for idx in row_indices:
        row_name = (job["rows"][idx].get("name") or "").lower()
        row_name_slug = re.sub(r"[^a-z0-9]", "", row_name)
        matched = []
        for exp in combined_exports:
            fn_stem = re.sub(r"[^a-z0-9]", "", (exp.get("filename") or "").lower().rsplit(".", 1)[0])
            if fn_stem == row_name_slug or row_name_slug in fn_stem or fn_stem in row_name_slug:
                matched.append(exp)
        job["rows"][idx]["exports"] = matched
        job["rows"][idx]["status"] = "done" if matched else "no_export"
        job["rows"][idx]["log"] = combined_log  # all rows share the log
        job["rows"][idx]["history"] = history
        assigned.update(id(e) for e in matched)

    # Any exports that didn't match a named row: assign to rows without exports in order
    unmatched = [e for e in combined_exports if id(e) not in assigned]
    no_export_indices = [i for i in row_indices if not job["rows"][i]["exports"]]
    for exp, idx in zip(unmatched, no_export_indices):
        job["rows"][idx]["exports"] = [exp]
        job["rows"][idx]["status"] = "done"

    _save_batch_jobs()


def start_batch_job(
    rows: list,
    model: str,
    pre_seeded_rows: Optional[dict] = None,
    batch_groups: Optional[list[dict]] = None,
    approval_required: bool = False,
    seed_history: Optional[list] = None,
    preview_exports: Optional[list] = None,
) -> str:
    """Create a batch job, launch a background thread, return job_id.

    pre_seeded_rows: optional dict {row_idx: row_state_dict} for rows already
    complete (e.g. the sample design ran ahead of the full batch). Those rows
    are skipped by the worker.

    batch_groups: optional list of {prompt: str, indices: [int, ...]} dicts.
    When provided, rows in the same group are run as ONE agent call (sharing
    backbone load, RE site checks, etc.) rather than N separate calls.

    seed_history: optional agent history from a prior sample run to pre-populate
    context so subsequent rows skip redundant backbone/RE-site lookups.
    """
    job_id = str(uuid.uuid4())
    pre_seeded_rows = pre_seeded_rows or {}
    job_rows: list[dict] = []
    for i, r in enumerate(rows):
        if i in pre_seeded_rows:
            job_rows.append(pre_seeded_rows[i])
        else:
            job_rows.append({
                "description": r.get("description", ""),
                "name": r.get("name", ""),
                "output_format": r.get("output_format", "genbank"),
                "status": "pending",
                "paused": False,
                "exports": [],
                "error": None,
            })
    job: dict = {
        "status": "running",
        "model": model,
        "rows": job_rows,
        "approval_required": approval_required,
        "preview_exports": preview_exports or [],
    }
    _batch_jobs[job_id] = job

    def worker():
        if batch_groups:
            # Indices covered by batch groups (run as combined agent calls)
            covered: set[int] = set()
            for grp in batch_groups:
                prompt   = grp["prompt"]
                indices  = [i for i in grp["indices"] if job["rows"][i].get("status") != "done"]
                if not indices:
                    continue
                covered.update(indices)
                if len(indices) == 1:
                    _run_batch_row(job_id, indices[0], rows[indices[0]], model, seed_history=seed_history)
                else:
                    _run_batch_group(job_id, indices, prompt, model, seed_history=seed_history)
            for idx, row in enumerate(rows):
                if idx not in covered and job["rows"][idx].get("status") != "done":
                    _run_batch_row(job_id, idx, row, model, seed_history=seed_history)
        else:
            for idx, row in enumerate(rows):
                if job["rows"][idx].get("status") == "done":
                    continue
                # For approval_required jobs, wait for user confirmation before row > 0
                if approval_required and idx > 0:
                    job["rows"][idx]["status"] = "waiting"
                    _save_batch_jobs()
                    _get_row_gate(job_id, idx).wait()
                    if job["rows"][idx].get("status") == "waiting":
                        job["rows"][idx]["status"] = "pending"
                _run_batch_row(job_id, idx, row, model, seed_history=seed_history)

        job["status"] = "done"
        _save_batch_jobs()

    threading.Thread(target=worker, daemon=True).start()
    return job_id

