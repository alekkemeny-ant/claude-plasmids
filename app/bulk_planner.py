"""
Bulk design planner — lightweight Haiku-powered analysis pass that runs before
the full agent batch.  Takes CSV rows (or a conversation-derived row list),
identifies the shared pattern, enriches each prompt, estimates cost, and
decides whether all rows can run in ONE batched agent call or need separate calls.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ── Model used for planning (cheap, fast) ──────────────────────────────────
PLANNER_MODEL = "claude-haiku-4-5-20251001"

# ── Pricing per million tokens (input_price, output_price) ────────────────
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00,  5.00),   # was 0.80/4.00
    "claude-sonnet-4-6":         (3.00, 15.00),   # ✓ correct
    "claude-opus-4-6":           (5.00, 25.00),   # was 15.00/75.00
    "claude-opus-4-7":           (5.00, 25.00),   # was 15.00/75.00
    "claude-opus-4-8":           (5.00, 25.00),   # new flagship, same rate
}

# ── Estimated tokens per design row, by complexity ────────────────────────
# Covers system prompt share + enriched prompt + tool round-trips + export output.
TOKENS_BY_COMPLEXITY: dict[str, tuple[int, int]] = {
    "simple":   (200_000,   1_400),
    "standard": (300_000, 8_000),
    "complex":  (450_000, 17_000),
}

# ── Cost thresholds ────────────────────────────────────────────────────────
COST_WARN_THRESHOLD  = 5.0
COST_SPLIT_THRESHOLD = 20.0


@dataclass
class BulkPlan:
    plan_id: str
    summary: str
    enriched_rows: list[dict]     # [{name, description, enriched_prompt, output_format}]
    job_groups: list[list[int]]   # indices grouped into cost sub-batches
    model_suggestion: str
    estimated_cost_usd: float
    complexity: str = "standard"
    # When True, all rows in a group can run in ONE agent conversation
    batch_eligible: bool = False
    # Single combined prompt for the whole batch (set when batch_eligible=True)
    batch_prompt: Optional[str] = None
    # Human-readable description of what is shared across rows
    shared_context: str = ""


# ── Internal helpers ───────────────────────────────────────────────────────

def _cost_per_row(model: str, complexity: str) -> float:
    input_t, output_t = TOKENS_BY_COMPLEXITY.get(complexity, TOKENS_BY_COMPLEXITY["standard"])
    input_p, output_p = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-6"])
    return (input_t * input_p + output_t * output_p) / 1_000_000


def estimate_cost(n_rows: int, model: str, complexity: str = "standard") -> float:
    return round(_cost_per_row(model, complexity) * n_rows, 4)


def build_job_groups(n_rows: int, model: str, complexity: str) -> list[list[int]]:
    """Split rows into cost sub-batches, each under COST_SPLIT_THRESHOLD."""
    cpr = _cost_per_row(model, complexity)
    max_per_group = max(1, int(COST_SPLIT_THRESHOLD / cpr)) if cpr > 0 else n_rows
    groups: list[list[int]] = []
    for start in range(0, n_rows, max_per_group):
        groups.append(list(range(start, min(start + max_per_group, n_rows))))
    return groups


# ── Planning prompt ────────────────────────────────────────────────────────

_PLANNING_SYSTEM = """You are a plasmid design planner. Given a list of construct descriptions, produce a JSON plan.

Instructions:
1. Identify the shared pattern (same backbone, same assembly method, same enzyme, etc.) and what varies per row.
2. Rewrite EVERY row as a COMPLETE, SELF-CONTAINED design prompt. Merge any user_context into every enriched prompt.
3. Classify complexity:
   - "simple"  → pure GG oligo annealing / trivial MCS, all parameters explicit
   - "standard" → MCS with NCBI/Addgene retrieval, some disambiguation needed
   - "complex"  → fusions, mutations, feature swaps, multiple unknowns
4. Suggest the best run model:
  
   - "claude-sonnet-4-6"         → pure oligo-annealing / simple GG or standard designs with gene retrieval
   - "claude-opus-4-7"           → complex designs, fusions, mutations
5. Decide if batch_eligible:
   - Set batch_eligible=true ONLY when ALL rows share the SAME backbone AND the SAME assembly method
     (e.g. all are BbsI Golden Gate with the same backbone, or all are simple MCS cloning into the same vector)
   - Set batch_eligible=false if rows differ in backbone, method, or complexity
6. If batch_eligible=true, provide:
   - shared_context: 1-2 sentences describing what is shared (backbone name, method, enzyme)
   - row_summaries: one short line per row describing only the VARIABLE parts (e.g. oligo sequences, gene name, construct name)

Respond ONLY with a JSON object (no markdown):
{
  "summary": "<1-2 sentence summary of the full batch>",
  "complexity": "simple|standard|complex",
  "model_suggestion": "<model id>",
  "batch_eligible": true|false,
  "shared_context": "<what is shared — backbone, method, enzyme>",
  "row_summaries": ["<variable parts for row 1>", "<variable parts for row 2>", ...],
  "enriched_rows": [
    {
      "name": "<construct name>",
      "enriched_prompt": "<complete self-contained design prompt>",
      "output_format": "genbank"
    }
  ]
}"""


def _make_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _build_batch_prompt(
    enriched_rows: list[dict],
    shared_context: str,
    row_summaries: list[str],
) -> str:
    """Build a single combined prompt for all rows to run in one agent call."""
    n = len(enriched_rows)
    rows_section = "\n\n".join(
        f"Row {i + 1} | Name: {r['name']}\n{row_summaries[i] if i < len(row_summaries) else r['description']}"
        for i, r in enumerate(enriched_rows)
    )
    export_names = ", ".join(r["name"] for r in enriched_rows)
    return (
        f"<!-- bulk-row-batch -->\n"
        f"You will design {n} construct{'s' if n > 1 else ''} in this single session.\n\n"
        f"SHARED SETUP (do this ONCE, not per row):\n{shared_context}\n\n"
        f"ROWS TO DESIGN ({n} total — process ALL of them before stopping):\n\n"
        f"{rows_section}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Load the shared backbone and verify the assembly enzyme recognition sites ONCE at the start.\n"
        f"- Then process each row in order: assemble, validate, and export as GenBank.\n"
        f"- Use the EXACT construct name for each export ({export_names}).\n"
        f"- After exporting one construct, IMMEDIATELY start the next row — do NOT pause.\n"
        f"- Only end your response after ALL {n} constructs have been exported."
    )


# ── Public API ─────────────────────────────────────────────────────────────

_TEMPLATE_SYSTEM = """You are a plasmid design assistant. A user has written a template design prompt and provided a table of per-construct details as CSV rows.

Your job: for each CSV row, produce a COMPLETE, SELF-CONTAINED design prompt by substituting the row's data into the template. Do not invent data — use only what is in the template and the row.

Rules:
- Column names in the CSV map to the corresponding references in the template (e.g. "Oligo 1" column → "Oligo 1" in the template).
- Keep all shared instructions from the template (backbone, enzyme, method, export format, etc.).
- The construct name should come from the CSV "name" or "Name" column if present.
- Prefix every enriched prompt with "<!-- bulk-row -->".

Respond ONLY with a JSON object (no markdown):
{
  "rows": [
    {"name": "<construct name>", "enriched_prompt": "<!-- bulk-row -->\\n<complete prompt>"},
    ...
  ]
}"""


def generate_from_template(
    template: str,
    csv_rows: list[dict],
    run_model: str = "claude-sonnet-4-6",
    planner_model: str = PLANNER_MODEL,
) -> BulkPlan:
    """Merge a chat template with raw CSV row data to produce a BulkPlan."""
    rows_text = "\n".join(
        f"{i + 1}. " + " | ".join(f"{k}={v}" for k, v in row.items())
        for i, row in enumerate(csv_rows)
    )
    user_msg = f"Template prompt:\n{template}\n\nCSV rows ({len(csv_rows)} total):\n{rows_text}"

    client = _make_client()
    response = client.messages.create(
        model=planner_model,
        max_tokens=4096,
        system=_TEMPLATE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw[raw.find("\n") + 1:]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()

    parsed: dict = json.loads(raw)
    planner_rows = parsed.get("rows", [])

    enriched: list[dict] = []
    for i, orig in enumerate(csv_rows):
        pr = planner_rows[i] if i < len(planner_rows) else {}
        name = pr.get("name") or orig.get("name") or orig.get("Name") or f"construct_{i + 1:03d}"
        ep   = pr.get("enriched_prompt") or f"<!-- bulk-row -->\n{template}\n\nRow data: {orig}"
        enriched.append({
            "name":            name,
            "description":     template,
            "enriched_prompt": ep,
            "output_format":   "genbank",
        })

    while len(enriched) < len(csv_rows):
        i = len(enriched)
        orig = csv_rows[i]
        name = orig.get("name") or orig.get("Name") or f"construct_{i + 1:03d}"
        enriched.append({
            "name":            name,
            "description":     template,
            "enriched_prompt": f"<!-- bulk-row -->\n{template}\n\nRow data: {orig}",
            "output_format":   "genbank",
        })

    cost   = estimate_cost(len(csv_rows), run_model, "simple")
    groups = [[i] for i in range(len(csv_rows))]  # sequential, one per group

    return BulkPlan(
        plan_id=str(uuid.uuid4()),
        summary=f"{len(csv_rows)} construct(s) from template",
        enriched_rows=enriched,
        job_groups=groups,
        model_suggestion="claude-sonnet-4-6",
        estimated_cost_usd=cost,
        complexity="simple",
        batch_eligible=False,
    )


def generate_bulk_plan(
    rows: list[dict],
    user_context: str = "",
    run_model: str = "claude-sonnet-4-6",
    planner_model: str = PLANNER_MODEL,
) -> BulkPlan:
    """Analyze *rows* and return a BulkPlan with optional batch execution mode."""
    rows_text = "\n".join(
        f"{i + 1}. name={r.get('name') or '(unnamed)'} | description={r.get('description', '')}"
        for i, r in enumerate(rows)
    )
    user_msg = f"Rows to design ({len(rows)} total):\n{rows_text}"
    if user_context.strip():
        user_msg += f"\n\nAdditional context from user: {user_context.strip()}"

    client = _make_client()
    response = client.messages.create(
        model=planner_model,
        max_tokens=8192,
        system=_PLANNING_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw[raw.find("\n") + 1:]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()

    parsed: dict = json.loads(raw)

    summary        = parsed.get("summary", f"{len(rows)} design(s)")
    complexity     = parsed.get("complexity", "standard")
    model_suggest  = parsed.get("model_suggestion", "claude-sonnet-4-6")
    batch_eligible = bool(parsed.get("batch_eligible", False))
    shared_ctx     = parsed.get("shared_context", "")
    row_summaries  = parsed.get("row_summaries", [])
    planner_rows   = parsed.get("enriched_rows", [])

    enriched: list[dict] = []
    for i, orig in enumerate(rows):
        pr = planner_rows[i] if i < len(planner_rows) else {}
        raw_prompt = pr.get("enriched_prompt") or orig.get("description", "")
        enriched.append({
            "name":            pr.get("name") or orig.get("name") or f"construct_{i + 1:03d}",
            "description":     orig.get("description", ""),
            "enriched_prompt": f"<!-- bulk-row -->\n{raw_prompt}",
            "output_format":   pr.get("output_format") or orig.get("output_format") or "genbank",
        })

    # Pad if planner returned fewer rows than input
    while len(enriched) < len(rows):
        i = len(enriched)
        orig = rows[i]
        enriched.append({
            "name":            orig.get("name") or f"construct_{i + 1:03d}",
            "description":     orig.get("description", ""),
            "enriched_prompt": f"<!-- bulk-row -->\n{orig.get('description', '')}",
            "output_format":   orig.get("output_format", "genbank"),
        })

    # Build the combined batch prompt if eligible
    batch_prompt: Optional[str] = None
    if batch_eligible and shared_ctx and len(enriched) > 1:
        batch_prompt = _build_batch_prompt(enriched, shared_ctx, row_summaries)

    cost   = estimate_cost(len(rows), run_model, complexity)
    groups = build_job_groups(len(rows), run_model, complexity)

    return BulkPlan(
        plan_id=str(uuid.uuid4()),
        summary=summary,
        enriched_rows=enriched,
        job_groups=groups,
        model_suggestion=model_suggest,
        estimated_cost_usd=cost,
        complexity=complexity,
        batch_eligible=batch_eligible,
        batch_prompt=batch_prompt,
        shared_context=shared_ctx,
    )
