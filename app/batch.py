#!/usr/bin/env python3
"""
Batch Plasmid Designer

Runs the plasmid design agent on each row of a CSV file and saves outputs.

CSV format (header row required):
  description     — free-text design prompt (required)
  name            — output filename prefix (optional, default: plasmid_001, ...)
  output_format   — genbank | fasta | both (optional, default: genbank)

Example CSV:
  description,name,output_format
  "Express EGFP in HEK293 cells using pcDNA3.1(+)",egfp_hek293,genbank
  "Put mCherry into a lentiviral backbone",mcherry_lenti,both
  "Tag GAPDH with FLAG at the C-terminus",gapdh_flag,fasta

Usage:
  python app/batch.py designs.csv
  python app/batch.py designs.csv --output ./outputs/
  python app/batch.py designs.csv --output ./outputs/ --model claude-sonnet-4-6
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Add project directories to path so imports resolve
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic

# Import shared tool definitions, executor, and system prompt from the web app
from app import execute_tool, TOOLS, SYSTEM_PROMPT, MODEL
from references import ReferenceTracker


def run_agent_for_row(prompt: str, model: str = MODEL, max_iterations: int = 15) -> dict:
    """
    Run the plasmid design agent for a single prompt.

    Returns a dict:
      text     — final assistant text (summary / references)
      exports  — list of {filename, content} dicts from export_construct calls
      error    — error string if something crashed, else None
    """
    client = anthropic.Anthropic()
    tracker = ReferenceTracker()
    history = [{"role": "user", "content": prompt}]
    exports: list[dict] = []
    text_parts: list[str] = []

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
            thinking={"type": "enabled", "budget_tokens": 5000},
        )

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input, tracker)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
            if block.name == "export_construct":
                fmt = block.input.get("output_format", "genbank")
                cname = block.input.get("construct_name", "construct")
                ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
                exports.append({"filename": cname + ext, "content": result})

        history.append({"role": "assistant", "content": response.content})
        history.append({"role": "user", "content": tool_results})

    refs = tracker.format_references()
    if refs:
        text_parts.append(refs)

    return {"text": "\n".join(text_parts), "exports": exports, "error": None}


def run_batch(csv_path: str, output_dir: str, model: str = MODEL) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty.")
        return

    if "description" not in rows[0]:
        print("Error: CSV must have a 'description' column.", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    successes = 0
    failures = 0

    for i, row in enumerate(rows, 1):
        description = row.get("description", "").strip()
        if not description:
            print(f"[{i}/{total}] Skipping empty row.")
            continue

        name = row.get("name", "").strip() or f"plasmid_{i:03d}"
        output_format = (row.get("output_format") or "genbank").strip().lower()

        # Append a clear export instruction so the agent always calls export_construct
        if output_format == "both":
            prompt = description + "\nPlease export the final construct in both GenBank and FASTA formats."
        elif output_format == "fasta":
            prompt = description + "\nPlease export the final construct in FASTA format."
        else:
            prompt = description + "\nPlease export the final construct in GenBank format."

        preview = description[:72] + ("..." if len(description) > 72 else "")
        print(f"[{i}/{total}] {preview}")
        t0 = time.time()

        try:
            result = run_agent_for_row(prompt, model=model)
            elapsed = time.time() - t0

            if result["exports"]:
                for export in result["exports"]:
                    # Use the row name as the filename prefix for predictability
                    orig_stem = Path(export["filename"]).stem
                    suffix = Path(export["filename"]).suffix
                    fname = f"{name}{suffix}" if orig_stem == "construct" else f"{name}_{orig_stem}{suffix}"
                    filepath = out / fname
                    filepath.write_text(export["content"])
                    print(f"  ✓ {fname}  ({elapsed:.0f}s)")
                successes += 1
            else:
                # Agent ran but produced no export — save its text output for inspection
                log_path = out / f"{name}_output.txt"
                log_path.write_text(result["text"])
                print(f"  ⚠ No export produced — saved output to {log_path.name}  ({elapsed:.0f}s)")
                failures += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ Failed ({elapsed:.0f}s): {e}")
            failures += 1

    print(f"\nDone: {successes}/{total} succeeded, {failures} failed.")
    print(f"Output: {out.resolve()}/")


def main():
    parser = argparse.ArgumentParser(
        description="Batch plasmid designer — run the agent on every row of a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", help="CSV file with a 'description' column")
    parser.add_argument(
        "--output", "-o", default="./batch_output",
        help="Output directory (default: ./batch_output)",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Model to use (default: {MODEL})",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    run_batch(args.csv, args.output, args.model)


if __name__ == "__main__":
    main()
