"""LAB-Bench (FutureHouse) external benchmark runner.

Runs the plasmid designer agent against LAB-Bench CloningScenarios and SeqQA
subsets. These are multiple-choice questions; we present shuffled options
and score by whether the agent's selection matches the ground-truth answer.

    python -m evals.lab_bench --subset CloningScenarios --model claude-opus-4-6
    python -m evals.lab_bench --subset SeqQA --limit 50

Dataset: https://huggingface.co/datasets/futurehouse/lab-bench
Paper:   https://arxiv.org/abs/2407.10362
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    ResultMessage,
    TextBlock,
)
from datasets import load_dataset  # noqa: E402

from src.tools import build_mcp_servers  # noqa: E402

SYSTEM_PROMPT = (PROJECT_ROOT / "app" / "system_prompt.md").read_text()


async def _approve(tool_name, tool_input, ctx):
    return PermissionResultAllow()


def build_prompt(question: str, options: list[str], labels: list[str]) -> str:
    opts = "\n".join(f"{lab}. {o}" for lab, o in zip(labels, options))
    return (
        f"{question}\n\n"
        f"Select the single best answer from the following options. "
        f"You may use any tools to verify sequences, simulate cloning steps, "
        f"or check your reasoning.\n\n"
        f"{opts}\n\n"
        f"End your response with a line of the form:\n"
        f"ANSWER: <letter>"
    )


def extract_choice(text: str, labels: list[str]) -> str | None:
    m = re.search(r"ANSWER:\s*([A-Za-z])", text)
    if m:
        c = m.group(1).upper()
        if c in labels:
            return c
    # Fallback: last standalone label letter in the text
    for lab in reversed(labels):
        if re.search(rf"\b{lab}\b", text):
            return lab
    return None


async def run_one(row: dict, model: str, max_turns: int, seed: int) -> dict:
    rng = random.Random(seed)
    options = [row["ideal"], *row["distractors"]]
    rng.shuffle(options)
    labels = [chr(ord("A") + i) for i in range(len(options))]
    correct_label = labels[options.index(row["ideal"])]

    prompt = build_prompt(row["question"], options, labels)
    opts = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers=build_mcp_servers(),
        model=model,
        max_turns=max_turns,
        cwd=str(PROJECT_ROOT),
        permission_mode="acceptEdits",
        can_use_tool=_approve,
        disallowed_tools=["Bash", "Write", "Edit", "NotebookEdit"],
    )

    text_out = ""
    cost = 0.0
    t0 = time.perf_counter()
    async with ClaudeSDKClient(options=opts) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        text_out += b.text
            elif isinstance(msg, ResultMessage):
                cost = getattr(msg, "total_cost_usd", None) or 0.0
                break

    choice = extract_choice(text_out, labels)
    correct = choice == correct_label
    return {
        "id": row["id"],
        "subtask": row.get("subtask", ""),
        "correct": correct,
        "choice": choice,
        "correct_label": correct_label,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "cost_usd": round(cost, 4),
        "answer_text": text_out[-500:],
    }


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="CloningScenarios",
                   choices=["CloningScenarios", "SeqQA"])
    p.add_argument("--model", default="claude-opus-4-6")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N examples")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for option order")
    p.add_argument("--output", type=str, default=None,
                   help="Write per-example results as JSONL")
    args = p.parse_args()

    ds = load_dataset("futurehouse/lab-bench", args.subset, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"LAB-Bench {args.subset}: {len(ds)} examples, model={args.model}")
    results = []
    out_f = open(args.output, "w") if args.output else None
    try:
        for i, row in enumerate(ds):
            r = await run_one(dict(row), args.model, args.max_turns, args.seed + i)
            results.append(r)
            mark = "✓" if r["correct"] else "✗"
            print(
                f"  [{i+1}/{len(ds)}] {mark} {r['subtask'] or args.subset:<24} "
                f"choice={r['choice']} (correct={r['correct_label']}) "
                f"{r['elapsed_s']}s ${r['cost_usd']}"
            )
            if out_f:
                out_f.write(json.dumps(r) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()

    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n else 0.0
    total_cost = sum(r["cost_usd"] for r in results)
    print(f"\n{'='*60}")
    print(f"LAB-Bench {args.subset} ({args.model})")
    print(f"Accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{n})")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
