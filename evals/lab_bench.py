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


def build_open_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        f"You may use any tools to verify sequences, simulate cloning steps, "
        f"or check your reasoning. End your response with a line of the form:\n"
        f"ANSWER: <your final answer>"
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def score_open(text: str, ideal: str, distractors: list[str]) -> tuple[bool, str]:
    m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    given = m.group(1).strip() if m else ""
    n_ideal = _norm(ideal)
    n_given = _norm(given)
    n_text = _norm(text)
    # Correct if the normalized ideal is the given answer or appears in it,
    # and no distractor is a closer match to the given answer.
    if not n_ideal:
        return False, given
    correct = n_ideal == n_given or n_ideal in n_given or n_ideal in n_text
    if correct:
        for d in distractors:
            if _norm(d) and _norm(d) == n_given:
                correct = False
                break
    return correct, given


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


async def run_one(
    row: dict, model: str, max_turns: int, seed: int, open_answer: bool = False
) -> dict:
    if open_answer:
        prompt = build_open_prompt(row["question"])
        labels: list[str] = []
        correct_label = row["ideal"]
    else:
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

    if open_answer:
        correct, choice = score_open(text_out, row["ideal"], row["distractors"])
    else:
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
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of examples to run concurrently")
    p.add_argument("--open-answer", action="store_true",
                   help="No options shown; score by normalized match to ideal")
    args = p.parse_args()

    ds = load_dataset("futurehouse/lab-bench", args.subset, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    rows = [dict(r) for r in ds]
    n = len(rows)

    mode = "open-answer" if args.open_answer else "MCQ"
    print(
        f"LAB-Bench {args.subset} ({mode}): {n} examples, model={args.model}, "
        f"parallel={args.parallel}", flush=True,
    )
    out_f = open(args.output, "w") if args.output else None
    sem = asyncio.Semaphore(args.parallel)
    done = 0

    async def worker(i: int, row: dict):
        nonlocal done
        async with sem:
            r = await run_one(
                row, args.model, args.max_turns, args.seed + i, args.open_answer
            )
        done += 1
        mark = "✓" if r["correct"] else "✗"
        choice_disp = (r["choice"] or "")[:30]
        label_disp = str(r["correct_label"])[:30]
        print(
            f"  [{done}/{n}] {mark} {r['subtask'] or args.subset:<24} "
            f"choice={choice_disp} (correct={label_disp}) "
            f"{r['elapsed_s']}s", flush=True,
        )
        if out_f:
            out_f.write(json.dumps(r) + "\n")
            out_f.flush()
        return r

    try:
        results = await asyncio.gather(
            *(worker(i, row) for i, row in enumerate(rows))
        )
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
