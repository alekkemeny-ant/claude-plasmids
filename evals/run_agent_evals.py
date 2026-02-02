#!/usr/bin/env python3
"""
End-to-End Agent Evaluation Runner (Claude Agent SDK)

Sends natural language prompts through the full Claude agent loop
using the Agent SDK's query() function and scores the final output
using the Allen Institute rubric.

The agent uses the exact same tools and system prompt as the
production agent (app/agent.py), so evals test the real system.

Usage:
    python -m evals.run_agent_evals                     # Run all agent eval cases
    python -m evals.run_agent_evals --case A1-001       # Run a single case
    python -m evals.run_agent_evals --verbose           # Show full agent trace
    python -m evals.run_agent_evals --model sonnet      # Use a different model
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "app" / ".env")
except ImportError:
    pass

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    PermissionResultAllow,
)
from claude_agent_sdk.types import UserMessage
from src.tools import create_plasmid_tools, ALL_TOOL_NAMES
from src.library import get_backbone_by_id, get_insert_by_id
from src.assembler import find_mcs_insertion_point
from evals.rubric import score_construct, RubricResult

logger = logging.getLogger(__name__)

# Load system prompt
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "app" / "system_prompt.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""


# ── Agent eval test cases ──────────────────────────────────────────────


@dataclass
class AgentTestCase:
    """A test case for the end-to-end agent eval."""
    id: str
    name: str
    prompt: str
    description: str
    expected_backbone_id: str
    expected_insert_id: str
    expected_insertion_position: Optional[int] = None
    expected_total_size: Optional[int] = None
    tags: list[str] = field(default_factory=list)


AGENT_CASES = [
    # ── A1: Explicit backbone + insert (pcDNA3.1(+) baseline) ─────────
    AgentTestCase(
        id="A1-001",
        name="Explicit EGFP in pcDNA3.1(+)",
        prompt="Design an EGFP expression plasmid using pcDNA3.1(+) as the backbone. Assemble the construct and give me the final sequence.",
        description="Straightforward request with both backbone and insert named explicitly.",
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["explicit", "mammalian", "fluorescent_protein"],
    ),
    AgentTestCase(
        id="A1-002",
        name="Explicit mCherry in pcDNA3.1(+)",
        prompt="Put mCherry into pcDNA3.1(+) and assemble the construct. Return the assembled sequence.",
        description="Direct request with both components specified.",
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mCherry",
        expected_insertion_position=895,
        expected_total_size=6139,
        tags=["explicit", "mammalian", "fluorescent_protein"],
    ),
    AgentTestCase(
        id="A1-003",
        name="EGFP in pUC19",
        prompt="Assemble EGFP into pUC19. Give me the complete construct sequence.",
        description="Bacterial backbone with a fluorescent protein insert.",
        expected_backbone_id="pUC19",
        expected_insert_id="EGFP",
        expected_insertion_position=396,
        expected_total_size=3406,
        tags=["explicit", "bacterial", "fluorescent_protein"],
    ),
    # ── A1: Explicit — new backbone coverage ──────────────────────────
    AgentTestCase(
        id="A1-004",
        name="EGFP in pBABE-puro",
        prompt="Assemble EGFP into pBABE-puro. Give me the assembled sequence.",
        description="Retroviral expression vector. Tests a non-pcDNA mammalian backbone.",
        expected_backbone_id="pBABE-puro",
        expected_insert_id="EGFP",
        expected_insertion_position=1260,
        expected_total_size=5806,
        tags=["explicit", "mammalian", "fluorescent_protein", "retroviral"],
    ),
    AgentTestCase(
        id="A1-005",
        name="EGFP in pGEX-4T-1",
        prompt="Insert EGFP into pGEX-4T-1 and assemble the construct. Return the sequence.",
        description="Bacterial GST-fusion vector. Tests tac promoter backbone.",
        expected_backbone_id="pGEX-4T-1",
        expected_insert_id="EGFP",
        expected_insertion_position=930,
        expected_total_size=5689,
        tags=["explicit", "bacterial", "fluorescent_protein"],
    ),
    AgentTestCase(
        id="A1-006",
        name="Luciferase in pAAV-CMV",
        prompt="Assemble firefly luciferase into the pAAV-CMV vector. Output the assembled sequence.",
        description="AAV gene therapy backbone with a reporter insert.",
        expected_backbone_id="pAAV-CMV",
        expected_insert_id="Firefly_Luciferase",
        expected_insertion_position=1200,
        expected_total_size=5905,
        tags=["explicit", "mammalian", "reporter", "aav"],
    ),
    AgentTestCase(
        id="A1-007",
        name="EGFP in pLKO.1-puro",
        prompt="Put EGFP into pLKO.1-puro and assemble. Return the final sequence.",
        description="Lentiviral vector. Tests a larger backbone (7050 bp) with U6 promoter.",
        expected_backbone_id="pLKO.1-puro",
        expected_insert_id="EGFP",
        expected_insertion_position=1878,
        expected_total_size=7770,
        tags=["explicit", "mammalian", "fluorescent_protein", "lentiviral"],
    ),
    AgentTestCase(
        id="A1-008",
        name="mCherry in pEGFP-N1",
        prompt="Insert mCherry into the MCS of pEGFP-N1 and assemble the construct. Return the sequence.",
        description="C-terminal EGFP fusion vector with mCherry at MCS. Tests pEGFP-N1 backbone.",
        expected_backbone_id="pEGFP-N1",
        expected_insert_id="mCherry",
        expected_insertion_position=591,
        expected_total_size=5444,
        tags=["explicit", "mammalian", "fluorescent_protein"],
    ),
    AgentTestCase(
        id="A1-009",
        name="EGFP in pcDNA3",
        prompt="Assemble EGFP into pcDNA3. Return the assembled construct sequence.",
        description="Older pcDNA3 vector (not pcDNA3.1). Agent must not confuse with pcDNA3.1.",
        expected_backbone_id="pCDNA3",
        expected_insert_id="EGFP",
        expected_insertion_position=900,
        expected_total_size=6173,
        tags=["explicit", "mammalian", "fluorescent_protein"],
    ),
    # ── A2: Alias / name resolution ───────────────────────────────────
    AgentTestCase(
        id="A2-001",
        name="Alias: 'eGFP' resolution",
        prompt="Insert eGFP into pcDNA3.1+ and assemble the construct. Output the raw sequence.",
        description="Uses common aliases: 'eGFP' for EGFP, 'pcDNA3.1+' for pcDNA3.1(+).",
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["alias", "name_resolution", "mammalian"],
    ),
    AgentTestCase(
        id="A2-002",
        name="Alias: 'GFP' resolution",
        prompt="I need a GFP expression plasmid in pcDNA3.1 plus. Assemble and return the sequence.",
        description="Uses 'GFP' (should resolve to EGFP) and 'pcDNA3.1 plus'.",
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["alias", "name_resolution", "mammalian"],
    ),
    AgentTestCase(
        id="A2-003",
        name="Alias: 'pBABE puro' resolution",
        prompt="Clone mCherry into pBABE puro and assemble. Give me the final sequence.",
        description="Uses 'pBABE puro' alias for pBABE-puro.",
        expected_backbone_id="pBABE-puro",
        expected_insert_id="mCherry",
        expected_insertion_position=1260,
        expected_total_size=5797,
        tags=["alias", "name_resolution", "mammalian", "retroviral"],
    ),
    AgentTestCase(
        id="A2-004",
        name="Alias: 'pGEX' resolution",
        prompt="Insert Renilla luciferase into pGEX and assemble. Return the construct.",
        description="Uses 'pGEX' alias for pGEX-4T-1.",
        expected_backbone_id="pGEX-4T-1",
        expected_insert_id="Renilla_Luciferase",
        expected_insertion_position=930,
        expected_total_size=5905,
        tags=["alias", "name_resolution", "bacterial", "reporter"],
    ),
    AgentTestCase(
        id="A2-005",
        name="Alias: 'pcDNA3.1-' resolution",
        prompt="Put EGFP into pcDNA3.1- and assemble. Return the assembled DNA sequence.",
        description="Uses 'pcDNA3.1-' alias for pcDNA3.1(-).",
        expected_backbone_id="pcDNA3.1(-)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6224,
        tags=["alias", "name_resolution", "mammalian"],
    ),
    # ── A3: Natural language / underspecified ──────────────────────────
    AgentTestCase(
        id="A3-001",
        name="Natural language: mammalian GFP",
        prompt=(
            "I want to express a green fluorescent protein in mammalian cells. "
            "Can you design the plasmid and assemble the full construct?"
        ),
        description=(
            "Underspecified request. Agent should pick a mammalian backbone "
            "(pcDNA3.1(+) is the default) and resolve GFP to EGFP."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["natural_language", "mammalian", "fluorescent_protein"],
    ),
    AgentTestCase(
        id="A3-002",
        name="Natural language: bacterial reporter",
        prompt=(
            "I need a luciferase reporter plasmid for E. coli. "
            "Pick an appropriate bacterial expression vector and assemble the construct."
        ),
        description=(
            "Agent must choose a bacterial backbone (e.g., pGEX-4T-1 or pUC19) "
            "and resolve 'luciferase' to Firefly_Luciferase. Multiple correct answers."
        ),
        expected_backbone_id="pGEX-4T-1",
        expected_insert_id="Firefly_Luciferase",
        expected_insertion_position=930,
        expected_total_size=6622,
        tags=["natural_language", "bacterial", "reporter"],
    ),
    AgentTestCase(
        id="A3-003",
        name="Natural language: retroviral red FP",
        prompt=(
            "I'm doing retroviral transduction and need a red fluorescent protein marker. "
            "Can you build the construct with puromycin selection?"
        ),
        description=(
            "Agent should choose pBABE-puro (retroviral, puromycin selection) "
            "and resolve 'red fluorescent protein' to mCherry."
        ),
        expected_backbone_id="pBABE-puro",
        expected_insert_id="mCherry",
        expected_insertion_position=1260,
        expected_total_size=5797,
        tags=["natural_language", "mammalian", "fluorescent_protein", "retroviral"],
    ),
    # ── A4: Specific insert types ─────────────────────────────────────
    AgentTestCase(
        id="A4-001",
        name="Luciferase reporter",
        prompt="Assemble a firefly luciferase reporter construct in pcDNA3.1(+). Return the sequence.",
        description="Tests a larger insert (1653 bp luciferase).",
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="Firefly_Luciferase",
        expected_insertion_position=895,
        expected_total_size=7081,
        tags=["explicit", "mammalian", "reporter"],
    ),
    AgentTestCase(
        id="A4-002",
        name="FLAG tag insert",
        prompt="Add a FLAG tag to pcDNA3.1(+). Assemble and output the construct.",
        description="Very short insert (24 bp FLAG tag).",
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="FLAG_tag",
        expected_insertion_position=895,
        expected_total_size=5452,
        tags=["explicit", "mammalian", "epitope_tag"],
    ),
    AgentTestCase(
        id="A4-003",
        name="HA tag in pcDNA3.1(-)",
        prompt="Clone an HA tag into pcDNA3.1(-) and assemble the construct. Give me the sequence.",
        description="Short epitope tag in the reverse-orientation backbone.",
        expected_backbone_id="pcDNA3.1(-)",
        expected_insert_id="HA_tag",
        expected_insertion_position=895,
        expected_total_size=5531,
        tags=["explicit", "mammalian", "epitope_tag"],
    ),
    AgentTestCase(
        id="A4-004",
        name="tdTomato large insert",
        prompt="Assemble tdTomato into pBABE-puro and return the sequence.",
        description="Large tandem dimer insert (1431 bp) in a retroviral vector.",
        expected_backbone_id="pBABE-puro",
        expected_insert_id="tdTomato",
        expected_insertion_position=1260,
        expected_total_size=6517,
        tags=["explicit", "mammalian", "fluorescent_protein", "retroviral"],
    ),
    # ── A5: Multi-step workflow ────────────────────────────────────────
    AgentTestCase(
        id="A5-001",
        name="GenBank export workflow",
        prompt=(
            "Design an EGFP expression plasmid using pcDNA3.1(+). "
            "Assemble the construct, validate it, and export as GenBank format."
        ),
        description=(
            "Tests the full 5-step workflow: retrieve, assemble, validate, export. "
            "Agent must call multiple tools in sequence."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["workflow", "mammalian", "fluorescent_protein"],
    ),
    AgentTestCase(
        id="A5-002",
        name="Full workflow: pBABE-puro + mCherry validated",
        prompt=(
            "Design and assemble a mCherry expression construct in pBABE-puro. "
            "Validate the assembled construct and export as FASTA format."
        ),
        description=(
            "Full workflow on a retroviral backbone. Agent must resolve backbone, "
            "assemble, validate, and export — hitting at least 3 tools."
        ),
        expected_backbone_id="pBABE-puro",
        expected_insert_id="mCherry",
        expected_insertion_position=1260,
        expected_total_size=5797,
        tags=["workflow", "mammalian", "fluorescent_protein", "retroviral"],
    ),
    AgentTestCase(
        id="A5-003",
        name="Full workflow: pGEX-4T-1 + EGFP validated",
        prompt=(
            "I need an EGFP-GST fusion construct for bacterial expression. "
            "Use pGEX-4T-1 as the backbone. Assemble, validate, and export as GenBank."
        ),
        description=(
            "Full workflow on a bacterial GST-fusion backbone. "
            "Agent must pick the right backbone from a descriptive request."
        ),
        expected_backbone_id="pGEX-4T-1",
        expected_insert_id="EGFP",
        expected_insertion_position=930,
        expected_total_size=5689,
        tags=["workflow", "bacterial", "fluorescent_protein"],
    ),
]


def get_agent_case_by_id(case_id: str) -> Optional[AgentTestCase]:
    for c in AGENT_CASES:
        if c.id == case_id:
            return c
    return None


def get_agent_cases_by_tag(tag: str) -> list[AgentTestCase]:
    return [c for c in AGENT_CASES if tag in c.tags]


# ── Sequence extraction ────────────────────────────────────────────────


def _find_dna_sequence_in_text(text: str) -> Optional[str]:
    """Find the longest DNA sequence in text output."""
    if not isinstance(text, str):
        return None

    # Look for "Assembled sequence (NNNN bp):" pattern from assemble_construct tool
    match = re.search(r'Assembled sequence \(\d+ bp\):\n([ATCGN\s]+)', text, re.IGNORECASE)
    if match:
        seq = re.sub(r'\s', '', match.group(1)).upper()
        if len(seq) > 100:
            return seq

    # Look for sequence in code blocks
    for block_match in re.finditer(r'```\n?([ATCGN\s]{100,})\n?```', text, re.IGNORECASE):
        seq = re.sub(r'\s', '', block_match.group(1)).upper()
        if set(seq) <= set('ATCGN') and len(seq) > 100:
            return seq

    # Look for a long stretch of pure DNA characters
    for dna_match in re.finditer(r'[ATCGN]{200,}', text, re.IGNORECASE):
        seq = dna_match.group(0).upper()
        return seq

    return None


# ── Agent runner ───────────────────────────────────────────────────────


@dataclass
class AgentTrace:
    """Record of an agent run."""
    prompt: str
    tool_calls: list[dict] = field(default_factory=list)
    assistant_text: str = ""
    assembled_sequence: Optional[str] = None
    total_turns: int = 0
    error: Optional[str] = None
    cost_usd: Optional[float] = None


async def _auto_approve(tool_name, tool_input, context):
    """Auto-approve all tool calls (MCP tools are safe, in-process)."""
    return PermissionResultAllow()


async def run_agent(
    prompt: str,
    model: str = "claude-opus-4-5-20251101",
    max_turns: int = 15,
    verbose: bool = False,
) -> AgentTrace:
    """Run the plasmid design agent on a single prompt using the Agent SDK."""
    trace = AgentTrace(prompt=prompt)
    server_config = create_plasmid_tools()

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"plasmid-library": server_config},
        allowed_tools=ALL_TOOL_NAMES,
        permission_mode="acceptEdits",
        model=model,
        max_turns=max_turns,
        cwd=str(PROJECT_ROOT),
        can_use_tool=_auto_approve,
    )

    all_text_parts = []

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    trace.total_turns += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            all_text_parts.append(block.text)
                            if verbose:
                                preview = block.text[:200]
                                print(f"    [text] {preview}{'...' if len(block.text) > 200 else ''}")
                        elif isinstance(block, ToolUseBlock):
                            trace.tool_calls.append({
                                "tool": block.name,
                                "input": block.input,
                            })
                            if verbose:
                                input_preview = json.dumps(block.input)
                                if len(input_preview) > 200:
                                    input_preview = input_preview[:200] + "..."
                                print(f"    [tool] {block.name}({input_preview})")
                        elif isinstance(block, ToolResultBlock):
                            content_str = str(block.content) if block.content else ""
                            all_text_parts.append(content_str)
                            if verbose:
                                preview = content_str[:200]
                                print(f"    [result] {preview}...")
                elif isinstance(message, UserMessage):
                    # Tool results come as UserMessage with ToolResultBlock items
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock) and isinstance(block.content, list):
                                for item in block.content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text_parts.append(item["text"])
                            elif isinstance(block, ToolResultBlock) and isinstance(block.content, str):
                                all_text_parts.append(block.content)
                elif isinstance(message, ResultMessage):
                    trace.cost_usd = message.total_cost_usd
                    break
    except Exception as e:
        trace.error = str(e)

    trace.assistant_text = "\n".join(all_text_parts)

    # Extract assembled sequence from all collected text
    longest = None
    for part in all_text_parts:
        seq = _find_dna_sequence_in_text(part)
        if seq and (longest is None or len(seq) > len(longest)):
            longest = seq
    trace.assembled_sequence = longest

    return trace


# ── Eval runner ────────────────────────────────────────────────────────


async def run_agent_eval_case(
    tc: AgentTestCase,
    model: str = "claude-opus-4-5-20251101",
    verbose: bool = False,
) -> tuple[Optional[RubricResult], AgentTrace]:
    """Run a single agent eval case."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Case: {tc.id} — {tc.name}")
        print(f"Prompt: {tc.prompt[:100]}...")
        print(f"  Running agent...")

    trace = await run_agent(prompt=tc.prompt, model=model, verbose=verbose)

    if trace.error:
        if verbose:
            print(f"  Agent ERROR: {trace.error}")
        return None, trace

    if not trace.assembled_sequence:
        if verbose:
            print(f"  No assembled sequence found in agent output")
            print(f"  Tool calls made: {[t['tool'] for t in trace.tool_calls]}")
        return None, trace

    if verbose:
        print(f"  Extracted sequence: {len(trace.assembled_sequence)} bp")
        print(f"  Tool calls: {len(trace.tool_calls)} ({', '.join(t['tool'] for t in trace.tool_calls)})")

    # Resolve expected values for scoring
    backbone_data = get_backbone_by_id(tc.expected_backbone_id)
    insert_data = get_insert_by_id(tc.expected_insert_id)

    if not backbone_data or not backbone_data.get("sequence"):
        if verbose:
            print(f"  SKIP: No backbone sequence for '{tc.expected_backbone_id}'")
        return None, trace

    if not insert_data or not insert_data.get("sequence"):
        if verbose:
            print(f"  SKIP: No insert sequence for '{tc.expected_insert_id}'")
        return None, trace

    backbone_seq = backbone_data["sequence"]
    insert_seq = insert_data["sequence"]

    insertion_pos = tc.expected_insertion_position
    if insertion_pos is None:
        insertion_pos = find_mcs_insertion_point(backbone_data)

    if insertion_pos is None:
        if verbose:
            print(f"  SKIP: Cannot determine insertion position")
        return None, trace

    # Score with rubric
    rubric_result = score_construct(
        construct_sequence=trace.assembled_sequence,
        expected_backbone_sequence=backbone_seq,
        expected_insert_sequence=insert_seq,
        expected_insert_position=insertion_pos,
        backbone_name=backbone_data["name"],
        insert_name=insert_data["name"],
        insert_category=insert_data.get("category"),
        backbone_features=backbone_data.get("features"),
    )

    if verbose:
        print(f"  Result: {rubric_result.summary()}")
        print()
        print(rubric_result.report())

    return rubric_result, trace


async def run_agent_eval_suite(
    cases: list[AgentTestCase],
    model: str = "claude-opus-4-5-20251101",
    verbose: bool = False,
) -> dict:
    """Run a suite of agent eval cases."""
    results = []
    passed = 0
    failed = 0
    errored = 0

    for tc in cases:
        start_time = time.time()
        rubric, trace = await run_agent_eval_case(tc, model=model, verbose=verbose)
        elapsed = time.time() - start_time

        if rubric is None:
            errored += 1
            results.append({
                "id": tc.id,
                "name": tc.name,
                "status": "ERROR",
                "score": None,
                "detail": trace.error or "No sequence extracted",
                "tool_calls": len(trace.tool_calls),
                "turns": trace.total_turns,
                "elapsed_s": round(elapsed, 1),
                "cost_usd": trace.cost_usd,
            })
        elif rubric.overall_pass:
            passed += 1
            results.append({
                "id": tc.id,
                "name": tc.name,
                "status": "PASS",
                "score": rubric.score_pct,
                "detail": rubric.summary(),
                "tool_calls": len(trace.tool_calls),
                "turns": trace.total_turns,
                "elapsed_s": round(elapsed, 1),
                "cost_usd": trace.cost_usd,
            })
        else:
            failed += 1
            results.append({
                "id": tc.id,
                "name": tc.name,
                "status": "FAIL",
                "score": rubric.score_pct,
                "detail": rubric.summary(),
                "critical_fail": rubric.critical_fail,
                "tool_calls": len(trace.tool_calls),
                "turns": trace.total_turns,
                "elapsed_s": round(elapsed, 1),
                "cost_usd": trace.cost_usd,
            })

    summary = {
        "total": len(cases),
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "pass_rate": round(passed / (passed + failed) * 100, 1) if (passed + failed) > 0 else 0,
        "model": model,
    }

    return {"results": results, "summary": summary}


def print_agent_summary_table(eval_output: dict):
    """Print a compact summary table."""
    results = eval_output["results"]
    summary = eval_output["summary"]

    print(f"\n{'='*80}")
    print(f"AGENT EVALUATION RESULTS (model: {summary['model']})")
    print(f"{'='*80}")
    print(f"{'ID':<8} {'Name':<35} {'Status':>6} {'Score':>7} {'Tools':>5} {'Time':>6}")
    print(f"{'-'*80}")

    for r in results:
        score_str = f"{r['score']}%" if r['score'] is not None else "—"
        time_str = f"{r['elapsed_s']}s"
        print(f"{r['id']:<8} {r['name']:<35} {r['status']:>6} {score_str:>7} {r['tool_calls']:>5} {time_str:>6}")

    print(f"{'-'*80}")
    print(
        f"Total: {summary['total']}  |  "
        f"Passed: {summary['passed']}  |  "
        f"Failed: {summary['failed']}  |  "
        f"Errors: {summary['errored']}  |  "
        f"Pass Rate: {summary['pass_rate']}%"
    )
    print(f"{'='*80}")


# ── CLI ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end agent evaluations (Agent SDK)")
    parser.add_argument("--case", type=str, help="Run a single test case by ID (e.g., A1-001)")
    parser.add_argument("--tag", type=str, help="Run cases matching this tag")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full agent trace")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--model", type=str, default="claude-opus-4-5-20251101",
        help="Model to use (default: sonnet)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(1)

    # Select cases
    if args.case:
        tc = get_agent_case_by_id(args.case)
        if not tc:
            print(f"Test case '{args.case}' not found.", file=sys.stderr)
            print(f"Available: {', '.join(c.id for c in AGENT_CASES)}", file=sys.stderr)
            sys.exit(1)
        cases = [tc]
    elif args.tag:
        cases = get_agent_cases_by_tag(args.tag)
    else:
        cases = AGENT_CASES

    if not cases:
        print("No test cases matched the filter.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(cases)} agent eval case(s) with model {args.model}...")

    eval_output = asyncio.run(run_agent_eval_suite(cases, model=args.model, verbose=args.verbose))

    if args.json:
        print(json.dumps(eval_output, indent=2))
    else:
        print_agent_summary_table(eval_output)


if __name__ == "__main__":
    main()
