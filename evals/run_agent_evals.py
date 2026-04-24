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
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

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
from src.tools import build_mcp_servers, create_plasmid_tools, ALL_TOOL_NAMES
from src.library import (
    get_backbone_by_id,
    get_insert_by_id,
    set_library_readonly,
    register_test_fixtures,
    clear_test_fixtures,
)
from src.assembler import find_mcs_insertion_point
from evals.rubric import score_construct, RubricResult, Check
from evals.simulated_user import SimulatedUser
from evals.llm_judge import LLMJudge, JudgeResult

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
    # For NCBI/fusion cases: provide the expected insert sequence directly
    # so rubric scoring works offline without needing an NCBI call at score time.
    expected_insert_sequence: Optional[str] = None
    # Transcript-level assertions: strings the agent's text output should contain.
    # Used for disambiguation evals where the agent should ASK clarifying questions
    # (e.g., "which species", "which TRAF"). Graded from the transcript, not the
    # assembled sequence, per the blog's conversational agent pattern.
    transcript_assertions: list[str] = field(default_factory=list)
    # Tool assertions: tools the agent SHOULD NOT call (negative cases).
    # Per the blog's "balanced problem sets" principle (Step 3): test both
    # directions to avoid one-sided optimization (e.g., agent over-triggering NCBI).
    tools_should_not_use: list[str] = field(default_factory=list)
    # Simulated user persona for multi-turn disambiguation evals.
    # When set, a SimulatedUser responds to the agent's clarifying questions.
    user_persona: Optional[str] = None
    # When True, the rubric expects the insert in reverse complement orientation.
    expect_reverse_complement: bool = False
    # Alternative valid backbones: list of dicts with backbone_id,
    # insertion_position, total_size. If the primary backbone fails scoring,
    # each alternative is tried and the best result is used.
    alternative_expected: list[dict] = field(default_factory=list)
    # Fusion parts for linker checks: list of dicts ordered N-terminal to
    # C-terminal. Each dict: {"name": str, "sequence": str, "type": "protein" | "tag"}.
    # When provided, rubric runs fusion linker checks (tag junctions skip linker
    # requirement, protein-protein junctions require a linker).
    fusion_parts: list[dict] = field(default_factory=list)
    # Grading mode: controls how the case is scored.
    #   "sequence"   — (default) full rubric against library backbone+insert
    #   "ncbi"       — extract insert from construct by length (library insert
    #                  may be unavailable or may differ species/isoform)
    #   "transcript" — no sequence scoring; pass/fail on transcript_assertions
    #                  + tools_should_not_use only (for analysis-only /
    #                  promoter-swap cases where no ground-truth insert exists)
    grading_mode: Literal["sequence", "ncbi", "transcript"] = "sequence"
    # Hard upper bound on tool calls. If set, exceeding this count adds a
    # Critical rubric failure. Use for cases that specifically test the agent
    # does NOT loop (e.g., bespoke-promoter "no loop" cases). Without this,
    # transcript-mode cases have no defense against the agent passing keyword
    # checks while burning 40+ tool calls in a search loop.
    max_tool_calls: Optional[int] = None


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
        expected_total_size=6148,  # pcDNA3.1(-) 5428bp + EGFP 720bp
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
        alternative_expected=[
            {"backbone_id": "pUC19", "insertion_position": 631, "total_size": 4339},
        ],
        user_persona=(
            "You want firefly luciferase. Any standard bacterial expression "
            "vector is fine — if asked to pick, say pGEX-4T-1. Proceed with "
            "assembly when the design is presented."
        ),
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
        user_persona=(
            "You want mCherry — it's the standard monomeric red FP. pBABE-puro "
            "is exactly right for retroviral + puromycin. Proceed with assembly."
        ),
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
        user_persona=(
            "You want the FLAG tag inserted into the MCS by itself, exactly as "
            "it is in the library. No fusion, no extra codons. Just the raw "
            "FLAG tag sequence into the backbone. If the agent says it needs "
            "ATG or stop codons, say no — just insert the tag as-is."
        ),
    ),
    AgentTestCase(
        id="A4-003",
        name="HA tag in pcDNA3.1(-)",
        prompt="Clone an HA tag into pcDNA3.1(-) and assemble the construct. Give me the sequence.",
        description="Short epitope tag in pcDNA3.1(-) backbone. (+)/(-) refers to MCS direction, not insert orientation.",
        expected_backbone_id="pcDNA3.1(-)",
        expected_insert_id="HA_tag",
        expected_insertion_position=895,
        expected_total_size=5455,  # pcDNA3.1(-) 5428bp + HA_tag 27bp
        tags=["explicit", "mammalian", "epitope_tag"],
        user_persona=(
            "You want the HA tag inserted into the MCS by itself, exactly as "
            "it is in the library. No fusion, no extra codons. Just the raw "
            "tag sequence into the backbone."
        ),
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
    # ── A6: NCBI Gene Retrieval (Allen Institute sample prompts) ─────────
    # These are the exact prompts from the Allen Institute reference doc.
    # They test NCBI gene retrieval, species disambiguation, gene family
    # disambiguation, variant disambiguation, and alternative name resolution.
    # Grading note: A6 cases test LLM agent behavior (tool selection,
    # clarification prompting, NCBI retrieval). Rubric scoring requires
    # expected_insert_sequence since these genes are not in the local library.
    # Disambiguation cases (A6-002 through A6-006) may ask clarifying questions
    # rather than immediately assembling — scoring should account for this.
    AgentTestCase(
        id="A6-001",
        name="Allen: sfGFP in pcDNA3.1(+)",
        prompt=(
            "Using pcDNA3.1(+) as a backbone, design a plasmid to express "
            "Super Folder GFP reporter expression in HEK293 Cells"
        ),
        description=(
            "Allen sample prompt #1. sfGFP is not in the local library. Agent must "
            "use NCBI to retrieve the sfGFP CDS. Since backbone is specified, agent "
            "should skip expression type/level questions (smart skip)."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="sfGFP",
        tags=["ncbi", "mammalian", "fluorescent_protein", "allen_sample", "smart_skip"],
    ),
    AgentTestCase(
        id="A6-002",
        name="Allen: MyD88 species disambiguation",
        prompt="Design a vector to allow expression of MyD88 in RAW 264 cells",
        description=(
            "Allen sample prompt #2. MyD88 exists in multiple species. RAW 264 cells "
            "are mouse macrophages. Agent should prompt for: (1) which organism the "
            "sequence should come from, (2) expression type and level. Should NOT "
            "return a tagged version (e.g., pCMV-HA-MyD88) without user confirmation."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="Myd88",
        tags=["ncbi", "mammalian", "species_disambiguation", "allen_sample"],
        transcript_assertions=[
            "species",      # agent should ask about species/organism
            "mouse",        # agent should recognize RAW 264 = mouse
        ],
        user_persona=(
            "You want to express mouse MyD88. When asked about species, "
            "specify mouse. When asked about expression, say transient "
            "high expression."
        ),
    ),
    AgentTestCase(
        id="A6-003",
        name="Allen: TRAF gene family disambiguation",
        prompt="Design a vector to allow transient overexpression of TRAF in Raw 264.7",
        description=(
            "Allen sample prompt #3. TRAF is a gene family (TRAF1-7). Agent should "
            "prompt for: (1) which TRAF protein, (2) which organism. Since the user "
            "specified transient overexpression, agent should pick a strong "
            "constitutive backbone directly (smart skip)."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="Traf6",
        tags=["ncbi", "mammalian", "gene_family_disambiguation", "allen_sample", "smart_skip"],
        transcript_assertions=[
            "TRAF",         # agent should mention TRAF family members
            "which",        # agent should ask which one
        ],
        user_persona=(
            "You want TRAF6 specifically. When asked which TRAF, say TRAF6. "
            "The sequence should come from mouse."
        ),
    ),
    AgentTestCase(
        id="A6-004",
        name="Allen: TRAF by full name (alternative name resolution)",
        prompt=(
            "Design a vector to allow transient overexpression of TNF receptor "
            "associated factor in Raw 264.7 Cells"
        ),
        description=(
            "Allen sample prompt #4. 'TNF receptor associated factor' is the full "
            "name for TRAF. Agent should resolve this to the TRAF family and produce "
            "the same outcome as A6-003. Must NOT retrieve TNF Receptor itself."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="Traf6",
        tags=["ncbi", "mammalian", "alternative_name", "allen_sample", "smart_skip"],
        transcript_assertions=[
            "TRAF",         # agent should recognize TNF receptor associated factor = TRAF
        ],
        user_persona=(
            "You want TRAF6 specifically. When asked which TRAF, say TRAF6. "
            "The sequence should come from mouse."
        ),
    ),
    AgentTestCase(
        id="A6-005",
        name="Allen: RFP variant disambiguation",
        prompt="Design an expression vector for expression of RFP in human cells",
        description=(
            "Allen sample prompt #5. RFP is ambiguous — could be mCherry, tdTomato, "
            "mScarlet, DsRed. Agent should prompt for: (1) which RFP variant, "
            "(2) expression type, (3) expression level."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mCherry",
        tags=["ncbi", "mammalian", "variant_disambiguation", "allen_sample"],
        transcript_assertions=[
            "mCherry",      # agent should mention specific variants
            "which",        # agent should ask which variant
        ],
        user_persona=(
            "You want mCherry as your RFP. When asked which variant, say "
            "mCherry. Expression: transient, high."
        ),
    ),
    AgentTestCase(
        id="A6-006",
        name="Allen: SERPINE1/PAI-1 alternative name",
        prompt=(
            "Design a plasmid to express PAI-1 in HEK293 cells using pcDNA3.1(+). "
            "Assemble and return the sequence."
        ),
        description=(
            "Allen testing requirement: alternative gene name resolution. PAI-1 is "
            "an alternative name for SERPINE1 (also known as Planh1). Agent should "
            "resolve PAI-1 to SERPINE1 via NCBI alias data and retrieve the CDS."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="SERPINE1",
        tags=["ncbi", "mammalian", "alternative_name", "allen_sample"],
        grading_mode="ncbi",
        user_persona=(
            "Human SERPINE1 (PAI-1). pcDNA3.1(+) is fine. Transient expression "
            "in HEK293. Proceed with assembly."
        ),
    ),
    AgentTestCase(
        id="A6-007",
        name="NL backbone: mammalian vector with neomycin resistance",
        prompt=(
            "I need a mammalian expression vector with neomycin resistance to "
            "express EGFP. Assemble the construct and return the sequence."
        ),
        description=(
            "Allen sprint goal: natural language backbone selection. Agent should "
            "select a backbone matching the criteria (e.g., pcDNA3.1(+) has CMV "
            "promoter and neomycin/G418 mammalian selection) rather than asking "
            "the user to pick a specific backbone."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["natural_language", "mammalian", "backbone_selection", "allen_sample"],
    ),
    # ── A7: Protein Tagging / Fusions ─────────────────────────────────
    AgentTestCase(
        id="A7-001",
        name="Fusion: N-terminal FLAG-EGFP",
        prompt=(
            "Add an N-terminal FLAG tag to EGFP in pcDNA3.1(+). "
            "Assemble the construct and return the sequence."
        ),
        description=(
            "Agent must fuse FLAG_tag + EGFP using fuse_inserts, then assemble "
            "the fusion CDS into pcDNA3.1(+). Tests the fuse_inserts tool and "
            "correct codon management. Agent adds ATG before FLAG (which lacks "
            "one) and EGFP provides the stop codon."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6172,
        # Fused CDS: ATG + FLAG_tag (24bp) + EGFP (720bp, remove ATG, keep stop) = 744 bp
        expected_insert_sequence=(
            "ATGGACTACAAGGACGACGATGACAAGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTG"
            "CCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGC"
            "GAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCC"
            "GTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCC"
            "GACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGC"
            "ACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGAC"
            "ACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGG"
            "CACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAAC"
            "GGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGAC"
            "CACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTG"
            "AGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAG"
            "TTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
        ),
        tags=["fusion", "mammalian", "epitope_tag", "n_terminal"],
        fusion_parts=[
            {"name": "FLAG_tag", "sequence": "GACTACAAGGACGACGATGACAAG", "type": "tag"},
            {"name": "EGFP", "sequence": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA", "type": "protein"},
        ],
    ),
    AgentTestCase(
        id="A7-002",
        name="Fusion: C-terminal mCherry-HA",
        prompt=(
            "Express a C-terminal HA-tagged mCherry in pcDNA3.1(+). "
            "Assemble the construct and return the sequence."
        ),
        description=(
            "Agent must fuse mCherry + HA_tag using fuse_inserts, then assemble "
            "into pcDNA3.1(+). Tests C-terminal fusion with correct codon management. "
            "Agent adds stop codon after HA (which lacks one)."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mCherry",
        expected_insertion_position=895,
        expected_total_size=6166,
        # Fused CDS: mCherry (711bp, remove stop) + HA_tag (27bp) + TAA stop = 738 bp
        expected_insert_sequence=(
            "ATGGTGAGCAAGGGCGAGGAGGATAACATGGCCATCATCAAGGAGTTCATGCGCTTCAAGGTG"
            "CACATGGAGGGCTCCGTGAACGGCCACGAGTTCGAGATCGAGGGCGAGGGCGAGGGCCGCCCC"
            "TACGAGGGCACCCAGACCGCCAAGCTGAAGGTGACCAAGGGTGGCCCCCTGCCCTTCGCCTGG"
            "GACATCCTGTCCCCTCAGTTCATGTACGGCTCCAAGGCCTACGTGAAGCACCCCGCCGACATC"
            "CCCGACTACTTGAAGCTGTCCTTCCCCGAGGGCTTCAAGTGGGAGCGCGTGATGAACTTCGAG"
            "GACGGCGGCGTGGTGACCGTGACCCAGGACTCCTCCCTGCAGGACGGCGAGTTCATCTACAAG"
            "GTGAAGCTGCGCGGCACCAACTTCCCCTCCGACGGCCCCGTAATGCAGAAGAAGACCATGGGC"
            "TGGGAGGCCTCCTCCGAGCGGATGTACCCCGAGGACGGCGCCCTGAAGGGCGAGATCAAGCAG"
            "AGGCTGAAGCTGAAGGACGGCGGCCACTACGACGCTGAGGTCAAGACCACCTACAAGGCCAAG"
            "AAGCCCGTGCAGCTGCCCGGCGCCTACAACGTCAACATCAAGTTGGACATCACCTCCCACAAC"
            "GAGGACTACACCATCGTGGAACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATG"
            "GACGAGCTGTACAAGTACCCATACGATGTTCCAGATTACGCTTAA"
        ),
        tags=["fusion", "mammalian", "epitope_tag", "c_terminal"],
        fusion_parts=[
            {"name": "mCherry", "sequence": "ATGGTGAGCAAGGGCGAGGAGGATAACATGGCCATCATCAAGGAGTTCATGCGCTTCAAGGTGCACATGGAGGGCTCCGTGAACGGCCACGAGTTCGAGATCGAGGGCGAGGGCGAGGGCCGCCCCTACGAGGGCACCCAGACCGCCAAGCTGAAGGTGACCAAGGGTGGCCCCCTGCCCTTCGCCTGGGACATCCTGTCCCCTCAGTTCATGTACGGCTCCAAGGCCTACGTGAAGCACCCCGCCGACATCCCCGACTACTTGAAGCTGTCCTTCCCCGAGGGCTTCAAGTGGGAGCGCGTGATGAACTTCGAGGACGGCGGCGTGGTGACCGTGACCCAGGACTCCTCCCTGCAGGACGGCGAGTTCATCTACAAGGTGAAGCTGCGCGGCACCAACTTCCCCTCCGACGGCCCCGTAATGCAGAAGAAGACCATGGGCTGGGAGGCCTCCTCCGAGCGGATGTACCCCGAGGACGGCGCCCTGAAGGGCGAGATCAAGCAGAGGCTGAAGCTGAAGGACGGCGGCCACTACGACGCTGAGGTCAAGACCACCTACAAGGCCAAGAAGCCCGTGCAGCTGCCCGGCGCCTACAACGTCAACATCAAGTTGGACATCACCTCCCACAACGAGGACTACACCATCGTGGAACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTAA", "type": "protein"},
            {"name": "HA_tag", "sequence": "TACCCATACGATGTTCCAGATTACGCT", "type": "tag"},
        ],
    ),
    AgentTestCase(
        id="A7-003",
        name="Fusion: H2B-EGFP (NCBI + fusion)",
        prompt=(
            "Create a mammalian expression plasmid for a fusion of H2B to eGFP, "
            "where eGFP is on the C-terminal end of H2B. Assemble the construct and return the sequence."
        ),
        description=(
            "Agent must retrieve H2B CDS from NCBI, and an appropriate linker sequence, then fuse H2B + linker + EGFP using "
            "fuse_inserts (H2B on N-terminal, EGFP on C-terminal), and assemble "
            "into a mammalian backbone. First eval combining NCBI retrieval with "
            "protein fusion. H2B is not in the local library."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="H2B",
        expected_insertion_position=895,
        expected_insert_sequence=(
            "atgccagagccagcgaagtctgctcccgccccgaaaaagggctccaagaaggcggtgactaaggcgcagaagaaaggcggcaagaagcgcaagcgcagccgcaaggagagctattccatctatgtgtacaaggttctgaagcaggtccaccctgacaccggcatttcgtccaaggccatgggcatcatgaattcgtttgtgaacgacattttcgagcgcatcgcaggtgaggcttcccgcctggcgcattacaacaagcgctcgaccatcacctccagggagatccagacggccgtgcgcctgctgctgcctggggagttggccaagcacgccgtgtccgagggtactaaggccatcaccaagtacaccagcgctaagGGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGCGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
        ),
        fusion_parts=[
            {"name": "H2B", "sequence": "atgccagagccagcgaagtctgctcccgccccgaaaaagggctccaagaaggcggtgactaagg"
            "cgcagaagaaaggcggcaagaagcgcaagcgcagccgcaaggagagctattccatctatgtgta"
            "caaggttctgaagcaggtccaccctgacaccggcatttcgtccaaggccatgggcatcatgaat"
            "tcgtttgtgaacgacattttcgagcgcatcgcaggtgaggcttcccgcctggcgcattacaaca"
            "agcgctcgaccatcacctccagggagatccagacggccgtgcgcctgctgctgcctggggagtt"
            "ggccaagcacgccgtgtccgagggtactaaggccatcaccaagtacaccagcgctaag", "type": "protein"},
            {"name": "Gly4Ser20_Flexible_linker", "sequence": "GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC", "type": "linker"},
            {"name": "EGFP", "sequence": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA", "type": "protein"}],
        tags=["fusion", "ncbi", "mammalian", "c_terminal"],
        expected_total_size=6592,
        user_persona=(
            "You want human H2B (H2B1B). There are many Histone H2B variants — make sure to get this one. "
            "You can use a common Glycine4 Serine flexible linker (e.g., GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC) between H2B and EGFP. "
            "When asked about species, say human. The backbone should be for "
            "mammalian expression — pcDNA3.1(+) is fine."
        ),
    ),
    AgentTestCase(
        id="A7-004",
        name="Fusion: H2B-EGFP with user-specified custom linker",
        prompt=(
            "Create a mammalian expression plasmid for H2B-eGFP. "
            "I want to use GATCCACCGGTC as the linker sequence between H2B and eGFP. "
            "Assemble the construct and return the sequence."
        ),
        description=(
            "Agent must retrieve H2B CDS from NCBI, use the user-provided custom linker "
            "(GATCCACCGGTC, 12 bp) instead of the default (GGGGS)x4, then fuse H2B + linker + EGFP "
            "using fuse_inserts and assemble into a mammalian backbone. Tests that the agent "
            "respects a non-default linker and that Kozak (GCCACC) is still appended after it."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="H2B",
        expected_insertion_position=895,
        expected_insert_sequence=(
            "ATGCCAGAGCCAGCGAAGTCTGCTCCCGCCCCGAAAAAGGGCTCCAAGAAGGCGGTGACTAAGG"
            "CGCAGAAGAAAGGCGGCAAGAAGCGCAAGCGCAGCCGCAAGGAGAGCTATTCCATCTATGTGTA"
            "CAAGGTTCTGAAGCAGGTCCACCCTGACACCGGCATTTCGTCCAAGGCCATGGGCATCATGAAT"
            "TCGTTTGTGAACGACATTTTCGAGCGCATCGCAGGTGAGGCTTCCCGCCTGGCGCATTACAACA"
            "AGCGCTCGACCATCACCTCCAGGGAGATCCAGACGGCCGTGCGCCTGCTGCTGCCTGGGGAGTT"
            "GGCCAAGCACGCCGTGTCCGAGGGTACTAAGGCCATCACCAAGTACACCAGCGCTAAG"
            "GATCCACCGGTCGCCACC"
            "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGC"
            "GAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACC"
            "CTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAG"
            "CGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGC"
            "ATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACG"
            "GCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCC"
            "CGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTG"
            "ACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
        ),
        fusion_parts=[
            {"name": "H2B", "sequence": "atgccagagccagcgaagtctgctcccgccccgaaaaagggctccaagaaggcggtgactaagg"
            "cgcagaagaaaggcggcaagaagcgcaagcgcagccgcaaggagagctattccatctatgtgta"
            "caaggttctgaagcaggtccaccctgacaccggcatttcgtccaaggccatgggcatcatgaat"
            "tcgtttgtgaacgacattttcgagcgcatcgcaggtgaggcttcccgcctggcgcattacaaca"
            "agcgctcgaccatcacctccagggagatccagacggccgtgcgcctgctgctgcctggggagtt"
            "ggccaagcacgccgtgtccgagggtactaaggccatcaccaagtacaccagcgctaag", "type": "protein"},
            {"name": "custom_linker", "sequence": "GATCCACCGGTC", "type": "linker"},
            {"name": "EGFP", "sequence": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA", "type": "protein"},
        ],
        tags=["fusion", "ncbi", "mammalian", "c_terminal", "custom_linker"],
        expected_total_size=6544,
        user_persona=(
            "You want human H2B (H2B1B). There are many Histone H2B variants — make sure to get this one. "
            "When asked about species, say human. The backbone should be for "
            "mammalian expression — pcDNA3.1(+) is fine."
        ),
    ),
    AgentTestCase(
        id="A7-005",
        name="Fusion: H2B-EGFP (NCBI + fusion), relaxed natural language",
        prompt=(
            "Create a mammalian expression plasmid for H2B-eGFP, "
            "Assemble the construct and return the sequence."
        ),
        description=(
            "Agent must retrieve H2B CDS from NCBI, and an appropriate linker sequence, then fuse H2B + linker + EGFP using "
            "fuse_inserts (H2B on N-terminal, EGFP on C-terminal), and assemble "
            "into a mammalian backbone. First eval combining NCBI retrieval with "
            "protein fusion. H2B is not in the local library."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="H2B",
        expected_insertion_position=895,
        expected_insert_sequence=(
            "atgccagagccagcgaagtctgctcccgccccgaaaaagggctccaagaaggcggtgactaaggcgcagaagaaaggcggcaagaagcgcaagcgcagccgcaaggagagctattccatctatgtgtacaaggttctgaagcaggtccaccctgacaccggcatttcgtccaaggccatgggcatcatgaattcgtttgtgaacgacattttcgagcgcatcgcaggtgaggcttcccgcctggcgcattacaacaagcgctcgaccatcacctccagggagatccagacggccgtgcgcctgctgctgcctggggagttggccaagcacgccgtgtccgagggtactaaggccatcaccaagtacaccagcgctaagGGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGCGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
        ),
        fusion_parts=[
            {"name": "H2B", "sequence": "atgccagagccagcgaagtctgctcccgccccgaaaaagggctccaagaaggcggtgactaagg"
            "cgcagaagaaaggcggcaagaagcgcaagcgcagccgcaaggagagctattccatctatgtgta"
            "caaggttctgaagcaggtccaccctgacaccggcatttcgtccaaggccatgggcatcatgaat"
            "tcgtttgtgaacgacattttcgagcgcatcgcaggtgaggcttcccgcctggcgcattacaaca"
            "agcgctcgaccatcacctccagggagatccagacggccgtgcgcctgctgctgcctggggagtt"
            "ggccaagcacgccgtgtccgagggtactaaggccatcaccaagtacaccagcgctaag", "type": "protein"},
            {"name": "Gly4Ser20_Flexible_linker", "sequence": "GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC", "type": "linker"},
            {"name": "EGFP", "sequence": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA", "type": "protein"}],
        tags=["fusion", "ncbi", "mammalian", "c_terminal"],
        expected_total_size=6592,
        user_persona=(
            "You want human H2B (H2B1B). There are many Histone H2B variants — make sure to get this one. "
            "You can use a common Glycine4 Serine flexible linker (e.g., GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC) between H2B and EGFP. "
            "When asked about species, say human. The backbone should be for "
            "mammalian expression — pcDNA3.1(+) is fine."
        ),
    ),
    # ── A8: Negative / balanced cases (blog Step 3) ───────────────────
    # Per the blog: "Test both the cases where a behavior should occur and
    # where it shouldn't. One-sided evals create one-sided optimization."
    # These test that the agent does NOT over-trigger NCBI or fuse_inserts
    # when the insert is already in the local library.
    AgentTestCase(
        id="A8-001",
        name="Negative: EGFP should NOT trigger NCBI",
        prompt=(
            "Put EGFP into pcDNA3.1(+) and assemble the construct. "
            "Return the assembled sequence."
        ),
        description=(
            "EGFP is in the local library. Agent should use get_insert, NOT "
            "search_gene/fetch_gene. Tests that NCBI is not over-triggered."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["negative", "balanced", "no_ncbi"],
        tools_should_not_use=["search_gene", "fetch_gene"],
    ),
    AgentTestCase(
        id="A8-002",
        name="Negative: mCherry should NOT trigger NCBI",
        prompt=(
            "Assemble mCherry into pcDNA3.1(+). Return the sequence."
        ),
        description=(
            "mCherry is in the local library. Agent should not call NCBI tools."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mCherry",
        expected_insertion_position=895,
        expected_total_size=6139,
        tags=["negative", "balanced", "no_ncbi"],
        tools_should_not_use=["search_gene", "fetch_gene"],
    ),
    AgentTestCase(
        id="A8-003",
        name="Negative: simple EGFP should NOT fuse",
        prompt=(
            "Design an EGFP expression plasmid using pcDNA3.1(+). "
            "Assemble and return the sequence."
        ),
        description=(
            "Plain EGFP expression — no tagging requested. Agent should NOT "
            "call fuse_inserts. Tests that fusion is not over-triggered."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["negative", "balanced", "no_fusion"],
        tools_should_not_use=["fuse_inserts"],
    ),

    # ══════════════════════════════════════════════════════════════════
    # P1: Phase 1 Acceptance Prompts (Allen Institute Reference Doc)
    # ══════════════════════════════════════════════════════════════════
    # These 7 prompts are the acceptance criteria for Phase 1 completion.
    # Each exercises a specific disambiguation or routing behavior.

    AgentTestCase(
        id="P1-SP1",
        name="sfGFP in pcDNA3.1(+) for HEK293",
        prompt=(
            "Design an expression vector for Super Folder GFP (sfGFP) "
            "using pcDNA3.1(+) for transient expression in HEK293 cells."
        ),
        description=(
            "Backbone explicitly specified → no backbone clarification "
            "needed. sfGFP IS in library. Should assemble directly."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="sfGFP",
        expected_insertion_position=895,
        tags=["phase1", "acceptance", "explicit_backbone"],
        tools_should_not_use=["search_addgene"],  # sfGFP is local
    ),

    AgentTestCase(
        id="P1-SP2",
        name="MyD88 in RAW 264 cells (species + backbone inference)",
        prompt="Design an expression vector for MyD88 in RAW 264 cells.",
        description=(
            "No backbone specified → agent must ask about expression "
            "type/level. RAW 264 = mouse → agent should infer or confirm "
            "mouse MyD88. Tests cell-line species inference."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="Myd88",
        tags=["phase1", "acceptance", "disambiguation", "cell_line", "multiturn"],
        grading_mode="ncbi",  # mouse MyD88 from NCBI, not library
        user_persona=(
            "You want mouse MyD88 (since RAW 264.7 is a mouse cell line). "
            "For expression: transient, strong constitutive, any standard "
            "mammalian backbone like pcDNA3.1(+) is fine."
        ),
        transcript_assertions=["species", "backbone"],  # should discuss both
    ),

    AgentTestCase(
        id="P1-SP3",
        name="TRAF transient overexpression in Raw 264.7 (family disambiguation)",
        prompt=(
            "Design a vector for TRAF transient overexpression in Raw 264.7 cells."
        ),
        description=(
            "TRAF is a family (1-7), not a gene. Agent MUST ask which "
            "family member. 'transient overexpression' answers the "
            "backbone-selection questions."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="TRAF6",
        tags=["phase1", "acceptance", "gene_family", "disambiguation", "multiturn"],
        grading_mode="ncbi",  # mouse TRAF6 from NCBI, not library
        user_persona=(
            "You want TRAF6 (the well-studied one in innate immunity). "
            "Mouse, since RAW 264.7 is mouse. pcDNA3.1(+) is fine."
        ),
        transcript_assertions=["TRAF6", "TRAF"],  # must present options + use choice
    ),

    AgentTestCase(
        id="P1-SP4",
        name="TNF receptor associated factor long-name recognition",
        prompt=(
            "Design a vector for TNF receptor associated factor "
            "transient overexpression in Raw 264.7."
        ),
        description=(
            "Same as SP3 but uses the long name. Agent must recognize "
            "this as TRAF family, NOT fetch the TNF Receptor gene. "
            "Critical alias resolution test."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="TRAF6",
        tags=["phase1", "acceptance", "alias", "gene_family", "multiturn"],
        grading_mode="ncbi",  # mouse TRAF6 from NCBI, not library
        user_persona=(
            "You want TRAF6. Mouse. pcDNA3.1(+) is fine."
        ),
        transcript_assertions=["TRAF"],
        # Must NOT fetch TNFR/TNFRSF genes by mistake
    ),

    AgentTestCase(
        id="P1-SP5",
        name="RFP in human cells (FP variant disambiguation)",
        prompt="Design an expression vector for RFP in human cells.",
        description=(
            "RFP is not a specific protein. Agent MUST ask which variant "
            "(mCherry, tdTomato, mScarlet, etc.). Also needs backbone info."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mCherry",
        tags=["phase1", "acceptance", "fp_variant", "disambiguation", "multiturn"],
        user_persona=(
            "You want mCherry — it's the standard choice. Transient, "
            "strong constitutive, pcDNA3.1(+) is fine."
        ),
        transcript_assertions=["mCherry"],
    ),

    AgentTestCase(
        id="P1-SP6",
        name="eGFP driven by SV40 in pcDNA3.1(+) (promoter conflict)",
        prompt=(
            "Design an expression vector for eGFP (driven by SV40) "
            "using pcDNA3.1(+)."
        ),
        description=(
            "pcDNA3.1(+) already contains SV40 driving NeoR. Agent should "
            "detect this conflict and offer alternatives. This tests "
            "backbone-feature awareness."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        tags=["phase1", "acceptance", "promoter_conflict", "multiturn"],
        user_persona=(
            "Good catch on the SV40 conflict. Just use the CMV promoter "
            "that's already in pcDNA3.1(+) instead."
        ),
        transcript_assertions=["SV40"],  # should mention the conflict
    ),

    AgentTestCase(
        id="P1-SP7",
        name="mRuby in HEK293 (FPbase routing, no hallucination, fail-closed)",
        prompt=(
            "Design an expression vector for mRuby in HEK293 cells, "
            "transient expression."
        ),
        description=(
            "mRuby is NOT in local library and NOT a natural gene. "
            "Previously the agent hallucinated or fetched wrong NCBI result. "
            "Now: FPbase routing confirms it's a real FP. Two valid outcomes: "
            "(a) FPbase has DNA -> agent retrieves and assembles directly; "
            "(b) FPbase has only AA -> agent tells user exactly what it found "
            "and asks for DNA (fail-closed, no synthesis). Both pass the "
            "'no hallucination' criterion — the key is the agent NEVER goes "
            "to NCBI Gene for an engineered FP and NEVER synthesizes DNA."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mRuby",
        tags=["phase1", "acceptance", "fpbase", "no_hallucination", "multiturn"],
        grading_mode="ncbi",  # mRuby from FPbase/Addgene extraction, not library
        user_persona=(
            "pcDNA3.1(+) is fine. If you find multiple mRuby variants, "
            "I want the original mRuby (not mRuby2 or mRuby3). "
            "If FPbase doesn't have the DNA sequence, suggest I look at "
            "Addgene plasmid #40260 (pcDNA3-mRuby) and you can extract the "
            "mRuby CDS from there."
        ),
        transcript_assertions=["FPbase", "mRuby"],  # must route via FPbase
        tools_should_not_use=["search_gene"],  # must NOT try NCBI Gene
    ),

    # ══════════════════════════════════════════════════════════════════
    # P2: Phase 2 Advanced Design Features (Anthropic-assigned)
    # ══════════════════════════════════════════════════════════════════
    # These cases exercise the 5 Anthropic-assigned features from the
    # Allen Institute Epics & Prioritization doc:
    #   - Design Confidence Score (cryptic signals, CAI, Kozak, etc.)
    #   - Bespoke Promoters (detect non-standard, offer 3 options, no loop)
    #   - Intelligent Fusion Design (disorder-based internal fusion sites)
    #   - Smart Mutation Design (curated GoF/LoF lookup + deterministic edit)
    #   - Troubleshooting Mode (diagnose failure → propose remediation)
    #
    # Most of these are TRANSCRIPT-assertion evals (did the agent say/do
    # the right thing?), not sequence-correctness evals. expected_backbone_id
    # and expected_insert_id are still required by the schema but the key
    # grading signal is transcript_assertions + tool usage.

    # ── P2-DCS: Design Confidence Score ────────────────────────────────
    AgentTestCase(
        id="P2-DCS1",
        name="Confidence score surfaces cryptic polyA warning",
        prompt=(
            "I have a custom CDS I want to express in HEK293 cells using "
            "pcDNA3.1(+). Before I proceed, can you check if there are any "
            "sequence-level problems? Here is the CDS:\n\n"
            "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTG"
            "GACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCC"
            "ACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCC"
            "TGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAATAAACTGCTTCAGCCGCT"
            "ACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACG"
            "TCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGG"
            "TGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCA"
            "AGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACG"
            "TCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCC"
            "ACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCA"
            "TCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCC"
            "TGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCG"
            "CCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
        ),
        description=(
            "The supplied CDS has an AATAAA cryptic polyA signal at ~position "
            "200 (in the insert body, not the last 150bp). Agent should call "
            "score_construct_confidence and surface the cryptic-polyA warning. "
            "Tests the Design Confidence Score feature from Epic 3."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",  # the CDS is a modified EGFP; use for size ref
        tags=["phase2", "confidence_score", "cryptic_polya"],
        grading_mode="transcript",  # analysis-only; no assembly expected
        transcript_assertions=["polyA", "cryptic"],  # must mention the signal
        user_persona=(
            "You just want an analysis of the sequence. If the agent finds "
            "the cryptic polyA signal, acknowledge it and say you will "
            "fix it by codon-optimizing that region. No assembly needed."
        ),
        # Negative assertion: agent should NOT blindly assemble without warning
    ),

    AgentTestCase(
        id="P2-DCS2",
        name="Confidence score on clean CDS returns high score",
        prompt=(
            "I want to express EGFP in HEK293 using pcDNA3.1(+). Before "
            "assembling, can you run a design confidence check on the EGFP "
            "CDS to make sure there are no red flags?"
        ),
        description=(
            "EGFP is a well-optimized CDS (CAI 0.91, good Kozak, no cryptic "
            "signals). Agent should score it, report high confidence (≥85), "
            "and proceed. Positive-path test for the confidence scorer."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        expected_insertion_position=895,
        tags=["phase2", "confidence_score", "positive"],
        transcript_assertions=["confidence", "100"],  # EGFP scores 100/100
        user_persona="Looks good, please proceed with assembly.",
    ),

    # ── P2-BP: Bespoke Promoters ───────────────────────────────────────
    AgentTestCase(
        id="P2-BP1",
        name="p65 promoter reporter — detect bespoke, offer options, no loop",
        prompt=(
            "Design a p65 promoter reporter construct for stable expression "
            "in HEK293 and RAW264.7 cells."
        ),
        description=(
            "Direct from the Allen Epics doc. p65 (RELA) is NOT a standard "
            "promoter. The agent should: (1) detect this is a bespoke "
            "promoter request, (2) NOT get stuck in a search loop, (3) offer "
            "the 3 options: research/Addgene, user-paste, or native upstream "
            "fetch. The agent should also ask which species since the user "
            "mentioned both human (HEK293) and mouse (RAW264.7) cells. "
            "Reference: Epics doc, 'Bespoke promoters' section."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",  # reporter gene
        tags=["phase2", "bespoke_promoter", "no_loop", "multiturn"],
        grading_mode="transcript",  # bespoke promoter swap; no canonical insert
        transcript_assertions=[
            "p65",           # agent acknowledges the promoter name
            "upstream",      # agent offers the native-upstream option
            "RELA",          # agent resolves p65 → RELA gene symbol
        ],
        # The entire point of this case is verifying the agent does NOT loop
        # searching for a promoter it will never find. A focused run is:
        # 1× get_backbone, 1× search (miss), 1× fetch_promoter_region,
        # 1× get_insert (reporter), 1× assemble, 1× validate + a few extras.
        # 12 is generous; the previous 44-tool trace was a genuine loop.
        max_tool_calls=12,
        user_persona=(
            "I want a human p65 (RELA) promoter reporter — EGFP as the "
            "reporter gene. I know there is a published p65 promoter on "
            "Addgene, but I do not have the catalog number handy. Just "
            "fetch the ~2kb upstream of the human RELA gene from NCBI "
            "and use that. pcDNA3.1(+) backbone is fine for the rest."
        ),
    ),

    AgentTestCase(
        id="P2-BP2",
        name="IFNβ promoter — user pastes sequence",
        prompt=(
            "I want to build an IFNβ promoter reporter. I have the IFNβ "
            "promoter sequence — should I paste it?"
        ),
        description=(
            "IFNβ (IFNB1) promoter is not standard. Agent should recognize "
            "the bespoke promoter pattern, confirm it wants the user to "
            "paste the sequence (option b in the decision tree), validate "
            "it, and use it. Tests the paste-sequence branch."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",  # persona specifies EGFP as reporter
        tags=["phase2", "bespoke_promoter", "paste_sequence", "multiturn"],
        grading_mode="transcript",  # bespoke promoter; can't score standard insert
        transcript_assertions=[
            "IFNβ",      # agent acknowledges the specific promoter
            "EGFP",      # agent uses the persona-specified reporter
            "validate",  # agent validates the pasted sequence before use
        ],
        max_tool_calls=12,
        user_persona=(
            "Yes, here is the ~300bp IFNβ minimal promoter:\n"
            "AGTTTCACTTTCCATTTCCCAGAGTCAGGAGACTTCCTAAGTGCCTCAAGGGCTCAG"
            "TTTAGAAATCCTACCAAGATGCGCACAGGCTGTTTCTCTCAGGCCTAGGCGGTGTCT"
            "CCTGCTGTCCTTCCTGCCACAGCATCTGCTGAGCCTTCCCACCGGGCGTGGAGGAGG"
            "AGCGCTCTCCTGATTTTCCTGCCGCTCCCCGGCAAAGCCTAGCACGGCGCGGAGCCT"
            "ACCTGCCGTCCGCGAAGGAGTCAATCAGCGGAAGTTCATC\n"
            "Use EGFP as the reporter, pcDNA3.1(+) backbone. Transient in "
            "HEK293."
        ),
    ),

    # ── P2-IF: Intelligent Fusion Design ───────────────────────────────
    AgentTestCase(
        id="P2-IF1",
        name="Internal fusion site for buried-terminus protein",
        prompt=(
            "I want to tag human TP53 with EGFP for live-cell imaging in "
            "U2OS cells. I've tried a C-terminal fusion before and it was "
            "non-functional — the C-terminus of p53 is critical for "
            "tetramerization. Can you suggest an internal insertion site "
            "in a disordered loop instead?"
        ),
        description=(
            "User explicitly asks for internal (not terminal) fusion. "
            "Agent should call predict_fusion_sites on TP53's protein "
            "sequence and offer ranked disordered regions. p53 has a "
            "well-known intrinsically disordered N-terminal transactivation "
            "domain (residues ~1-60) that should be detected. Tests the "
            "Smart Fusion Design feature from Epic 3."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="TP53",
        tags=["phase2", "intelligent_fusion", "internal_fusion", "multiturn"],
        grading_mode="ncbi",  # internal EGFP fusion into TP53; extract from construct
        transcript_assertions=[
            "disorder",      # agent should reference disorder prediction
            "residue",       # agent should give specific residue positions
        ],
        user_persona=(
            "Human TP53. The N-terminal disordered region sounds right — "
            "let's insert EGFP into the most disordered window you find "
            "there. Use the default (GGGGS)x4 linker on both sides of EGFP. "
            "pcDNA3.1(+) is fine."
        ),
    ),

    # ── P2-SM: Smart Mutation Design ───────────────────────────────────
    AgentTestCase(
        id="P2-SM1",
        name="Constitutively active BRAF — curated V600E lookup",
        prompt=(
            "I want to express a constitutively active human BRAF in "
            "HEK293 cells to study MEK/ERK signaling. What mutation should "
            "I use?"
        ),
        description=(
            "Agent should call lookup_known_mutations('BRAF', 'GoF') and "
            "return V600E (and V600K) with phenotype + PMID. Then offer "
            "to apply the mutation and assemble. Tests the Smart Mutation "
            "feature — Tier A curated database."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="BRAF",
        tags=["phase2", "smart_mutation", "gof", "curated_db", "multiturn"],
        grading_mode="ncbi",  # mutated BRAF; extract from construct
        transcript_assertions=[
            "V600E",         # must surface the canonical mutation
            "constitutive",  # phenotype from the DB
        ],
        user_persona=(
            "V600E is perfect. Please apply it to the human BRAF CDS and "
            "assemble in pcDNA3.1(+) for transient expression. Show me "
            "the original→new codon change."
        ),
    ),

    AgentTestCase(
        id="P2-SM2",
        name="Loss-of-function PTEN — curated + premature stop fallback",
        prompt=(
            "I need a kinase-dead PTEN (loss of function) for a rescue "
            "experiment in PTEN-null cells. What are my options?"
        ),
        description=(
            "Agent should look up curated PTEN LoF mutations (C124S "
            "catalytic-dead, R130G phosphatase-dead) and ALSO mention the "
            "premature-stop option. Tests both Tier A (curated) and Tier B "
            "(de novo LoF design)."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="PTEN",
        tags=["phase2", "smart_mutation", "lof", "curated_db", "multiturn"],
        grading_mode="ncbi",  # mutated PTEN; extract from construct
        transcript_assertions=[
            "C124S",          # curated catalytic-dead
            "phosphatase",    # phenotype
        ],
        user_persona=(
            "C124S is the one I want (catalytic-dead). Human PTEN. "
            "pcDNA3.1(+). Apply the mutation and assemble."
        ),
    ),

    # ── P2-TM: Troubleshooting Mode ────────────────────────────────────
    AgentTestCase(
        id="P2-TM1",
        name="Troubleshooting — no expression, agent diagnoses cryptic polyA",
        prompt=(
            "I previously designed a pcDNA3.1(+)-myGene construct with you "
            "and tested it in the lab. I got zero fluorescence in HEK293, "
            "even though transfection efficiency looked normal (control "
            "plasmid worked). Here is the insert CDS I used:\n\n"
            "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTG"
            "GACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCC"
            "ACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCC"
            "TGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAATAAACTGCTTCAGCCGCT"
            "ACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACG"
            "TCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGG"
            "TGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCA"
            "AGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACG"
            "TCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCC"
            "ACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCA"
            "TCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCC"
            "TGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCG"
            "CCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA\n\n"
            "What went wrong and how do I fix it?"
        ),
        description=(
            "The CDS has a cryptic AATAAA polyA signal at ~pos 200 that "
            "would cause premature transcription termination → truncated/ "
            "no protein → no fluorescence. Agent should: (1) acknowledge "
            "the prior failure, (2) run score_construct_confidence or "
            "manually inspect, (3) identify the cryptic polyA, (4) propose "
            "specific remediation (codon-optimize around that position to "
            "eliminate AATAAA). Tests Troubleshooting Mode from Epic 5. "
            "Same CDS as P2-DCS1 but framed as a post-lab-failure scenario."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",  # modified EGFP, for size ref
        tags=["phase2", "troubleshooting", "diagnosis", "multiturn"],
        transcript_assertions=[
            "polyA",         # must identify the cryptic polyA
            "premature",     # explain the mechanism (premature termination)
            "codon",         # propose codon-optimization as fix
        ],
        user_persona=(
            "That makes sense — the cryptic polyA explains the truncated "
            "transcript. Please log this outcome so I can refer back to "
            "it. I will codon-optimize that region and come back with a "
            "revised sequence."
        ),
    ),

    AgentTestCase(
        id="P2-TM2",
        name="Troubleshooting — fusion misfolding, agent suggests internal site",
        prompt=(
            "Follow-up on a fusion design: I made a C-terminal EGFP fusion "
            "to human Lamin A/C (LMNA) and it mislocalized — EGFP was "
            "cytoplasmic instead of at the nuclear envelope. The LMNA "
            "C-terminus has the CaaX farnesylation motif that targets it to "
            "the membrane. I think the fusion blocked it. What should I try?"
        ),
        description=(
            "Classic C-terminus-critical case: LMNA's C-terminal CaaX box "
            "is essential for nuclear lamina targeting. Agent should "
            "diagnose (C-term fusion buried the CaaX), and propose either "
            "(a) N-terminal fusion, or (b) internal fusion via "
            "predict_fusion_sites. Tests the troubleshooting-mode → "
            "intelligent-fusion pipeline."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="LMNA",
        tags=["phase2", "troubleshooting", "intelligent_fusion", "multiturn"],
        grading_mode="ncbi",  # N-term EGFP-LMNA fusion; extract from construct
        transcript_assertions=[
            "CaaX",          # agent acknowledges the farnesylation motif
            "N-terminal",    # suggests N-term alternative
        ],
        user_persona=(
            "N-terminal fusion sounds safer for this case. Use the "
            "standard (GGGGS)x4 linker. Human LMNA, transient in U2OS, "
            "pcDNA3.1(+)."
        ),
    ),

    # ── A9: Golden Gate assembly (Allen Institute modular system) ─────────
    AgentTestCase(
        id="A9-001",
        name="Golden Gate: X0001 backbone + three AICS parts (explicit IDs)",
        prompt=(
            "Using Golden Gate assembly, design a vector using the backbone X0001 "
            "with inserts AICS_SynP000X, AICS_SynP000Y, AICS_SynP000Z. "
            "Assemble the construct and export the final sequence."
        ),
        description=(
            "Explicit Golden Gate request with backbone and part IDs. "
            "Agent must call assemble_golden_gate (not assemble_construct) with "
            "backbone AICS_X0001_pTwist_Kan_B and the three part carrier vectors. "
            "Expected assembly: SynP000X → SynP000Y → SynP000Z, 5383 bp total, "
            "Esp3I enzyme, junctions CACC / CTGG / ATCC / AACG."
        ),
        # For multi-part GG assembly there is no single insert_id; grading is
        # transcript-based. backbone_id and expected_insert_id are set to the
        # primary components for bookkeeping purposes only.
        expected_backbone_id="AICS_X0001_pTwist_Kan_B",
        expected_insert_id="AICS_SynP000X",
        expected_total_size=5383,
        grading_mode="transcript",
        transcript_assertions=[
            "5383",           # correct assembled size reported
            "Golden Gate",    # GG assembly method identified
            "AICS_SynP000X",  # all three parts addressed
            "AICS_SynP000Y",
            "AICS_SynP000Z",
        ],
        tools_should_not_use=["assemble_construct"],  # must use GG path, not standard
        tags=["golden_gate", "allen_institute", "modular_system", "multi_part"],
    ),
    AgentTestCase(
        id="A9-002",
        name="Golden Gate: compound construct name (ABC_XYZ-XXYYY_LMN_OhP)",
        prompt=(
            "Using Golden Gate assembly, design the following vector: ABC_XYZ-XXYYY_LMN_OhP."
        ),
        description=(
            "Name-based Golden Gate request. The construct name encodes the three Allen "
            "Institute modular parts by their display names: "
            "  ABC_XYZ  → AICS_SynP000X, "
            "  XXYYY    → AICS_SynP000Y, "
            "  LMN_OhP  → AICS_SynP000Z. "
            "Agent must parse the compound name, resolve each component to its library "
            "ID, identify an appropriate X0001 backbone, call assemble_golden_gate, "
            "and export the assembled sequence. "
            "Expected assembly: 5383 bp total, same junctions as A9-001."
        ),
        expected_backbone_id="AICS_X0001_pTwist_Kan_B",
        expected_insert_id="AICS_SynP000X",
        expected_total_size=5383,
        grading_mode="transcript",
        transcript_assertions=[
            "5383",       # correct assembled size
            "Golden Gate",
            "ABC_XYZ",    # agent acknowledges the part names from the construct label
            "XXYYY",
            "LMN_OhP",
        ],
        tools_should_not_use=["assemble_construct"],
        max_tool_calls=35,
        tags=["golden_gate", "allen_institute", "modular_system", "multi_part",
              "name_resolution"],
        user_persona=(
            "You are a researcher who submitted the construct name ABC_XYZ-XXYYY_LMN_OhP. "
            "If the agent asks you to confirm how it parsed the name, confirm that: "
            "ABC_XYZ is the first part (AICS_SynP000X), XXYYY is the second part (AICS_SynP000Y), "
            "and LMN_OhP is the third part (AICS_SynP000Z). "
            "If the agent asks which backbone to use, say you want the Kanamycin backbone (X0001). "
            "If the agent asks any other clarifying question, give a short direct answer consistent "
            "with this setup. If the agent presents a design summary or asks to proceed, say 'Yes, proceed.'"
        ),
    ),

    # ── A10: Insert extraction from Addgene plasmids ──────────────────
    AgentTestCase(
        id="A10-001",
        name="Extract mCerulean CDS from Addgene #27796",
        prompt=(
            "Download the plasmid 27796 from addgene and extract the mCerulean "
            "coding sequence"
        ),
        description=(
            "Agent must fetch Addgene plasmid 27796, then call "
            "extract_insert_from_plasmid to locate and return the mCerulean CDS. "
            "No backbone assembly — the deliverable is the extracted insert sequence. "
            "Expected sequence is 720 bp (mCerulean, a cyan FP). "
            "Graded on transcript: agent must output the correct CDS."
        ),
        # Extraction-only: no backbone assembly. IDs are for bookkeeping only.
        expected_backbone_id="addgene_27796",
        expected_insert_id="mCerulean",
        expected_insert_sequence=(
            "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGT"
            "AAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAA"
            "GTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTGGGGCGT"
            "GCAGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGG"
            "CTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTT"
            "CGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCT"
            "GGGGCACAAGCTGGAGTACAACGCCATCAGCGACAACGTCTATATCACCGCCGACAAGCAGAAGAACGGC"
            "ATCAAGGCCAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGC"
            "AGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCAAGCT"
            "GAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACT"
            "CTCGGCATGGACGAGCTGTACAAG"
        ),
        grading_mode="transcript",
        transcript_assertions=[
            "mCerulean",          # agent identifies the gene by name
            "27796",              # agent fetched the correct Addgene plasmid
            "ATGGTGAGCAAGGGCGAGGAG",  # first 22 bp of expected CDS
            "720",                # correct insert length in bp
        ],
        max_tool_calls=20,
        tags=["addgene", "extraction", "fluorescent_protein", "single_insert"],
    ),
    AgentTestCase(
        id="A10-002",
        name="Download Addgene #244170 and export as GenBank",
        prompt="Download the plasmid 244170 and export as a genbank file",
        description=(
            "Agent must call fetch_addgene_sequence_with_metadata(244170), then export_construct with "
            "sequence_cache_key and output_format='genbank'. No assembly. "
            "Graded on transcript: GenBank output must be non-empty and contain "
            "expected feature annotations (mNeonGreen CDS, CMV promoter/enhancer, "
            "f1 ori, EF-1α intron A, neo CDS) from pLannotate annotation of the "
            "7792 bp PolyTX-mNeonGreen plasmid."
        ),
        expected_backbone_id="addgene_244170",
        expected_insert_id="mNeonGreen",
        grading_mode="transcript",
        transcript_assertions=[
            "244170",           # correct Addgene plasmid fetched
            "FEATURES",         # GenBank format header present → file is non-empty
            "mNeonGreen",       # key insert CDS annotated
            "CMV",              # CMV promoter/enhancer annotated
            "neo",              # neomycin resistance annotated
            "f1 ori",           # f1 origin of replication annotated
        ],
        max_tool_calls=15,
        tags=["addgene", "export", "genbank", "whole_plasmid"],
    ),
    AgentTestCase(
        id="A10-003",
        name="mCerulean-His transient overexpression in mammalian vector",
        prompt=(
            "Design a mammalian expression vector for transient overexpression of "
            "mCerulean-His. Create a fresh design."
        ),
        description=(
            "Agent must fetch mCerulean from Addgene (e.g. #27796), extract the CDS "
            "without stop codon (717 bp), append a C-terminal His tag, and assemble "
            "into a suitable mammalian backbone (e.g. pcDNA3.1(+)) at the MCS. "
            "Key length check: mCerulean is 720 bp with stop codon, 717 bp without. "
            "The 717 bp form is required for in-frame His-tag fusion. "
            "Graded on transcript: correct insert length cited, His tag appended, "
            "and assembly into a mammalian backbone confirmed."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mCerulean",
        grading_mode="transcript",
        transcript_assertions=[
            "mCerulean",   # correct insert identified
            "717",         # mCerulean without stop codon for C-terminal fusion
            "His",         # His tag appended
            "pcDNA",       # mammalian backbone selected
        ],
        user_persona=(
            "You want mCerulean with a C-terminal 6xHis tag for transient "
            "overexpression in mammalian cells. pcDNA3.1(+) or any standard "
            "mammalian expression vector is fine. Proceed when the design is presented."
            "If the agent asks for the mCerulean sequence, tell it to look up a sequence from Addgene."
        ),
        max_tool_calls=25,
        tags=["addgene", "extraction", "fusion", "mammalian", "his_tag"],
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
    simulated_user_exchanges: list[dict] = field(default_factory=list)
    judge_result: Optional[JudgeResult] = None
    # Disambiguates why a case produced no rubric (set by run_agent_eval_case):
    #   None       — scoring succeeded
    #   "ERROR"    — trace.error is set (agent crashed / SDK exception)
    #   "SKIP"     — scorer couldn't obtain reference sequences
    #   "NO_OUTPUT"— agent produced no assembled sequence (and case isn't transcript-mode)
    skip_reason: Optional[str] = None


# ── Grading-mode helpers ───────────────────────────────────────────────

# Default simulated-user persona for cases without an explicit one.
# The original prompt already states the user's intent; this persona only
# unblocks the agent when it asks for confirmation or a trivial choice.
DEFAULT_PROCEED_PERSONA = (
    "You have already told the agent exactly what you want in your first "
    "message. If the agent presents a design summary or asks to proceed, "
    "say 'Yes, proceed with the assembly.' If it asks a clarifying question, "
    "give a direct 1-line answer consistent with the original request, "
    "picking the most common/standard option."
)


def _normalize_tool_name(name: str) -> str:
    """Strip MCP namespace prefix so test-case tool names match trace entries.

    Trace records show 'mcp__plasmid-library__search_gene' but test cases
    list 'search_gene'. Without normalization the tool-negative check never
    triggers.
    """
    return name.removeprefix("mcp__plasmid-library__")


def _max_tool_calls_check(
    max_tool_calls: Optional[int],
    actual_tool_calls: int,
) -> Optional[Check]:
    """Return a Critical check if max_tool_calls is set. None otherwise."""
    if max_tool_calls is None:
        return None
    passed = actual_tool_calls <= max_tool_calls
    return Check(
        section="Tool Routing",
        name=f"Tool calls within budget (≤{max_tool_calls})",
        severity="Critical",
        passed=passed,
        detail=f"{actual_tool_calls} tool calls"
        + ("" if passed else f" — exceeds budget, likely search loop"),
    )


def _build_transcript_rubric(
    transcript_results: list[tuple[str, bool]],
    tool_violation_results: list[tuple[str, bool]],
    max_tool_check: Optional[Check],
) -> RubricResult:
    """Build a RubricResult from transcript assertions + tool checks only.

    Used for grading_mode="transcript" cases where no sequence-level
    ground truth exists (analysis-only tasks, bespoke promoter swaps).
    Transcript assertions are Critical (the case's PASS/FAIL hinges on
    whether the agent said the right thing); tool violations are Major.
    """
    result = RubricResult()
    for assertion, found in transcript_results:
        result.checks.append(Check(
            section="Transcript",
            name=f"Agent mentions '{assertion}'",
            severity="Critical",
            passed=found,
            detail="found in transcript" if found else "not found in transcript",
        ))
    for tool_name, was_used in tool_violation_results:
        result.checks.append(Check(
            section="Tool Routing",
            name=f"Agent should NOT use {tool_name}",
            severity="Major",
            passed=not was_used,
            detail=f"Agent {'used' if was_used else 'correctly avoided'} {tool_name}",
        ))
    if max_tool_check is not None:
        result.checks.append(max_tool_check)
    return result


async def _auto_approve(tool_name, tool_input, context):
    """Auto-approve all tool calls (MCP tools are safe, in-process)."""
    return PermissionResultAllow()


def _has_tool_call(message) -> bool:
    """Check if an AssistantMessage contains any actionable tool use blocks.

    Excludes AskUserQuestion since that tool IS a clarifying question and
    should trigger the simulated user path, not short-circuit it.
    """
    if not isinstance(message, AssistantMessage):
        return False
    return any(
        isinstance(block, ToolUseBlock) and block.name != "AskUserQuestion"
        for block in message.content
    )


def _get_assistant_text(message) -> str:
    """Extract text content from an AssistantMessage."""
    if not isinstance(message, AssistantMessage):
        return ""
    parts = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "\n".join(parts)


async def run_agent(
    prompt: str,
    model: str = "claude-opus-4-7",
    max_turns: int = 15,
    verbose: bool = False,
    simulated_user: Optional[SimulatedUser] = None,
    max_user_exchanges: int = 3,
) -> AgentTrace:
    """Run the plasmid design agent on a single prompt using the Agent SDK.

    If a SimulatedUser is provided, the agent loop supports multi-turn
    conversation: when the agent responds with text only (no tool calls),
    it's treated as a clarifying question. The simulated user generates a
    response, which is fed back to the agent via client.query().
    """
    trace = AgentTrace(prompt=prompt)

    # Evals must be deterministic and network-independent. Force external MCP
    # servers OFF regardless of ambient env — PubMed defaults on otherwise and
    # would make eval scores vary with network availability.
    os.environ["PLASMID_ENABLE_PUBMED"] = "0"
    os.environ.pop("BENCHLING_SUBDOMAIN", None)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers=build_mcp_servers(),
        permission_mode="acceptEdits",
        model=model,
        max_turns=max_turns,
        cwd=str(PROJECT_ROOT),
        can_use_tool=_auto_approve,
    )

    all_text_parts = []
    user_exchanges = 0
    # Track conversation history for simulated user context.
    # Seed with the original user prompt so the simulated user has context.
    sim_conversation: list[dict] = [{"role": "user", "content": prompt}]

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            while True:
                last_assistant_msg = None
                got_result = False

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        trace.total_turns += 1
                        last_assistant_msg = message
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
                        got_result = True
                        break

                # Check if the agent asked a question (text-only, no tool calls)
                # and we have a simulated user available.
                # This must be checked BEFORE the ResultMessage exit, because the
                # SDK sends a ResultMessage when the agent ends with text-only
                # (no tool calls), which would otherwise cause early exit.
                if (
                    last_assistant_msg is not None
                    and simulated_user
                    and not _has_tool_call(last_assistant_msg)
                    and user_exchanges < max_user_exchanges
                ):
                    agent_text = _get_assistant_text(last_assistant_msg)
                    if agent_text.strip():
                        if verbose:
                            print(f"    [simulated_user] Agent asked a question, generating response...")

                        # Build conversation history for context
                        sim_conversation.append({
                            "role": "assistant",
                            "content": agent_text,
                        })

                        # SimulatedUser.respond() is a blocking Anthropic SDK
                        # call — wrap in to_thread so parallel evals don't
                        # stall the event loop waiting on Haiku.
                        user_response = await asyncio.to_thread(
                            simulated_user.respond,
                            agent_text,
                            sim_conversation[:-1] if len(sim_conversation) > 1 else None,
                        )

                        sim_conversation.append({
                            "role": "user",
                            "content": user_response,
                        })

                        trace.simulated_user_exchanges.append({
                            "agent_question": agent_text[:500],
                            "user_response": user_response,
                        })
                        user_exchanges += 1

                        if verbose:
                            print(f"    [simulated_user] Response: {user_response[:200]}")

                        # Feed the simulated user's response back to the agent
                        await client.query(user_response)
                        continue  # Continue the outer loop to process the next response

                # If we got a ResultMessage or no assistant message, we're done
                if got_result or last_assistant_msg is None:
                    break

                # No simulated user interaction needed — we're done
                break

    except Exception as e:
        trace.error = str(e)
        traceback.print_exc()

    trace.assistant_text = "\n".join(all_text_parts)

    # Extract assembled sequence from all collected text.
    # Use the LAST valid sequence found, since the agent may reassemble
    # after corrections (e.g., removing ATG/stop codons per user request).
    last_seq = None
    for part in all_text_parts:
        seq = _find_dna_sequence_in_text(part)
        if seq:
            last_seq = seq
    trace.assembled_sequence = last_seq

    return trace


# ── Eval runner ────────────────────────────────────────────────────────


FIXTURES_PATH = PROJECT_ROOT / "tests" / "fixtures"


def _load_fixtures_for_case(tc: AgentTestCase) -> bool:
    """Load test fixtures for cases that need parts not in the main library.

    Currently loads tests/fixtures/{backbones,inserts}.json for golden_gate
    cases.  Returns True if fixtures were registered (caller must call
    clear_test_fixtures() when done).
    """
    if "golden_gate" not in tc.tags:
        return False
    backbones_path = FIXTURES_PATH / "backbones.json"
    inserts_path = FIXTURES_PATH / "inserts.json"
    extra_backbones: list[dict] = []
    extra_inserts: list[dict] = []
    if backbones_path.exists():
        with open(backbones_path) as f:
            extra_backbones = json.load(f).get("backbones", [])
    if inserts_path.exists():
        with open(inserts_path) as f:
            extra_inserts = json.load(f).get("inserts", [])
    if extra_backbones or extra_inserts:
        register_test_fixtures(backbones=extra_backbones, inserts=extra_inserts)
        return True
    return False


async def run_agent_eval_case(
    tc: AgentTestCase,
    model: str = "claude-opus-4-7",
    verbose: bool = False,
    use_judge: bool = False,
    judge_model: str = "claude-sonnet-4-6",
) -> tuple[Optional[RubricResult], AgentTrace]:
    """Run a single agent eval case."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Case: {tc.id} — {tc.name}")
        print(f"Prompt: {tc.prompt[:100]}...")
        if tc.user_persona:
            print(f"  Simulated user persona: {tc.user_persona[:80]}...")
        print(f"  Running agent...")

    # Load test fixtures for cases that need parts not in the main library
    # (e.g. Golden Gate AICS parts in tests/fixtures/).
    fixtures_loaded = _load_fixtures_for_case(tc)
    if fixtures_loaded and verbose:
        print(f"  Loaded test fixtures from {FIXTURES_PATH}")

    # Always have a simulated user. If the case doesn't specify a persona,
    # use the default "proceed" persona so the agent doesn't stall at the
    # design-summary confirmation (Pattern #1/#2 safety net).
    sim_user = SimulatedUser(persona=tc.user_persona or DEFAULT_PROCEED_PERSONA)

    try:
        trace = await run_agent(
            prompt=tc.prompt,
            model=model,
            verbose=verbose,
            simulated_user=sim_user,
        )
    finally:
        if fixtures_loaded:
            clear_test_fixtures()

    if verbose and trace.simulated_user_exchanges:
        print(f"  Simulated user exchanges: {len(trace.simulated_user_exchanges)}")
        for i, ex in enumerate(trace.simulated_user_exchanges):
            print(f"    Exchange {i+1}:")
            print(f"      Agent: {ex['agent_question'][:100]}...")
            print(f"      User:  {ex['user_response'][:100]}")

    if trace.error:
        if verbose:
            print(f"  Agent ERROR: {trace.error}")
        trace.skip_reason = "ERROR"
        return None, trace

    # ── Transcript assertions (conversational quality grading) ────────
    transcript_results = []
    if tc.transcript_assertions:
        for assertion in tc.transcript_assertions:
            found = assertion.lower() in trace.assistant_text.lower()
            transcript_results.append((assertion, found))
            if verbose:
                status = "FOUND" if found else "MISSING"
                print(f"  [transcript] '{assertion}': {status}")

    # ── Tool negative assertions (balanced eval grading) ──────────────
    # Normalize MCP-prefixed tool names so test-case short names match.
    tool_violation_results = []
    if tc.tools_should_not_use:
        tools_used = {_normalize_tool_name(t["tool"]) for t in trace.tool_calls}
        for forbidden_tool in tc.tools_should_not_use:
            was_used = _normalize_tool_name(forbidden_tool) in tools_used
            tool_violation_results.append((forbidden_tool, was_used))
            if verbose:
                status = "VIOLATION — used" if was_used else "OK — not used"
                print(f"  [tool_negative] {forbidden_tool}: {status}")

    # ── Tool budget assertion (loop detection for all grading modes) ──
    max_tool_check = _max_tool_calls_check(tc.max_tool_calls, len(trace.tool_calls))
    if max_tool_check is not None and verbose:
        status = "OK" if max_tool_check.passed else "VIOLATION"
        print(f"  [max_tool_calls] {status}: {max_tool_check.detail}")

    # ── Transcript-mode grading (no sequence scoring) ─────────────────
    # For analysis-only / promoter-swap cases, PASS/FAIL is based entirely
    # on whether the agent said the right things + avoided forbidden tools.
    if tc.grading_mode == "transcript":
        rubric_result = _build_transcript_rubric(
            transcript_results, tool_violation_results, max_tool_check
        )
        if verbose:
            print(f"  [grading_mode=transcript] Result: {rubric_result.summary()}")
            print(rubric_result.report())
        _maybe_run_judge(tc, trace, rubric_result, use_judge, judge_model, verbose)
        return rubric_result, trace

    # ── Sequence / NCBI grading modes ─────────────────────────────────
    if not trace.assembled_sequence:
        if verbose:
            print(f"  No assembled sequence found in agent output")
            print(f"  Tool calls made: {[t['tool'] for t in trace.tool_calls]}")
        if tc.transcript_assertions and transcript_results:
            passed_assertions = sum(1 for _, found in transcript_results if found)
            if verbose:
                print(
                    f"  Transcript assertions: {passed_assertions}/"
                    f"{len(transcript_results)} passed"
                )
        _maybe_run_judge(tc, trace, None, use_judge, judge_model, verbose)
        trace.skip_reason = "NO_OUTPUT"
        return None, trace

    if verbose:
        print(f"  Extracted sequence: {len(trace.assembled_sequence)} bp")
        print(
            f"  Tool calls: {len(trace.tool_calls)} "
            f"({', '.join(t['tool'] for t in trace.tool_calls)})"
        )

    backbone_data = get_backbone_by_id(tc.expected_backbone_id)
    if not backbone_data or not backbone_data.get("sequence"):
        if verbose:
            print(f"  SKIP: No backbone sequence for '{tc.expected_backbone_id}'")
        trace.skip_reason = "SKIP"
        return None, trace

    backbone_seq = backbone_data["sequence"]
    insertion_pos = tc.expected_insertion_position
    if insertion_pos is None:
        insertion_pos = find_mcs_insertion_point(backbone_data)
    if insertion_pos is None:
        if verbose:
            print(f"  SKIP: Cannot determine insertion position")
        trace.skip_reason = "SKIP"
        return None, trace

    # Resolve insert sequence. Priority: inline → library → extract-from-construct
    # (ncbi mode only). Guard against get_insert_by_id() returning a
    # disambiguation dict ({"needs_disambiguation": True, ...}) — that's a
    # valid return value but has no sequence/name keys and should be treated
    # the same as "not found" for scoring purposes.
    insert_seq = tc.expected_insert_sequence
    insert_data = None
    if not insert_seq:
        lookup = get_insert_by_id(tc.expected_insert_id)
        if lookup and not lookup.get("needs_disambiguation") and lookup.get("sequence"):
            insert_data = lookup
            insert_seq = lookup["sequence"]

    # NCBI mode: library insert may be missing entirely OR may be a different
    # species/isoform than what the agent fetched. In either case, extract
    # the insert from the construct by length (construct - backbone).
    if tc.grading_mode == "ncbi" and trace.assembled_sequence:
        needs_extraction = not insert_seq or insert_seq not in trace.assembled_sequence
        if needs_extraction:
            insert_len = len(trace.assembled_sequence) - len(backbone_seq)
            if insert_len > 0:
                extracted = trace.assembled_sequence[insertion_pos:insertion_pos + insert_len]
                if verbose:
                    print(
                        f"  [grading_mode=ncbi] library insert unavailable/mismatched; "
                        f"extracted {insert_len} bp from construct"
                    )
                insert_seq = extracted

    # Legacy tag-based extraction (kept for backward compatibility with any
    # untagged test cases that relied on "ncbi" in tags).
    elif (
        "ncbi" in tc.tags
        and insert_seq
        and insert_seq not in trace.assembled_sequence
    ):
        insert_len = len(trace.assembled_sequence) - len(backbone_seq)
        if insert_len > 0:
            extracted = trace.assembled_sequence[insertion_pos:insertion_pos + insert_len]
            if verbose:
                print(
                    f"  NCBI case: library insert not found, extracted "
                    f"{insert_len} bp from construct"
                )
            insert_seq = extracted

    if not insert_seq:
        if verbose:
            print(f"  SKIP: No insert sequence for '{tc.expected_insert_id}'")
        trace.skip_reason = "SKIP"
        return None, trace

    # insert_data is only set when we have a real library/FPbase/NCBI record
    # (never a disambiguation dict), so .get() here is defensive only.
    insert_name = (insert_data or {}).get("name") or tc.expected_insert_id
    insert_category = (insert_data or {}).get("category")

    rubric_result = score_construct(
        construct_sequence=trace.assembled_sequence,
        expected_backbone_sequence=backbone_seq,
        expected_insert_sequence=insert_seq,
        expected_insert_position=insertion_pos,
        backbone_name=backbone_data["name"],
        insert_name=insert_name,
        insert_category=insert_category,
        backbone_features=backbone_data.get("features"),
        expect_reverse_complement=tc.expect_reverse_complement,
        fusion_parts=tc.fusion_parts or None,
    )

    # Try alternative backbones if primary scoring fails
    if not rubric_result.overall_pass and tc.alternative_expected:
        for alt in tc.alternative_expected:
            alt_backbone_data = get_backbone_by_id(alt["backbone_id"])
            if not alt_backbone_data or not alt_backbone_data.get("sequence"):
                continue
            alt_insertion_pos = alt.get("insertion_position")
            if alt_insertion_pos is None:
                alt_insertion_pos = find_mcs_insertion_point(alt_backbone_data)
            if alt_insertion_pos is None:
                continue
            alt_result = score_construct(
                construct_sequence=trace.assembled_sequence,
                expected_backbone_sequence=alt_backbone_data["sequence"],
                expected_insert_sequence=insert_seq,
                expected_insert_position=alt_insertion_pos,
                backbone_name=alt_backbone_data["name"],
                insert_name=insert_name,
                insert_category=insert_category,
                backbone_features=alt_backbone_data.get("features"),
                expect_reverse_complement=tc.expect_reverse_complement,
            )
            if alt_result.score_pct > rubric_result.score_pct:
                rubric_result = alt_result
                if verbose:
                    print(
                        f"  Alternative backbone '{alt['backbone_id']}' scored "
                        f"better: {alt_result.summary()}"
                    )
                if rubric_result.overall_pass:
                    break

    # Add tool violation checks to the rubric
    if tool_violation_results:
        for tool_name, was_used in tool_violation_results:
            rubric_result.checks.append(Check(
                section="Tool Routing",
                name=f"Agent should NOT use {tool_name}",
                severity="Major",
                passed=not was_used,
                detail=(
                    f"Agent {'used' if was_used else 'correctly avoided'} "
                    f"{tool_name}"
                ),
            ))

    # Add tool-budget check (applies to sequence/ncbi modes too)
    if max_tool_check is not None:
        rubric_result.checks.append(max_tool_check)

    if verbose:
        print(f"  Result: {rubric_result.summary()}")
        print()
        print(rubric_result.report())

    _maybe_run_judge(tc, trace, rubric_result, use_judge, judge_model, verbose)
    return rubric_result, trace


def _maybe_run_judge(
    tc: AgentTestCase,
    trace: AgentTrace,
    rubric_result: Optional[RubricResult],
    use_judge: bool,
    judge_model: str,
    verbose: bool,
) -> None:
    """Run LLM judge if enabled. Mutates trace.judge_result."""
    if not use_judge or not trace.assistant_text.strip():
        return
    if verbose:
        print(f"  Running LLM judge ({judge_model})...")
    judge = LLMJudge(model=judge_model)
    judge_result = judge.evaluate(
        case_id=tc.id,
        case_name=tc.name,
        case_description=tc.description,
        case_prompt=tc.prompt,
        expected_backbone=tc.expected_backbone_id,
        expected_insert=tc.expected_insert_id,
        transcript=trace.assistant_text,
        tool_calls=trace.tool_calls,
        transcript_assertions=tc.transcript_assertions or None,
        rubric_result=rubric_result,
    )
    trace.judge_result = judge_result
    if verbose:
        print(f"  {judge_result.summary()}")
        for s in judge_result.scores:
            print(f"    {s.dimension}: {s.score}/5 — {s.explanation[:80]}")


def _build_result_dict(
    tc: AgentTestCase,
    rubric: Optional[RubricResult],
    trace: AgentTrace,
    elapsed: float,
) -> dict:
    """Build the per-case result dict for the summary table / JSONL output."""
    judge_score = None
    if trace.judge_result and trace.judge_result.scores:
        judge_score = round(trace.judge_result.overall_score, 1)

    base = {
        "id": tc.id,
        "name": tc.name,
        "grading_mode": tc.grading_mode,
        "tool_calls": len(trace.tool_calls),
        "turns": trace.total_turns,
        "elapsed_s": round(elapsed, 1),
        "cost_usd": trace.cost_usd,
        "judge_score": judge_score,
    }

    if rubric is None:
        # Disambiguate ERROR vs SKIP vs NO_OUTPUT via skip_reason
        status = trace.skip_reason or "ERROR"
        if status == "ERROR":
            detail = trace.error or "Unknown error"
        elif status == "SKIP":
            detail = "Scorer could not obtain reference sequences"
        else:  # NO_OUTPUT
            detail = "Agent produced no assembled sequence"
        return {**base, "status": status, "score": None, "detail": detail}

    if rubric.overall_pass:
        return {**base, "status": "PASS", "score": rubric.score_pct, "detail": rubric.summary()}

    return {
        **base,
        "status": "FAIL",
        "score": rubric.score_pct,
        "detail": rubric.summary(),
        "critical_fail": rubric.critical_fail,
    }


async def run_agent_eval_suite(
    cases: list[AgentTestCase],
    model: str = "claude-opus-4-7",
    verbose: bool = False,
    use_judge: bool = False,
    judge_model: str = "claude-sonnet-4-6",
    concurrency: int = 1,
    output_path: Optional[Path] = None,
) -> dict:
    """Run a suite of agent eval cases, optionally in parallel.

    Args:
        concurrency: Max concurrent cases. 1 = sequential (default).
        output_path: If set, stream full per-case traces as JSONL lines.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[Optional[dict]] = [None] * len(cases)
    output_file = open(output_path, "w") if output_path else None
    output_lock = asyncio.Lock()
    completed = [0]  # mutable counter for progress line

    async def _run_one(idx: int, tc: AgentTestCase):
        async with sem:
            start = time.time()
            rubric, trace = await run_agent_eval_case(
                tc, model=model, verbose=verbose,
                use_judge=use_judge, judge_model=judge_model,
            )
            elapsed = time.time() - start
            result = _build_result_dict(tc, rubric, trace, elapsed)
            results[idx] = result

            if output_file is not None:
                trace_record = {
                    **result,
                    "prompt": tc.prompt,
                    "tool_calls_full": trace.tool_calls,
                    "assistant_text": trace.assistant_text,
                    "simulated_user_exchanges": trace.simulated_user_exchanges,
                    "rubric_report": rubric.report() if rubric else None,
                    "error": trace.error,
                }
                async with output_lock:
                    output_file.write(json.dumps(trace_record) + "\n")
                    output_file.flush()

            if not verbose:
                completed[0] += 1
                score_str = f"{result['score']}%" if result['score'] is not None else "—"
                print(
                    f"  [{completed[0]}/{len(cases)}] [{result['status']:>9}] "
                    f"{tc.id:<8} {score_str:>6}  ({elapsed:.1f}s)",
                    flush=True,
                )

    try:
        await asyncio.gather(*[_run_one(i, tc) for i, tc in enumerate(cases)])
    finally:
        if output_file:
            output_file.close()

    # Aggregate — results is now fully populated in original case order.
    final_results = [r for r in results if r is not None]
    passed = sum(1 for r in final_results if r["status"] == "PASS")
    failed = sum(1 for r in final_results if r["status"] == "FAIL")
    errored = sum(1 for r in final_results if r["status"] == "ERROR")
    skipped = sum(1 for r in final_results if r["status"] == "SKIP")
    no_output = sum(1 for r in final_results if r["status"] == "NO_OUTPUT")

    scored = passed + failed
    summary = {
        "total": len(final_results),
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "skipped": skipped,
        "no_output": no_output,
        "pass_rate": round(passed / scored * 100, 1) if scored > 0 else 0,
        "model": model,
        "concurrency": concurrency,
    }

    return {"results": final_results, "summary": summary}


def print_agent_summary_table(eval_output: dict):
    """Print a compact summary table."""
    results = eval_output["results"]
    summary = eval_output["summary"]

    has_judge = any(r.get("judge_score") is not None for r in results)

    width = 92 if has_judge else 84
    print(f"\n{'='*width}")
    print(f"AGENT EVALUATION RESULTS (model: {summary['model']})")
    print(f"{'='*width}")

    if has_judge:
        print(
            f"{'ID':<8} {'Name':<35} {'Status':>9} {'Score':>7} "
            f"{'Judge':>7} {'Tools':>5} {'Time':>6}"
        )
    else:
        print(
            f"{'ID':<8} {'Name':<35} {'Status':>9} {'Score':>7} "
            f"{'Tools':>5} {'Time':>6}"
        )
    print(f"{'-'*width}")

    for r in results:
        score_str = f"{r['score']}%" if r['score'] is not None else "—"
        time_str = f"{r['elapsed_s']}s"
        name_trunc = r['name'][:35]
        if has_judge:
            judge_str = f"{r['judge_score']}/5" if r.get('judge_score') is not None else "—"
            print(
                f"{r['id']:<8} {name_trunc:<35} {r['status']:>9} {score_str:>7} "
                f"{judge_str:>7} {r['tool_calls']:>5} {time_str:>6}"
            )
        else:
            print(
                f"{r['id']:<8} {name_trunc:<35} {r['status']:>9} {score_str:>7} "
                f"{r['tool_calls']:>5} {time_str:>6}"
            )

    print(f"{'-'*width}")
    parts = [
        f"Total: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
    ]
    if summary.get("errored"):
        parts.append(f"Errors: {summary['errored']}")
    if summary.get("skipped"):
        parts.append(f"Skipped: {summary['skipped']}")
    if summary.get("no_output"):
        parts.append(f"NoOutput: {summary['no_output']}")
    parts.append(f"Pass Rate: {summary['pass_rate']}%")
    print("  |  ".join(parts))
    print(f"{'='*width}")


# ── CLI ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end agent evaluations (Agent SDK)"
    )
    parser.add_argument("--case", type=str, help="Run a single test case by ID (e.g., A1-001)")
    parser.add_argument("--tag", type=str, help="Run cases matching this tag")
    parser.add_argument(
        "--filter", type=str,
        help="Run cases whose ID starts with this prefix (e.g., P1-SP)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full agent trace",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON to stdout",
    )
    parser.add_argument(
        "--model", type=str, default="claude-opus-4-6",
        help="Model to use (default: opus)",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Enable LLM-as-judge grading (off by default to save cost)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="claude-sonnet-4-6",
        help="Model for LLM judge (default: sonnet)",
    )
    parser.add_argument(
        "--parallel", "-j", type=int, default=1,
        help="Max concurrent cases (default: 1, sequential). Try 4-8 for full suite.",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Write full per-case traces as JSONL to this path",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(1)

    # Disable library write-back: evals should not pollute library/*.json.
    # This is also a hard prerequisite for parallel execution (the library
    # caching layer does unprotected read-modify-write on JSON files).
    set_library_readonly(True)

    if args.verbose and args.parallel > 1:
        print(
            "Warning: --verbose with --parallel>1 will interleave output across "
            "cases. Consider --parallel 1 for readable traces.",
            file=sys.stderr,
        )

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
    elif args.filter:
        cases = [c for c in AGENT_CASES if c.id.startswith(args.filter)]
    else:
        cases = AGENT_CASES

    if not cases:
        print("No test cases matched the filter.", file=sys.stderr)
        sys.exit(1)

    judge_info = f", judge={args.judge_model}" if args.judge else ""
    parallel_info = f", parallel={args.parallel}" if args.parallel > 1 else ""
    output_info = f", output={args.output}" if args.output else ""
    print(
        f"Running {len(cases)} agent eval case(s) with model "
        f"{args.model}{judge_info}{parallel_info}{output_info}..."
    )

    eval_output = asyncio.run(run_agent_eval_suite(
        cases, model=args.model, verbose=args.verbose,
        use_judge=args.judge, judge_model=args.judge_model,
        concurrency=args.parallel, output_path=args.output,
    ))

    if args.json:
        print(json.dumps(eval_output, indent=2))
    else:
        print_agent_summary_table(eval_output)


if __name__ == "__main__":
    main()
