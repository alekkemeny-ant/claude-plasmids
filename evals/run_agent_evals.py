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
            "correct codon management (FLAG provides ATG, EGFP provides stop)."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="EGFP",
        tags=["fusion", "mammalian", "epitope_tag", "n_terminal"],
    ),
    AgentTestCase(
        id="A7-002",
        name="Fusion: C-terminal HA-mCherry",
        prompt=(
            "Express a C-terminal HA-tagged mCherry in pcDNA3.1(+). "
            "Assemble the construct and return the sequence."
        ),
        description=(
            "Agent must fuse mCherry + HA_tag using fuse_inserts, then assemble "
            "into pcDNA3.1(+). Tests C-terminal fusion with correct codon management."
        ),
        expected_backbone_id="pcDNA3.1(+)",
        expected_insert_id="mCherry",
        tags=["fusion", "mammalian", "epitope_tag", "c_terminal"],
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

    # ── Transcript assertions (conversational quality grading) ────────
    # For disambiguation cases, check that the agent asked the right
    # clarifying questions in its text output.
    transcript_results = []
    if tc.transcript_assertions:
        for assertion in tc.transcript_assertions:
            found = assertion.lower() in trace.assistant_text.lower()
            transcript_results.append((assertion, found))
            if verbose:
                status = "FOUND" if found else "MISSING"
                print(f"  [transcript] '{assertion}': {status}")

    # ── Tool negative assertions (balanced eval grading) ──────────────
    # Check that the agent did NOT call tools it shouldn't have.
    tool_violation_results = []
    if tc.tools_should_not_use:
        tools_used = {t["tool"] for t in trace.tool_calls}
        for forbidden_tool in tc.tools_should_not_use:
            was_used = forbidden_tool in tools_used
            tool_violation_results.append((forbidden_tool, was_used))
            if verbose:
                status = "VIOLATION — used" if was_used else "OK — not used"
                print(f"  [tool_negative] {forbidden_tool}: {status}")

    if not trace.assembled_sequence:
        if verbose:
            print(f"  No assembled sequence found in agent output")
            print(f"  Tool calls made: {[t['tool'] for t in trace.tool_calls]}")
        # For disambiguation cases, no assembled sequence is expected
        # (agent should ask questions, not guess). Still report transcript results.
        if tc.transcript_assertions and transcript_results:
            passed_assertions = sum(1 for _, found in transcript_results if found)
            total_assertions = len(transcript_results)
            if verbose:
                print(f"  Transcript assertions: {passed_assertions}/{total_assertions} passed")
        return None, trace

    if verbose:
        print(f"  Extracted sequence: {len(trace.assembled_sequence)} bp")
        print(f"  Tool calls: {len(trace.tool_calls)} ({', '.join(t['tool'] for t in trace.tool_calls)})")

    # Resolve expected values for scoring
    backbone_data = get_backbone_by_id(tc.expected_backbone_id)

    if not backbone_data or not backbone_data.get("sequence"):
        if verbose:
            print(f"  SKIP: No backbone sequence for '{tc.expected_backbone_id}'")
        return None, trace

    backbone_seq = backbone_data["sequence"]

    # Resolve insert sequence: prefer inline ground truth, fall back to library
    insert_seq = tc.expected_insert_sequence
    insert_data = None
    if not insert_seq:
        insert_data = get_insert_by_id(tc.expected_insert_id)
        if not insert_data or not insert_data.get("sequence"):
            if verbose:
                print(f"  SKIP: No insert sequence for '{tc.expected_insert_id}'")
            return None, trace
        insert_seq = insert_data["sequence"]

    insertion_pos = tc.expected_insertion_position
    if insertion_pos is None:
        insertion_pos = find_mcs_insertion_point(backbone_data)

    if insertion_pos is None:
        if verbose:
            print(f"  SKIP: Cannot determine insertion position")
        return None, trace

    insert_name = insert_data["name"] if insert_data else tc.expected_insert_id
    insert_category = insert_data.get("category") if insert_data else None

    # Score with rubric
    rubric_result = score_construct(
        construct_sequence=trace.assembled_sequence,
        expected_backbone_sequence=backbone_seq,
        expected_insert_sequence=insert_seq,
        expected_insert_position=insertion_pos,
        backbone_name=backbone_data["name"],
        insert_name=insert_name,
        insert_category=insert_category,
        backbone_features=backbone_data.get("features"),
    )

    # ── Add tool violation checks to rubric (negative/balanced cases) ────
    if tool_violation_results:
        from evals.rubric import Check
        for tool_name, was_used in tool_violation_results:
            rubric_result.checks.append(Check(
                section="Tool Routing",
                name=f"Agent should NOT use {tool_name}",
                severity="Major",
                passed=not was_used,
                detail=f"Agent {'used' if was_used else 'correctly avoided'} {tool_name}",
            ))

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
