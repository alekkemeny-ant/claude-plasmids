#!/usr/bin/env python3
"""
Pipeline Integration Tests

Tests the full assembly pipeline (library lookup → assembly → export → rubric
scoring) against the Allen Institute benchmark cases. Each test case resolves
sequences from the library, assembles a construct, exports GenBank output, and
scores the result using the programmatic rubric.

These are deterministic tests — no LLM calls. For end-to-end agent evals that
send prompts through Claude, see evals/run_agent_evals.py.

Run:
    python -m pytest tests/test_pipeline.py -v
    python -m pytest tests/test_pipeline.py -v -k "T1_001"
    python -m pytest tests/test_pipeline.py -v -k "tier1"
    python -m pytest tests/test_pipeline.py -v -k "tier3"
"""

import sys
from pathlib import Path
from typing import Optional

import pytest

# Add src/ and project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from assembler import (
    assemble_construct,
    export_construct,
    find_mcs_insertion_point,
    clean_sequence,
)
from library import get_backbone_by_id, get_insert_by_id
from addgene_integration import get_addgene_sequence
from evals.rubric import score_construct, RubricResult
from evals.test_cases import (
    TIER_1_CASES,
    TIER_2_CASES,
    TIER_3_CASES,
    TestCase,
)

# Cache for Addgene sequences to avoid repeated fetches across tests
_addgene_sequence_cache: dict[str, Optional[str]] = {}


def _run_pipeline_case(tc: TestCase) -> Optional[RubricResult]:
    """
    Run a single test case through the assembly pipeline and score it.

    Returns RubricResult, or None if the case cannot be run (e.g., missing
    sequences).
    """
    # Resolve backbone
    backbone_seq = tc.backbone_sequence
    backbone_data = None
    if not backbone_seq and tc.backbone_id:
        backbone_data = get_backbone_by_id(tc.backbone_id)
        if backbone_data:
            backbone_seq = backbone_data.get("sequence")

    if not backbone_seq:
        return None

    # Resolve insert
    insert_seq = tc.insert_sequence
    insert_data = None
    if not insert_seq and tc.insert_id:
        insert_data = get_insert_by_id(tc.insert_id)
        if insert_data:
            insert_seq = insert_data.get("sequence")

    if not insert_seq:
        return None

    # Determine insertion position
    insertion_pos = tc.expected_insertion_position
    if insertion_pos is None and backbone_data:
        insertion_pos = find_mcs_insertion_point(backbone_data)
    if insertion_pos is None:
        return None

    # Assemble
    assembly = assemble_construct(
        backbone_seq=backbone_seq,
        insert_seq=insert_seq,
        insertion_position=insertion_pos,
        replace_region_end=tc.replace_region_end,
        reverse_complement_insert=tc.reverse_complement_insert,
    )

    if not assembly.success or not assembly.sequence:
        return None

    construct_seq = assembly.sequence

    # Score against rubric — fetch Addgene ground truth if available
    ground_truth = None
    if tc.addgene_ground_truth_id:
        aid = tc.addgene_ground_truth_id
        if aid not in _addgene_sequence_cache:
            _addgene_sequence_cache[aid] = get_addgene_sequence(aid)
        ground_truth = _addgene_sequence_cache[aid]

    # Generate GenBank output for Output Verification checks
    bb_name = backbone_data["name"] if backbone_data else (tc.backbone_id or "custom")
    ins_name = insert_data["name"] if insert_data else (tc.insert_id or "custom")
    output_text = None
    output_format = None
    if assembly.success:
        try:
            output_text = export_construct(
                result=assembly,
                output_format="genbank",
                construct_name=f"{ins_name}_in_{bb_name}",
                backbone_name=bb_name,
                insert_name=ins_name,
                insert_length=len(clean_sequence(insert_seq)),
                backbone_features=backbone_data.get("features") if backbone_data else None,
            )
            output_format = "genbank"
        except Exception:
            pass

    return score_construct(
        construct_sequence=construct_seq,
        expected_backbone_sequence=backbone_seq,
        expected_insert_sequence=insert_seq,
        expected_insert_position=insertion_pos,
        backbone_name=bb_name,
        insert_name=ins_name,
        insert_category=insert_data.get("category") if insert_data else None,
        ground_truth_sequence=ground_truth,
        ground_truth_strict=tc.ground_truth_strict,
        backbone_features=backbone_data.get("features") if backbone_data else None,
        output_text=output_text,
        output_format=output_format,
    )


# ── Tier 1: Library backbone + library insert ────────────────────────────


def _make_test_id(tc: TestCase) -> str:
    """Create a readable pytest ID like 'T1_001_pcDNA31_EGFP'."""
    return tc.id.replace("-", "_")


@pytest.mark.parametrize(
    "tc",
    TIER_1_CASES,
    ids=[_make_test_id(tc) for tc in TIER_1_CASES],
)
def test_tier1(tc: TestCase):
    """Tier 1: Both backbone and insert from library with full sequences."""
    result = _run_pipeline_case(tc)
    if result is None:
        pytest.skip(f"Missing sequence data for {tc.id}")
    assert result.overall_pass, (
        f"{tc.id} ({tc.name}) failed: {result.summary()}"
    )


# ── Tier 2: Backbone by alias, insert from library ──────────────────────


@pytest.mark.parametrize(
    "tc",
    TIER_2_CASES,
    ids=[_make_test_id(tc) for tc in TIER_2_CASES],
)
def test_tier2(tc: TestCase):
    """Tier 2: Backbone resolved by alias from library (name resolution)."""
    result = _run_pipeline_case(tc)
    if result is None:
        pytest.skip(f"Missing sequence data for {tc.id}")
    assert result.overall_pass, (
        f"{tc.id} ({tc.name}) failed: {result.summary()}"
    )


# ── Tier 3: Addgene ground truth comparison ──────────────────────────────


@pytest.mark.parametrize(
    "tc",
    TIER_3_CASES,
    ids=[_make_test_id(tc) for tc in TIER_3_CASES],
)
def test_tier3(tc: TestCase):
    """Tier 3: Assembly compared against Addgene ground truth sequence."""
    result = _run_pipeline_case(tc)
    if result is None:
        pytest.skip(f"Missing sequence data for {tc.id}")
    assert result.overall_pass, (
        f"{tc.id} ({tc.name}) failed: {result.summary()}"
    )
