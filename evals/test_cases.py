#!/usr/bin/env python3
"""
Benchmark Test Cases for Plasmid Assembly Evaluation

Each test case specifies:
  - inputs (backbone + insert, by library ID or raw sequence)
  - expected assembly parameters (insertion position, orientation)
  - optional ground truth (Addgene ID of known-correct plasmid)

Test case difficulty tiers (from Allen Institute presentation):
  Tier 1: Both backbone and insert from library with full sequences
  Tier 2: Backbone by name (needs library/Addgene lookup), insert from library
  Tier 3: Backbone from Addgene ID, insert from library
  Tier 4: Natural language ("make a GFP expression plasmid")

This module defines Tier 1-3 test cases. Tier 4 requires LLM orchestration
and is tested via the full agent pipeline, not the assembly engine alone.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestCase:  # noqa: pytest collection disabled via __init__
    """A single benchmark test case."""
    __test__ = False  # Prevent pytest from collecting this as a test class
    id: str
    name: str
    description: str
    tier: int  # 1-4

    # Inputs — at least one of (backbone_id, backbone_sequence) required
    backbone_id: Optional[str] = None
    backbone_sequence: Optional[str] = None
    insert_id: Optional[str] = None
    insert_sequence: Optional[str] = None

    # Expected assembly parameters
    expected_insertion_position: Optional[int] = None  # 0-based; None = use MCS start
    reverse_complement_insert: bool = False
    replace_region_end: Optional[int] = None

    # Ground truth
    addgene_ground_truth_id: Optional[str] = None  # Addgene ID of known-correct plasmid
    ground_truth_strict: bool = False  # If True, exact match is Critical; if False, Info only
    expected_total_size: Optional[int] = None

    # Tags for filtering
    tags: list[str] = field(default_factory=list)


# ── Tier 1: Library backbone + library insert ───────────────────────────

TIER_1_CASES = [
    TestCase(
        id="T1-001",
        name="pcDNA3.1(+) + EGFP",
        description=(
            "Primary benchmark. CMV-driven EGFP expression in mammalian cells. "
            "Both sequences available in library. This is the same test case that "
            "CRISPR-GPT scored 96% on (off by 1 nt)."
        ),
        tier=1,
        backbone_id="pcDNA3.1(+)",
        insert_id="EGFP",
        expected_insertion_position=895,  # MCS start
        expected_total_size=6148,  # 5428 + 720
        tags=["primary", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-002",
        name="pcDNA3.1(+) + mCherry",
        description="CMV-driven mCherry expression. Tests a different FP insert.",
        tier=1,
        backbone_id="pcDNA3.1(+)",
        insert_id="mCherry",
        expected_insertion_position=895,
        expected_total_size=6139,  # 5428 + 711
        tags=["mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-003",
        name="pcDNA3.1(+) + mNeonGreen",
        description="CMV-driven mNeonGreen expression. Brightest monomeric green FP.",
        tier=1,
        backbone_id="pcDNA3.1(+)",
        insert_id="mNeonGreen",
        expected_insertion_position=895,
        expected_total_size=6142,  # 5428 + 714
        tags=["mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-004",
        name="pcDNA3.1(+) + tdTomato",
        description="CMV-driven tdTomato expression. Larger tandem dimer insert (1431 bp).",
        tier=1,
        backbone_id="pcDNA3.1(+)",
        insert_id="tdTomato",
        expected_insertion_position=895,
        expected_total_size=6859,  # 5428 + 1431
        tags=["mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-005",
        name="pcDNA3.1(+) + mTagBFP2",
        description="CMV-driven mTagBFP2 expression. Blue fluorescent protein.",
        tier=1,
        backbone_id="pcDNA3.1(+)",
        insert_id="mTagBFP2",
        expected_insertion_position=895,
        expected_total_size=6142,  # 5428 + 714
        tags=["mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-006",
        name="pcDNA3.1(+) + Firefly Luciferase",
        description="CMV-driven firefly luciferase reporter. Larger insert (1653 bp).",
        tier=1,
        backbone_id="pcDNA3.1(+)",
        insert_id="Firefly_Luciferase",
        expected_insertion_position=895,
        expected_total_size=7081,  # 5428 + 1653
        tags=["mammalian", "reporter"],
    ),
    TestCase(
        id="T1-007",
        name="pUC19 + EGFP",
        description="Bacterial cloning vector with EGFP. Tests a different backbone.",
        tier=1,
        backbone_id="pUC19",
        insert_id="EGFP",
        expected_insertion_position=396,  # pUC19 MCS start
        expected_total_size=3406,  # 2686 + 720
        tags=["bacterial", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-008",
        name="pcDNA3.1(+) + FLAG tag",
        description="CMV vector with FLAG epitope tag. Very short insert (24 bp).",
        tier=1,
        backbone_id="pcDNA3.1(+)",
        insert_id="FLAG_tag",
        expected_insertion_position=895,
        expected_total_size=5452,  # 5428 + 24
        tags=["mammalian", "epitope_tag"],
    ),
    # ── New backbone coverage ──────────────────────────────────────────
    TestCase(
        id="T1-009",
        name="pcDNA3.1(-) + EGFP",
        description="Reverse-orientation MCS mammalian vector with EGFP. Tests pcDNA3.1(-) backbone.",
        tier=1,
        backbone_id="pcDNA3.1(-)",
        insert_id="EGFP",
        expected_insertion_position=895,  # MCS start
        expected_total_size=6224,  # 5504 + 720
        tags=["mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-010",
        name="pEGFP-N1 + mCherry",
        description=(
            "C-terminal fusion vector with mCherry inserted into MCS upstream of EGFP. "
            "Tests pEGFP-N1 backbone (CMV promoter, Kanamycin resistance)."
        ),
        tier=1,
        backbone_id="pEGFP-N1",
        insert_id="mCherry",
        expected_insertion_position=591,  # MCS start
        expected_total_size=5444,  # 4733 + 711
        tags=["mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-011",
        name="pcDNA3 + EGFP",
        description="Earlier pcDNA3 vector with EGFP. Tests a different CMV backbone variant.",
        tier=1,
        backbone_id="pCDNA3",
        insert_id="EGFP",
        expected_insertion_position=900,  # MCS start
        expected_total_size=6173,  # 5453 + 720
        tags=["mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-012",
        name="pGEX-4T-1 + EGFP",
        description=(
            "Bacterial GST-fusion vector with EGFP. Tests a bacterial expression backbone "
            "with tac promoter and GST fusion reading frame."
        ),
        tier=1,
        backbone_id="pGEX-4T-1",
        insert_id="EGFP",
        expected_insertion_position=930,  # MCS start
        expected_total_size=5689,  # 4969 + 720
        tags=["bacterial", "fluorescent_protein"],
    ),
    TestCase(
        id="T1-013",
        name="pBABE-puro + EGFP",
        description="Retroviral expression vector with EGFP. Tests LTR-driven backbone with puromycin selection.",
        tier=1,
        backbone_id="pBABE-puro",
        insert_id="EGFP",
        expected_insertion_position=1260,  # MCS start
        expected_total_size=5806,  # 5086 + 720
        tags=["mammalian", "fluorescent_protein", "retroviral"],
    ),
    TestCase(
        id="T1-014",
        name="pAAV-CMV + Firefly Luciferase",
        description="AAV transfer vector with luciferase reporter. Tests AAV backbone for gene therapy applications.",
        tier=1,
        backbone_id="pAAV-CMV",
        insert_id="Firefly_Luciferase",
        expected_insertion_position=1200,  # MCS start
        expected_total_size=5905,  # 4252 + 1653
        tags=["mammalian", "reporter", "aav"],
    ),
    TestCase(
        id="T1-015",
        name="pLKO.1-puro + EGFP",
        description=(
            "Lentiviral shRNA vector with EGFP inserted at the shRNA cloning site. "
            "Tests lentiviral backbone with U6 promoter."
        ),
        tier=1,
        backbone_id="pLKO.1-puro",
        insert_id="EGFP",
        expected_insertion_position=1878,  # shRNA cloning site start
        expected_total_size=7770,  # 7050 + 720
        tags=["mammalian", "fluorescent_protein", "lentiviral"],
    ),
    TestCase(
        id="T1-016",
        name="pcDNA3.1(-) + HA tag",
        description="Reverse-orientation MCS vector with HA epitope tag. Tests short insert in pcDNA3.1(-).",
        tier=1,
        backbone_id="pcDNA3.1(-)",
        insert_id="HA_tag",
        expected_insertion_position=895,
        expected_total_size=5531,  # 5504 + 27
        tags=["mammalian", "epitope_tag"],
    ),
]

# ── Tier 2: Backbone by name (needs lookup), insert from library ────────
# These test that the system can resolve a backbone from a name/description
# without the user providing a raw sequence. The assembly itself is the same,
# but the orchestration layer must find the right backbone.

TIER_2_CASES = [
    TestCase(
        id="T2-001",
        name="pcDNA3.1+ (by name) + EGFP",
        description=(
            "Same as T1-001 but backbone specified by name string only. "
            "Tests the system's ability to resolve 'pcDNA3.1+' to the correct "
            "library entry. This is analogous to the CRISPR-GPT Test 2 that failed."
        ),
        tier=2,
        backbone_id="pcDNA3.1+",  # alias, not canonical ID
        insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["name_resolution", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T2-002",
        name="pcDNA3.1 plus (by name) + mCherry",
        description="Backbone specified as 'pcDNA3.1 plus' — tests alias matching.",
        tier=2,
        backbone_id="pcDNA3.1 plus",  # another alias
        insert_id="mCherry",
        expected_insertion_position=895,
        expected_total_size=6139,
        tags=["name_resolution", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T2-003",
        name="'eGFP' (alias) in pcDNA3.1(+)",
        description="Insert specified as 'eGFP' alias — tests insert name resolution.",
        tier=2,
        backbone_id="pcDNA3.1(+)",
        insert_id="eGFP",  # alias
        expected_insertion_position=895,
        expected_total_size=6148,
        tags=["name_resolution", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T2-004",
        name="'pcDNA3.1-' (alias) + EGFP",
        description="Backbone specified as 'pcDNA3.1-' alias — tests resolution to pcDNA3.1(-).",
        tier=2,
        backbone_id="pcDNA3.1-",  # alias
        insert_id="EGFP",
        expected_insertion_position=895,
        expected_total_size=6224,
        tags=["name_resolution", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T2-005",
        name="'pET28a' (alias) + EGFP",
        description="Backbone specified as 'pET28a' alias — tests resolution to pET-28a(+). Will SKIP (no sequence).",
        tier=2,
        backbone_id="pET28a",  # alias
        insert_id="EGFP",
        expected_insertion_position=158,
        expected_total_size=6089,
        tags=["name_resolution", "bacterial", "fluorescent_protein"],
    ),
    TestCase(
        id="T2-006",
        name="'pBABE puro' (alias) + mCherry",
        description="Backbone specified as 'pBABE puro' alias — tests resolution to pBABE-puro.",
        tier=2,
        backbone_id="pBABE puro",  # alias
        insert_id="mCherry",
        expected_insertion_position=1260,
        expected_total_size=5797,
        tags=["name_resolution", "mammalian", "fluorescent_protein", "retroviral"],
    ),
    TestCase(
        id="T2-007",
        name="'pGEX' (alias) + Renilla Luciferase",
        description="Backbone specified as 'pGEX' alias — tests resolution to pGEX-4T-1.",
        tier=2,
        backbone_id="pGEX",  # alias
        insert_id="Renilla_Luciferase",
        expected_insertion_position=930,
        expected_total_size=5905,  # 4969 + 936
        tags=["name_resolution", "bacterial", "reporter"],
    ),
]

# ── Tier 3: Addgene ground truth comparison ─────────────────────────────
# These use known Addgene plasmids as ground truth. The assembly output is
# compared against the actual deposited sequence. Requires fetching the
# Addgene sequence at eval time (or pre-cached).

TIER_3_CASES = [
    TestCase(
        id="T3-001",
        name="pcDNA3-EGFP (Addgene #13031)",
        description=(
            "Reconstruct pcDNA3-EGFP and compare against Addgene deposited sequence. "
            "From Allen Institute benchmark list. Note: backbone is pcDNA3 (not 3.1), "
            "may need Addgene import for backbone sequence."
        ),
        tier=3,
        backbone_id="pcDNA3.1(+)",  # closest available; real pcDNA3 may differ
        insert_id="EGFP",
        addgene_ground_truth_id="13031",
        tags=["addgene_benchmark", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T3-002",
        name="pcDNA3-mRFP (Addgene #13032)",
        description="Reconstruct pcDNA3-mRFP. From Allen Institute benchmark list.",
        tier=3,
        backbone_id="pcDNA3.1(+)",
        insert_id="mCherry",  # closest to mRFP in our library
        addgene_ground_truth_id="13032",
        tags=["addgene_benchmark", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T3-003",
        name="pcDNA3-CFP (Addgene #13030)",
        description=(
            "Reconstruct pcDNA3-CFP. From Allen Institute benchmark list. "
            "CFP not in our library — uses EGFP as proxy insert (same size GFP variant). "
            "Ground truth comparison expected to fail due to insert mismatch."
        ),
        tier=3,
        backbone_id="pcDNA3.1(+)",
        insert_id="EGFP",  # proxy — CFP not in library
        addgene_ground_truth_id="13030",
        tags=["addgene_benchmark", "mammalian", "fluorescent_protein"],
    ),
    TestCase(
        id="T3-004",
        name="pcDNA3-YFP (Addgene #13033)",
        description=(
            "Reconstruct pcDNA3-YFP. From Allen Institute benchmark list. "
            "YFP not in our library — uses EGFP as proxy insert (same size GFP variant). "
            "Ground truth comparison expected to fail due to insert mismatch."
        ),
        tier=3,
        backbone_id="pcDNA3.1(+)",
        insert_id="EGFP",  # proxy — YFP not in library
        addgene_ground_truth_id="13033",
        tags=["addgene_benchmark", "mammalian", "fluorescent_protein"],
    ),
]


# ── All cases ───────────────────────────────────────────────────────────

ALL_CASES = TIER_1_CASES + TIER_2_CASES + TIER_3_CASES


def get_cases_by_tier(tier: int) -> list[TestCase]:
    return [c for c in ALL_CASES if c.tier == tier]


def get_cases_by_tag(tag: str) -> list[TestCase]:
    return [c for c in ALL_CASES if tag in c.tags]


def get_case_by_id(case_id: str) -> Optional[TestCase]:
    for c in ALL_CASES:
        if c.id == case_id:
            return c
    return None
