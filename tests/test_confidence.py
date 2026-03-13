#!/usr/bin/env python3
"""Tests for the Design Confidence Score module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from confidence import (
    ConfidenceCheck,
    ConfidenceReport,
    check_cai,
    check_cryptic_polya,
    check_cryptic_splice,
    check_fusion_linker,
    check_gc_content,
    check_kozak,
    check_promoter_count,
    check_repeat_runs,
    compute_cai,
    compute_confidence,
    format_confidence_report,
)
from codon_tables import HUMAN_CODON_W, HUMAN_OPTIMAL_CODONS


# ── Helper: build a "clean" CDS of given length using optimal codons ───


def _make_optimal_cds(n_codons: int) -> str:
    """Build a CDS using the human-optimal codon for Ala (GCC) repeated."""
    return "GCC" * n_codons


def _make_poor_cds(n_codons: int) -> str:
    """Build a CDS using a rare codon for Ala (GCG, w~0.27)."""
    return "GCG" * n_codons


def _make_balanced_gc_seq(length: int) -> str:
    """Return a sequence with ~50% GC content."""
    unit = "ATGC"
    return (unit * (length // 4 + 1))[:length]


# ── Codon tables sanity ───────────────────────────────────────────────


class TestCodonTables:
    def test_all_61_sense_codons(self):
        assert len(HUMAN_CODON_W) == 61

    def test_optimal_codons_have_w_1(self):
        for aa, codon in HUMAN_OPTIMAL_CODONS.items():
            assert HUMAN_CODON_W[codon] == 1.0, f"{aa} optimal codon {codon} w={HUMAN_CODON_W[codon]}"

    def test_serine_grouped_correctly(self):
        """Serine has 6 codons (TCN + AGT/AGC); max should be AGC (19.5)."""
        ser_codons = ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"]
        ws = [HUMAN_CODON_W[c] for c in ser_codons]
        assert max(ws) == 1.0
        assert HUMAN_CODON_W["AGC"] == 1.0

    def test_arginine_grouped_correctly(self):
        """Arg has 6 codons (CGN + AGA/AGG); max should be AGA (12.2)."""
        arg_codons = ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"]
        ws = [HUMAN_CODON_W[c] for c in arg_codons]
        assert max(ws) == 1.0
        assert HUMAN_CODON_W["AGA"] == 1.0

    def test_stop_codons_not_in_table(self):
        for stop in ["TAA", "TAG", "TGA"]:
            assert stop not in HUMAN_CODON_W


# ── Cryptic polyA ─────────────────────────────────────────────────────


class TestCrypticPolyA:
    def test_clean_sequence_passes(self):
        seq = _make_balanced_gc_seq(500)
        result = check_cryptic_polya(seq)
        assert result.passed is True
        assert result.score_delta == 0

    def test_aataaa_in_body_fails(self):
        # Place AATAAA at position 100 in a 500bp seq
        seq = "G" * 100 + "AATAAA" + "G" * 394
        result = check_cryptic_polya(seq)
        assert result.passed is False
        assert result.severity == "critical"
        assert result.score_delta == -15
        assert result.position == 100

    def test_attaaa_in_body_fails(self):
        seq = "G" * 50 + "ATTAAA" + "G" * 444
        result = check_cryptic_polya(seq)
        assert result.passed is False
        assert result.score_delta == -15

    def test_polya_in_last_150bp_passes(self):
        """A polyA signal in the last 150bp is expected/harmless."""
        seq = "G" * 400 + "AATAAA" + "G" * 94
        assert len(seq) == 500
        result = check_cryptic_polya(seq)
        assert result.passed is True

    def test_short_sequence_no_search_region(self):
        seq = "AATAAA" + "G" * 50
        result = check_cryptic_polya(seq)
        # Sequence is <= 150bp so search region is empty
        assert result.passed is True

    def test_multiple_signals_reported(self):
        seq = "AATAAA" + "G" * 100 + "ATTAAA" + "G" * 388
        result = check_cryptic_polya(seq)
        assert result.passed is False
        assert "2" in result.message


# ── Cryptic splice sites ─────────────────────────────────────────────


class TestCrypticSplice:
    def test_clean_sequence_passes(self):
        seq = _make_balanced_gc_seq(500)
        result = check_cryptic_splice(seq)
        assert result.passed is True
        assert result.score_delta == 0

    def test_strong_donor_fails(self):
        # Strong 5' donor: [AC]AGGT[AG]AGT
        seq = "G" * 100 + "AAGGTAAGT" + "G" * 100
        result = check_cryptic_splice(seq)
        assert result.passed is False
        assert "splice donor" in result.message

    def test_strong_acceptor_fails(self):
        # Strong 3' acceptor: [CT]{6,}[ACGT]CAGG
        seq = "G" * 100 + "CCCCCCACAGG" + "G" * 100
        result = check_cryptic_splice(seq)
        assert result.passed is False
        assert "acceptor" in result.message

    def test_penalty_capped_at_minus_20(self):
        # 3 donors = -24, but should be capped at -20
        donor = "AAGGTAAGT"
        seq = donor + "G" * 50 + donor + "G" * 50 + donor + "G" * 50
        result = check_cryptic_splice(seq)
        assert result.score_delta == -20


# ── CAI ──────────────────────────────────────────────────────────────


class TestCAI:
    def test_all_optimal_codons_near_1(self):
        # All-optimal codons should give CAI ≈ 1.0
        # Use a mix of optimal codons (all have w=1.0)
        optimal = "".join(HUMAN_OPTIMAL_CODONS[aa] for aa in "MFLIVAGPTSYCHQNKDEW")
        cai = compute_cai(optimal)
        assert cai > 0.99

    def test_poor_codons_below_0_5(self):
        # All-rare codons: use the worst for each AA
        poor_seq = "GCG" * 20  # GCG w≈0.267
        cai = compute_cai(poor_seq)
        assert cai < 0.5

    def test_empty_sequence(self):
        assert compute_cai("") == 0.0

    def test_short_sequence(self):
        assert compute_cai("AT") == 0.0  # < 3bp

    def test_single_codon(self):
        cai = compute_cai("GCC")  # optimal Ala, w=1.0
        assert cai == 1.0

    def test_check_cai_good(self):
        seq = _make_optimal_cds(100)
        result = check_cai(seq)
        assert result.passed is True
        assert result.score_delta == 0
        assert "good" in result.message.lower()

    def test_check_cai_poor(self):
        seq = _make_poor_cds(100)
        result = check_cai(seq)
        assert result.passed is False
        assert result.score_delta == -10

    def test_check_cai_moderate(self):
        # Mix optimal and poor to get moderate CAI (0.6-0.8)
        # GCC (w=1.0) and GCA (w≈0.570) → geometric mean ≈ 0.755
        seq = ("GCC" + "GCA") * 50
        result = check_cai(seq)
        assert result.passed is True
        assert result.score_delta == -3


# ── GC content ───────────────────────────────────────────────────────


class TestGCContent:
    def test_50_percent_passes(self):
        seq = "ATGC" * 100
        result = check_gc_content(seq)
        assert result.passed is True
        assert result.score_delta == 0

    def test_20_percent_fails(self):
        # 20% GC: 80 A/T + 20 G/C
        seq = "A" * 80 + "G" * 20
        result = check_gc_content(seq)
        assert result.passed is False
        assert result.score_delta == -5
        assert "below" in result.message.lower()

    def test_80_percent_fails(self):
        seq = "G" * 80 + "A" * 20
        result = check_gc_content(seq)
        assert result.passed is False
        assert result.score_delta == -5
        assert "above" in result.message.lower()

    def test_boundary_35_passes(self):
        # Exactly 35% GC
        seq = "A" * 65 + "G" * 35
        result = check_gc_content(seq)
        assert result.passed is True

    def test_boundary_65_passes(self):
        seq = "G" * 65 + "A" * 35
        result = check_gc_content(seq)
        assert result.passed is True

    def test_empty_sequence(self):
        result = check_gc_content("")
        assert result.passed is True


# ── Kozak context ────────────────────────────────────────────────────


class TestKozak:
    def test_strong_kozak(self):
        # GCCACCATGG: -3=A (purine), +4=G
        seq = "GCCACCATGG" + "G" * 100
        result = check_kozak(seq)
        assert result.passed is True
        assert result.score_delta == 0
        assert "strong" in result.message.lower()

    def test_weak_kozak_no_upstream(self):
        # ATG at position 0 — no upstream context
        seq = "ATGC" + "G" * 100
        result = check_kozak(seq)
        # -3 is empty, +4 is C — neither condition met
        # Actually +4 is seq[3] = 'C', not G, and -3 is empty
        assert result.score_delta < 0

    def test_no_atg(self):
        seq = "GGGCCCTTTT" * 10
        result = check_kozak(seq)
        assert result.passed is False
        assert "No ATG" in result.message

    def test_moderate_kozak_minus3_only(self):
        # -3 is A (purine) but +4 is not G
        seq = "CCCACCATGC" + "G" * 100  # -3=A, +4=C
        result = check_kozak(seq)
        assert result.score_delta == -2

    def test_moderate_kozak_plus4_only(self):
        # -3 is T (not purine) but +4 is G
        seq = "CCCTCCATGG" + "G" * 100  # -3=T, +4=G
        result = check_kozak(seq)
        assert result.score_delta == -2


# ── Repeat runs ──────────────────────────────────────────────────────


class TestRepeatRuns:
    def test_10_a_run_fails(self):
        seq = "GCC" * 20 + "A" * 10 + "GCC" * 20
        result = check_repeat_runs(seq)
        assert result.passed is False
        assert result.score_delta == -3
        assert "10" in result.message

    def test_7_a_run_passes(self):
        seq = "GCC" * 20 + "A" * 7 + "GCC" * 20
        result = check_repeat_runs(seq)
        assert result.passed is True
        assert result.score_delta == 0

    def test_8_run_passes(self):
        """Exactly 8 is at the boundary — passes (>8 triggers)."""
        seq = "GCC" * 20 + "A" * 8 + "GCC" * 20
        result = check_repeat_runs(seq)
        assert result.passed is True

    def test_9_run_fails(self):
        seq = "GCC" * 20 + "T" * 9 + "GCC" * 20
        result = check_repeat_runs(seq)
        assert result.passed is False

    def test_multiple_runs(self):
        seq = "A" * 10 + "GCC" * 10 + "T" * 12 + "GCC" * 10
        result = check_repeat_runs(seq)
        assert result.passed is False
        assert "2" in result.message  # 2 runs
        assert "12" in result.message  # longest = 12


# ── Fusion linker ────────────────────────────────────────────────────


class TestFusionLinker:
    def test_single_insert_passes(self):
        result = check_fusion_linker(None)
        assert result.passed is True

    def test_empty_list_passes(self):
        result = check_fusion_linker([])
        assert result.passed is True

    def test_two_large_domains_no_linker_fails(self):
        parts = [
            {"name": "GFP", "aa_length": 240, "is_linker": False},
            {"name": "mCherry", "aa_length": 230, "is_linker": False},
        ]
        result = check_fusion_linker(parts)
        assert result.passed is False
        assert result.score_delta == -8

    def test_two_large_domains_with_20aa_linker_passes(self):
        parts = [
            {"name": "GFP", "aa_length": 240, "is_linker": False},
            {"name": "linker", "aa_length": 20, "is_linker": True},
            {"name": "mCherry", "aa_length": 230, "is_linker": False},
        ]
        result = check_fusion_linker(parts)
        assert result.passed is True
        assert result.score_delta == 0

    def test_two_large_domains_short_linker_warning(self):
        parts = [
            {"name": "GFP", "aa_length": 240, "is_linker": False},
            {"name": "linker", "aa_length": 10, "is_linker": True},
            {"name": "mCherry", "aa_length": 230, "is_linker": False},
        ]
        result = check_fusion_linker(parts)
        assert result.passed is True
        assert result.score_delta == -2

    def test_small_domains_no_penalty(self):
        parts = [
            {"name": "tag1", "aa_length": 30, "is_linker": False},
            {"name": "tag2", "aa_length": 50, "is_linker": False},
        ]
        result = check_fusion_linker(parts)
        assert result.passed is True
        assert result.score_delta == 0


# ── Promoter count ───────────────────────────────────────────────────


class TestPromoterCount:
    def test_no_backbone_passes(self):
        result = check_promoter_count(None)
        assert result.passed is True

    def test_two_promoters_passes(self):
        backbone = {
            "features": [
                {"type": "promoter", "name": "CMV"},
                {"type": "promoter", "name": "SV40"},
            ]
        }
        result = check_promoter_count(backbone)
        assert result.passed is True
        assert result.score_delta == 0

    def test_four_promoters_fails(self):
        backbone = {
            "features": [
                {"type": "promoter", "name": "CMV"},
                {"type": "promoter", "name": "SV40"},
                {"type": "promoter", "name": "EF1a"},
                {"type": "promoter", "name": "PGK"},
            ]
        }
        result = check_promoter_count(backbone)
        assert result.passed is False
        assert result.score_delta == -5
        assert "4" in result.message

    def test_promoter_detected_by_name(self):
        backbone = {
            "features": [
                {"type": "regulatory", "name": "CMV promoter"},
                {"type": "regulatory", "name": "SV40 promoter"},
                {"type": "regulatory", "name": "EF1a promoter"},
            ]
        }
        result = check_promoter_count(backbone)
        assert result.passed is False

    def test_empty_features_passes(self):
        backbone = {"features": []}
        result = check_promoter_count(backbone)
        assert result.passed is True


# ── Composite confidence ─────────────────────────────────────────────


class TestComputeConfidence:
    def test_perfect_sequence_high_score(self):
        # Optimal codons, good GC, strong Kozak, no issues
        kozak = "GCCACC"
        cds = "".join(HUMAN_OPTIMAL_CODONS[aa] for aa in "MLIVAGPTSYCHQNKDEW" * 5)
        # Pad to >150bp and ensure no polyA / repeats
        seq = kozak + cds
        while len(seq) < 300:
            seq += "GCC"
        report = compute_confidence(seq)
        assert report.overall_score >= 85
        assert "High" in report.summary

    def test_bad_sequence_low_score(self):
        # polyA signal in body + poor CAI + bad GC
        bad_cds = "GCG" * 50  # poor CAI (~0.27)
        seq = "AATAAA" + bad_cds + "A" * 200  # polyA in body + low GC
        report = compute_confidence(seq)
        assert report.overall_score < 70
        assert any(c.severity == "critical" for c in report.checks)

    def test_score_clamped_to_0_100(self):
        # Even with many penalties, score shouldn't go negative
        seq = "A" * 500  # low GC, polyA everywhere, repeat runs, no ATG
        report = compute_confidence(seq)
        assert 0 <= report.overall_score <= 100

    def test_backbone_and_fusion_parts(self):
        seq = "GCCACC" + "GCC" * 100
        backbone = {
            "features": [
                {"type": "promoter", "name": "P1"},
                {"type": "promoter", "name": "P2"},
                {"type": "promoter", "name": "P3"},
            ]
        }
        fusion = [
            {"name": "A", "aa_length": 200, "is_linker": False},
            {"name": "B", "aa_length": 200, "is_linker": False},
        ]
        report = compute_confidence(seq, backbone=backbone, fusion_parts=fusion)
        # Should have penalties from promoters and fusion linker
        promoter_check = next(c for c in report.checks if c.name == "Promoter count")
        linker_check = next(c for c in report.checks if c.name == "Fusion linker adequacy")
        assert promoter_check.passed is False
        assert linker_check.passed is False

    def test_recommendation_mentions_critical(self):
        seq = "AATAAA" + "G" * 400
        report = compute_confidence(seq)
        assert "Critical" in report.recommendation or "critical" in report.recommendation.lower()

    def test_whitespace_stripped(self):
        seq1 = "GCC" * 100
        seq2 = "GCC " * 50 + "\nGCC" * 50
        r1 = compute_confidence(seq1)
        r2 = compute_confidence(seq2)
        assert r1.overall_score == r2.overall_score


# ── Format report ────────────────────────────────────────────────────


class TestFormatReport:
    def test_contains_header(self):
        seq = "GCCACC" + "GCC" * 100
        report = compute_confidence(seq)
        text = format_confidence_report(report)
        assert "Design Confidence:" in text

    def test_contains_check_marks(self):
        seq = "GCCACC" + "GCC" * 100
        report = compute_confidence(seq)
        text = format_confidence_report(report)
        # Should have at least one check mark (passed checks)
        assert "\u2713" in text or "\u26a0" in text

    def test_contains_recommendation(self):
        seq = "GCCACC" + "GCC" * 100
        report = compute_confidence(seq)
        text = format_confidence_report(report)
        assert "Recommendation:" in text

    def test_category_labels_present(self):
        seq = "GCCACC" + "GCC" * 100
        report = compute_confidence(seq)
        text = format_confidence_report(report)
        assert "Expression optimality:" in text
