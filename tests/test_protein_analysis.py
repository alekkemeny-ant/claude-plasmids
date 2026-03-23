#!/usr/bin/env python3
"""Tests for protein-level analysis (translation, disorder, fusion sites)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein_analysis import (
    translate,
    predict_disorder,
    find_fusion_sites,
    CODON_TABLE,
    STOP_CODONS,
)


# ── Translation tests ──────────────────────────────────────────────────


class TestTranslate:
    def test_simple_cds(self):
        """ATG GCA TTT TAG → MAF (stops at TAG)."""
        assert translate("ATGGCATTTTAG") == "MAF"

    def test_to_stop_false(self):
        """With to_stop=False, include stop codon as '*' and continue."""
        assert translate("ATGTAGATG", to_stop=False) == "M*M"

    def test_frame_shift(self):
        """Frame 1: skip first base, read from position 1."""
        # CATGGCA → frame 1 → ATG GCA → MA
        assert translate("CATGGCA", frame=1) == "MA"

    def test_unknown_codon(self):
        """Codons with N → 'X' (unknown residue)."""
        # ATG NNN TAA → M then X then stop
        assert translate("ATGNNNTAA") == "MX"

    def test_empty(self):
        assert translate("") == ""

    def test_egfp_known(self):
        """First 18 bp of EGFP → known N-terminus MVSKGE."""
        egfp_start = "ATGGTGAGCAAGGGCGAGGAG"
        result = translate(egfp_start)
        assert result.startswith("MVSKGE")

    def test_whitespace_handling(self):
        """Whitespace and newlines should be stripped."""
        assert translate("ATG GCA\nTTT TAG") == "MAF"

    def test_all_stop_codons(self):
        """TAA, TAG, TGA are all stop codons."""
        assert "TAA" in STOP_CODONS
        assert "TAG" in STOP_CODONS
        assert "TGA" in STOP_CODONS
        assert len(STOP_CODONS) == 3

    def test_codon_table_completeness(self):
        """Standard genetic code has 64 codons."""
        assert len(CODON_TABLE) == 64


# ── Disorder prediction tests ──────────────────────────────────────────


class TestPredictDisorder:
    def test_length_preserved(self):
        """Output length must equal input length."""
        seq = "MVSKGEELFTGVVPILVELDG"
        scores = predict_disorder(seq)
        assert len(scores) == len(seq)

    def test_empty(self):
        assert predict_disorder("") == []

    def test_glycine_rich_disordered(self):
        """Poly-glycine (flexible, low hydrophobicity) → high disorder."""
        seq = "G" * 30
        scores = predict_disorder(seq)
        mean_score = sum(scores) / len(scores)
        assert mean_score > 0.5, f"Poly-G mean disorder {mean_score} should be > 0.5"

    def test_hydrophobic_ordered(self):
        """Poly-isoleucine (very hydrophobic) → low disorder."""
        seq = "I" * 30
        scores = predict_disorder(seq)
        mean_score = sum(scores) / len(scores)
        assert mean_score < 0.3, f"Poly-I mean disorder {mean_score} should be < 0.3"

    def test_mixed_sequence(self):
        """Hydrophobic core flanked by charged tails → tails more disordered."""
        core = "IVLFFCMA" * 3       # 24 hydrophobic residues
        tail = "EEKKRRDD" * 3       # 24 charged residues
        seq = tail + core + tail
        scores = predict_disorder(seq)
        n_tail = len(tail)
        n_core = len(core)
        tail_scores = scores[:n_tail] + scores[n_tail + n_core:]
        core_scores = scores[n_tail:n_tail + n_core]
        mean_tail = sum(tail_scores) / len(tail_scores)
        mean_core = sum(core_scores) / len(core_scores)
        assert mean_tail > mean_core, (
            f"Tail disorder ({mean_tail:.3f}) should exceed core ({mean_core:.3f})"
        )

    def test_scores_in_range(self):
        """All scores must be in [0, 1]."""
        seq = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL"
        scores = predict_disorder(seq)
        for s in scores:
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] range"


# ── Fusion site finding tests ──────────────────────────────────────────


class TestFindFusionSites:
    def test_no_sites_in_ordered(self):
        """Poly-Ile (fully ordered) → no fusion sites."""
        seq = "I" * 50
        sites = find_fusion_sites(seq)
        assert sites == []

    def test_finds_disordered_loop(self):
        """Ordered flanks with a disordered G-stretch → one site found."""
        seq = "I" * 20 + "G" * 25 + "I" * 20
        sites = find_fusion_sites(seq)
        assert len(sites) >= 1
        site = sites[0]
        # The G-stretch is at positions 20..44
        assert site["start"] >= 15   # may include some boundary
        assert site["end"] <= 50
        assert site["length"] >= 10

    def test_min_window_filter(self):
        """Short disordered stretch below min_window → not reported."""
        seq = "I" * 20 + "G" * 5 + "I" * 20
        sites = find_fusion_sites(seq, min_window=10)
        assert sites == []

    def test_sorted_by_suitability(self):
        """Two disordered regions: the longer/more disordered one ranks first."""
        short = "G" * 15
        long = "G" * 30
        seq = "I" * 20 + short + "I" * 20 + long + "I" * 20
        sites = find_fusion_sites(seq, min_window=10)
        assert len(sites) >= 2
        # First site should have higher suitability (mean_disorder * length)
        s0 = sites[0]["mean_disorder"] * sites[0]["length"]
        s1 = sites[1]["mean_disorder"] * sites[1]["length"]
        assert s0 >= s1

    def test_site_has_required_keys(self):
        """Each site dict has start, end, length, mean_disorder, context."""
        seq = "I" * 20 + "G" * 25 + "I" * 20
        sites = find_fusion_sites(seq)
        assert len(sites) >= 1
        required = {"start", "end", "length", "mean_disorder", "context"}
        assert required.issubset(sites[0].keys())

    def test_threshold_parameter(self):
        """Higher threshold → fewer or no sites."""
        seq = "I" * 20 + "G" * 25 + "I" * 20
        sites_low = find_fusion_sites(seq, threshold=0.3)
        sites_high = find_fusion_sites(seq, threshold=0.9)
        assert len(sites_low) >= len(sites_high)
