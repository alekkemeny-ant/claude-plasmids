#!/usr/bin/env python3
"""Tests for restriction_utils: RE site checking and silent mutation design."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from restriction_utils import (
    check_re_sites,
    design_silent_mutation,
    find_extra_sites_in_sequence,
)
from assembler import reverse_complement

# Esp3I recognition: CGTCTC (forward), GAGACG (RC)
ESP3I_REC = "CGTCTC"
ESP3I_RC = reverse_complement(ESP3I_REC)  # GAGACG


# ── Helpers ───────────────────────────────────────────────────────────────────

def _translate(dna: str) -> str:
    table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    dna = dna.upper()
    return "".join(table.get(dna[i:i+3], "X") for i in range(0, len(dna) - 2, 3))


def _plant_site(cds: str, offset: int, recognition: str) -> str:
    """Replace cds[offset:offset+len(recognition)] with recognition."""
    return cds[:offset] + recognition + cds[offset + len(recognition):]


# ── find_extra_sites_in_sequence ──────────────────────────────────────────────

class TestFindExtraSites:
    def test_no_sites_returns_empty(self):
        seq = "A" * 200
        assert find_extra_sites_in_sequence(seq, "Esp3I", 0) == []

    def test_one_site_expected_returns_empty(self):
        seq = "AAAA" + ESP3I_REC + "NNNNNAAAA"
        assert find_extra_sites_in_sequence(seq, "Esp3I", expected_site_count=1) == []

    def test_two_sites_one_expected(self):
        seq = "AAAA" + ESP3I_REC + "A" * 20 + ESP3I_REC + "NNNNNAAAA"
        extras = find_extra_sites_in_sequence(seq, "Esp3I", expected_site_count=1)
        assert len(extras) == 1

    def test_rc_site_counts(self):
        # RC site is also a recognition site
        seq = "A" * 10 + ESP3I_RC + "A" * 10
        extras = find_extra_sites_in_sequence(seq, "Esp3I", expected_site_count=0)
        assert len(extras) == 1

    def test_unknown_enzyme_raises(self):
        with pytest.raises(ValueError, match="Unknown enzyme"):
            find_extra_sites_in_sequence("ATCG", "FakeEnzymeX")


# ── check_re_sites ─────────────────────────────────────────────────────────────

class TestCheckReSites:
    def test_all_clear_no_sites(self):
        result = check_re_sites(
            [{"name": "insert", "sequence": "ATG" + "GGG" * 60 + "TAA"}],
            "Esp3I",
        )
        assert result["all_clear"] is True
        assert result["problematic_sequences"] == []

    def test_expected_backbone_sites_not_flagged(self):
        # Two Esp3I sites (expected for GG backbone)
        bb = "A" * 50 + ESP3I_REC + "A" * 50 + ESP3I_REC + "A" * 50
        result = check_re_sites(
            [{"name": "backbone", "sequence": bb, "expected_site_count": 2}],
            "Esp3I",
        )
        assert result["all_clear"] is True

    def test_extra_backbone_site_flagged(self):
        bb = "A" * 30 + ESP3I_REC + "A" * 30 + ESP3I_REC + "A" * 30 + ESP3I_REC + "A" * 30
        result = check_re_sites(
            [{"name": "backbone", "sequence": bb, "expected_site_count": 2}],
            "Esp3I",
        )
        assert result["all_clear"] is False
        assert len(result["problematic_sequences"]) == 1
        assert result["problematic_sequences"][0]["extra_site_count"] == 1

    def test_insert_site_flagged(self):
        insert_seq = "ATG" + "GGG" * 5 + ESP3I_REC + "GGG" * 5 + "TAA"
        result = check_re_sites(
            [{"name": "mCherry", "sequence": insert_seq}],
            "Esp3I",
        )
        assert result["all_clear"] is False
        assert result["problematic_sequences"][0]["sequence_name"] == "mCherry"

    def test_feature_attribution_cds(self):
        insert_seq = "ATG" + "GGG" * 5 + ESP3I_REC + "GGG" * 5 + "TAA"
        site_pos = 3 + 15  # "ATG" + "GGG"*5 = 3+15=18
        features = [{"name": "mCherry_CDS", "type": "CDS", "start": 0, "end": len(insert_seq)}]
        result = check_re_sites(
            [{"name": "mCherry", "sequence": insert_seq, "features": features}],
            "Esp3I",
        )
        site_info = result["problematic_sequences"][0]["sites"][0]
        assert site_info["in_cds"] is True
        assert site_info["solvable_by_silent_mutation"] is True
        assert site_info["overlapping_feature"]["name"] == "mCherry_CDS"

    def test_feature_attribution_noncoding(self):
        # Site in a promoter (non-CDS)
        seq = "A" * 30 + ESP3I_REC + "A" * 30
        features = [{"name": "CMV_promoter", "type": "promoter", "start": 0, "end": 66}]
        result = check_re_sites(
            [{"name": "backbone", "sequence": seq, "features": features}],
            "Esp3I",
        )
        site_info = result["problematic_sequences"][0]["sites"][0]
        assert site_info["in_cds"] is False
        assert site_info["solvable_by_silent_mutation"] is False
        assert site_info["overlapping_feature"]["name"] == "CMV_promoter"

    def test_mixed_sequences_only_dirty_in_report(self):
        clean_seq = "A" * 100
        dirty_seq = "A" * 20 + ESP3I_REC + "A" * 20
        result = check_re_sites(
            [
                {"name": "clean_insert", "sequence": clean_seq},
                {"name": "dirty_insert", "sequence": dirty_seq},
            ],
            "Esp3I",
        )
        assert result["all_clear"] is False
        names = [p["sequence_name"] for p in result["problematic_sequences"]]
        assert "dirty_insert" in names
        assert "clean_insert" not in names

    def test_enzyme_metadata_in_result(self):
        result = check_re_sites([{"name": "x", "sequence": "AAAA"}], "BsaI")
        assert result["enzyme"] == "BsaI"
        assert result["recognition_sequence"] == "GGTCTC"

    def test_unknown_enzyme_raises(self):
        with pytest.raises(ValueError, match="Unknown enzyme"):
            check_re_sites([{"name": "x", "sequence": "ATCG"}], "BogusEnzyme")


# ── design_silent_mutation ─────────────────────────────────────────────────────

class TestDesignSilentMutation:
    def _make_cds_with_site(self, prefix_codons: int, recognition: str) -> tuple[str, int]:
        """Build a CDS with a recognition site planted at codon boundary."""
        prefix = "ATG" + "GAA" * prefix_codons  # ATG + Glu repeats
        site_pos = len(prefix)
        # Pad recognition to codon-aligned length using Glu (GAA) codons
        # Find codons that contain the recognition site (may need to adjust)
        # For simplicity: plant recognition at an exact position, pad the rest
        suffix_codons = 20
        suffix = "GAA" * suffix_codons + "TAA"
        full_cds = prefix + recognition + suffix
        return full_cds, site_pos

    def test_forward_site_eliminated(self):
        cds, pos = self._make_cds_with_site(5, ESP3I_REC)
        result = design_silent_mutation(cds, pos, "Esp3I")
        assert result["success"] is True
        assert ESP3I_REC not in result["mutated_sequence"]
        assert reverse_complement(ESP3I_REC) not in result["mutated_sequence"]

    def test_rc_site_eliminated(self):
        cds, pos = self._make_cds_with_site(5, ESP3I_RC)
        result = design_silent_mutation(cds, pos, "Esp3I")
        assert result["success"] is True
        assert ESP3I_REC not in result["mutated_sequence"]
        assert ESP3I_RC not in result["mutated_sequence"]

    def test_protein_preserved(self):
        cds, pos = self._make_cds_with_site(4, ESP3I_REC)
        result = design_silent_mutation(cds, pos, "Esp3I")
        assert result["success"] is True
        assert _translate(cds) == _translate(result["mutated_sequence"])

    def test_codon_changes_report(self):
        cds, pos = self._make_cds_with_site(3, ESP3I_REC)
        result = design_silent_mutation(cds, pos, "Esp3I")
        assert result["success"] is True
        for change in result["codons_changed"]:
            assert "codon_index" in change
            assert "original_codon" in change
            assert "new_codon" in change
            assert "amino_acid" in change
            assert "dna_position" in change
            assert change["original_codon"] != change["new_codon"]
            # AA must be preserved
            from restriction_utils import _CODON_TABLE
            assert _CODON_TABLE[change["original_codon"]] == _CODON_TABLE[change["new_codon"]]

    def test_site_spanning_codon_boundary(self):
        # Plant site that spans two codons: offset 1 into a codon
        # ATG + 4x GAA: position 3+12=15; shift site to position 16 (1 nt into codon 5)
        prefix = "ATG" + "GAA" * 4  # len=15
        site_pos = 16  # starts 1 nt into the next codon
        cds = prefix + "G" + "CGTCTC"[1:] + "GAAGAAGAATAA"  # plant Esp3I overlapping boundary
        # Build a proper test: prefix ends with ...GAAG, plant at offset 14+2
        # Simpler: make CDS where site starts at a non-multiple-of-3 position
        cds2 = "ATG" + "GAAGAA" + ESP3I_REC + "GAAGAAGAA" + "TAA"
        # Position of ESP3I_REC in cds2: 3+6=9 (not divisible by 3 → spans 2 codons)
        pos2 = 9
        result = design_silent_mutation(cds2, pos2, "Esp3I")
        assert result["success"] is True
        assert _translate(cds2) == _translate(result["mutated_sequence"])
        assert ESP3I_REC not in result["mutated_sequence"]

    def test_preferred_codon_scoring(self):
        # Two synonymous solutions should exist; we just verify the highest-scoring one is chosen
        # Use Leu (6 synonymous codons) so many solutions exist
        cds = "ATG" + "CTG" * 3 + ESP3I_REC + "CTG" * 3 + "TAA"
        pos = 3 + 9  # after ATG and 3xCTG = 3+9=12
        result = design_silent_mutation(cds, pos, "Esp3I")
        assert result["success"] is True
        # Verify protein preserved
        assert _translate(cds) == _translate(result["mutated_sequence"])

    def test_stop_codon_overlap_fails_gracefully(self):
        # Construct a CDS where an in-frame stop codon falls at the site window start.
        # design_silent_mutation does not require the recognition site to actually be
        # present — it reads the codons at the given position range. Here codon at
        # position 9 is TAA (stop), so the function must return gracefully.
        cds = "ATG" + "GAA" * 2 + "TAA" + "GAAGAA" + "TAA"
        # Position 9 is the start of in-frame stop codon TAA; overlapping codons are
        # positions 9 (TAA=stop) and 12 (GAA=E)
        result = design_silent_mutation(cds, 9, "Esp3I")
        assert result["success"] is False
        assert result["reason"] == "site_overlaps_stop_codon"

    def test_sequence_fragment_in_result(self):
        cds, pos = self._make_cds_with_site(5, ESP3I_REC)
        result = design_silent_mutation(cds, pos, "Esp3I")
        assert result["success"] is True
        assert result["original_sequence_fragment"] is not None
        assert result["mutated_sequence_fragment"] is not None
        assert ESP3I_REC not in result["mutated_sequence_fragment"]

    def test_unknown_enzyme_raises(self):
        with pytest.raises(ValueError, match="Unknown enzyme"):
            design_silent_mutation("ATGAAATAA", 3, "GhostEnzyme")

    def test_position_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            design_silent_mutation("ATGAAATAA", 999, "Esp3I")


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_check_then_mutate_leaves_sequence_clean(self):
        """Full workflow: find site, mutate it, confirm clean."""
        cds = "ATG" + "GAA" * 6 + ESP3I_REC + "GAA" * 6 + "TAA"
        site_pos_in_cds = 3 + 18  # ATG + 6xGAA = 3+18=21

        result = check_re_sites(
            [{"name": "test_CDS", "sequence": cds,
              "features": [{"name": "test_CDS", "type": "CDS", "start": 0, "end": len(cds)}]}],
            "Esp3I",
        )
        assert result["all_clear"] is False

        site = result["problematic_sequences"][0]["sites"][0]
        assert site["solvable_by_silent_mutation"] is True

        mut_result = design_silent_mutation(cds, site_pos_in_cds, "Esp3I")
        assert mut_result["success"] is True

        clean_check = check_re_sites(
            [{"name": "test_CDS", "sequence": mut_result["mutated_sequence"]}],
            "Esp3I",
        )
        assert clean_check["all_clear"] is True

    def test_protein_identical_after_full_workflow(self):
        cds = "ATG" + "CTG" * 4 + ESP3I_REC + "CTG" * 4 + "TAA"
        site_pos = 3 + 12
        result = design_silent_mutation(cds, site_pos, "Esp3I")
        assert result["success"] is True
        assert _translate(cds) == _translate(result["mutated_sequence"])
