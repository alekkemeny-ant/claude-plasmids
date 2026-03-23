#!/usr/bin/env python3
"""Tests for the Smart Mutation Design module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from mutations import (
    lookup_known_mutations,
    apply_point_mutation,
    design_premature_stop,
    parse_mutation_notation,
    KNOWN_MUTATIONS,
    PREFERRED_CODONS,
    _CODON_TABLE,
)


# ── Database integrity ─────────────────────────────────────────────────


class TestKnownMutations:
    """Verify curated mutation database is well-formed."""

    REQUIRED_KEYS = {"mutation", "type", "phenotype", "reference", "codon_change"}

    def test_every_entry_has_required_keys(self):
        for gene, muts in KNOWN_MUTATIONS.items():
            for m in muts:
                missing = self.REQUIRED_KEYS - m.keys()
                assert not missing, f"{gene} {m.get('mutation','?')}: missing {missing}"

    def test_codon_change_format(self):
        """Every codon_change must be 'XXX>YYY' with valid 3-letter codons."""
        for gene, muts in KNOWN_MUTATIONS.items():
            for m in muts:
                cc = m["codon_change"]
                parts = cc.split(">")
                assert len(parts) == 2, f"{gene} {m['mutation']}: bad format '{cc}'"
                old, new = parts
                assert len(old) == 3 and len(new) == 3, (
                    f"{gene} {m['mutation']}: codons not 3nt '{cc}'"
                )
                assert old.upper() in _CODON_TABLE, (
                    f"{gene} {m['mutation']}: unknown old codon '{old}'"
                )
                assert new.upper() in _CODON_TABLE, (
                    f"{gene} {m['mutation']}: unknown new codon '{new}'"
                )

    def test_type_is_gof_or_lof(self):
        for gene, muts in KNOWN_MUTATIONS.items():
            for m in muts:
                assert m["type"] in ("GoF", "LoF"), (
                    f"{gene} {m['mutation']}: type '{m['type']}' not GoF/LoF"
                )

    def test_preferred_codons_cover_all_amino_acids(self):
        """PREFERRED_CODONS should have an entry for every AA in the codon table."""
        all_aas = set(_CODON_TABLE.values())
        assert all_aas == set(PREFERRED_CODONS.keys())

    def test_preferred_codons_encode_correct_aa(self):
        """Each preferred codon must actually encode its amino acid."""
        for aa, codon in PREFERRED_CODONS.items():
            assert _CODON_TABLE[codon] == aa, (
                f"PREFERRED_CODONS['{aa}'] = '{codon}' but translates to '{_CODON_TABLE[codon]}'"
            )


# ── Lookup ─────────────────────────────────────────────────────────────


class TestLookupKnownMutations:
    def test_braf_gof_contains_v600e(self):
        results = lookup_known_mutations("BRAF", "GoF")
        names = [r["mutation"] for r in results]
        assert "V600E" in names

    def test_case_insensitive_gene(self):
        assert lookup_known_mutations("braf") == lookup_known_mutations("BRAF")

    def test_tp53_lof_only(self):
        results = lookup_known_mutations("TP53", "LoF")
        assert all(r["type"] == "LoF" for r in results)
        assert len(results) > 0

    def test_unknown_gene_returns_empty(self):
        assert lookup_known_mutations("UNKNOWN_GENE_XYZ") == []

    def test_filter_gof_excludes_lof(self):
        results = lookup_known_mutations("TP53", "GoF")
        assert results == []  # TP53 only has LoF entries


# ── Point mutation ─────────────────────────────────────────────────────


class TestApplyPointMutation:
    # ATG GTG AGC AAG  →  M V S K
    SEQ = "ATGGTGAGCAAG"

    def test_v_to_e_at_pos2(self):
        result = apply_point_mutation(self.SEQ, 2, "E")
        assert result["original_aa"] == "V"
        assert result["new_aa"] == "E"
        assert result["original_codon"] == "GTG"
        assert result["new_codon"] == "GAG"  # preferred E codon
        # Only codon 2 changed
        assert result["sequence"][:3] == "ATG"      # codon 1 preserved
        assert result["sequence"][3:6] == "GAG"      # codon 2 mutated
        assert result["sequence"][6:] == "AGCAAG"    # codons 3-4 preserved

    def test_length_preserved(self):
        result = apply_point_mutation(self.SEQ, 2, "E")
        assert len(result["sequence"]) == len(self.SEQ)

    def test_frame_preserved(self):
        result = apply_point_mutation(self.SEQ, 2, "E")
        assert len(result["sequence"]) % 3 == 0

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            apply_point_mutation(self.SEQ, 100, "E")

    def test_zero_position_raises(self):
        with pytest.raises(ValueError, match="≥1"):
            apply_point_mutation(self.SEQ, 0, "E")

    def test_invalid_aa_raises(self):
        with pytest.raises(ValueError, match="Invalid amino acid"):
            apply_point_mutation(self.SEQ, 2, "Z")

    def test_mutation_to_stop(self):
        result = apply_point_mutation(self.SEQ, 2, "*")
        assert result["new_codon"] == "TGA"
        assert result["new_aa"] == "*"

    def test_dna_position_echoed(self):
        result = apply_point_mutation(self.SEQ, 3, "A")
        assert result["dna_position"] == 6  # (3-1)*3
        assert result["aa_position"] == 3


# ── Premature stop ────────────────────────────────────────────────────


class TestDesignPrematureStop:
    # 10-codon CDS (30 nt)
    SEQ = "ATG" + "GCC" * 8 + "TAA"  # M A A A A A A A A *

    def test_result_has_tga_at_reported_position(self):
        result = design_premature_stop(self.SEQ)
        pos = result["stop_position_dna"]
        assert result["sequence"][pos:pos + 3] == "TGA"

    def test_length_preserved(self):
        result = design_premature_stop(self.SEQ)
        assert len(result["sequence"]) == len(self.SEQ)

    def test_fraction_half_near_middle(self):
        result = design_premature_stop(self.SEQ, position_fraction=0.5)
        n_codons = len(self.SEQ) // 3
        mid = n_codons // 2
        # Should be within ±1 of middle
        assert abs(result["stop_position_aa"] - 1 - mid) <= 1

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            design_premature_stop("ATGCCCTAA")  # only 3 codons

    def test_fraction_out_of_range_raises(self):
        with pytest.raises(ValueError, match="position_fraction"):
            design_premature_stop(self.SEQ, position_fraction=1.5)

    def test_skips_start_codon(self):
        """Even with fraction=0, the stop should not overwrite the ATG."""
        result = design_premature_stop(self.SEQ, position_fraction=0.0)
        assert result["stop_position_aa"] >= 2  # 1-indexed, so >=2 means codon index >=1
        assert result["sequence"][:3] == "ATG"


# ── Notation parser ────────────────────────────────────────────────────


class TestParseMutationNotation:
    def test_v600e(self):
        parsed = parse_mutation_notation("V600E")
        assert parsed == {"original_aa": "V", "position": 600, "new_aa": "E"}

    def test_r175h(self):
        parsed = parse_mutation_notation("R175H")
        assert parsed == {"original_aa": "R", "position": 175, "new_aa": "H"}

    def test_lowercase_input(self):
        parsed = parse_mutation_notation("v600e")
        assert parsed == {"original_aa": "V", "position": 600, "new_aa": "E"}

    def test_invalid_returns_none(self):
        assert parse_mutation_notation("invalid") is None
        assert parse_mutation_notation("") is None
        assert parse_mutation_notation("600VE") is None

    def test_stop_codon_notation(self):
        parsed = parse_mutation_notation("R100*")
        assert parsed == {"original_aa": "R", "position": 100, "new_aa": "*"}
