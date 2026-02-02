#!/usr/bin/env python3
"""Tests for the sequence assembly engine."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assembler import (
    assemble_construct,
    clean_sequence,
    validate_dna,
    reverse_complement,
    find_mcs_insertion_point,
    export_construct,
    AssemblyResult,
)
from library import get_backbone_by_id, get_insert_by_id


# ── Unit tests ──────────────────────────────────────────────────────────


class TestCleanSequence:
    def test_removes_whitespace(self):
        assert clean_sequence("ATG CGC\nTAA") == "ATGCGCTAA"

    def test_uppercases(self):
        assert clean_sequence("atgcgc") == "ATGCGC"


class TestValidateDna:
    def test_valid(self):
        ok, errs = validate_dna("ATCGN")
        assert ok and not errs

    def test_empty(self):
        ok, errs = validate_dna("")
        assert not ok

    def test_invalid_chars(self):
        ok, errs = validate_dna("ATCGX")
        assert not ok
        assert "Invalid characters" in errs[0]


class TestReverseComplement:
    def test_basic(self):
        assert reverse_complement("ATCG") == "CGAT"

    def test_palindrome(self):
        assert reverse_complement("AATT") == "AATT"


# ── Assembly tests ──────────────────────────────────────────────────────


class TestAssembleConstruct:
    def test_simple_insert(self):
        backbone = "AAAAAAAAAA"  # 10 bp
        insert = "ATGCCCTAA"     # 9 bp, has start+stop, multiple of 3
        result = assemble_construct(backbone, insert, insertion_position=5)

        assert result.success
        # backbone[:5] + insert + backbone[5:] = AAAAA + ATGCCCTAA + AAAAA = 19 bp
        assert result.sequence == "AAAAAATGCCCTAAAAAAA"
        assert result.total_size_bp == 19
        assert result.backbone_preserved
        assert result.insert_preserved
        assert result.insert_has_start_codon
        assert result.insert_has_stop_codon
        assert result.insert_length_valid

    def test_replace_region(self):
        backbone = "AAAGGGCCCAAA"  # 12 bp
        insert = "ATGTAA"          # 6 bp
        result = assemble_construct(backbone, insert, insertion_position=3, replace_region_end=9)

        assert result.success
        # backbone[:3] + insert + backbone[9:] = AAA + ATGTAA + AAA = 12 bp
        assert result.sequence == "AAAATGTAAAAA"
        assert result.total_size_bp == 12

    def test_invalid_backbone(self):
        result = assemble_construct("ATCGXYZ", "ATGTAA", 3)
        assert not result.success
        assert any("Backbone" in e for e in result.errors)

    def test_invalid_insert(self):
        result = assemble_construct("ATCGATCG", "XYZ", 3)
        assert not result.success
        assert any("Insert" in e for e in result.errors)

    def test_position_out_of_range(self):
        result = assemble_construct("ATCG", "ATG", 10)
        assert not result.success
        assert any("out of range" in e for e in result.errors)

    def test_negative_position(self):
        result = assemble_construct("ATCG", "ATG", -1)
        assert not result.success

    def test_warnings_no_start_codon(self):
        result = assemble_construct("AAAAAAAAAA", "CCCCCC", 5)
        assert result.success
        assert not result.insert_has_start_codon
        assert any("start codon" in w for w in result.warnings)

    def test_warnings_not_multiple_of_3(self):
        result = assemble_construct("AAAAAAAAAA", "ATGCC", 5)
        assert result.success
        assert not result.insert_length_valid
        assert any("multiple of 3" in w for w in result.warnings)

    def test_reverse_complement_insert(self):
        backbone = "AAAAAAAAAA"
        insert = "ATCG"
        result = assemble_construct(backbone, insert, 5, reverse_complement_insert=True)
        assert result.success
        assert result.sequence == "AAAAACGATAAAAA"


# ── Integration test with real library data ─────────────────────────────


class TestPcDNA31EGFP:
    """Primary benchmark: pcDNA3.1(+) + EGFP at MCS start."""

    def test_assembly(self):
        backbone = get_backbone_by_id("pcDNA3.1(+)")
        insert = get_insert_by_id("EGFP")

        assert backbone is not None, "pcDNA3.1(+) not found in library"
        assert insert is not None, "EGFP not found in library"
        assert backbone.get("sequence"), "pcDNA3.1(+) has no sequence"
        assert insert.get("sequence"), "EGFP has no sequence"

        insertion_pos = find_mcs_insertion_point(backbone)
        assert insertion_pos == 895

        result = assemble_construct(
            backbone_seq=backbone["sequence"],
            insert_seq=insert["sequence"],
            insertion_position=insertion_pos,
        )

        assert result.success, f"Assembly failed: {result.errors}"
        assert result.total_size_bp == 5428 + 720  # 6148
        assert result.backbone_preserved
        assert result.insert_preserved
        assert result.insert_has_start_codon
        assert result.insert_has_stop_codon
        assert result.insert_length_valid
        assert len(result.warnings) == 0

        # Verify the insert is at the right position
        seq = result.sequence
        assert seq[895:895 + 3] == "ATG"  # EGFP starts with ATG
        assert seq[895 + 720 - 3:895 + 720] == "TAA"  # EGFP ends with TAA

        # Verify backbone flanking regions are intact
        bb_seq = clean_sequence(backbone["sequence"])
        assert seq[:895] == bb_seq[:895]  # upstream preserved
        assert seq[895 + 720:] == bb_seq[895:]  # downstream preserved


# ── Export tests ────────────────────────────────────────────────────────


class TestExport:
    def _make_result(self) -> AssemblyResult:
        result = assemble_construct("AAAAAAAAAA", "ATGCCCTAA", 5)
        assert result.success
        return result

    def test_raw(self):
        r = self._make_result()
        out = export_construct(r, "raw")
        assert out == r.sequence

    def test_fasta(self):
        r = self._make_result()
        out = export_construct(r, "fasta", construct_name="test", insert_name="GFP", backbone_name="pTest")
        assert out.startswith(">test")
        assert "GFP in pTest" in out
        # Sequence on second line
        lines = out.strip().split("\n")
        assert lines[1] == r.sequence

    def test_genbank(self):
        r = self._make_result()
        out = export_construct(
            r, "genbank",
            construct_name="test",
            backbone_name="pTest",
            insert_name="GFP",
            insert_length=9,
        )
        assert "LOCUS" in out
        assert "ORIGIN" in out
        assert "//" in out
        assert "GFP" in out

    def test_invalid_format(self):
        r = self._make_result()
        try:
            export_construct(r, "pdf")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_failed_result_raises(self):
        r = AssemblyResult(success=False)
        try:
            export_construct(r, "raw")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
