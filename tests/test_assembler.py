#!/usr/bin/env python3
"""Tests for the sequence assembly engine."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assembler import (
    assemble_construct,
    fuse_sequences,
    clean_sequence,
    validate_dna,
    reverse_complement,
    find_mcs_insertion_point,
    export_construct,
    AssemblyResult,
    DEFAULT_FUSION_LINKER,
    KOZAK,
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


# ── Fusion tests ────────────────────────────────────────────────────────


class TestFuseSequences:
    """Unit tests for fuse_sequences() — deterministic codon management."""

    def test_basic_two_sequence_fusion_no_linker(self):
        """With linker="", first keeps ATG, loses stop; last keeps ATG+stop."""
        seq1 = "ATGAAACCCTAA"  # 12 bp CDS with start+stop
        seq2 = "ATGGGGTTTGACTGA"  # 15 bp CDS with start+stop
        result = fuse_sequences([
            {"sequence": seq1, "name": "A"},
            {"sequence": seq2, "name": "B"},
        ], linker="")
        # seq1 without stop: ATGAAACCC (9bp)
        # seq2 kept intact: ATGGGGTTTGACTGA (15bp)
        assert result == "ATGAAACCCATGGGGTTTGACTGA"
        assert result[:3] == "ATG"  # keeps first start
        assert result[-3:] == "TGA"  # keeps last stop
        assert len(result) == 24

    def test_three_sequence_fusion_no_linker(self):
        """With linker="", middle sequence loses stop only, keeps ATG."""
        seq1 = "ATGAAATAA"   # 9 bp
        seq2 = "ATGCCCTGA"   # 9 bp
        seq3 = "ATGGGGTAG"   # 9 bp
        result = fuse_sequences([
            {"sequence": seq1, "name": "first"},
            {"sequence": seq2, "name": "middle"},
            {"sequence": seq3, "name": "last"},
        ], linker="")
        # first without stop: ATGAAA (6bp)
        # middle without stop: ATGCCC (6bp)
        # last intact: ATGGGGTAG (9bp)
        assert result == "ATGAAAATGCCCATGGGGTAG"
        assert result[:3] == "ATG"
        assert result[-3:] == "TAG"

    def test_fusion_with_explicit_linker(self):
        """Explicit linker DNA + KOZAK inserted between each pair."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        linker = "GGCGGC"  # encodes GG
        result = fuse_sequences(
            [{"sequence": seq1}, {"sequence": seq2}],
            linker=linker,
        )
        # first without stop: ATGAAA
        # linker + KOZAK: GGCGGCGCCACC
        # last intact: ATGCCCTGA
        assert result == "ATGAAA" + "GGCGGC" + KOZAK + "ATGCCCTGA"

    def test_default_linker_is_ggggs_x4(self):
        """When no linker is specified, (GGGGS)x4 + KOZAK is used."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences([
            {"sequence": seq1, "name": "A"},
            {"sequence": seq2, "name": "B"},
        ])
        # first without stop: ATGAAA
        # default linker + KOZAK: DEFAULT_FUSION_LINKER + GCCACC
        # last intact: ATGCCCTGA
        expected = "ATGAAA" + DEFAULT_FUSION_LINKER + KOZAK + "ATGCCCTGA"
        assert result == expected

    def test_kozak_between_linker_and_second_gene(self):
        """GCCACC appears right after the linker and before the second gene."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences([
            {"sequence": seq1},
            {"sequence": seq2},
        ])
        # Find the position of GCCACC
        kozak_pos = result.index(KOZAK)
        # GCCACC should be immediately followed by the second gene (ATGCCCTGA)
        after_kozak = result[kozak_pos + len(KOZAK):]
        assert after_kozak == "ATGCCCTGA"
        # GCCACC should be immediately preceded by the linker
        before_kozak = result[:kozak_pos]
        assert before_kozak.endswith(DEFAULT_FUSION_LINKER)

    def test_empty_linker_no_kozak(self):
        """linker="" produces direct concatenation with no KOZAK."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences([
            {"sequence": seq1},
            {"sequence": seq2},
        ], linker="")
        assert KOZAK not in result
        assert result == "ATGAAA" + "ATGCCCTGA"

    def test_no_stop_codon_on_first(self):
        """If first seq lacks a stop, nothing extra removed."""
        seq1 = "ATGAAACCC"   # no stop
        seq2 = "ATGGGGTGA"
        result = fuse_sequences([
            {"sequence": seq1, "name": "A"},
            {"sequence": seq2, "name": "B"},
        ], linker="")
        assert result == "ATGAAACCCATGGGGTGA"

    def test_no_start_codon_on_last(self):
        """If last seq lacks ATG start, nothing extra removed."""
        seq1 = "ATGAAATAA"
        seq2 = "CCCGGGTGA"  # no ATG start
        result = fuse_sequences([
            {"sequence": seq1},
            {"sequence": seq2},
        ], linker="")
        assert result == "ATGAAACCCGGGTGA"

    def test_fewer_than_two_raises(self):
        """Must provide at least 2 sequences."""
        try:
            fuse_sequences([{"sequence": "ATGTAA"}])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "2 sequences" in str(e)

    def test_invalid_dna_raises(self):
        """Invalid DNA in any input raises ValueError."""
        try:
            fuse_sequences([
                {"sequence": "ATGXYZ", "name": "bad"},
                {"sequence": "ATGTAA", "name": "good"},
            ], linker="")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "bad" in str(e)

    def test_invalid_linker_raises(self):
        try:
            fuse_sequences(
                [{"sequence": "ATGTAA"}, {"sequence": "ATGTGA"}],
                linker="XYZ",
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "linker" in str(e).lower()

    def test_flag_egfp_fusion(self):
        """Integration: FLAG + EGFP from library with no linker (tag fusion).

        FLAG_tag is an epitope tag without its own start/stop codons.
        EGFP has ATG start and TAA stop. In an N-terminal tag fusion
        (linker=""):
        - FLAG (first): has no stop to remove, kept as-is
        - EGFP (last): kept intact (ATG + stop)
        Result: FLAG + EGFP(full) — direct concatenation for tag fusions.
        """
        flag = get_insert_by_id("FLAG_tag")
        egfp = get_insert_by_id("EGFP")
        assert flag and flag.get("sequence")
        assert egfp and egfp.get("sequence")

        result = fuse_sequences([
            {"sequence": flag["sequence"], "name": "FLAG"},
            {"sequence": egfp["sequence"], "name": "EGFP"},
        ], linker="")
        flag_seq = clean_sequence(flag["sequence"])
        egfp_seq = clean_sequence(egfp["sequence"])

        # FLAG has no stop codon, so it's kept as-is (first position)
        assert flag_seq[-3:] not in ("TAA", "TAG", "TGA")
        # EGFP starts with ATG — kept intact (no ATG removal)
        assert egfp_seq[:3] == "ATG"

        expected = flag_seq + egfp_seq  # FLAG + EGFP (full, ATG kept)
        assert result == expected
        assert result[-3:] in ("TAA", "TAG", "TGA")  # EGFP stop preserved
