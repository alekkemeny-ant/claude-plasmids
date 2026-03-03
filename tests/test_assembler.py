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
        """With linker="", first keeps ATG+loses stop; last (protein) loses ATG+keeps stop."""
        seq1 = "ATGAAACCCTAA"  # 12 bp CDS with start+stop
        seq2 = "ATGGGGTTTGACTGA"  # 15 bp CDS with start+stop
        result = fuse_sequences([
            {"sequence": seq1, "name": "A"},
            {"sequence": seq2, "name": "B"},
        ], linker="")
        # seq1 without stop: ATGAAACCC (9bp)
        # seq2 (protein, last): ATG removed + keep stop → GGGTTTGACTGA (12bp)
        assert result == "ATGAAACCC" + "GGGTTTGACTGA"
        assert result[:3] == "ATG"   # first ATG preserved
        assert result[-3:] == "TGA"  # last stop preserved
        assert len(result) == 21

    def test_three_sequence_fusion_no_linker(self):
        """With linker="", middle loses ATG+stop; last loses ATG+keeps stop."""
        seq1 = "ATGAAATAA"   # 9 bp
        seq2 = "ATGCCCTGA"   # 9 bp
        seq3 = "ATGGGGTAG"   # 9 bp
        result = fuse_sequences([
            {"sequence": seq1, "name": "first"},
            {"sequence": seq2, "name": "middle"},
            {"sequence": seq3, "name": "last"},
        ], linker="")
        # first: remove stop → ATGAAA (6bp)
        # middle (protein): remove ATG + stop → CCC (3bp)
        # last (protein): remove ATG, keep stop → GGGTAG (6bp)
        assert result == "ATGAAA" + "CCC" + "GGGTAG"
        assert result[:3] == "ATG"
        assert result[-3:] == "TAG"
        assert len(result) == 15

    def test_fusion_with_explicit_linker(self):
        """Explicit linker inserted between parts; no Kozak for protein-protein junction."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        linker = "GGCGGC"  # encodes GG
        result = fuse_sequences(
            [{"sequence": seq1}, {"sequence": seq2}],
            linker=linker,
        )
        # first: remove stop → ATGAAA
        # seq2 (protein, last): remove ATG, keep stop → CCCTGA
        # no Kozak because seq2 is a protein (ATG was removed)
        assert result == "ATGAAA" + "GGCGGC" + "CCCTGA"

    def test_default_linker_is_ggggs_x4(self):
        """When no linker is specified, (GGGGS)x4 is used; no Kozak for protein-protein."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences([
            {"sequence": seq1, "name": "A"},
            {"sequence": seq2, "name": "B"},
        ])
        # first: remove stop → ATGAAA
        # seq2 (protein): remove ATG, keep stop → CCCTGA
        # no Kozak because protein ATG was removed
        expected = "ATGAAA" + DEFAULT_FUSION_LINKER + "CCCTGA"
        assert result == expected

    def test_none_linker_uses_default(self):
        """Passing linker=None explicitly should use the default linker (not direct concat)."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences([
            {"sequence": seq1},
            {"sequence": seq2},
        ], linker=None)
        expected = "ATGAAA" + DEFAULT_FUSION_LINKER + "CCCTGA"
        assert result == expected

    def test_no_kozak_in_protein_protein_fusion(self):
        """Kozak (GCCACC) is NOT inserted between protein-protein junctions.

        The ATG is removed from the second protein instead, making Kozak unnecessary.
        """
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences([{"sequence": seq1}, {"sequence": seq2}])
        assert KOZAK not in result
        # Junction: linker immediately followed by seq2 without ATG
        assert result.endswith(DEFAULT_FUSION_LINKER + "CCCTGA")

    def test_kozak_added_before_tag_with_atg(self):
        """When a tag with ATG follows with a linker, Kozak IS inserted before it."""
        seq1 = "ATGAAATAA"  # protein
        seq2 = "ATGCCC"     # tag (with ATG — kept)
        linker = "GGCGGC"
        result = fuse_sequences(
            [{"sequence": seq1, "type": "protein"},
             {"sequence": seq2, "type": "tag"}],
            linker=linker,
        )
        # seq1: remove stop → ATGAAA
        # seq2 (tag): keep ATG → ATGCCC; Kozak added before it
        assert result == "ATGAAA" + linker + KOZAK + "ATGCCC"

    def test_empty_linker_no_kozak(self):
        """linker="" produces direct concatenation; Kozak never added without linker."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences([
            {"sequence": seq1},
            {"sequence": seq2},
        ], linker="")
        assert KOZAK not in result
        # seq2 (protein): ATG removed → CCCTGA
        assert result == "ATGAAA" + "CCCTGA"

    def test_no_stop_codon_on_first(self):
        """If first seq lacks a stop, nothing extra removed from it; second protein loses ATG."""
        seq1 = "ATGAAACCC"   # no stop
        seq2 = "ATGGGGTGA"
        result = fuse_sequences([
            {"sequence": seq1, "name": "A"},
            {"sequence": seq2, "name": "B"},
        ], linker="")
        # seq1: no stop to remove → ATGAAACCC
        # seq2 (protein): remove ATG, keep stop → GGGTGA
        assert result == "ATGAAACCC" + "GGGTGA"

    def test_no_start_codon_on_last(self):
        """If last seq already lacks ATG, nothing extra removed from it."""
        seq1 = "ATGAAATAA"
        seq2 = "CCCGGGTGA"  # no ATG start
        result = fuse_sequences([
            {"sequence": seq1},
            {"sequence": seq2},
        ], linker="")
        # seq2 has no ATG, so nothing to remove — same as before
        assert result == "ATGAAA" + "CCCGGGTGA"

    def test_tag_type_preserves_atg(self):
        """type='tag' prevents ATG removal even for non-first sequences."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCC"     # this tag has an ATG
        result = fuse_sequences([
            {"sequence": seq1, "type": "protein"},
            {"sequence": seq2, "type": "tag"},
        ], linker="")
        # seq1: remove stop → ATGAAA
        # seq2 (tag): ATG KEPT → ATGCCC
        assert result == "ATGAAA" + "ATGCCC"

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
        """Integration: FLAG (N-terminal tag, no ATG) + EGFP (C-terminal protein).

        FLAG_tag is an epitope tag without its own start/stop codons.
        EGFP has ATG start and TAA stop.
        - FLAG (first): kept as-is (no stop to remove)
        - EGFP (last, type="protein"): ATG removed, stop preserved

        Result: FLAG + EGFP_without_ATG. The fuse_inserts tool then prepends
        ATG to produce an expressible fusion.
        """
        flag = get_insert_by_id("FLAG_tag")
        egfp = get_insert_by_id("EGFP")
        assert flag and flag.get("sequence")
        assert egfp and egfp.get("sequence")

        result = fuse_sequences([
            {"sequence": flag["sequence"], "name": "FLAG", "type": "protein"},
            {"sequence": egfp["sequence"], "name": "EGFP", "type": "protein"},
        ], linker="")
        flag_seq = clean_sequence(flag["sequence"])
        egfp_seq = clean_sequence(egfp["sequence"])

        # FLAG has no stop codon — kept as-is at first position
        assert flag_seq[-3:] not in ("TAA", "TAG", "TGA")
        # EGFP's ATG is removed (non-first protein)
        assert egfp_seq[:3] == "ATG"
        egfp_no_atg = egfp_seq[3:]

        expected = flag_seq + egfp_no_atg
        assert result == expected
        assert result[-3:] in ("TAA", "TAG", "TGA")  # EGFP stop preserved

    def test_flag_egfp_tag_type_preserves_egfp_atg(self):
        """With EGFP marked as type='tag', its ATG is preserved."""
        flag = get_insert_by_id("FLAG_tag")
        egfp = get_insert_by_id("EGFP")
        assert flag and flag.get("sequence")
        assert egfp and egfp.get("sequence")

        result = fuse_sequences([
            {"sequence": flag["sequence"], "name": "FLAG"},
            {"sequence": egfp["sequence"], "name": "EGFP", "type": "tag"},
        ], linker="")
        flag_seq = clean_sequence(flag["sequence"])
        egfp_seq = clean_sequence(egfp["sequence"])

        # EGFP ATG preserved because type="tag"
        expected = flag_seq + egfp_seq
        assert result == expected

    # ── remove_internal_atg=False tests ─────────────────────────────────

    def test_remove_internal_atg_false_preserves_atg(self):
        """remove_internal_atg=False: non-first protein sequences keep their ATG."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        result = fuse_sequences(
            [{"sequence": seq1}, {"sequence": seq2}],
            linker="",
            remove_internal_atg=False,
        )
        # seq1: remove stop → ATGAAA
        # seq2 (protein, remove_internal_atg=False): keep ATG, keep stop → ATGCCCTGA
        assert result == "ATGAAA" + "ATGCCCTGA"
        assert result[:3] == "ATG"
        assert result[-3:] == "TGA"

    def test_remove_internal_atg_false_adds_kozak_with_linker(self):
        """remove_internal_atg=False + linker: Kozak is inserted before retained ATG."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        linker = "GGCGGC"
        result = fuse_sequences(
            [{"sequence": seq1}, {"sequence": seq2}],
            linker=linker,
            remove_internal_atg=False,
        )
        # seq1: remove stop → ATGAAA
        # seq2 (protein, ATG retained): Kozak added before it
        assert result == "ATGAAA" + linker + KOZAK + "ATGCCCTGA"

    def test_remove_internal_atg_false_three_sequences(self):
        """remove_internal_atg=False with three proteins: all middle/last ATGs kept."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        seq3 = "ATGGGGTAG"
        result = fuse_sequences(
            [
                {"sequence": seq1, "name": "first"},
                {"sequence": seq2, "name": "middle"},
                {"sequence": seq3, "name": "last"},
            ],
            linker="",
            remove_internal_atg=False,
        )
        # first: remove stop → ATGAAA
        # middle (protein, ATG kept, no linker so no Kozak): remove stop → ATGCCC
        # last (protein, ATG kept, no linker so no Kozak): keep stop → ATGGGGTAG
        assert result == "ATGAAA" + "ATGCCC" + "ATGGGGTAG"
        assert result[:3] == "ATG"
        assert result[-3:] == "TAG"

    def test_remove_internal_atg_true_is_default(self):
        """Default behavior (remove_internal_atg=True) unchanged: ATG stripped."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCCTGA"
        explicit_true = fuse_sequences(
            [{"sequence": seq1}, {"sequence": seq2}],
            linker="",
            remove_internal_atg=True,
        )
        default = fuse_sequences(
            [{"sequence": seq1}, {"sequence": seq2}],
            linker="",
        )
        assert explicit_true == default
        # ATG removed from seq2
        assert explicit_true == "ATGAAA" + "CCCTGA"

    def test_remove_internal_atg_false_no_effect_on_tags(self):
        """remove_internal_atg=False does not change tag behaviour (tags always keep ATG)."""
        seq1 = "ATGAAATAA"
        seq2 = "ATGCCC"
        result_false = fuse_sequences(
            [{"sequence": seq1, "type": "protein"}, {"sequence": seq2, "type": "tag"}],
            linker="",
            remove_internal_atg=False,
        )
        result_true = fuse_sequences(
            [{"sequence": seq1, "type": "protein"}, {"sequence": seq2, "type": "tag"}],
            linker="",
            remove_internal_atg=True,
        )
        # Both should produce the same result — tag ATG is never removed
        assert result_false == result_true
        assert result_false == "ATGAAA" + "ATGCCC"
