#!/usr/bin/env python3
"""Unit tests for Golden Gate assembly functions.

Covers:
  - find_gg_sites()    — Type IIS site detection (forward + reverse)
  - _excise_insert()   — insert excision from carrier vector
  - assemble_golden_gate() — full in-silico assembly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assembler import (
    GG_ENZYMES,
    GoldenGateResult,
    find_gg_sites,
    _excise_insert,
    assemble_golden_gate,
    reverse_complement,
    clean_sequence,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_scenario_a_backbone(
    bb_left: str,
    left_oh: str,
    dropout: str,
    right_oh: str,
    bb_right: str,
    spacer: str = "N",
) -> str:
    """Build a Scenario A backbone: FWD site on left, REV site on right.

    Layout:
        bb_left + CGTCTC + spacer + left_oh + dropout + right_oh + X + GAGACG + bb_right

    Esp3I cuts downstream of its recognition sequence (d_top=1, d_bot=5), so:
      - FWD (CGTCTC) site produces left overhang = left_oh at [rec_end+1 : rec_end+5]
      - REV (GAGACG) site produces right overhang = right_oh at [rec_start-5 : rec_start-1]
      - One filler nucleotide 'X' is needed between right_oh and GAGACG so that
        cut_top (rec_start-1) falls just outside right_oh.

    After digestion:
        bb_left_body = bb_left + "CGTCTC" + spacer   (backbone keeps the FWD rec seq)
        bb_right_body = "X" + "GAGACG" + bb_right    (backbone keeps the REV rec seq)
    """
    fwd = "CGTCTC"
    rev = "GAGACG"
    filler = "A"  # the one nt at cut_top position (goes with bb_right_body)
    return bb_left + fwd + spacer + left_oh + dropout + right_oh + filler + rev + bb_right


def _make_scenario_b_backbone(
    bb_left: str,
    left_oh: str,
    dropout: str,
    right_oh: str,
    bb_right: str,
    spacer: str = "N",
) -> str:
    """Build a Scenario B backbone: REV site on left, FWD site on right.

    This is the Allen Institute AICS vector orientation.  The recognition
    sequences end up on the dropout; the 4-nt overhangs remain on the scaffold.

    Layout:
        bb_left + left_oh + filler + GAGACG + dropout + CGTCTC + spacer + right_oh + bb_right

    For GAGACG at position p (p = len(bb_left)+len(left_oh)+len(filler)):
        cut_top    = p - 1
        cut_bottom = p - 5
        bb_left_body = seq[:cut_bottom] = bb_left     (backbone keeps left of cloning window)
        left_oh      = seq[cut_bottom:cut_top] = left_oh

    For CGTCTC at position q (q = p+6+len(dropout)):
        cut_top    = q + 7
        cut_bottom = q + 11
        right_oh      = seq[cut_top:cut_bottom] = right_oh
        bb_right_body = seq[cut_bottom:] = bb_right   (backbone keeps right of cloning window)
    """
    fwd = "CGTCTC"
    rev = "GAGACG"
    filler = "A"  # one nt at cut_top position (between left_oh and GAGACG)
    return bb_left + left_oh + filler + rev + dropout + fwd + spacer + right_oh + bb_right


def _make_scenario_a_carrier(
    carrier_backbone: str,
    left_oh: str,
    insert_body: str,
    right_oh: str,
    carrier_suffix: str = "PPPP",
    spacer: str = "N",
) -> str:
    """Build a Scenario A carrier vector for a part (FWD left, REV right)."""
    fwd = "CGTCTC"
    rev = "GAGACG"
    filler = "A"
    return carrier_backbone + fwd + spacer + left_oh + insert_body + right_oh + filler + rev + carrier_suffix


# ── TestFindGgSites ──────────────────────────────────────────────────────────


class TestFindGgSites:
    """Tests for find_gg_sites() — site detection and cut math."""

    def test_unknown_enzyme_raises(self):
        try:
            find_gg_sites("ATCG", "UltraRestrict9000")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown enzyme" in str(e)

    def test_no_sites_returns_empty(self):
        seq = "AAAAAAAAAAAAAAAAAA"
        sites = find_gg_sites(seq, "Esp3I")
        assert sites == []

    def test_forward_site_cut_positions(self):
        # CGTCTC at position 4, d_top=1, d_bot=5
        # cut_top = 4+6+1 = 11, cut_bottom = 4+6+5 = 15
        # overhang = seq[11:15]
        oh = "AATG"
        seq = "AAAA" + "CGTCTC" + "N" + oh + "CCCCCCCCC"
        sites = find_gg_sites(seq, "Esp3I")

        fwd = [s for s in sites if s["strand"] == "+"]
        assert len(fwd) == 1
        s = fwd[0]
        assert s["rec_start"] == 4
        assert s["cut_top"] == 11
        assert s["cut_bottom"] == 15
        assert s["overhang"] == oh

    def test_reverse_site_cut_positions(self):
        # We need GAGACG at position p=20 so that:
        #   cut_top    = p-1 = 19
        #   cut_bottom = p-5 = 15
        #   overhang   = seq[15:19] = "TTAA"
        # Layout: 15 A's + "TTAA" + "X" + "GAGACG" + padding
        #          0-14           15-18  19    20-25
        oh = "TTAA"
        seq = "AAAAAAAAAAAAAAA" + oh + "X" + "GAGACG" + "CCCC"
        assert seq[15:19] == oh          # sanity check
        assert seq[20:26] == "GAGACG"   # sanity check
        sites = find_gg_sites(seq, "Esp3I")

        rev = [s for s in sites if s["strand"] == "-"]
        assert len(rev) == 1
        s = rev[0]
        assert s["rec_start"] == 20
        assert s["cut_top"] == 19
        assert s["cut_bottom"] == 15
        assert s["overhang"] == oh

    def test_both_strands_detected(self):
        # Build a sequence with one FWD site and one REV site (no overlap)
        fwd_seq = "CGTCTCNAATGCCCCC"  # FWD at 0
        rev_seq = "TTTTTTTTTTTTGAGACG"  # REV at 12
        seq = fwd_seq + rev_seq
        sites = find_gg_sites(seq, "Esp3I")

        strands = {s["strand"] for s in sites}
        assert "+" in strands
        assert "-" in strands

    def test_site_sorted_by_rec_start(self):
        # Two FWD sites: one at 0, one at 30
        seq = "CGTCTCNAATG" + "A" * 19 + "CGTCTCNTTGG" + "CCCC"
        sites = find_gg_sites(seq, "Esp3I")
        starts = [s["rec_start"] for s in sites]
        assert starts == sorted(starts)

    def test_overhang_length_always_four(self):
        seq = "CGTCTCNAATGCCCCC" + "TTTTTTTTTTTTGAGACG"
        sites = find_gg_sites(seq, "Esp3I")
        for s in sites:
            assert len(s["overhang"]) == 4

    def test_site_too_close_to_edge_excluded(self):
        # CGTCTC at position 0; cut_bottom = 0+6+5=11; cut_bottom <= len("CGTCTCNA") = 8
        # Sequence is too short for the full cut — site should be excluded
        seq = "CGTCTCNA"  # only 8 chars; cut_bottom would be 11 (out of range)
        sites = find_gg_sites(seq, "Esp3I")
        fwd = [s for s in sites if s["strand"] == "+"]
        assert len(fwd) == 0

    def test_bsai_enzyme(self):
        # BsaI: GGTCTC, d_top=1, d_bot=5 (same offsets as Esp3I but different recognition)
        oh = "GCGC"
        seq = "AAAA" + "GGTCTC" + "N" + oh + "TTTTTTTT"
        sites = find_gg_sites(seq, "BsaI")
        fwd = [s for s in sites if s["strand"] == "+"]
        assert len(fwd) == 1
        assert fwd[0]["overhang"] == oh

    def test_bbsi_enzyme_different_offsets(self):
        # BbsI: GAAGAC, d_top=2, d_bot=6
        # cut_top = 0+6+2=8, cut_bottom = 0+6+6=12
        seq = "GAAGAC" + "NN" + "AAAACCCC"
        sites = find_gg_sites(seq, "BbsI")
        fwd = [s for s in sites if s["strand"] == "+"]
        assert len(fwd) == 1
        s = fwd[0]
        assert s["cut_top"] == 8
        assert s["cut_bottom"] == 12
        assert s["overhang"] == "AAAA"


# ── TestExciseInsert ─────────────────────────────────────────────────────────


class TestExciseInsert:
    """Tests for _excise_insert() — insert excision from carrier vector."""

    INSERT_BODY = "CCCCCCCCCC"  # 10 chars
    LEFT_OH = "AATG"
    RIGHT_OH = "TTAA"

    def _carrier_a(self, insert_body=None):
        """Scenario A carrier: FWD left, REV right."""
        return _make_scenario_a_carrier(
            carrier_backbone="AAAAA",
            left_oh=self.LEFT_OH,
            insert_body=insert_body or self.INSERT_BODY,
            right_oh=self.RIGHT_OH,
        )

    def test_basic_excision_scenario_a(self):
        carrier = self._carrier_a()
        result = _excise_insert(carrier, "Esp3I")
        assert result is not None
        left_oh, body, right_oh = result
        assert left_oh == self.LEFT_OH
        assert body == self.INSERT_BODY
        assert right_oh == self.RIGHT_OH

    def test_excision_returns_none_with_fewer_than_two_sites(self):
        # Only one Esp3I site
        seq = "AAAA" + "CGTCTCNAATGCCCCCC"
        result = _excise_insert(seq, "Esp3I")
        assert result is None

    def test_overhang_matching_selects_correct_sites(self):
        # Carrier has an extra Esp3I site (e.g. in AmpR) that should be ignored
        # when overhang hints are provided.
        extra_fwd = "CGTCTCN" + "XXXX"  # spurious FWD site with overhang XXXX
        carrier = extra_fwd + self._carrier_a()
        result = _excise_insert(
            carrier, "Esp3I",
            expected_left_oh=self.LEFT_OH,
            expected_right_oh=self.RIGHT_OH,
        )
        assert result is not None
        left_oh, body, right_oh = result
        assert left_oh == self.LEFT_OH
        assert body == self.INSERT_BODY
        assert right_oh == self.RIGHT_OH

    def test_positional_fallback_without_overhang_hints(self):
        # Without hints, code picks leftmost/rightmost by rec_start.
        # With exactly two sites flanking insert, this should work correctly.
        carrier = self._carrier_a()
        result = _excise_insert(carrier, "Esp3I")
        assert result is not None
        _, body, _ = result
        assert body == self.INSERT_BODY

    def test_different_insert_bodies_produce_correct_excision(self):
        for insert in ["ATGCATGCATGC", "TTTTTT", "GCGCGC"]:
            carrier = self._carrier_a(insert_body=insert)
            result = _excise_insert(carrier, "Esp3I")
            assert result is not None
            _, body, _ = result
            assert body == insert, f"Expected {insert}, got {body}"

    def test_insert_body_size_matches_original(self):
        body = "ATATAT" * 5  # 30 chars
        carrier = self._carrier_a(insert_body=body)
        result = _excise_insert(carrier, "Esp3I")
        assert result is not None
        _, excised_body, _ = result
        assert len(excised_body) == len(body)

    def test_overhang_length_is_four(self):
        carrier = self._carrier_a()
        result = _excise_insert(carrier, "Esp3I")
        assert result is not None
        left_oh, _, right_oh = result
        assert len(left_oh) == 4
        assert len(right_oh) == 4


# ── TestAssembleGoldenGate ───────────────────────────────────────────────────


# Shared constants for assembly tests
_LEFT_OH = "AATG"
_RIGHT_OH = "TTAA"
_INSERT = "CCCCCCCCCC"
_BB_LEFT_MARKER = "GGGGG"
_BB_RIGHT_MARKER = "TTTTT"


def _make_backbone_a():
    """Scenario A backbone with known markers."""
    return _make_scenario_a_backbone(
        bb_left=_BB_LEFT_MARKER,
        left_oh=_LEFT_OH,
        dropout="NNNNNNNNNN",
        right_oh=_RIGHT_OH,
        bb_right=_BB_RIGHT_MARKER,
    )


def _make_part(left_oh=_LEFT_OH, right_oh=_RIGHT_OH, insert_body=_INSERT):
    return {
        "name": "test_part",
        "plasmid_sequence": _make_scenario_a_carrier(
            carrier_backbone="AAAAA",
            left_oh=left_oh,
            insert_body=insert_body,
            right_oh=right_oh,
        ),
        "overhang_l": left_oh,
        "overhang_r": right_oh,
    }


class TestAssembleGoldenGate:
    """Tests for assemble_golden_gate() — full in-silico assembly."""

    def test_unknown_enzyme_returns_error(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq="ATCGATCGATCGATCG",
            parts=[_make_part()],
            enzyme_name="FakeEnzyme",
        )
        assert not result.success
        assert any("Unknown enzyme" in e for e in result.errors)

    def test_backbone_with_no_reverse_site_returns_error(self):
        # Backbone with only FWD sites — missing a REV site
        backbone = "AAAA" + "CGTCTCNAATG" + "CCCCCCCCCC" + "TTTT"
        result = assemble_golden_gate(
            backbone_plasmid_seq=backbone,
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert not result.success

    def test_part_with_no_plasmid_sequence_returns_error(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[{"name": "empty_part"}],
            enzyme_name="Esp3I",
        )
        assert not result.success
        assert any("plasmid_sequence" in e for e in result.errors)

    def test_part_with_no_cut_sites_returns_error(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[{"name": "no_sites", "plasmid_sequence": "ATCGATCGATCGATCG"}],
            enzyme_name="Esp3I",
        )
        assert not result.success
        assert any("no_sites" in e or "cut site" in e.lower() for e in result.errors)

    def test_single_part_assembly_succeeds(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success, f"Assembly failed: {result.errors}"

    def test_single_part_assembled_sequence_contains_insert(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success
        assert _INSERT in result.sequence

    def test_single_part_assembled_sequence_contains_backbone_markers(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success
        assert _BB_LEFT_MARKER in result.sequence
        assert _BB_RIGHT_MARKER in result.sequence

    def test_single_part_assembled_sequence_excludes_dropout(self):
        dropout = "NNNNNNNNNN"
        backbone = _make_scenario_a_backbone(
            bb_left=_BB_LEFT_MARKER,
            left_oh=_LEFT_OH,
            dropout=dropout,
            right_oh=_RIGHT_OH,
            bb_right=_BB_RIGHT_MARKER,
        )
        result = assemble_golden_gate(
            backbone_plasmid_seq=backbone,
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success
        assert dropout not in result.sequence

    def test_assembled_size_matches_sequence_length(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success
        assert result.total_size_bp == len(result.sequence)

    def test_result_contains_assembly_order(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success
        assert "test_part" in result.assembly_order

    def test_result_contains_junction_overhangs(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success
        assert _LEFT_OH in result.junction_overhangs
        assert _RIGHT_OH in result.junction_overhangs

    def test_assembled_sequence_is_valid_dna(self):
        from assembler import validate_dna
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success
        ok, errs = validate_dna(result.sequence)
        assert ok, f"Assembled sequence contains invalid DNA: {errs}"

    def test_multi_part_assembly_ordered_by_overhangs(self):
        # Two parts with compatible overhangs:
        #   Part A: left_oh=AATG, right_oh=GGCC
        #   Part B: left_oh=GGCC, right_oh=TTAA
        # Assembly order: backbone_left → AATG → A → GGCC → B → TTAA → backbone_right
        oh_l_a = "AATG"
        oh_r_a = "GGCC"  # Part A right = Part B left
        oh_l_b = "GGCC"
        oh_r_b = "TTAA"

        backbone = _make_scenario_a_backbone(
            bb_left=_BB_LEFT_MARKER,
            left_oh=oh_l_a,
            dropout="NNNNNNNNNN",
            right_oh=oh_r_b,
            bb_right=_BB_RIGHT_MARKER,
        )
        part_a = {
            "name": "part_A",
            "plasmid_sequence": _make_scenario_a_carrier(
                carrier_backbone="AAAAA",
                left_oh=oh_l_a,
                insert_body="AAAAAAAAAA",
                right_oh=oh_r_a,
            ),
            "overhang_l": oh_l_a,
            "overhang_r": oh_r_a,
        }
        part_b = {
            "name": "part_B",
            "plasmid_sequence": _make_scenario_a_carrier(
                carrier_backbone="GCGCG",
                left_oh=oh_l_b,
                insert_body="GCGCGCGCGC",
                right_oh=oh_r_b,
            ),
            "overhang_l": oh_l_b,
            "overhang_r": oh_r_b,
        }

        # Provide parts in WRONG order (B then A) to test overhang sorting
        result = assemble_golden_gate(
            backbone_plasmid_seq=backbone,
            parts=[part_b, part_a],
            enzyme_name="Esp3I",
        )

        assert result.success, f"Assembly failed: {result.errors}"
        assert result.assembly_order == ["part_A", "part_B"]
        # Both inserts present in assembled sequence
        assert "AAAAAAAAAA" in result.sequence
        assert "GCGCGCGCGC" in result.sequence
        # Parts appear in correct order
        pos_a = result.sequence.index("AAAAAAAAAA")
        pos_b = result.sequence.index("GCGCGCGCGC")
        assert pos_a < pos_b

    def test_mismatched_last_part_right_oh_produces_warning(self):
        # Last part's right_oh doesn't match backbone right_oh — should warn
        wrong_right_oh = "CCCC"  # backbone expects TTAA
        part = {
            "name": "mismatched_part",
            "plasmid_sequence": _make_scenario_a_carrier(
                carrier_backbone="AAAAA",
                left_oh=_LEFT_OH,
                insert_body=_INSERT,
                right_oh=wrong_right_oh,
            ),
            "overhang_l": _LEFT_OH,
            "overhang_r": wrong_right_oh,
        }
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[part],
            enzyme_name="Esp3I",
        )
        assert any("overhang" in w.lower() for w in result.warnings)

    def test_scenario_b_backbone_assembly(self):
        """Allen Institute vector style: REV site on left, FWD site on right."""
        backbone = _make_scenario_b_backbone(
            bb_left=_BB_LEFT_MARKER,
            left_oh=_LEFT_OH,
            dropout="NNNNNNNNNN",
            right_oh=_RIGHT_OH,
            bb_right=_BB_RIGHT_MARKER,
        )
        result = assemble_golden_gate(
            backbone_plasmid_seq=backbone,
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert result.success, f"Scenario B assembly failed: {result.errors}"
        assert _INSERT in result.sequence
        assert _BB_LEFT_MARKER in result.sequence
        assert _BB_RIGHT_MARKER in result.sequence

    def test_return_type_is_golden_gate_result(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq=_make_backbone_a(),
            parts=[_make_part()],
            enzyme_name="Esp3I",
        )
        assert isinstance(result, GoldenGateResult)

    def test_failed_result_has_empty_sequence(self):
        result = assemble_golden_gate(
            backbone_plasmid_seq="ATCGATCG",
            parts=[{"name": "p", "plasmid_sequence": "ATCGATCG"}],
            enzyme_name="Esp3I",
        )
        assert not result.success
        assert result.sequence is None

    def test_all_gg_enzyme_names_accepted(self):
        """All enzymes in GG_ENZYMES dict should be accepted (given valid sites)."""
        for enzyme_name, enzyme in GG_ENZYMES.items():
            rec = enzyme["recognition"]
            rec_rc = reverse_complement(rec)
            d_top = enzyme["cut_top"]
            d_bot = enzyme["cut_bottom"]

            # Build a minimal backbone with FWD site followed by a REV site
            oh = "AATG"
            spacer = "N"
            # FWD site at position 0: cut_top=rec_len+d_top, cut_bottom=rec_len+d_bot
            fwd_prefix = rec + spacer  # spacer fills up to cut_top
            rec_len = len(rec)
            # OH starts at cut_top; body_start=cut_bottom
            body = "CCCCCCCCCC"
            right_oh = "TTAA"
            filler = "A"
            # REV site follows right_oh + filler
            backbone = (
                "XXXXX"
                + rec + spacer + oh
                + body
                + right_oh + filler + rec_rc
                + "YYYYY"
            )
            carrier = (
                "AAAAA"
                + rec + spacer + oh
                + body
                + right_oh + filler + rec_rc
                + "PPPPP"
            )
            part = {
                "name": f"part_{enzyme_name}",
                "plasmid_sequence": carrier,
                "overhang_l": oh,
                "overhang_r": right_oh,
            }
            result = assemble_golden_gate(
                backbone_plasmid_seq=backbone,
                parts=[part],
                enzyme_name=enzyme_name,
            )
            # The sequence math may not perfectly produce body for all enzymes
            # (d_top and d_bot vary), but assembly should at least not error out
            # on the enzyme recognition step.
            assert "Unknown enzyme" not in str(result.errors), (
                f"Enzyme {enzyme_name} was rejected"
            )
