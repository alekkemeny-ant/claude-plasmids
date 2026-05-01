#!/usr/bin/env python3
"""
Tests for src/gg_denovo.py — Golden Gate de novo oligo design.
"""

import pytest
from src.gg_denovo import (
    BASE_POOL,
    _rc,
    _oh_conflicts_enzyme,
    _design_pcr_primers,
    _design_annealing_oligos,
    _design_gblock,
    _design_part_in_vector,
    select_overhangs,
    design_golden_gate_oligos,
    AnnealingOligo,
)
from src.assembler import GG_ENZYMES, reverse_complement
from src.library import get_backbone_by_id


# ── Helpers ────────────────────────────────────────────────────────────────────

SHORT_FRAG = "ATCGATCGATCGATCGATCG"         # 20 bp
MEDIUM_FRAG = "ATCG" * 30                    # 120 bp
LONG_FRAG = "ATCGATCG" * 70                  # 560 bp

EGFP_STUB = "ATGGTGAGCAAGGGCGAGGAG" + "ATCG" * 50 + "TTACTTGTACAGCTCGTCCAT"  # ~230 bp
MCHERRY_STUB = "ATGGTGAGCAAGGGCGAGGAG" + "GCTA" * 50 + "TTACTTGTACAGCTCGTCCAT"


# ── TestBuildBasePool ──────────────────────────────────────────────────────────

class TestBuildBasePool:
    def test_no_palindromes(self):
        for oh in BASE_POOL:
            assert oh != _rc(oh), f"{oh} is palindromic"

    def test_no_rc_pair_conflicts(self):
        pool_set = set(BASE_POOL)
        for oh in BASE_POOL:
            r = _rc(oh)
            assert r not in pool_set, f"RC pair conflict: {oh} and {r} both in pool"

    def test_gc_content(self):
        for oh in BASE_POOL:
            gc = sum(1 for b in oh if b in "GC")
            assert 1 <= gc <= 3, f"{oh} has GC={gc}, expected 1–3"

    def test_pool_is_nonempty(self):
        assert len(BASE_POOL) > 50, f"Pool only has {len(BASE_POOL)} entries"

    def test_all_length_four(self):
        for oh in BASE_POOL:
            assert len(oh) == 4


# ── TestOhConflictsEnzyme ──────────────────────────────────────────────────────

class TestOhConflictsEnzyme:
    def test_bsai_conflict(self):
        # GGTC is a prefix of GGTCTC (BsaI)
        assert _oh_conflicts_enzyme("GGTC", "BsaI")

    def test_bsai_no_conflict(self):
        assert not _oh_conflicts_enzyme("AACC", "BsaI")

    def test_bbsi_conflict(self):
        assert _oh_conflicts_enzyme("GAAG", "BbsI")

    def test_paqci_conflict(self):
        assert _oh_conflicts_enzyme("CACC", "PaqCI")

    def test_esp3i_conflict(self):
        assert _oh_conflicts_enzyme("CGTC", "Esp3I")


# ── TestSelectOverhangs ────────────────────────────────────────────────────────

class TestSelectOverhangs:
    def test_returns_n_plus_one(self):
        for n in range(2, 6):
            result = select_overhangs(n, "BsaI")
            assert len(result) == n + 1, f"Expected {n+1} overhangs for {n} fragments, got {len(result)}"

    def test_no_palindromes(self):
        result = select_overhangs(4, "BsaI")
        for oh in result:
            assert oh != _rc(oh), f"Palindrome in result: {oh}"

    def test_no_rc_conflicts(self):
        result = select_overhangs(5, "BsaI")
        for i, oh in enumerate(result):
            for j, other in enumerate(result):
                if i != j:
                    assert oh != _rc(other), f"RC conflict: {oh} and {other}"

    def test_fixed_endpoints_respected(self):
        result = select_overhangs(3, "BsaI", fixed_left="AATG", fixed_right="CCAT")
        assert result[0] == "AATG"
        assert result[-1] == "CCAT"
        assert len(result) == 4

    def test_fixed_left_only(self):
        result = select_overhangs(2, "BsaI", fixed_left="AATG")
        assert result[0] == "AATG"
        assert len(result) == 3

    def test_fixed_right_only(self):
        result = select_overhangs(2, "BsaI", fixed_right="CCAT")
        assert result[-1] == "CCAT"
        assert len(result) == 3

    def test_deterministic(self):
        r1 = select_overhangs(3, "BsaI")
        r2 = select_overhangs(3, "BsaI")
        assert r1 == r2

    def test_raises_for_too_few(self):
        with pytest.raises(ValueError, match="2–10"):
            select_overhangs(1, "BsaI")

    def test_raises_for_too_many(self):
        with pytest.raises(ValueError, match="2–10"):
            select_overhangs(11, "BsaI")

    def test_raises_palindromic_fixed_left(self):
        with pytest.raises(ValueError, match="palindromic"):
            select_overhangs(2, "BsaI", fixed_left="AATT")

    def test_raises_rc_endpoint_pair(self):
        with pytest.raises(ValueError, match="RC of each other"):
            select_overhangs(2, "BsaI", fixed_left="AATG", fixed_right="CATT")  # CATT = RC(AATG)

    def test_raises_unknown_enzyme(self):
        with pytest.raises(ValueError, match="Unknown enzyme"):
            select_overhangs(2, "FakeEnzyme")

    def test_works_all_enzymes(self):
        for enzyme in GG_ENZYMES:
            result = select_overhangs(3, enzyme)
            assert len(result) == 4

    def test_no_enzyme_conflict_bsai(self):
        result = select_overhangs(5, "BsaI")
        for oh in result:
            assert not _oh_conflicts_enzyme(oh, "BsaI"), f"Enzyme conflict in {oh}"


# ── TestDesignPCRPrimers ───────────────────────────────────────────────────────

class TestDesignPCRPrimers:
    def test_bsai_prefix(self):
        fwd, rev, _, _, _ = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert fwd.startswith("GGTCTCN")
        assert rev.startswith("GGTCTCN")

    def test_bbsi_prefix(self):
        fwd, rev, _, _, _ = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "BbsI")
        assert fwd.startswith("GAAGACNN")
        assert rev.startswith("GAAGACNN")

    def test_paqci_prefix(self):
        fwd, rev, _, _, _ = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "PaqCI")
        assert fwd.startswith("CACCTGCNNNN")
        assert rev.startswith("CACCTGCNNNN")

    def test_fwd_contains_oh_left(self):
        fwd, _, _, _, _ = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert "AACC" in fwd

    def test_rev_contains_rc_oh_right(self):
        _, rev, _, _, _ = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert reverse_complement("TTGG") in rev

    def test_fwd_binding(self):
        fwd, _, fwd_b, _, _ = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "BsaI", binding_length=20)
        assert fwd_b == MEDIUM_FRAG[:20]
        assert fwd.endswith(fwd_b)

    def test_rev_binding(self):
        _, rev, _, rev_b, _ = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "BsaI", binding_length=20)
        assert rev_b == reverse_complement(MEDIUM_FRAG[-20:])
        assert rev.endswith(rev_b)

    def test_amplicon_size(self):
        _, _, _, _, amp = _design_pcr_primers(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        enzyme = GG_ENZYMES["BsaI"]
        tail = len(enzyme["recognition"]) + enzyme["cut_top"] + 4
        expected = len(MEDIUM_FRAG) + 2 * tail
        assert amp == expected


# ── TestDesignAnnealingOligos ─────────────────────────────────────────────────

class TestDesignAnnealingOligos:
    def test_short_fragment_two_oligos(self):
        oligos = _design_annealing_oligos(SHORT_FRAG, "AACC", "TTGG", "TEST", max_oligo_len=60)
        assert len(oligos) == 2
        strands = {o.strand for o in oligos}
        assert strands == {"top", "bottom"}

    def test_top_oligo_starts_with_oh_left(self):
        oligos = _design_annealing_oligos(SHORT_FRAG, "AACC", "TTGG", "TEST", max_oligo_len=60)
        top = next(o for o in oligos if o.strand == "top")
        assert top.sequence.startswith("AACC")

    def test_bottom_oligo_starts_with_oh_right(self):
        oligos = _design_annealing_oligos(SHORT_FRAG, "AACC", "TTGG", "TEST", max_oligo_len=60)
        bot = next(o for o in oligos if o.strand == "bottom")
        assert bot.sequence.startswith("TTGG")

    def test_annealing_complement_single_pair(self):
        oligos = _design_annealing_oligos(SHORT_FRAG, "AACC", "TTGG", "TEST", max_oligo_len=60)
        top = next(o for o in oligos if o.strand == "top")
        bot = next(o for o in oligos if o.strand == "bottom")
        # The fragment portions should be RC of each other
        top_frag = top.sequence[4:]   # strip oh_left
        bot_frag = bot.sequence[4:]   # strip oh_right
        assert bot_frag == reverse_complement(top_frag)

    def test_long_fragment_multiple_oligos(self):
        oligos = _design_annealing_oligos(LONG_FRAG, "AACC", "TTGG", "TEST",
                                           max_oligo_len=60, overlap_len=20)
        assert len(oligos) > 2
        for o in oligos:
            assert len(o.sequence) <= 60

    def test_oligo_names_labeled(self):
        oligos = _design_annealing_oligos(MEDIUM_FRAG, "AACC", "TTGG", "EGFP",
                                           max_oligo_len=60)
        for o in oligos:
            assert o.name.startswith("EGFP_")
            assert o.strand in ("top", "bottom")


# ── TestDesignGBlock ───────────────────────────────────────────────────────────

class TestDesignGBlock:
    def test_contains_fragment(self):
        gb = _design_gblock(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert MEDIUM_FRAG in gb

    def test_starts_with_enzyme_recognition(self):
        gb = _design_gblock(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert gb.startswith("GGTCTC")

    def test_ends_with_rc_recognition(self):
        gb = _design_gblock(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert gb.endswith(reverse_complement("GGTCTC"))

    def test_contains_oh_left(self):
        gb = _design_gblock(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert "AACC" in gb

    def test_contains_rc_oh_right(self):
        gb = _design_gblock(MEDIUM_FRAG, "AACC", "TTGG", "BsaI")
        assert reverse_complement("TTGG") in gb


# ── TestDesignGoldenGateOligos (integration) ──────────────────────────────────

class TestDesignGoldenGateOligos:
    def _two_frags(self, output_format="oligos", **kwargs):
        return design_golden_gate_oligos(
            fragments=[
                {"name": "FragA", "sequence": EGFP_STUB},
                {"name": "FragB", "sequence": MCHERRY_STUB},
            ],
            enzyme_name="BsaI",
            output_format=output_format,
            **kwargs,
        )

    def test_success_two_fragments(self):
        result = self._two_frags()
        assert result.success
        assert len(result.fragments) == 2

    def test_overhang_chain_consistent(self):
        result = self._two_frags()
        assert result.fragments[0].oh_right == result.fragments[1].oh_left

    def test_junction_map_has_correct_keys(self):
        result = self._two_frags()
        keys = list(result.junction_map.keys())
        assert keys[0].startswith("start →")
        assert "→" in keys[1]
        assert keys[-1].endswith("→ end")

    def test_output_primers_only(self):
        result = self._two_frags(output_format="primers")
        assert result.success
        for fd in result.fragments:
            assert fd.fwd_primer != ""
            assert fd.rev_primer != ""
            assert fd.annealing_oligos == []
            assert fd.synthesis_seq == ""

    def test_output_oligos_only(self):
        result = self._two_frags(output_format="oligos")
        assert result.success
        for fd in result.fragments:
            assert len(fd.annealing_oligos) >= 2
            assert fd.fwd_primer == ""
            assert fd.synthesis_seq == ""

    def test_output_gblocks_only(self):
        result = self._two_frags(output_format="gblocks")
        assert result.success
        for fd in result.fragments:
            assert fd.synthesis_seq != ""
            assert fd.fwd_primer == ""
            assert fd.annealing_oligos == []

    def test_output_both(self):
        carrier = get_backbone_by_id("pUC19")
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "FragA", "sequence": EGFP_STUB},
                {"name": "FragB", "sequence": MCHERRY_STUB},
            ],
            enzyme_name="BsaI",
            output_format="both",
            carrier_backbone=carrier,
        )
        assert result.success, result.errors
        for fd in result.fragments:
            assert fd.fwd_primer != ""
            assert len(fd.annealing_oligos) >= 2
            assert fd.synthesis_seq != ""
            assert fd.plasmid_seq != ""

    def test_three_fragments_overhang_chain(self):
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": EGFP_STUB},
                {"name": "B", "sequence": MEDIUM_FRAG},
                {"name": "C", "sequence": MCHERRY_STUB},
            ],
            output_format="oligos",
            enzyme_name="BsaI",
        )
        assert result.success
        assert result.fragments[0].oh_right == result.fragments[1].oh_left
        assert result.fragments[1].oh_right == result.fragments[2].oh_left

    def test_error_too_few_fragments(self):
        result = design_golden_gate_oligos(
            fragments=[{"name": "A", "sequence": EGFP_STUB}],
            output_format="oligos",
        )
        assert not result.success
        assert any("2–10" in e for e in result.errors)

    def test_error_too_many_fragments(self):
        frags = [{"name": f"F{i}", "sequence": MEDIUM_FRAG} for i in range(11)]
        result = design_golden_gate_oligos(fragments=frags, output_format="oligos")
        assert not result.success

    def test_error_empty_sequence(self):
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": ""},
                {"name": "B", "sequence": MEDIUM_FRAG},
            ],
            output_format="oligos",
        )
        assert not result.success

    def test_error_invalid_output_format(self):
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": EGFP_STUB},
                {"name": "B", "sequence": MCHERRY_STUB},
            ],
            output_format="invalid",
        )
        assert not result.success

    def test_custom_binding_length(self):
        result = self._two_frags(output_format="primers", binding_length=25)
        assert result.success
        for fd in result.fragments:
            assert fd.fwd_binding == fd.fragment_seq[:25]

    def test_custom_max_oligo_len(self):
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": LONG_FRAG},
                {"name": "B", "sequence": LONG_FRAG},
            ],
            output_format="oligos",
            max_oligo_len=80,
        )
        assert result.success
        for fd in result.fragments:
            for o in fd.annealing_oligos:
                assert len(o.sequence) <= 80

    def test_all_enzymes_work(self):
        for enzyme in GG_ENZYMES:
            result = design_golden_gate_oligos(
                fragments=[
                    {"name": "A", "sequence": EGFP_STUB},
                    {"name": "B", "sequence": MCHERRY_STUB},
                ],
                enzyme_name=enzyme,
                output_format="primers",
            )
            assert result.success, f"Failed for enzyme {enzyme}: {result.errors}"

    def test_backbone_seq_sets_endpoints(self):
        # Build a minimal synthetic backbone with two BsaI sites and known overhangs
        # Backbone structure: [fwd BsaI site + AACC overhang] + spacer + [TTGG + rev BsaI site]
        # BsaI (GGTCTC, cut_top=1): forward site leaves overhang starting 1nt after recognition
        # Forward site: GGTCTCN + AACC ... → leaves AACC overhang
        # Reverse site (on bottom strand): must leave TTGG overhang
        #   → on top strand: ...[RC(TTGG)] + N + [RC(GGTCTC)] = CCAA + N + GAGACC
        from src.assembler import find_gg_sites
        backbone = "NNNN" + "GGTCTCNAACC" + "N" * 100 + "CCAANGAGACC" + "NNNN"
        sites = find_gg_sites(backbone, "BsaI")
        if len(sites) >= 2:
            result = design_golden_gate_oligos(
                fragments=[
                    {"name": "A", "sequence": EGFP_STUB},
                    {"name": "B", "sequence": MCHERRY_STUB},
                ],
                output_format="oligos",
                enzyme_name="BsaI",
                backbone_seq=backbone,
            )
            assert result.success
            # Endpoint overhangs should match what the backbone provides
            assert result.backbone_left_oh is not None
            assert result.backbone_right_oh is not None
            assert result.fragments[0].oh_left == result.backbone_left_oh
            assert result.fragments[-1].oh_right == result.backbone_right_oh

    def test_output_part_in_vector(self):
        carrier = get_backbone_by_id("pUC19")
        assert carrier, "pUC19 must be in the library for this test"
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": EGFP_STUB},
                {"name": "B", "sequence": MCHERRY_STUB},
            ],
            output_format="part_in_vector",
            enzyme_name="BsaI",
            carrier_backbone=carrier,
        )
        assert result.success, result.errors
        for fd in result.fragments:
            assert fd.plasmid_seq != ""
            assert fd.plasmid_size_bp > len(fd.fragment_seq)
            assert fd.carrier_backbone_id == "pUC19"
            # Plasmid should contain the fragment sequence
            assert fd.fragment_seq in fd.plasmid_seq
            # Plasmid should contain enzyme recognition site (from flanking cassette)
            assert "GGTCTC" in fd.plasmid_seq
            # No primers or annealing oligos
            assert fd.fwd_primer == ""
            assert fd.annealing_oligos == []

    def test_part_in_vector_requires_carrier(self):
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": EGFP_STUB},
                {"name": "B", "sequence": MCHERRY_STUB},
            ],
            output_format="part_in_vector",
            enzyme_name="BsaI",
            carrier_backbone=None,
        )
        assert not result.success
        assert any("carrier_backbone_id" in e for e in result.errors)

    def test_part_in_vector_plasmid_contains_overhangs(self):
        carrier = get_backbone_by_id("pUC19")
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": EGFP_STUB},
                {"name": "B", "sequence": MCHERRY_STUB},
            ],
            output_format="part_in_vector",
            enzyme_name="BsaI",
            carrier_backbone=carrier,
        )
        assert result.success
        for fd in result.fragments:
            assert fd.oh_left in fd.plasmid_seq
            assert fd.oh_right in fd.plasmid_seq or reverse_complement(fd.oh_right) in fd.plasmid_seq

    def test_backbone_no_sites_returns_error(self):
        result = design_golden_gate_oligos(
            fragments=[
                {"name": "A", "sequence": EGFP_STUB},
                {"name": "B", "sequence": MCHERRY_STUB},
            ],
            output_format="oligos",
            enzyme_name="BsaI",
            backbone_seq="ATCGATCGATCGATCG",  # no BsaI sites
        )
        assert not result.success
        assert any("site" in e.lower() or "required" in e.lower() for e in result.errors)
