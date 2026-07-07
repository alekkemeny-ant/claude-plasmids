#!/usr/bin/env python3
"""
Golden Gate (Type IIS) assembly engine.

In-silico Golden Gate assembly: digests a backbone plasmid and part-in-vector
plasmids at their Type IIS sites, orders the parts by overhang matching, and
ligates them into the final construct. Like the core assembler, every operation
is deterministic string math on verified sequences — no LLM involvement.

The basic sequence utilities (clean_sequence, reverse_complement, validate_dna)
live in src.assembler and are imported here.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.assembler import clean_sequence, reverse_complement, validate_dna


GG_ENZYMES = {
    "Esp3I":  {"recognition": "CGTCTC", "cut_top": 1, "cut_bottom": 5},
    "BsmBI":  {"recognition": "CGTCTC", "cut_top": 1, "cut_bottom": 5},
    "BsaI":   {"recognition": "GGTCTC", "cut_top": 1, "cut_bottom": 5},
    "BbsI":   {"recognition": "GAAGAC", "cut_top": 2, "cut_bottom": 6},
    # PaqCI: CACCTGC(4/8), 4-nt 5' overhang (NEB #R0745)
    "PaqCI":  {"recognition": "CACCTGC", "cut_top": 4, "cut_bottom": 8},
}


@dataclass
class GoldenGateResult:
    """Result of a Golden Gate assembly operation."""
    success: bool
    sequence: Optional[str] = None
    total_size_bp: Optional[int] = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    assembly_order: list = field(default_factory=list)   # names in ligation order
    junction_overhangs: list = field(default_factory=list)  # 4-nt overhangs at each junction


def find_gg_sites(sequence: str, enzyme_name: str) -> list:
    """Find all Golden Gate cut sites for a given enzyme in a sequence.

    For each recognition site (forward or reverse), calculates where the
    enzyme cuts and the resulting 4-nt 5' overhang.

    Returns a list of dicts:
        strand    : "+" (top-strand recognition) or "-" (bottom-strand)
        rec_start : position where recognition sequence starts (0-based)
        cut_top   : position of top-strand cut
        cut_bottom: position of bottom-strand cut
        overhang  : 4-nt overhang sequence (always reported 5'→3' on top strand)
    """
    enzyme = GG_ENZYMES.get(enzyme_name)
    if enzyme is None:
        raise ValueError(f"Unknown enzyme: {enzyme_name!r}. Supported: {list(GG_ENZYMES)}")

    rec = enzyme["recognition"]
    d_top = enzyme["cut_top"]
    d_bot = enzyme["cut_bottom"]
    rec_len = len(rec)
    seq = clean_sequence(sequence)
    seq_len = len(seq)
    sites = []

    # Forward strand
    for m in re.finditer(rec, seq):
        p = m.start()
        ct = p + rec_len + d_top
        cb = p + rec_len + d_bot
        if cb <= seq_len:
            overhang = seq[ct:cb]
            sites.append({"strand": "+", "rec_start": p, "cut_top": ct,
                          "cut_bottom": cb, "overhang": overhang})

    # Reverse strand (RC of recognition)
    rec_rc = reverse_complement(rec)
    for m in re.finditer(rec_rc, seq):
        p = m.start()
        # Cuts are measured back from the recognition site start
        ct = p - d_top
        cb = p - d_bot
        if ct >= 0 and cb >= 0:
            lo, hi = min(ct, cb), max(ct, cb)
            overhang = seq[lo:hi]
            sites.append({"strand": "-", "rec_start": p, "cut_top": ct,
                          "cut_bottom": cb, "overhang": overhang})

    sites.sort(key=lambda s: s["rec_start"])
    return sites


def _excise_insert(
    plasmid_seq: str,
    enzyme_name: str,
    expected_left_oh: Optional[str] = None,
    expected_right_oh: Optional[str] = None,
) -> Optional[tuple]:
    """Extract the insert body + flanking overhangs from a part-in-vector plasmid.

    Expects at least one forward and one reverse cut site. When the expected
    overhangs are provided (from library metadata), the correct flanking sites
    are identified by overhang matching rather than by position — this avoids
    including carrier-vector sequence if there are extra Type IIS sites
    elsewhere in the vector (e.g. in the AmpR gene).

    Cut-position math for Esp3I at a reverse site (GAGACG at top-strand pos p):
        cut_top    = p - 1   (top-strand nick; insert body ends before this)
        cut_bottom = p - 5   (bottom-strand nick; right overhang = seq[p-5:p-1])
    So insert_body ends at rev_site["cut_bottom"] (NOT cut_top).

    Returns (left_overhang, insert_body, right_overhang) or None on failure.
    """
    sites = find_gg_sites(plasmid_seq, enzyme_name)
    if len(sites) < 2:
        return None

    # ── Site selection ─────────────────────────────────────────────────────
    # Search across ALL sites (forward and reverse) because the recognition
    # site orientation varies: some vectors put the FWD site before the insert
    # (standard) while others put the REV site first (recognition internal to
    # backbone/dropout, overhangs on the scaffold side).
    #
    # Prefer overhang-matching to find the correct flanking sites when the
    # library metadata provides expected_left_oh / expected_right_oh.
    # Fall back to leftmost / rightmost by recognition position if not.

    if expected_left_oh:
        left_matches = [s for s in sites if s["overhang"].upper() == expected_left_oh.upper()]
        left_site = left_matches[0] if left_matches else min(sites, key=lambda s: s["rec_start"])
    else:
        left_site = min(sites, key=lambda s: s["rec_start"])

    remaining = [s for s in sites if s is not left_site]
    if expected_right_oh:
        right_matches = [s for s in remaining if s["overhang"].upper() == expected_right_oh.upper()]
        right_site = right_matches[0] if right_matches else max(remaining, key=lambda s: s["rec_start"])
    else:
        right_site = max(remaining, key=lambda s: s["rec_start"])

    left_oh = left_site["overhang"]
    right_oh = right_site["overhang"]

    # ── Insert body boundaries ─────────────────────────────────────────────
    # The body starts just AFTER the left overhang and ends just BEFORE the
    # right overhang, regardless of which strand carries the recognition site.
    #
    # For a FWD left site:  body_start = cut_bottom (larger of the two cuts)
    # For a REV left site:  body_start = cut_top    (larger of the two cuts)
    # → universally: body_start = max(left_site cut_top, cut_bottom)
    #
    # For a REV right site: body_end = cut_bottom (smaller)
    # For a FWD right site: body_end = cut_top    (smaller)
    # → universally: body_end = min(right_site cut_top, cut_bottom)
    body_start = max(left_site["cut_top"], left_site["cut_bottom"])
    body_end   = min(right_site["cut_top"], right_site["cut_bottom"])

    if body_end > body_start:
        insert_body = plasmid_seq[body_start:body_end]
    else:
        # Wrap-around: insert spans the circular origin
        insert_body = plasmid_seq[body_start:] + plasmid_seq[:body_end]

    return left_oh, insert_body, right_oh


def assemble_golden_gate(
    backbone_plasmid_seq: str,
    parts: list,
    enzyme_name: str,
    dropout_sequence: Optional[str] = None,  # noqa: ARG001 — kept for documentation
) -> "GoldenGateResult":
    """Perform in-silico Golden Gate assembly.

    The backbone plasmid must carry exactly 2 Type IIS cut sites flanking the
    cloning window (plus optionally a dropout/negative-selection cassette
    between them that will be discarded).

    Each part dict must have:
        name            : str — display name
        plasmid_sequence: str — full circular plasmid sequence carrying the part
        (optional) overhang_l / overhang_r: pre-computed overhangs (4-nt).
                          If absent they are detected from the cut sites.

    Args:
        backbone_plasmid_seq: Full circular backbone sequence.
        parts               : List of part dicts (see above). Order may be
                              inferred from overhang matching.
        enzyme_name         : One of GG_ENZYMES keys.
        dropout_sequence    : Optional — the negative-selection cassette
                              sequence that will be discarded (unused in
                              sequence math, kept for documentation).

    Returns:
        GoldenGateResult
    """
    result = GoldenGateResult(success=False)

    if enzyme_name not in GG_ENZYMES:
        result.errors.append(
            f"Unknown enzyme {enzyme_name!r}. Supported: {list(GG_ENZYMES)}"
        )
        return result

    # ── 1. Find all backbone cut sites (used later after part overhangs known) ─
    bb_seq = clean_sequence(backbone_plasmid_seq)
    bb_sites = find_gg_sites(bb_seq, enzyme_name)

    # Early-exit guard: backbone must have at least one site of each strand
    if not any(s["strand"] == "+" for s in bb_sites) or \
       not any(s["strand"] == "-" for s in bb_sites):
        result.errors.append(
            f"Could not find both a forward and reverse {enzyme_name} site in the backbone."
        )
        return result

    # ── 2. Digest each part first so we know the terminal overhangs ──────────
    # We need the first part's left_oh and last part's right_oh to select the
    # correct backbone cut sites (avoiding spurious Type IIS sites elsewhere).

    excised = []  # list of (name, left_oh, insert_body, right_oh)
    for part in parts:
        name = part.get("name", "unnamed_part")
        ps = part.get("plasmid_sequence") or part.get("sequence", "")
        if not ps:
            result.errors.append(f"Part {name!r} has no plasmid_sequence.")
            return result

        # Pass the library overhangs to _excise_insert so it finds the
        # correct flanking sites even if the carrier vector has extra Esp3I
        # sites (e.g. in the AmpR gene).
        left_oh = part.get("overhang_l")
        right_oh = part.get("overhang_r")
        excised_tuple = _excise_insert(
            clean_sequence(ps), enzyme_name,
            expected_left_oh=left_oh,
            expected_right_oh=right_oh,
        )
        if excised_tuple is None:
            result.errors.append(
                f"Could not find {enzyme_name} cut sites in part {name!r}."
            )
            return result
        detected_left, insert_body, detected_right = excised_tuple
        # Use library overhangs when available; fall back to detected values.
        excised.append((
            name,
            (left_oh or detected_left).upper(),
            insert_body,
            (right_oh or detected_right).upper(),
        ))

    # ── 3. Digest the backbone ───────────────────────────────────────────────
    # Use the first part's left_oh and last part's right_oh to identify the
    # correct backbone cut sites.  Search ALL sites (both strands) because the
    # recognition-site orientation varies by vector design:
    #   • Standard (Scenario A): FWD site on left, REV site on right
    #   • Allen Institute backbones (Scenario B): REV site on left, FWD on right
    #     — recognition sequences end up on the dropout, overhangs on the scaffold

    first_part_left_oh = excised[0][1] if excised else None
    last_part_right_oh = excised[-1][3] if excised else None

    if len(bb_sites) < 2:
        result.errors.append(
            f"Could not find at least 2 {enzyme_name} sites in the backbone."
        )
        return result

    if first_part_left_oh:
        left_matches = [s for s in bb_sites if s["overhang"].upper() == first_part_left_oh]
        bb_left_site = left_matches[0] if left_matches else min(bb_sites, key=lambda s: s["rec_start"])
    else:
        bb_left_site = min(bb_sites, key=lambda s: s["rec_start"])

    remaining_bb = [s for s in bb_sites if s is not bb_left_site]
    if last_part_right_oh:
        right_matches = [s for s in remaining_bb if s["overhang"].upper() == last_part_right_oh]
        bb_right_site = right_matches[0] if right_matches else max(remaining_bb, key=lambda s: s["rec_start"])
    else:
        bb_right_site = max(remaining_bb, key=lambda s: s["rec_start"])

    # Body boundaries — orientation-agnostic:
    #   left body ends   at min(cut_top, cut_bottom) of the left site
    #   right body starts at max(cut_top, cut_bottom) of the right site
    # This gives the correct result for both FWD-left (Scenario A) and
    # REV-left (Scenario B) backbones without any strand-specific logic.
    bb_left_body = bb_seq[:min(bb_left_site["cut_top"], bb_left_site["cut_bottom"])]
    bb_left_oh   = bb_left_site["overhang"]
    bb_right_oh  = bb_right_site["overhang"]
    bb_right_body = bb_seq[max(bb_right_site["cut_top"], bb_right_site["cut_bottom"]):]

    # ── 3. Order parts by overhang matching ─────────────────────────────────
    # Build a map from left_overhang → part
    oh_map = {item[1]: item for item in excised}

    ordered = []
    current_oh = bb_left_oh
    visited = set()
    while current_oh in oh_map and current_oh not in visited:
        part_entry = oh_map[current_oh]
        ordered.append(part_entry)
        visited.add(current_oh)
        current_oh = part_entry[3]  # right_oh of this part

    if len(ordered) != len(excised):
        unmatched = [e[0] for e in excised if e[0] not in {o[0] for o in ordered}]
        result.warnings.append(
            f"Could not chain all parts via overhangs. "
            f"Unmatched parts: {unmatched}. "
            f"Using provided order instead."
        )
        ordered = excised  # fall back to given order

    # Verify the last part's right_oh matches the backbone right
    if ordered and ordered[-1][3] != bb_right_oh:
        result.warnings.append(
            f"Last part right overhang ({ordered[-1][3]!r}) does not match "
            f"backbone right overhang ({bb_right_oh!r}). Ligation may fail."
        )

    # ── 4. Ligate ────────────────────────────────────────────────────────────
    # Formula: bb_left_body + OH_left + part_body + OH_right + ... + bb_right_body

    assembled = bb_left_body
    junctions = [bb_left_oh]
    assembly_order = []
    for name, left_oh, insert_body, right_oh in ordered:
        assembled += left_oh + insert_body
        junctions.append(right_oh)
        assembly_order.append(name)
    assembled += ordered[-1][3] + bb_right_body if ordered else bb_right_body

    # ── 5. Validate ──────────────────────────────────────────────────────────
    valid, errors = validate_dna(assembled)
    if not valid:
        result.errors.extend(errors)
        return result

    result.success = True
    result.sequence = assembled
    result.total_size_bp = len(assembled)
    result.assembly_order = assembly_order
    result.junction_overhangs = junctions
    return result
