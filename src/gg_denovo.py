#!/usr/bin/env python3
"""
Golden Gate De Novo Oligo/Fragment Design

Given a list of raw fragment sequences, designs orthogonal 4-nt overhangs for
all junctions and produces one of three outputs:
  - PCR primers  : forward + reverse primers to amplify each fragment from a template
  - Annealing oligos : tiled top/bottom oligos that anneal into a ready-to-ligate ds fragment
  - gBlock sequences : full synthesis-ready sequences with flanking Type IIS sites

No LLM involvement — all operations are deterministic sequence design.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

from .assembler import GG_ENZYMES, find_gg_sites, reverse_complement, clean_sequence, find_mcs_insertion_point


# ── Overhang pool ─────────────────────────────────────────────────────────────

def _rc(s: str) -> str:
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(comp[b] for b in reversed(s))


def _build_base_pool() -> list[str]:
    """
    All 4-nt overhangs satisfying:
      - Not palindromic (oh != RC(oh))
      - GC content 1–3 of 4
      - Only one representative per RC pair (oh, RC(oh))
    Returns ~108 candidates.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for oh in (''.join(x) for x in itertools.product('ACGT', repeat=4)):
        if oh == _rc(oh):
            continue
        gc = sum(1 for b in oh if b in 'GC')
        if gc < 1 or gc > 3:
            continue
        r = _rc(oh)
        if oh in seen or r in seen:
            continue
        candidates.append(oh)
        seen.add(oh)
        seen.add(r)
    return candidates


BASE_POOL: list[str] = _build_base_pool()


# ── Enzyme conflict detection ──────────────────────────────────────────────────

def _oh_conflicts_enzyme(oh: str, enzyme_name: str) -> bool:
    """
    Return True if the overhang is a 4-nt window within the enzyme's recognition
    sequence or its RC.

    Since overhangs are exactly 4 nt, only 4-nt windows of the recognition site
    can reconstruct a cut site at a junction — shorter matches are not actionable.
    """
    rec = GG_ENZYMES[enzyme_name]["recognition"]
    rec_rc = _rc(rec)
    oh_len = len(oh)
    for site in (rec, rec_rc):
        for i in range(len(site) - oh_len + 1):
            if site[i:i + oh_len] == oh:
                return True
    return False


# ── Overhang selection ─────────────────────────────────────────────────────────

def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def select_overhangs(
    n_fragments: int,
    enzyme_name: str,
    fixed_left: Optional[str] = None,
    fixed_right: Optional[str] = None,
) -> list[str]:
    """
    Select N+1 orthogonal 4-nt overhangs for N fragments.

    Returns [oh_0, oh_1, ..., oh_N] where:
      oh_0 = left end of assembly (= fixed_left if provided)
      oh_N = right end of assembly (= fixed_right if provided)
      oh_1..oh_{N-1} = designed internal junction overhangs

    Raises ValueError for invalid inputs or if the pool is exhausted.
    """
    if n_fragments < 2 or n_fragments > 10:
        raise ValueError(f"n_fragments must be 2–10, got {n_fragments}")
    if enzyme_name not in GG_ENZYMES:
        raise ValueError(f"Unknown enzyme {enzyme_name!r}. Supported: {list(GG_ENZYMES)}")

    # Validate fixed endpoints
    for label, oh in (("fixed_left", fixed_left), ("fixed_right", fixed_right)):
        if oh is not None:
            oh_upper = oh.upper()
            if oh_upper == _rc(oh_upper):
                raise ValueError(f"{label}={oh!r} is palindromic — cannot be used as a sticky end")

    if fixed_left and fixed_right:
        fl, fr = fixed_left.upper(), fixed_right.upper()
        if fl == _rc(fr):
            raise ValueError(
                f"fixed_left={fixed_left!r} and fixed_right={fixed_right!r} are RC of each other — "
                "they would ligate to each other instead of to the insert"
            )

    # Build filtered pool
    pool = [oh for oh in BASE_POOL if not _oh_conflicts_enzyme(oh, enzyme_name)]

    # Initialise chosen set and excluded set (chosen + their RCs)
    chosen: list[str] = []
    excluded: set[str] = set()

    def _add(oh: str) -> None:
        oh = oh.upper()
        chosen.append(oh)
        excluded.add(oh)
        excluded.add(_rc(oh))

    if fixed_left:
        _add(fixed_left)
    if fixed_right:
        _add(fixed_right)

    # Number of additional overhangs needed
    # Total needed = N+1; endpoints already accounted for
    endpoints_given = (1 if fixed_left else 0) + (1 if fixed_right else 0)
    n_needed = (n_fragments + 1) - endpoints_given

    for _ in range(n_needed):
        candidates = [oh for oh in pool if oh not in excluded]
        if not candidates:
            raise ValueError(
                f"Pool exhausted after selecting {len(chosen)} overhangs. "
                "Try fewer fragments or a different enzyme."
            )
        # Greedy: maximise minimum Hamming distance from all chosen + their RCs
        def _score(oh: str) -> tuple[int, int, str]:
            if not chosen:
                gc = sum(1 for b in oh if b in 'GC')
                return (4, gc, oh)
            min_h = min(_hamming(oh, c) for c in excluded)
            gc = sum(1 for b in oh if b in 'GC')
            return (min_h, gc, oh)

        best = max(candidates, key=_score)
        _add(best)

    # Assemble final list: [left, ...internal..., right]
    internal = [oh for oh in chosen if oh != (fixed_left.upper() if fixed_left else None)
                and oh != (fixed_right.upper() if fixed_right else None)]

    result: list[str] = []
    if fixed_left:
        result.append(fixed_left.upper())
    result.extend(internal[:n_fragments - 1])
    if fixed_right:
        result.append(fixed_right.upper())

    # If no fixed endpoints, we chose all N+1 overhangs; first and last are the endpoints
    if not fixed_left and not fixed_right:
        result = chosen[:]  # all chosen, no fixed constraints
    elif fixed_left and not fixed_right:
        result = [fixed_left.upper()] + internal
    elif not fixed_left and fixed_right:
        result = internal + [fixed_right.upper()]

    if len(result) != n_fragments + 1:
        raise ValueError(
            f"Expected {n_fragments + 1} overhangs but assembled {len(result)}. "
            "This is a bug — please report it."
        )
    return result


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class AnnealingOligo:
    name: str       # e.g. "EGFP_top_1"
    strand: str     # "top" or "bottom"
    sequence: str   # 5'→3' sequence to order
    position: int   # 0-based start position within the assembled fragment


@dataclass
class FragmentDesign:
    fragment_name: str
    fragment_seq: str       # cleaned, uppercase
    oh_left: str            # 4-nt left overhang
    oh_right: str           # 4-nt right overhang
    # PCR primers
    fwd_primer: str = ""
    rev_primer: str = ""
    fwd_binding: str = ""
    rev_binding: str = ""
    amplicon_size_bp: int = 0
    # Annealing oligos
    annealing_oligos: list = field(default_factory=list)
    # gBlock synthesis
    synthesis_seq: str = ""
    synthesis_size_bp: int = 0
    # Full part-in-vector plasmid
    plasmid_seq: str = ""
    plasmid_size_bp: int = 0
    carrier_backbone_id: str = ""


@dataclass
class GoldenGateDeNovoResult:
    success: bool
    enzyme_name: str
    output_format: str      # "primers" | "oligos" | "gblocks" | "part_in_vector" | "both"
    fragments: list[FragmentDesign] = field(default_factory=list)
    junction_map: dict = field(default_factory=dict)
    backbone_id: Optional[str] = None
    backbone_left_oh: Optional[str] = None
    backbone_right_oh: Optional[str] = None
    carrier_backbone_id: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Per-fragment design functions ──────────────────────────────────────────────

def _design_pcr_primers(
    fragment_seq: str,
    oh_left: str,
    oh_right: str,
    enzyme_name: str,
    binding_length: int = 20,
) -> tuple[str, str, str, str, int]:
    """Return (fwd_primer, rev_primer, fwd_binding, rev_binding, amplicon_size)."""
    enzyme = GG_ENZYMES[enzyme_name]
    rec = enzyme["recognition"]
    cut_top = enzyme["cut_top"]
    prefix = rec + "N" * cut_top

    fwd_binding = fragment_seq[:binding_length]
    rev_binding = reverse_complement(fragment_seq[-binding_length:])

    fwd_primer = prefix + oh_left + fwd_binding
    rev_primer = prefix + reverse_complement(oh_right) + rev_binding

    # Amplicon = fragment + both tails (prefix + overhang on each side)
    tail_len = len(rec) + cut_top + 4  # rec + spacer N's + overhang
    amplicon_size = len(fragment_seq) + 2 * tail_len

    return fwd_primer, rev_primer, fwd_binding, rev_binding, amplicon_size


def _design_annealing_oligos(
    fragment_seq: str,
    oh_left: str,
    oh_right: str,
    name: str,
    max_oligo_len: int = 60,
    overlap_len: int = 20,
) -> list[AnnealingOligo]:
    """
    Tile the fragment into annealing oligos.

    The annealed result is:
      top:    5'-[oh_left]-[fragment_seq]-3'          (5' overhang = oh_left)
      bottom: 5'-[oh_right]-[RC(fragment_seq)]-3'     (5' overhang = oh_right)

    For short fragments (full top strand fits in one oligo): 2 oligos.
    For longer fragments: tiled sets of top and bottom oligos.
    """
    top_full = oh_left + fragment_seq
    bot_full = oh_right + reverse_complement(fragment_seq)

    oligos: list[AnnealingOligo] = []
    step = max_oligo_len - overlap_len

    def _tile(seq: str, strand: str) -> list[AnnealingOligo]:
        tiles = []
        for i, start in enumerate(range(0, len(seq), step)):
            chunk = seq[start:start + max_oligo_len]
            # position is 0-based within the assembled fragment (oh_left starts at -4)
            pos = start - len(oh_left) if strand == "top" else -(start + len(chunk) - len(oh_right))
            tiles.append(AnnealingOligo(
                name=f"{name}_{strand}_{i + 1}",
                strand=strand,
                sequence=chunk,
                position=pos,
            ))
            if start + max_oligo_len >= len(seq):
                break
        return tiles

    if len(top_full) <= max_oligo_len:
        # Single pair
        oligos.append(AnnealingOligo(
            name=f"{name}_top_1",
            strand="top",
            sequence=top_full,
            position=-len(oh_left),
        ))
        oligos.append(AnnealingOligo(
            name=f"{name}_bot_1",
            strand="bottom",
            sequence=bot_full,
            position=0,
        ))
    else:
        oligos.extend(_tile(top_full, "top"))
        oligos.extend(_tile(bot_full, "bottom"))

    return oligos


def _design_gblock(
    fragment_seq: str,
    oh_left: str,
    oh_right: str,
    enzyme_name: str,
) -> str:
    """
    Full synthesis-ready sequence:
      [prefix][oh_left][fragment_seq][RC(oh_right)][spacer][RC(recognition)]
    After BsaI (or other enzyme) digestion, yields the fragment with oh_left and
    oh_right as 4-nt 5' overhangs.
    """
    enzyme = GG_ENZYMES[enzyme_name]
    rec = enzyme["recognition"]
    cut_top = enzyme["cut_top"]
    prefix = rec + "N" * cut_top
    suffix = "N" * cut_top + reverse_complement(rec)
    return prefix + oh_left + fragment_seq + reverse_complement(oh_right) + suffix


# ── Part-in-vector plasmid design ─────────────────────────────────────────────

def _design_part_in_vector(
    fragment_seq: str,
    oh_left: str,
    oh_right: str,
    enzyme_name: str,
    carrier_backbone: dict,
) -> str:
    """
    Build a full circular plasmid sequence by inserting the fragment (with flanking
    Type IIS sites and overhangs) into a carrier backbone at its MCS.

    The resulting plasmid is equivalent to the 'part_in_vector' format used by the
    Allen Institute modular system — it can be ordered as a whole-plasmid synthesis
    and later used directly in Golden Gate assembly to excise the insert.

    Spacer 'N' characters in the enzyme prefix/suffix are replaced with 'A' so the
    sequence is fully defined.
    """
    carrier_seq = clean_sequence(carrier_backbone.get("sequence", ""))
    if not carrier_seq:
        raise ValueError(
            f"Carrier backbone {carrier_backbone.get('id', '?')!r} has no sequence."
        )

    insertion_point = find_mcs_insertion_point(carrier_backbone)

    # Build the insert cassette: enzyme_site + overhang + fragment + RC(overhang) + RC(enzyme_site)
    # Replace 'N' spacers with 'A' for a fully defined plasmid sequence
    enzyme = GG_ENZYMES[enzyme_name]
    rec = enzyme["recognition"]
    cut_top = enzyme["cut_top"]
    spacer = "A" * cut_top  # concrete base instead of N

    cassette = (
        rec + spacer + oh_left
        + fragment_seq
        + reverse_complement(oh_right) + spacer + reverse_complement(rec)
    )

    return carrier_seq[:insertion_point] + cassette + carrier_seq[insertion_point:]


# ── Backbone endpoint extraction ───────────────────────────────────────────────

def _extract_backbone_overhangs(
    backbone_seq: str,
    enzyme_name: str,
) -> tuple[str, str, list[str]]:
    """
    Extract the left and right overhangs from an existing Golden Gate backbone.
    Returns (left_oh, right_oh, warnings).
    """
    warnings: list[str] = []
    sites = find_gg_sites(backbone_seq, enzyme_name)

    if len(sites) < 2:
        raise ValueError(
            f"Backbone has {len(sites)} {enzyme_name} site(s); at least 2 are required "
            "to define a cloning window."
        )

    if len(sites) > 2:
        warnings.append(
            f"Backbone has {len(sites)} {enzyme_name} sites. Using the outermost flanking "
            "sites as assembly endpoints. Verify this matches your intended cloning window."
        )

    # Left = site with smallest rec_start, right = site with largest rec_start
    sites_sorted = sorted(sites, key=lambda s: s["rec_start"])
    left_oh = sites_sorted[0]["overhang"]
    right_oh = sites_sorted[-1]["overhang"]

    return left_oh, right_oh, warnings


# ── Main entry point ───────────────────────────────────────────────────────────

def design_golden_gate_oligos(
    fragments: list[dict],
    enzyme_name: str = "BsaI",
    backbone_seq: Optional[str] = None,
    output_format: str = "oligos",
    binding_length: int = 20,
    max_oligo_len: int = 60,
    overlap_len: int = 20,
    carrier_backbone: Optional[dict] = None,
) -> GoldenGateDeNovoResult:
    """
    Design overhangs and output sequences for de novo Golden Gate assembly.

    Args:
        fragments       : ordered list of {"name": str, "sequence": str}
        enzyme_name     : key in GG_ENZYMES (default "BsaI")
        backbone_seq    : if provided, extract existing overhangs from backbone's Type IIS sites
        output_format   : "primers" | "oligos" | "gblocks" | "part_in_vector" | "both"
        binding_length  : gene-specific nt per PCR primer (primers output only)
        max_oligo_len   : max oligo length for annealing oligos
        overlap_len     : overlap between tiled annealing oligos
        carrier_backbone: backbone dict (from library) to use as carrier for part_in_vector output

    Returns:
        GoldenGateDeNovoResult
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Basic validation
    if len(fragments) < 2 or len(fragments) > 10:
        return GoldenGateDeNovoResult(
            success=False,
            enzyme_name=enzyme_name,
            output_format=output_format,
            errors=[f"Requires 2–10 fragments, got {len(fragments)}."],
        )

    if enzyme_name not in GG_ENZYMES:
        return GoldenGateDeNovoResult(
            success=False,
            enzyme_name=enzyme_name,
            output_format=output_format,
            errors=[f"Unknown enzyme {enzyme_name!r}. Supported: {list(GG_ENZYMES)}"],
        )

    valid_formats = {"primers", "oligos", "gblocks", "part_in_vector", "both"}
    if output_format not in valid_formats:
        return GoldenGateDeNovoResult(
            success=False,
            enzyme_name=enzyme_name,
            output_format=output_format,
            errors=[f"output_format must be one of {sorted(valid_formats)}, got {output_format!r}"],
        )

    # Clean and validate fragment sequences
    cleaned_fragments: list[dict] = []
    for frag in fragments:
        name = frag.get("name", "unnamed")
        raw_seq = frag.get("sequence", "")
        if not raw_seq:
            errors.append(f"Fragment {name!r} has an empty sequence.")
            continue
        try:
            seq = clean_sequence(raw_seq)
        except Exception as e:
            errors.append(f"Fragment {name!r} has invalid sequence: {e}")
            continue
        if len(seq) < 10:
            errors.append(f"Fragment {name!r} sequence is too short ({len(seq)} bp); minimum is 10 bp.")
            continue
        cleaned_fragments.append({"name": name, "sequence": seq})

    if errors:
        return GoldenGateDeNovoResult(
            success=False,
            enzyme_name=enzyme_name,
            output_format=output_format,
            errors=errors,
        )

    # Extract backbone overhangs if provided
    fixed_left: Optional[str] = None
    fixed_right: Optional[str] = None
    backbone_left_oh: Optional[str] = None
    backbone_right_oh: Optional[str] = None

    if backbone_seq:
        try:
            fixed_left, fixed_right, bb_warns = _extract_backbone_overhangs(backbone_seq, enzyme_name)
            backbone_left_oh = fixed_left
            backbone_right_oh = fixed_right
            warnings.extend(bb_warns)
        except ValueError as e:
            return GoldenGateDeNovoResult(
                success=False,
                enzyme_name=enzyme_name,
                output_format=output_format,
                errors=[str(e)],
            )

    # Select overhangs
    try:
        overhangs = select_overhangs(
            n_fragments=len(cleaned_fragments),
            enzyme_name=enzyme_name,
            fixed_left=fixed_left,
            fixed_right=fixed_right,
        )
    except ValueError as e:
        return GoldenGateDeNovoResult(
            success=False,
            enzyme_name=enzyme_name,
            output_format=output_format,
            errors=[f"Overhang selection failed: {e}"],
        )

    # Build junction map
    junction_map: dict[str, str] = {}
    n = len(cleaned_fragments)
    frag_names = [f["name"] for f in cleaned_fragments]
    junction_map["start → " + frag_names[0]] = overhangs[0]
    for i in range(n - 1):
        junction_map[frag_names[i] + " → " + frag_names[i + 1]] = overhangs[i + 1]
    junction_map[frag_names[-1] + " → end"] = overhangs[n]

    # Validate carrier backbone for part_in_vector output
    want_part_in_vector = output_format in ("part_in_vector", "both")
    if want_part_in_vector and not carrier_backbone:
        return GoldenGateDeNovoResult(
            success=False,
            enzyme_name=enzyme_name,
            output_format=output_format,
            errors=[
                "output_format='part_in_vector' requires a carrier_backbone. "
                "Provide a carrier_backbone_id (e.g. 'pUC19') in the tool call."
            ],
        )
    if want_part_in_vector and carrier_backbone:
        if not carrier_backbone.get("sequence"):
            return GoldenGateDeNovoResult(
                success=False,
                enzyme_name=enzyme_name,
                output_format=output_format,
                errors=[
                    f"Carrier backbone {carrier_backbone.get('id', '?')!r} has no sequence."
                ],
            )

    # Design per-fragment outputs
    want_primers = output_format in ("primers", "both")
    want_oligos = output_format in ("oligos", "both")
    want_gblocks = output_format in ("gblocks", "both")

    designed: list[FragmentDesign] = []
    for i, frag in enumerate(cleaned_fragments):
        seq = frag["sequence"]
        name = frag["name"]
        oh_l = overhangs[i]
        oh_r = overhangs[i + 1]

        fd = FragmentDesign(
            fragment_name=name,
            fragment_seq=seq,
            oh_left=oh_l,
            oh_right=oh_r,
        )

        if want_primers:
            fwd, rev, fwd_b, rev_b, amp = _design_pcr_primers(
                seq, oh_l, oh_r, enzyme_name, binding_length
            )
            fd.fwd_primer = fwd
            fd.rev_primer = rev
            fd.fwd_binding = fwd_b
            fd.rev_binding = rev_b
            fd.amplicon_size_bp = amp

        if want_oligos:
            fd.annealing_oligos = _design_annealing_oligos(
                seq, oh_l, oh_r, name, max_oligo_len, overlap_len
            )

        if want_gblocks:
            gb_seq = _design_gblock(seq, oh_l, oh_r, enzyme_name)
            fd.synthesis_seq = gb_seq
            fd.synthesis_size_bp = len(gb_seq)

        if want_part_in_vector:
            try:
                piv_seq = _design_part_in_vector(seq, oh_l, oh_r, enzyme_name, carrier_backbone)
                fd.plasmid_seq = piv_seq
                fd.plasmid_size_bp = len(piv_seq)
                fd.carrier_backbone_id = carrier_backbone.get("id", "")
            except ValueError as e:
                errors.append(f"Part-in-vector design failed for {name!r}: {e}")

        designed.append(fd)

    if errors:
        return GoldenGateDeNovoResult(
            success=False,
            enzyme_name=enzyme_name,
            output_format=output_format,
            errors=errors,
        )

    return GoldenGateDeNovoResult(
        success=True,
        enzyme_name=enzyme_name,
        output_format=output_format,
        fragments=designed,
        junction_map=junction_map,
        backbone_left_oh=backbone_left_oh,
        backbone_right_oh=backbone_right_oh,
        carrier_backbone_id=carrier_backbone.get("id") if carrier_backbone else None,
        warnings=warnings,
    )
