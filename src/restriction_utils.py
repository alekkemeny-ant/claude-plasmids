"""
Restriction enzyme cut site checking and silent mutation design.

Pre-flight safety for Golden Gate and RE cloning: finds unexpected recognition
sites in backbone and insert sequences, and designs synonymous codon swaps to
eliminate them while preserving the encoded amino acid sequence.
"""

import itertools
import re
from functools import lru_cache
from typing import Optional

try:
    from .assembler import find_gg_sites, reverse_complement, clean_sequence, GG_ENZYMES
    from .codon_tables import HUMAN_CODON_W
except ImportError:
    from assembler import find_gg_sites, reverse_complement, clean_sequence, GG_ENZYMES
    from codon_tables import HUMAN_CODON_W


# Standard genetic code (AA → codon, local copy to avoid circular imports)
_CODON_TABLE: dict[str, str] = {
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


@lru_cache(maxsize=1)
def _synonymous_codon_sets() -> dict[str, list[str]]:
    """Return AA → list of synonymous coding codons (no stop codons)."""
    result: dict[str, list[str]] = {}
    for codon, aa in _CODON_TABLE.items():
        if aa == "*":
            continue
        result.setdefault(aa, []).append(codon)
    return result


def _site_in_feature(site_start: int, site_end: int, features: list[dict]) -> Optional[dict]:
    """Return the first feature whose span overlaps [site_start, site_end)."""
    for feat in features:
        f_start = feat.get("start", 0)
        f_end = feat.get("end", 0)
        if f_start < site_end and f_end > site_start:
            return feat
    return None


def _site_in_cds(site_start: int, site_end: int, features: list[dict]) -> Optional[dict]:
    """Return the first CDS feature overlapping [site_start, site_end), or None."""
    for feat in features:
        if feat.get("type", "").upper() != "CDS":
            continue
        f_start = feat.get("start", 0)
        f_end = feat.get("end", 0)
        if f_start < site_end and f_end > site_start:
            return feat
    return None


def find_extra_sites_in_sequence(
    sequence: str,
    enzyme_name: str,
    expected_site_count: int = 0,
) -> list[dict]:
    """Find recognition sites in sequence beyond the expected count.

    Args:
        sequence: DNA string
        enzyme_name: Key from GG_ENZYMES (e.g., "Esp3I")
        expected_site_count: Sites that are normal/expected (0 for inserts; 2 for GG backbones)

    Returns:
        List of extra site dicts from find_gg_sites, sorted by rec_start.
    """
    all_sites = find_gg_sites(sequence, enzyme_name)
    if len(all_sites) <= expected_site_count:
        return []
    # Return sites beyond the expected count. For GG backbones the first N sites
    # (sorted by position) are the expected cloning-window flanks; extras are the rest.
    return all_sites[expected_site_count:]


def check_re_sites(
    sequences: list[dict],
    enzyme_name: str,
) -> dict:
    """Check one or more sequences for unexpected assembly enzyme recognition sites.

    Args:
        sequences: List of dicts with:
            - name (str): display name
            - sequence (str): DNA string
            - expected_site_count (int, default 0): sites that are OK
            - features (list[dict], optional): feature dicts with name/type/start/end
        enzyme_name: Key from GG_ENZYMES

    Returns:
        {
          enzyme, recognition_sequence, all_clear,
          problematic_sequences: [{sequence_name, extra_site_count, sites: [
            {position, strand, overlapping_feature, in_cds, solvable_by_silent_mutation}
          ]}]
        }
    """
    if enzyme_name not in GG_ENZYMES:
        raise ValueError(f"Unknown enzyme: {enzyme_name!r}. Supported: {list(GG_ENZYMES)}")

    rec_seq = GG_ENZYMES[enzyme_name]["recognition"]
    site_len = len(rec_seq)
    problematic: list[dict] = []

    for seq_entry in sequences:
        name = seq_entry["name"]
        raw_seq = seq_entry["sequence"]
        expected = seq_entry.get("expected_site_count", 0)
        features = seq_entry.get("features") or []

        extra_sites = find_extra_sites_in_sequence(raw_seq, enzyme_name, expected)
        if not extra_sites:
            continue

        annotated_sites = []
        for site in extra_sites:
            pos = site["rec_start"]
            strand = site["strand"]
            site_end = pos + site_len

            cds_feat = _site_in_cds(pos, site_end, features)
            overlapping = cds_feat
            if overlapping is None:
                overlapping = _site_in_feature(pos, site_end, features)

            in_cds = cds_feat is not None
            annotated_sites.append({
                "position": pos,
                "strand": strand,
                "overlapping_feature": {
                    "name": overlapping["name"],
                    "type": overlapping["type"],
                    "start": overlapping["start"],
                    "end": overlapping["end"],
                } if overlapping else None,
                "in_cds": in_cds,
                "solvable_by_silent_mutation": in_cds,
            })

        problematic.append({
            "sequence_name": name,
            "extra_site_count": len(extra_sites),
            "sites": annotated_sites,
        })

    return {
        "enzyme": enzyme_name,
        "recognition_sequence": rec_seq,
        "all_clear": len(problematic) == 0,
        "problematic_sequences": problematic,
    }


def design_silent_mutation(
    cds_sequence: str,
    site_position_in_cds: int,
    enzyme_name: str,
    cds_frame_offset: int = 0,
) -> dict:
    """Design synonymous codon substitutions to eliminate a recognition site in a CDS.

    Finds all codons overlapping the recognition site and tries every combination of
    synonymous alternatives, choosing the one that (a) eliminates the site on both
    strands and (b) has the highest sum of HUMAN_CODON_W scores.

    Args:
        cds_sequence: CDS DNA string in sense (5'→3') orientation
        site_position_in_cds: 0-based start of recognition site within cds_sequence
        enzyme_name: Key from GG_ENZYMES
        cds_frame_offset: nt before first in-frame codon (almost always 0)

    Returns:
        {success, mutated_sequence, codons_changed, original_sequence_fragment,
         mutated_sequence_fragment, reason}
    """
    if enzyme_name not in GG_ENZYMES:
        raise ValueError(f"Unknown enzyme: {enzyme_name!r}")

    cds = clean_sequence(cds_sequence)
    rec = GG_ENZYMES[enzyme_name]["recognition"]
    rec_rc = reverse_complement(rec)
    site_len = len(rec)

    if site_position_in_cds < 0 or site_position_in_cds + site_len > len(cds):
        raise ValueError(
            f"site_position_in_cds={site_position_in_cds} out of range for CDS length {len(cds)}"
        )

    # Identify overlapping codon indices
    pos = site_position_in_cds
    coding_start = cds_frame_offset
    if pos < coding_start:
        return {
            "success": False, "mutated_sequence": None, "codons_changed": [],
            "original_sequence_fragment": None, "mutated_sequence_fragment": None,
            "reason": "site_in_5utr_before_coding_start",
        }

    first_codon_idx = (pos - coding_start) // 3
    last_codon_idx = (pos + site_len - 1 - coding_start) // 3

    syns = _synonymous_codon_sets()

    # Collect synonymous options for each overlapping codon
    codon_options: list[list[str]] = []
    codon_positions: list[int] = []  # absolute 0-based position in cds for each codon

    for ci in range(first_codon_idx, last_codon_idx + 1):
        codon_pos = coding_start + ci * 3
        if codon_pos + 3 > len(cds):
            break
        codon = cds[codon_pos:codon_pos + 3]
        aa = _CODON_TABLE.get(codon, "X")

        if aa in ("*", "X"):
            return {
                "success": False, "mutated_sequence": None, "codons_changed": [],
                "original_sequence_fragment": None, "mutated_sequence_fragment": None,
                "reason": "site_overlaps_stop_codon",
            }
        synonyms = syns.get(aa, [codon])
        codon_options.append(synonyms)
        codon_positions.append(codon_pos)

    if not codon_positions:
        return {
            "success": False, "mutated_sequence": None, "codons_changed": [],
            "original_sequence_fragment": None, "mutated_sequence_fragment": None,
            "reason": "no_coding_codons_overlap_site",
        }

    # Context window for checking: site position ± 20 nt (clamped)
    ctx_start = max(0, pos - 20)
    ctx_end = min(len(cds), pos + site_len + 20)

    orig_codons = [cds[cp:cp + 3] for cp in codon_positions]

    best_seq: Optional[str] = None
    best_score = -1.0
    best_n_changes = len(codon_positions) + 1  # sentinel: worse than any real count
    best_changes: list[dict] = []

    for combo in itertools.product(*codon_options):
        # Build candidate CDS with this combo substituted
        candidate = list(cds)
        for cp, new_codon in zip(codon_positions, combo):
            candidate[cp:cp + 3] = list(new_codon)
        candidate_str = "".join(candidate)

        window = candidate_str[ctx_start:ctx_end]
        if re.search(rec, window) or re.search(rec_rc, window):
            continue

        n_changes = sum(1 for orig, new in zip(orig_codons, combo) if orig != new)
        score = sum(HUMAN_CODON_W.get(c, 0.0) for c in combo)

        # Primary: prefer fewest codon changes. Secondary: prefer higher codon frequency.
        if n_changes < best_n_changes or (n_changes == best_n_changes and score > best_score):
            best_n_changes = n_changes
            best_score = score
            best_seq = candidate_str
            best_changes = [
                {
                    "codon_index": first_codon_idx + i,
                    "original_codon": orig,
                    "new_codon": new,
                    "amino_acid": _CODON_TABLE.get(orig, "X"),
                    "dna_position": cp,
                }
                for i, (orig, new, cp) in enumerate(zip(orig_codons, combo, codon_positions))
                if orig != new
            ]

    if best_seq is None:
        return {
            "success": False, "mutated_sequence": None, "codons_changed": [],
            "original_sequence_fragment": None, "mutated_sequence_fragment": None,
            "reason": "no_synonymous_escape",
        }

    ctx_orig = cds[ctx_start:ctx_end]
    ctx_mut = best_seq[ctx_start:ctx_end]

    return {
        "success": True,
        "mutated_sequence": best_seq,
        "codons_changed": best_changes,
        "original_sequence_fragment": ctx_orig,
        "mutated_sequence_fragment": ctx_mut,
        "reason": None,
    }
