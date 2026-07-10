#!/usr/bin/env python3
"""
Sequence Assembly Engine for Expression Plasmid Design

Deterministic sequence assembly: splices an insert into a backbone at a
specified position. No LLM involvement — all operations are string-based
on verified sequences.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from src.utils.sequence_utils import clean_sequence, validate_dna, reverse_complement
from src.cloning.multiple_cloning_site_handler import MCSHandler
from src.config import DEFAULT_FUSION_LINKER, KOZAK

logger = logging.getLogger(__name__)


def find_mcs_insertion_point(backbone: dict) -> Optional[int]:
    """Module-level alias for MCSHandler.find_mcs_insertion_point.

    Kept for backward compatibility so callers can continue to
    `from src.cloning.assembler import find_mcs_insertion_point`.
    """
    return MCSHandler.find_mcs_insertion_point(backbone)


@dataclass
class AssemblyResult:
    """Result of a construct assembly operation."""
    success: bool
    sequence: Optional[str] = None
    total_size_bp: Optional[int] = None
    insert_position: Optional[int] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Structural checks (sequence-level)
    backbone_preserved: bool = False
    insert_preserved: bool = False
    # Expressed-sequence biology (always on the sense strand, regardless of RC)
    insert_has_start_codon: bool = False
    insert_has_stop_codon: bool = False
    insert_length_valid: bool = False
    # Positional / orientation checks (require backbone context)
    insertion_in_mcs: Optional[bool] = None        # None = could not determine
    insertion_disrupts_feature: Optional[str] = None  # name of disrupted feature, or None
    orientation_correct: Optional[bool] = None     # None = could not determine




def _check_insertion_in_mcs(
    insertion_position: int,
    backbone: dict,
    backbone_seq: str,
) -> tuple[Optional[bool], Optional[str]]:
    """Return (in_mcs, error_message). in_mcs=None means inconclusive."""
    mcs = backbone.get("mcs_position")
    if mcs:
        start, end = mcs["start"], mcs["end"]
        if start <= insertion_position <= end:
            return True, None
        return False, (
            f"Insertion at position {insertion_position} is outside the MCS "
            f"({start}–{end}). The insert will not be in the expression cassette "
            f"and will not be transcribed from the intended promoter."
        )
    bounds = MCSHandler.find_mcs_boundaries(backbone_seq)
    if bounds:
        start, end = bounds
        if start <= insertion_position <= end:
            return True, None
        return False, (
            f"Insertion at position {insertion_position} is outside the detected "
            f"MCS cluster ({start}–{end}). The insert may not be expressed."
        )
    return None, None  # can't determine


# Feature types that must not be disrupted by an insert
_PROTECTED_FEATURE_TYPES = {"CDS", "rep_origin"}


def _check_feature_disruption(
    insertion_position: int,
    backbone: dict,
    backbone_len: int,
) -> tuple[Optional[str], Optional[str]]:
    """Return (disrupted_feature_name, error_message) or (None, None) if clean.

    Features whose span covers more than 60 % of the backbone are skipped —
    these are malformed whole-plasmid annotations sometimes produced by
    Addgene's GenBank parser (e.g. rep_origin 0..N spanning the full sequence).
    """
    for feat in backbone.get("features") or []:
        if feat.get("type") not in _PROTECTED_FEATURE_TYPES:
            continue
        span = feat["end"] - feat["start"]
        if span > backbone_len * 0.6:
            continue  # malformed whole-plasmid annotation — skip
        # Strictly inside (not at boundary) — boundary insertions are fine
        if feat["start"] < insertion_position < feat["end"]:
            name = feat.get("name", "unknown")
            ftype = feat.get("type", "feature")
            return name, (
                f"Insertion at {insertion_position} disrupts '{name}' "
                f"({ftype}, {feat['start']}–{feat['end']}). "
                f"This will inactivate a critical plasmid element."
            )
    return None, None


def _check_orientation(
    insertion_position: int,
    reverse_complement_insert: bool,
    backbone: dict,
    backbone_seq: str,
) -> tuple[Optional[bool], Optional[str]]:
    """Check that the insert is oriented consistently with the MCS transcription direction.

    Compares the auto-detected MCS direction against what was actually done
    (reverse_complement_insert). A mismatch means the gene is pointing away
    from the promoter and will not be expressed.
    Returns (orientation_correct, error_message). None = could not determine.
    """
    features = backbone.get("features")
    mcs = backbone.get("mcs_position")
    mcs_bounds = (mcs["start"], mcs["end"]) if mcs else MCSHandler.find_mcs_boundaries(backbone_seq)
    if not mcs_bounds or not features:
        return None, None

    expected_direction = MCSHandler.detect_mcs_direction(mcs_bounds, features)
    expected_rc = (expected_direction == "reverse")

    if expected_rc == reverse_complement_insert:
        return True, None

    if expected_rc and not reverse_complement_insert:
        return False, (
            f"Orientation mismatch: the MCS is in reverse orientation "
            f"(promoter is downstream of position {insertion_position}), "
            f"but the insert was NOT reverse-complemented. "
            f"The gene will be transcribed antisense and will not be expressed."
        )
    # expected forward but RC was applied
    return False, (
        f"Orientation mismatch: the MCS is in forward orientation, "
        f"but the insert was reverse-complemented. "
        f"Verify this is intentional."
    )


def assemble_construct(
    backbone_seq: str,
    insert_seq: str,
    insertion_position: int,
    replace_region_end: Optional[int] = None,
    reverse_complement_insert: bool = False,
    backbone: Optional[dict] = None,
) -> AssemblyResult:
    """
    Assemble an expression construct by inserting a sequence into a backbone.

    The default mode inserts at a single position (no backbone sequence removed).
    If replace_region_end is provided, the backbone region from insertion_position
    to replace_region_end is replaced by the insert.

    Args:
        backbone_seq: Complete backbone DNA sequence.
        insert_seq: Insert DNA sequence (e.g., EGFP CDS) in expressed orientation.
        insertion_position: 0-based position in backbone to insert at.
        replace_region_end: If set, backbone[insertion_position:replace_region_end]
                           is replaced by the insert. Use this when replacing the
                           full MCS or an existing insert.
        reverse_complement_insert: If True, reverse-complement the insert before
                                   insertion (for reverse-orientation backbones).
        backbone: Optional backbone dict from the library. When provided, enables
                  three additional biological checks:
                  - Insertion is within the MCS bounds
                  - Insertion does not disrupt a CDS or origin of replication
                  - Insert orientation matches the MCS transcription direction

    Returns:
        AssemblyResult with the assembled sequence and validation details.
    """
    result = AssemblyResult(success=False)

    # --- Clean and validate inputs ---
    backbone_seq = clean_sequence(backbone_seq)
    insert_seq = clean_sequence(insert_seq)

    bb_valid, bb_errors = validate_dna(backbone_seq)
    if not bb_valid:
        result.errors.extend([f"Backbone: {e}" for e in bb_errors])
        return result

    ins_valid, ins_errors = validate_dna(insert_seq)
    if not ins_valid:
        result.errors.extend([f"Insert: {e}" for e in ins_errors])
        return result

    # --- Validate insertion position ---
    if insertion_position < 0 or insertion_position > len(backbone_seq):
        result.errors.append(
            f"Insertion position {insertion_position} is out of range "
            f"(backbone length: {len(backbone_seq)})"
        )
        return result

    if replace_region_end is not None:
        if replace_region_end < insertion_position:
            result.errors.append(
                f"replace_region_end ({replace_region_end}) must be >= "
                f"insertion_position ({insertion_position})"
            )
            return result
        if replace_region_end > len(backbone_seq):
            result.errors.append(
                f"replace_region_end ({replace_region_end}) exceeds "
                f"backbone length ({len(backbone_seq)})"
            )
            return result

    # --- Optionally reverse-complement the insert ---
    # Keep the original (expressed-orientation) sequence for biology checks below.
    expressed_seq = insert_seq

    if reverse_complement_insert:
        insert_seq = reverse_complement(insert_seq)

    # --- Assemble ---
    if replace_region_end is not None:
        # Replace mode: remove backbone[insertion_position:replace_region_end]
        upstream = backbone_seq[:insertion_position]
        downstream = backbone_seq[replace_region_end:]
    else:
        # Insert mode: splice insert into backbone at position
        upstream = backbone_seq[:insertion_position]
        downstream = backbone_seq[insertion_position:]

    assembled = upstream + insert_seq + downstream

    # --- Validate the assembled construct ---
    result.sequence = assembled
    result.total_size_bp = len(assembled)
    result.insert_position = insertion_position


    # Verify backbone preservation
    expected_backbone_len = len(backbone_seq)
    if replace_region_end is not None:
        replaced_len = replace_region_end - insertion_position
        expected_backbone_len -= replaced_len

    backbone_upstream_ok = assembled[:insertion_position] == backbone_seq[:insertion_position]
    if replace_region_end is not None:
        backbone_downstream_ok = assembled[insertion_position + len(insert_seq):] == backbone_seq[replace_region_end:]
    else:
        backbone_downstream_ok = assembled[insertion_position + len(insert_seq):] == backbone_seq[insertion_position:]

    result.backbone_preserved = backbone_upstream_ok and backbone_downstream_ok
    if not result.backbone_preserved:
        result.errors.append("Backbone sequence was not preserved during assembly")
        return result

    # Verify insert preservation
    extracted_insert = assembled[insertion_position:insertion_position + len(insert_seq)]
    result.insert_preserved = extracted_insert == insert_seq
    if not result.insert_preserved:
        result.errors.append("Insert sequence was not preserved during assembly")
        return result

    # Check insert biology on the expressed (sense) orientation, not the
    # potentially RC'd sequence that was spliced in.
    result.insert_has_start_codon = expressed_seq[:3] == "ATG"
    result.insert_has_stop_codon = expressed_seq[-3:] in ("TAA", "TAG", "TGA")
    result.insert_length_valid = len(expressed_seq) % 3 == 0

    if not result.insert_has_start_codon:
        result.warnings.append("Insert does not start with ATG (start codon)")
    if not result.insert_has_stop_codon:
        result.warnings.append("Insert does not end with a stop codon (TAA/TAG/TGA)")
    if not result.insert_length_valid:
        result.warnings.append(
            f"Insert length ({len(expressed_seq)} bp) is not a multiple of 3 — "
            f"may be out of reading frame"
        )

    # --- Biological context checks (require backbone dict) ---
    if backbone:
        # Check 1: insertion is within the MCS
        in_mcs, mcs_msg = _check_insertion_in_mcs(insertion_position, backbone, backbone_seq)
        result.insertion_in_mcs = in_mcs
        if in_mcs is False:
            result.errors.append(mcs_msg)

        # Check 2: insertion does not disrupt a protected feature
        disrupted, feat_msg = _check_feature_disruption(insertion_position, backbone, len(backbone_seq))
        result.insertion_disrupts_feature = disrupted
        if disrupted:
            result.errors.append(feat_msg)

        # Check 3: insert orientation matches MCS transcription direction
        orient_ok, orient_msg = _check_orientation(
            insertion_position, reverse_complement_insert, backbone, backbone_seq
        )

        result.orientation_correct = orient_ok
        if orient_ok is False:
            result.errors.append(orient_msg)

        if any(e for e in result.errors if result.errors):
            result.success = False
            return result
    # Expected size check
    expected_size = expected_backbone_len + len(insert_seq)
    if result.total_size_bp != expected_size:
        result.errors.append(
            f"Assembled size ({result.total_size_bp}) does not match expected "
            f"({expected_size} = {expected_backbone_len} backbone + {len(insert_seq)} insert)"
        )
        return result

    result.success = True
    return result


def fuse_sequences(sequences: list[dict], linker: Optional[str] = DEFAULT_FUSION_LINKER) -> str:
    """Fuse multiple coding sequences into a single CDS.

    Handles start/stop codon management at junctions:
    - First sequence: keep start codon (ATG), remove stop codon
    - Middle sequences (type="protein"): remove start codon AND stop codon
    - Last sequence (type="protein"): remove start codon, keep stop codon
    - Middle/last sequences (type="tag"): keep start codon (if any), manage stop only
    - Linker DNA inserted between each junction by default

    Start codons are removed from non-first sequences when their type is "protein"
    (the default). This is biologically required: in a fusion protein the ribosome
    translates from the first ATG only, so internal ATGs in subsequent CDS parts
    must be removed to keep the reading frame correct.

    Set type="tag" to preserve ATG (used for small epitope tags such as HA or Myc
    that may lack their own start codon, or fluorescent-protein tags appended
    C-terminally where you want to preserve the initiator Met context).
    Kozak (GCCACC) is inserted before the linker junction only when the following
    sequence is a "tag" that carries an ATG.

    The default linker is (GGGGS)x4 for protein-protein fusions. Pass
    linker="" for direct concatenation (e.g., short epitope tag fusions).

    Args:
        sequences: List of dicts, each with:
            - sequence: DNA sequence (required)
            - name: Name of the sequence (optional)
            - type: "protein" (default) or "tag". Non-first "protein" sequences
                    have their ATG removed. "tag" sequences are left as-is.
        linker: Linker DNA sequence. Defaults to (GGGGS)x4. Pass "" for
                direct concatenation (tag fusions).

    Returns:
        Fused CDS DNA sequence.

    Raises:
        ValueError: If fewer than 2 sequences provided or invalid DNA.
    """
    if linker is None:
        linker = DEFAULT_FUSION_LINKER

    if len(sequences) < 2:
        raise ValueError("At least 2 sequences are required for fusion")

    parts_seqs = []
    parts_types = []
    for i, seq_dict in enumerate(sequences):
        seq = clean_sequence(seq_dict["sequence"])
        valid, errors = validate_dna(seq)
        if not valid:
            name = seq_dict.get("name", f"sequence_{i}")
            raise ValueError(f"Invalid DNA in {name}: {'; '.join(errors)}")

        seq_type = seq_dict.get("type", "protein")
        is_first = (i == 0)
        is_last = (i == len(sequences) - 1)

        # Remove stop codon from all but the last sequence
        if not is_last:
            if seq[-3:] in ("TAA", "TAG", "TGA"):
                seq = seq[:-3]

        # Remove start codon from non-first protein sequences.
        # Tags are left unchanged — they either lack ATG or intentionally keep it.
        if not is_first and seq_type == "protein" and seq[:3] == "ATG":
            seq = seq[3:]

        parts_seqs.append(seq)
        parts_types.append(seq_type)

    # Join with optional linker
    if linker:
        cleaned_linker = clean_sequence(linker)
        valid, errors = validate_dna(cleaned_linker)
        if not valid:
            raise ValueError(f"Invalid linker DNA: {'; '.join(errors)}")

        # Build the result part by part.
        # Kozak (GCCACC) is inserted only when the next sequence is a tag that
        # keeps its ATG — protein parts had their ATG removed, so no Kozak needed.
        result = parts_seqs[0]
        for i in range(1, len(parts_seqs)):
            seq_str = parts_seqs[i]
            seq_type = parts_types[i]
            if seq_type == "tag" and seq_str[:3] == "ATG":
                result += cleaned_linker + KOZAK + seq_str
            else:
                result += cleaned_linker + seq_str
        return result
    else:
        return "".join(parts_seqs)









