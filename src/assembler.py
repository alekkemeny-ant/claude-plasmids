#!/usr/bin/env python3
"""
Sequence Assembly Engine for Expression Plasmid Design

Deterministic sequence assembly: splices an insert into a backbone at a
specified position. No LLM involvement — all operations are string-based
on verified sequences.
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
import io
from typing import Optional
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
import json

# pLannotate and bokeh are conda-only (not on PyPI). Guard the imports so the
# module remains importable in pip-based environments; annotation functions
# will raise a clear error when called if these are unavailable.
try:
    from bokeh.embed import json_item
    from plannotate.annotate import annotate
    from plannotate.bokeh_plot import get_bokeh
    from plannotate.resources import get_seq_record
    _PLANNOTATE_AVAILABLE = True
except ImportError:
    json_item = None
    annotate = None
    get_bokeh = None
    get_seq_record = None
    _PLANNOTATE_AVAILABLE = False

_PLANNOTATE_MISSING_MSG = (
    "pLannotate not available — install via conda using environment.yml "
    "(plannotate is not on PyPI)"
)


logger = logging.getLogger(__name__)

# (GGGGS)x4 linker — default for protein-protein fusions
DEFAULT_FUSION_LINKER = "GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC"
KOZAK = "GCCACC"


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


def clean_sequence(sequence: str) -> str:
    """Remove whitespace and normalize to uppercase."""
    return re.sub(r'\s', '', sequence.upper())


def validate_dna(sequence: str) -> tuple[bool, list[str]]:
    """Validate that a string is valid DNA. Returns (is_valid, errors)."""
    errors = []
    if not sequence:
        errors.append("Sequence is empty")
        return False, errors

    invalid_chars = set(sequence) - set('ATCGN')
    if invalid_chars:
        errors.append(f"Invalid characters in sequence: {sorted(invalid_chars)}")
        return False, errors

    return True, []


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(sequence))


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


def resolve_insertion_point(
    backbone: dict,
    backbone_seq: str,
) -> tuple[Optional[int], bool]:
    """Return (insertion_position, reverse_complement_insert).

    First tries the pre-stored mcs_position in the backbone dict.
    If that is absent, runs MCSHandler on the raw sequence to detect
    the MCS cluster and promoter direction automatically.
    Returns (None, False) if no position can be determined.
    """
    pos = find_mcs_insertion_point(backbone)
    if pos is not None:
        return pos, False

    bounds = MCSHandler.find_mcs_boundaries(backbone_seq)
    if bounds is None:
        return None, False

    features = backbone.get("features")
    direction = MCSHandler.detect_mcs_direction(bounds, features)
    if direction == "reverse":
        return bounds[1], True  # end of MCS cluster, RC insert
    else:
        return bounds[0], False  # start of MCS cluster, no RC


def find_mcs_insertion_point(backbone: dict) -> Optional[int]:
    """
    Determine the best insertion point within the MCS of a backbone.

    For expression vectors, the insert should go at the start of the MCS
    (immediately downstream of the promoter). Returns a 0-based position.

    Args:
        backbone: Backbone dict from the library (must have mcs_position).

    Returns:
        Insertion position (0-based), or None if no MCS info available.
    """
    mcs = backbone.get("mcs_position")
    if not mcs:
        return None

    # Default: insert at the start of the MCS
    # This places the insert immediately downstream of the promoter/5' UTR
    return mcs["start"]


class MCSHandler:
    """Handles finding and inserting genes into plasmid MCS (Multiple Cloning Site)."""

    # Common MCS recognition patterns (restriction sites commonly found in MCS)
    COMMON_MCS_PATTERNS = {
        "EcoRI": "GAATTC",
        "BamHI": "GGATCC",
        "KpnI": "GGTACC",
        "XbaI": "TCTAGA",
        "SalI": "GTCGAC",
        "PstI": "CTGCAG",
        "NotI": "GCGGCCGC",
        "XhoI": "CTCGAG",
        "NheI": "GCTAGC",
        "SmaI": "CCCGGG",
        "ApaI": "GGGCCC",
    }

    @staticmethod
    def find_mcs_sites(backbone_seq: str) -> list:
        """
        Find common restriction sites in the backbone that likely define the MCS.

        Args:
            backbone_seq: Backbone sequence string

        Returns:
            List of dicts: {name, position, end_position, pattern}
        """
        sites = []
        backbone_upper = backbone_seq.upper()

        for site_name, pattern in MCSHandler.COMMON_MCS_PATTERNS.items():
            matches = re.finditer(pattern, backbone_upper)
            for match in matches:
                sites.append({
                    "name": site_name,
                    "position": match.start(),
                    "end_position": match.end(),
                    "pattern": pattern
                })

        # Sort by position
        sites.sort(key=lambda x: x["position"])
        return sites

    @staticmethod
    def find_mcs_boundaries(backbone_seq: str, max_gap: int = 40) -> Optional[tuple]:
        """
        Find the MCS boundaries by identifying the densest cluster of
        restriction sites. An MCS is characterized by many unique sites
        packed into a short region. Sites separated by more than max_gap bp
        (measured from end of one site to start of the next) are considered
        outside the cluster.

        Args:
            backbone_seq: Backbone sequence string
            max_gap: Maximum bp gap between the end of one site and the
                     start of the next to be considered part of the same
                     cluster. Default 40 bp — real MCS sites are typically
                     0-30 bp apart.

        Returns:
            Tuple of (start_position, end_position) of the best cluster,
            or None if fewer than 3 sites found.
        """
        sites = MCSHandler.find_mcs_sites(backbone_seq)

        if len(sites) < 3:
            logger.warning("Could not find enough restriction sites for MCS")
            return None

        # Group sites into clusters based on max_gap between consecutive sites
        # Gap is measured from end_position of previous site to position of next
        clusters = []
        current_cluster = [sites[0]]

        for i in range(1, len(sites)):
            gap = sites[i]["position"] - current_cluster[-1]["end_position"]
            if gap <= max_gap:
                current_cluster.append(sites[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sites[i]]
        clusters.append(current_cluster)

        # Pick the cluster with the most unique enzyme sites
        best_cluster = max(clusters, key=lambda c: len(set(s["name"] for s in c)))

        if len(best_cluster) < 3:
            logger.warning("No dense restriction site cluster found")
            return None

        start = best_cluster[0]["position"]
        end = best_cluster[-1]["end_position"]
        return (start, end)

    @staticmethod
    def detect_mcs_direction(mcs_bounds: tuple, features: Optional[list[dict]] = None) -> str:
        """
        Detect whether the MCS is in forward or reverse orientation by
        looking at the nearest promoter relative to the MCS cluster.

        Forward: promoter is upstream (lower bp) of MCS → insert at MCS start
        Reverse: promoter is downstream (higher bp) of MCS → insert at MCS end

        Args:
            mcs_bounds: Tuple of (start_position, end_position) from find_mcs_boundaries.
            features: List of feature dicts with name, type, start, end keys.

        Returns:
            "forward" or "reverse"
        """
        if not features:
            return "forward"

        mcs_center = (mcs_bounds[0] + mcs_bounds[1]) / 2

        # Find promoter features
        promoters = []
        for feat in features:
            feat_type = feat.get("type", "").lower()
            feat_name = feat.get("name", "").lower()
            # Skip bacterial resistance promoters (e.g. bla)
            if "bla" in feat_name or "resistance" in feat_name:
                continue
            if feat_type == "promoter" or "promoter" in feat_name:
                promoters.append(feat)

        if not promoters:
            return "forward"

        # Find the promoter closest to the MCS
        closest_promoter = min(
            promoters,
            key=lambda p: min(abs(p["start"] - mcs_center), abs(p["end"] - mcs_center))
        )

        # If the promoter center is at a higher bp position than the MCS center,
        # the expression cassette runs in reverse
        promoter_center = (closest_promoter["start"] + closest_promoter["end"]) / 2
        if promoter_center > mcs_center:
            return "reverse"

        return "forward"

    @staticmethod
    def insert_gene_at_mcs(
        backbone_seq: str,
        gene_seq: str,
        insertion_point: Optional[int] = None,
        features: Optional[list[dict]] = None,
    ) -> dict:
        """
        Insert gene sequence into plasmid at MCS or specified position.
        Automatically detects MCS direction from features to place the
        insert at the correct end of the MCS (closest to the promoter).

        Args:
            backbone_seq: Backbone sequence string
            gene_seq: Gene sequence to insert
            insertion_point: Optional specific position to insert. If None, use MCS detection.
            features: Optional list of backbone feature dicts (for direction detection).

        Returns:
            Dictionary with:
            - final_sequence: The resulting construct
            - insertion_position: Where the gene was inserted
            - method: How the insertion was performed
            - direction: "forward" or "reverse"
            - mcs_sites: All restriction sites found
        """
        if not backbone_seq or not gene_seq:
            return {
                "final_sequence": None,
                "insertion_position": None,
                "method": "error",
                "direction": None,
                "error": "Missing backbone or gene sequence"
            }

        direction = "forward"

        # Try to find MCS boundaries
        if insertion_point is None:
            mcs_bounds = MCSHandler.find_mcs_boundaries(backbone_seq)
            if mcs_bounds:
                direction = MCSHandler.detect_mcs_direction(mcs_bounds, features)
                if direction == "reverse":
                    insertion_point = mcs_bounds[1]  # End of cluster (closer to promoter)
                else:
                    insertion_point = mcs_bounds[0]  # Start of cluster (closer to promoter)
                method = "mcs"
                logger.info(f"MCS detected ({direction}): inserting at position {insertion_point}")
            else:
                # Fallback: try to find promoter and insert after it
                promoter_match = re.search(r"CMV|SV40|EF1A|UBC", backbone_seq.upper())
                if promoter_match:
                    insertion_point = promoter_match.end() + 100  # Insert 100bp after promoter start
                    method = "after_promoter"
                    logger.info(f"MCS not found, inserting after promoter at position {insertion_point}")
                else:
                    # Default: concatenate
                    insertion_point = len(backbone_seq)
                    method = "concatenation"
                    logger.warning("Could not find MCS or promoter, using concatenation")
        else:
            method = "custom_position"

        # Insert the gene
        if insertion_point < 0 or insertion_point > len(backbone_seq):
            insertion_point = len(backbone_seq)

        if direction == "reverse":
            gene_seq = reverse_complement(gene_seq)

        final_sequence = backbone_seq[:insertion_point] + gene_seq + backbone_seq[insertion_point:]

        return {
            "final_sequence": final_sequence,
            "insertion_position": insertion_point,
            "method": method,
            "direction": direction,
            "mcs_sites": MCSHandler.find_mcs_sites(backbone_seq)
        }


def format_as_fasta(sequence: str, name: str, description: str = "") -> str:
    """Format a sequence as FASTA with 80-character line wrapping."""
    header = f">{name}"
    if description:
        header += f" {description}"

    lines = [header]
    for i in range(0, len(sequence), 80):
        lines.append(sequence[i:i + 80])

    return "\n".join(lines) + "\n"


def _build_annotated_record(
    sequence: str,
    df,
    name: str,
    backbone_name: str,
    insert_name: str,
    insert_position: int,
    insert_length: int,
    reverse_complement_insert: bool,
):
    """Build a BioPython SeqRecord from a pLannotate df, adding the insert feature if needed."""
    if not _PLANNOTATE_AVAILABLE:
        raise RuntimeError(_PLANNOTATE_MISSING_MSG)
    record = get_seq_record(df, sequence, is_linear=False)
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "circular"

    locus_name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)[:16]
    record.name = locus_name
    record.id = locus_name
    record.description = f"{insert_name} in {backbone_name}" if backbone_name else name

    if insert_length > 0:
        insert_start = insert_position
        insert_end = insert_position + insert_length
        already_annotated = any(
            int(f.location.start) < insert_end and int(f.location.end) > insert_start
            for f in record.features
            if f.type not in ("source", "rep_origin")
        )
        if not already_annotated:
            strand = -1 if reverse_complement_insert else 1
            record.features.append(SeqFeature(
                FeatureLocation(insert_start, insert_end, strand=strand),
                type="CDS",
                qualifiers={
                    "label": [insert_name],
                    "note": [f"Insert: {insert_name}"],
                }
            ))
    return record


def format_as_genbank(
    sequence: str,
    name: str,
    backbone_name: str = "",
    insert_name: str = "",
    insert_position: int = 0,
    insert_length: int = 0,
    reverse_complement_insert: bool = False,
    features: Optional[list[dict]] = None,
) -> str:
    """Format an assembled construct as a GenBank flat file.

    When pLannotate is available (conda environment), uses BLAST-based
    annotation for rich feature identification. Otherwise falls back to
    a minimal hand-written GenBank with just the insert + backbone features.
    """
    if not _PLANNOTATE_AVAILABLE:
        return _format_as_genbank_fallback(
            sequence=sequence, name=name, backbone_name=backbone_name,
            insert_name=insert_name, insert_position=insert_position,
            insert_length=insert_length, features=features,
        )
    df = annotate(sequence, linear=False)
    record = _build_annotated_record(
        sequence, df, name, backbone_name, insert_name,
        insert_position, insert_length, reverse_complement_insert,
    )
    handle = io.StringIO()
    SeqIO.write(record, handle, "genbank")
    return handle.getvalue()


def _format_as_genbank_fallback(
    sequence: str,
    name: str,
    backbone_name: str = "",
    insert_name: str = "",
    insert_position: int = 0,
    insert_length: int = 0,
    features: Optional[list[dict]] = None,
) -> str:
    """Minimal GenBank writer for environments without pLannotate.

    Produces a valid GenBank flat file with the insert CDS and any
    explicitly-passed backbone features, but no BLAST-based annotation.
    """
    # Truncate locus name to 16 chars per GenBank spec
    locus_name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)[:16]
    total_len = len(sequence)

    lines = []

    # LOCUS line
    lines.append(
        f"LOCUS       {locus_name:<16} {total_len:>5} bp    DNA     circular   UNK"
    )

    # DEFINITION
    lines.append(f"DEFINITION  Expression construct: {insert_name} in {backbone_name}.")

    # FEATURES
    lines.append("FEATURES             Location/Qualifiers")

    # Source feature
    lines.append(f"     source          1..{total_len}")
    lines.append('                     /mol_type="other DNA"')
    lines.append('                     /note="Assembled construct"')

    # Insert feature
    if insert_length > 0:
        ins_start_1based = insert_position + 1
        ins_end_1based = insert_position + insert_length
        lines.append(f"     CDS             {ins_start_1based}..{ins_end_1based}")
        lines.append(f'                     /label="{insert_name}"')
        lines.append(f'                     /note="Insert: {insert_name}"')

    # Additional features (offset those that come after the insertion point)
    if features:
        for feat in features:
            feat_start = feat["start"]
            feat_end = feat["end"]
            # Offset features downstream of the insert
            if feat_start >= insert_position:
                feat_start += insert_length
                feat_end += insert_length
            feat_type = feat.get("type", "misc_feature")
            feat_name = feat.get("name", "unknown")
            # Pad feature type to match GenBank format
            lines.append(f"     {feat_type:<16}{feat_start + 1}..{feat_end}")
            lines.append(f'                     /label="{feat_name}"')

    # ORIGIN + sequence
    lines.append("ORIGIN")
    seq_lower = sequence.lower()
    for i in range(0, len(seq_lower), 60):
        # Format: position (right-justified 9 chars), then 6 groups of 10 bases
        chunk = seq_lower[i:i + 60]
        groups = [chunk[j:j + 10] for j in range(0, len(chunk), 10)]
        lines.append(f"{i + 1:>9} {' '.join(groups)}")

    lines.append("//")

    return "\n".join(lines) + "\n"


def get_plasmid_plot_json(df, linear: bool = False) -> str:
    """Generate an interactive Bokeh plasmid map from a pLannotate annotation DataFrame.

    Args:
        df: DataFrame returned by plannotate.annotate.annotate()
        linear: If True, render as linear map; otherwise circular.

    Returns:
        JSON string suitable for Bokeh.embed.embed_item() in the browser.
    """
    if not _PLANNOTATE_AVAILABLE:
        raise RuntimeError(_PLANNOTATE_MISSING_MSG)
    plot = get_bokeh(df, linear=linear)
    plot.plot_width = 600
    plot.plot_height = 600
    plot.sizing_mode = "stretch_width"
    return json.dumps(json_item(plot))


def export_genbank_with_plot(
    sequence: str,
    name: str,
    backbone_name: str = "",
    insert_name: str = "",
    insert_position: int = 0,
    insert_length: int = 0,
    reverse_complement_insert: bool = False,
) -> tuple[str, str]:
    """Annotate a sequence, returning both a GenBank string and a Bokeh plot JSON.

    Runs pLannotate once and reuses the result for both outputs.

    Returns:
        (genbank_str, plot_json_str)
    """
    if not _PLANNOTATE_AVAILABLE:
        raise RuntimeError(_PLANNOTATE_MISSING_MSG)
    df = annotate(sequence, linear=False)
    record = _build_annotated_record(
        sequence, df, name, backbone_name, insert_name,
        insert_position, insert_length, reverse_complement_insert,
    )
    handle = io.StringIO()
    SeqIO.write(record, handle, "genbank")
    gbk = handle.getvalue()
    plot_json = get_plasmid_plot_json(df, linear=False)
    return gbk, plot_json


def export_construct(
    result: AssemblyResult,
    output_format: str,
    construct_name: str = "construct",
    backbone_name: str = "",
    insert_name: str = "",
    reverse_complement_insert: bool = False,
    insert_length: int = 0,
    backbone_features: Optional[list[dict]] = None,
) -> str:
    """
    Export an assembled construct in the requested format.

    Args:
        result: A successful AssemblyResult.
        output_format: One of "fasta", "genbank", "raw".
        construct_name: Name for the output record.
        backbone_name: Name of backbone (for GenBank annotation).
        insert_name: Name of insert (for GenBank annotation).
        insert_length: Length of insert (for GenBank annotation).
        backbone_features: Original backbone features (for GenBank annotation).
        reverse_complement_insert: bool = False
    Returns:
        Formatted sequence string.

    Raises:
        ValueError: If result is not successful or format is unknown.
    """
    if not result.success or not result.sequence:
        raise ValueError("Cannot export a failed assembly result")

    fmt = output_format.lower().strip()

    if fmt == "raw":
        return result.sequence

    elif fmt == "fasta":
        desc = f"{insert_name} in {backbone_name}, {result.total_size_bp} bp"
        return format_as_fasta(result.sequence, construct_name, desc)

    elif fmt in ("genbank", "gb"):
        return format_as_genbank(
            sequence=result.sequence,
            name=construct_name,
            backbone_name=backbone_name,
            insert_name=insert_name,
            insert_position=result.insert_position or 0,
            insert_length=insert_length,
            features=backbone_features,
            reverse_complement_insert=reverse_complement_insert,
        )

    else:
        raise ValueError(f"Unknown output format: {output_format!r}. Use 'raw', 'fasta', or 'genbank'.")
