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
    # Validation details
    backbone_preserved: bool = False
    insert_preserved: bool = False
    insert_has_start_codon: bool = False
    insert_has_stop_codon: bool = False
    insert_length_valid: bool = False


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


def assemble_construct(
    backbone_seq: str,
    insert_seq: str,
    insertion_position: int,
    replace_region_end: Optional[int] = None,
    reverse_complement_insert: bool = False,
) -> AssemblyResult:
    """
    Assemble an expression construct by inserting a sequence into a backbone.

    The default mode inserts at a single position (no backbone sequence removed).
    If replace_region_end is provided, the backbone region from insertion_position
    to replace_region_end is replaced by the insert.

    Args:
        backbone_seq: Complete backbone DNA sequence.
        insert_seq: Insert DNA sequence (e.g., EGFP CDS).
        insertion_position: 0-based position in backbone to insert at.
        replace_region_end: If set, backbone[insertion_position:replace_region_end]
                           is replaced by the insert. Use this when replacing the
                           full MCS or an existing insert.
        reverse_complement_insert: If True, reverse-complement the insert before
                                   insertion (for reverse-orientation backbones).

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

    # Check insert biology
    result.insert_has_start_codon = insert_seq[:3] == "ATG"
    result.insert_has_stop_codon = insert_seq[-3:] in ("TAA", "TAG", "TGA")
    result.insert_length_valid = len(insert_seq) % 3 == 0

    if not result.insert_has_start_codon:
        result.warnings.append("Insert does not start with ATG (start codon)")
    if not result.insert_has_stop_codon:
        result.warnings.append("Insert does not end with a stop codon (TAA/TAG/TGA)")
    if not result.insert_length_valid:
        result.warnings.append(
            f"Insert length ({len(insert_seq)} bp) is not a multiple of 3 — "
            f"may be out of reading frame"
        )

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
    - Middle sequences: remove stop codon
    - Last sequence: keep stop codon
    - Linker DNA + Kozak (GCCACC) inserted between each junction by default

    The default linker is (GGGGS)x4 for protein-protein fusions. Pass
    linker="" for direct concatenation (e.g., epitope tag fusions).

    Args:
        sequences: List of dicts, each with:
            - sequence: DNA sequence (required)
            - name: Name of the sequence (optional)
            - position: 'n_terminal', 'c_terminal', or 'middle' (optional,
              auto-determined from order if not specified)
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

    parts = []
    for i, seq_dict in enumerate(sequences):
        seq = clean_sequence(seq_dict["sequence"])
        valid, errors = validate_dna(seq)
        if not valid:
            name = seq_dict.get("name", f"sequence_{i}")
            raise ValueError(f"Invalid DNA in {name}: {'; '.join(errors)}")

        is_last = (i == len(sequences) - 1)

        # Remove stop codon from all but the last sequence
        if not is_last:
            if seq[-3:] in ("TAA", "TAG", "TGA"):
                seq = seq[:-3]

        parts.append(seq)

    # Join with optional linker
    if linker:
        linker = clean_sequence(linker)
        # Add kozak sequence (GCCACC) if linker is present and we're fusing two proteins
        linker = linker + "GCCACC"
        valid, errors = validate_dna(linker)
        if not valid:
            raise ValueError(f"Invalid linker DNA: {'; '.join(errors)}")
        return linker.join(parts)
    else:
        return "".join(parts)


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


def format_as_genbank(
    sequence: str,
    name: str,
    backbone_name: str = "",
    insert_name: str = "",
    insert_position: int = 0,
    insert_length: int = 0,
    features: Optional[list[dict]] = None,
) -> str:
    """
    Format an assembled construct as a minimal GenBank flat file.

    Args:
        sequence: The assembled DNA sequence.
        name: Locus name for the GenBank record.
        backbone_name: Name of the backbone used.
        insert_name: Name of the insert used.
        insert_position: 0-based start position of the insert.
        insert_length: Length of the insert in bp.
        features: Optional list of feature dicts with name, start, end, type keys.

    Returns:
        GenBank-formatted string.
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
    lines.append(f'                     /mol_type="other DNA"')
    lines.append(f'                     /note="Assembled construct"')

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


def export_construct(
    result: AssemblyResult,
    output_format: str,
    construct_name: str = "construct",
    backbone_name: str = "",
    insert_name: str = "",
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
        )

    else:
        raise ValueError(f"Unknown output format: {output_format!r}. Use 'raw', 'fasta', or 'genbank'.")
