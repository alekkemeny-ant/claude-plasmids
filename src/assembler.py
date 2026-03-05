#!/usr/bin/env python3
"""
Sequence Assembly Engine for Expression Plasmid Design

Deterministic sequence assembly: splices an insert into a backbone at a
specified position. No LLM involvement — all operations are string-based
on verified sequences.
"""

import re
from dataclasses import dataclass, field
import io
from typing import Optional
from Bio import SeqIO                                                                                                     
from Bio.SeqFeature import SeqFeature, FeatureLocation
from bokeh.embed import json_item
import json
from plannotate.annotate import annotate
from plannotate.bokeh_plot import get_bokeh                                                                        
from plannotate.resources import get_seq_record                                                                           
   
    
    
                                                                                                                     


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
        # Add Kozak sequence (GCCACC) after the linker
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
    """Format an assembled construct as a GenBank flat file using pLannotate for annotation."""
    df = annotate(sequence, linear=False)
    record = _build_annotated_record(
        sequence, df, name, backbone_name, insert_name,
        insert_position, insert_length, reverse_complement_insert,
    )
    handle = io.StringIO()
    SeqIO.write(record, handle, "genbank")
    return handle.getvalue()


def get_plasmid_plot_json(df, linear: bool = False) -> str:
    """Generate an interactive Bokeh plasmid map from a pLannotate annotation DataFrame.

    Args:
        df: DataFrame returned by plannotate.annotate.annotate()
        linear: If True, render as linear map; otherwise circular.

    Returns:
        JSON string suitable for Bokeh.embed.embed_item() in the browser.
    """
    
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
