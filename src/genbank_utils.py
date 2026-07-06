"""GenBank utilities — parsing, formatting, export, and annotation plots.

Single home for GenBank-related helpers:
  * Regex parsers for GenBank flat files (used by user_library.py / plasmid_intake.py).
  * BioPython-based export of annotated .gb files (part-in-vector designs).
  * Assembled-construct formatting via pLannotate (with a pLannotate-free fallback)
    and interactive Bokeh plasmid-map JSON.

pLannotate and bokeh are conda-only (not on PyPI). Their imports are guarded so
this module stays importable in pip-based environments; the annotation functions
raise a clear error when called if those packages are unavailable.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# BioPython — hard dependency for export/formatting; guarded so pure-regex
# parsing still works if it is somehow missing.
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    BIOPYTHON_AVAILABLE = True
except ImportError:
    SeqIO = None
    Seq = None
    SeqRecord = None
    SeqFeature = None
    FeatureLocation = None
    BIOPYTHON_AVAILABLE = False

# pLannotate + bokeh (conda-only) for BLAST-based annotation and plasmid maps.
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

# Optional custom annotation DB (BYOA — bring your own annotations)
try:
    from .custom_annotations import setup_custom_annotations, query_custom_db, merge_annotation_results
    _CUSTOM_ANNOTATIONS_AVAILABLE = True
except ImportError:
    try:
        from custom_annotations import setup_custom_annotations, query_custom_db, merge_annotation_results
        _CUSTOM_ANNOTATIONS_AVAILABLE = True
    except ImportError:
        _CUSTOM_ANNOTATIONS_AVAILABLE = False

if _CUSTOM_ANNOTATIONS_AVAILABLE:
    setup_custom_annotations()

# Recognised GenBank file extensions (shared by library scanners).
GENBANK_EXTENSIONS = (".gb", ".gbk", ".genbank")


def parse_genbank_sequence(content: str) -> Optional[str]:
    """Extract DNA sequence from GenBank format content (ORIGIN section)."""
    origin_match = re.search(r'ORIGIN\s*\n(.*?)(?://|\Z)', content, re.DOTALL)
    if not origin_match:
        return None
    origin_section = origin_match.group(1)
    sequence = re.sub(r'[^atcgATCGnN]', '', origin_section).upper()
    return sequence if len(sequence) > 5 else None


def parse_genbank_location(location_str: str) -> tuple[int, int]:
    """Parse a GenBank location string and return (start, end) as 0-based.

    Handles simple (100..200), complement(100..200),
    and join(100..200,201..300) formats. Returns the overall span.
    """
    inner = location_str.strip()
    if inner.startswith("complement(") and inner.endswith(")"):
        inner = inner[len("complement("):-1]
    if inner.startswith("join(") and inner.endswith(")"):
        inner = inner[len("join("):-1]

    coords: list[int] = []
    for part in inner.split(","):
        part = part.strip()
        m = re.search(r'(\d+)\.\.(\d+)', part)
        if m:
            coords.extend([int(m.group(1)), int(m.group(2))])

    if not coords:
        m = re.search(r'(\d+)', location_str)
        if m:
            pos = int(m.group(1))
            return pos - 1, pos
        return 0, 0

    return min(coords) - 1, max(coords)


def parse_genbank_features(content: str) -> tuple[List[Dict], Optional[Dict]]:
    """Parse feature annotations from GenBank content.

    Returns:
        (features, mcs_position) where features is a list of dicts
        matching the backbone library format, and mcs_position is a dict
        with 'start', 'end', 'description' keys (or None).
    """
    RELEVANT_TYPES = {
        "promoter", "CDS", "polyA_signal", "rep_origin",
        "misc_feature", "regulatory", "enhancer",
    }

    features: List[Dict] = []
    mcs_position: Optional[Dict] = None

    feat_match = re.search(
        r'^FEATURES\s+Location/Qualifiers\s*\n(.*?)(?=^ORIGIN|\Z)',
        content, re.MULTILINE | re.DOTALL,
    )
    if not feat_match:
        return features, mcs_position

    feat_text = feat_match.group(1)
    blocks = re.split(r'\n(?=     \S)', feat_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        header_match = re.match(r'(\S+)\s+([\S]+)', block)
        if not header_match:
            continue
        feat_type = header_match.group(1)
        location_str = header_match.group(2)

        if feat_type not in RELEVANT_TYPES:
            continue

        start, end = parse_genbank_location(location_str)

        label = note = gene = product = ""
        for qual_match in re.finditer(r'/(\w+)="((?:[^"\\]|"")*)"', block, re.DOTALL):
            key = qual_match.group(1)
            val = re.sub(r'\s+', ' ', qual_match.group(2).replace("\n", " ").strip())
            if key == "label":
                label = val
            elif key == "note":
                note = val
            elif key == "gene":
                gene = val
            elif key == "product":
                product = val

        name = label or gene or product or feat_type

        is_mcs = False
        if feat_type == "misc_feature":
            combined = (label + " " + note).lower()
            if "multiple cloning site" in combined or "mcs" in combined.split():
                is_mcs = True

        if is_mcs:
            mcs_position = {
                "start": start,
                "end": end,
                "description": note or f"Multiple cloning site ({label})",
            }
            features.append({"name": name, "type": "misc_feature", "start": start, "end": end})
        else:
            features.append({"name": name, "type": feat_type, "start": start, "end": end})

    return features, mcs_position


def parse_genbank_locus_name(content: str) -> Optional[str]:
    """Extract the LOCUS name (first token after LOCUS keyword)."""
    m = re.search(r'^LOCUS\s+(\S+)', content, re.MULTILINE)
    return m.group(1) if m else None


def parse_genbank(content: str) -> Optional[Dict[str, Any]]:
    """Composite parser: extract everything needed for a library entry.

    Returns a dict with keys: locus_name, sequence, size_bp, features, mcs_position.
    Returns None if no valid sequence found.
    """
    sequence = parse_genbank_sequence(content)
    if not sequence:
        return None
    features, mcs_position = parse_genbank_features(content)
    return {
        "locus_name": parse_genbank_locus_name(content),
        "sequence": sequence,
        "size_bp": len(sequence),
        "features": features,
        "mcs_position": mcs_position,
    }


def is_circular(content: str) -> bool:
    """Return True if the GenBank LOCUS line declares the topology as circular."""
    m = re.search(r'^LOCUS\s+\S.*?(circular|linear)', content, re.MULTILINE | re.IGNORECASE)
    return bool(m and m.group(1).lower() == "circular")


# ── GenBank export (annotated .gb file from a plasmid sequence) ──────────────
# Used to deliver part-in-vector designs built from vendor backbones as
# downloadable GenBank files.

def _rc(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def _safe_locus(name: str) -> str:
    """GenBank LOCUS name: ≤16 chars, alphanumeric + underscore."""
    slug = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return slug[:16]


def export_plasmid_genbank(
    plasmid_seq: str,
    name: str,
    output_path: Optional[str] = None,
    description: Optional[str] = None,
    enzyme_name: Optional[str] = None,
    fragments: Optional[list[dict]] = None,
    backbone_name: Optional[str] = None,
) -> str:
    """
    Write a plasmid sequence to an annotated GenBank (.gb) file.

    Args:
        plasmid_seq   : full circular plasmid sequence (DNA string)
        name          : plasmid name (used as LOCUS name and .gb filename)
        output_path   : directory or full .gb path; defaults to current directory
        description   : DEFINITION field text
        enzyme_name   : Type IIS enzyme — recognition sites are annotated
        fragments     : [{"name": str, "sequence": str}] annotated as misc_feature
        backbone_name : carrier backbone name (annotated as a feature)

    Returns:
        Absolute path of the written .gb file.
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("biopython is required for GenBank export: pip install biopython")

    plasmid_seq = plasmid_seq.upper()
    locus = _safe_locus(name)
    record = SeqRecord(
        Seq(plasmid_seq),
        id=locus,
        name=locus,
        description=description or f"{name} — designed by claude-plasmids",
        annotations={
            "molecule_type": "DNA",
            "topology": "circular",
            "date": datetime.now().strftime("%d-%b-%Y").upper(),
        },
    )

    # Annotate each insert fragment
    if fragments:
        for frag in fragments:
            frag_seq = frag.get("sequence", "").upper()
            frag_name = frag.get("name", "insert")
            if frag_seq and frag_seq in plasmid_seq:
                start = plasmid_seq.index(frag_seq)
                record.features.append(SeqFeature(
                    FeatureLocation(start, start + len(frag_seq), strand=1),
                    type="misc_feature",
                    qualifiers={"label": [frag_name]},
                ))

    # Annotate enzyme recognition sites
    if enzyme_name:
        try:
            from .assembler import GG_ENZYMES
        except ImportError:
            from assembler import GG_ENZYMES
        if enzyme_name in GG_ENZYMES:
            rec_site = GG_ENZYMES[enzyme_name]["recognition"]
            for site, strand in [(rec_site, 1), (_rc(rec_site), -1)]:
                pos = 0
                while True:
                    idx = plasmid_seq.find(site, pos)
                    if idx == -1:
                        break
                    record.features.append(SeqFeature(
                        FeatureLocation(idx, idx + len(site), strand=strand),
                        type="misc_feature",
                        qualifiers={"label": [f"{enzyme_name} site"]},
                    ))
                    pos = idx + 1

    # Annotate backbone region (everything outside the insert cassette)
    if backbone_name and fragments:
        # Backbone = sequence not covered by any fragment — annotate as a simple feature at pos 0
        record.features.insert(0, SeqFeature(
            FeatureLocation(0, len(plasmid_seq), strand=1),
            type="rep_origin",
            qualifiers={"label": [backbone_name]},
        ))

    # Resolve output path
    if output_path is None:
        import os
        output_path = os.getcwd()
    out = Path(output_path)
    if out.is_dir():
        safe_name = re.sub(r"[^\w.-]", "_", name)
        out = out / f"{safe_name}.gb"

    with open(out, "w") as f:
        SeqIO.write(record, f, "genbank")

    return str(out.resolve())


# ── Assembled-construct formatting + annotation plots ────────────────────────

def _format_provenance_comment(provenance: list[dict]) -> str:
    """Format a list of part provenance dicts into a GenBank COMMENT string."""
    lines = ["Assembly provenance:"]
    insert_idx = 0
    for p in provenance:
        part_type = p.get("part_type", "insert")
        name = p.get("part_name", "unknown")
        if part_type == "backbone":
            label = f"  backbone: {name}"
        else:
            insert_idx += 1
            label = f"  insert {insert_idx}: {name}"

        src = p.get("source_system") or "local"
        fields = [label, src]

        addgene_id = p.get("addgene_id")
        accession = p.get("genbank_accession")
        doi = p.get("source_doi")
        url = p.get("source_url")

        if addgene_id and "Addgene" in src:
            fields.append(f"catalog: #{addgene_id}")
        if accession and "NCBI" in src:
            fields.append(f"accession: {accession}")
        if doi:
            fields.append(f"DOI: {doi}")
        if url:
            fields.append(url)

        lines.append(" | ".join(fields))
    return "\n".join(lines)


def _build_annotated_record(
    sequence: str,
    df,
    name: str,
    backbone_name: str,
    insert_name: str,
    insert_position: int,
    insert_length: int,
    reverse_complement_insert: bool,
    linear: bool = False,
    provenance: Optional[list[dict]] = None,
):
    """Build a BioPython SeqRecord from a pLannotate df, adding the insert feature if needed."""
    if not _PLANNOTATE_AVAILABLE:
        raise RuntimeError(_PLANNOTATE_MISSING_MSG)
    record = get_seq_record(df, sequence, is_linear=linear)
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "linear" if linear else "circular"

    locus_name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)[:16]
    record.name = locus_name
    record.id = locus_name
    record.description = f"{insert_name} in {backbone_name}" if backbone_name else name

    if provenance:
        record.annotations["comment"] = _format_provenance_comment(provenance)

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
    linear: bool = False,
    provenance: Optional[list[dict]] = None,
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
            insert_length=insert_length, features=features, linear=linear,
            provenance=provenance,
        )

    df = annotate(sequence, linear=linear)
    if _CUSTOM_ANNOTATIONS_AVAILABLE:
        custom_df = query_custom_db(sequence)
        if custom_df is not None:
            df = merge_annotation_results(df, custom_df)
    record = _build_annotated_record(
        sequence, df, name, backbone_name, insert_name,
        insert_position, insert_length, reverse_complement_insert, linear=linear,
        provenance=provenance,
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
    linear: bool = False,
    provenance: Optional[list[dict]] = None,
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
        f"LOCUS       {locus_name:<16} {total_len:>5} bp    DNA     {'linear  ' if linear else 'circular'}   UNK"
    )

    # DEFINITION
    lines.append(f"DEFINITION  Expression construct: {insert_name} in {backbone_name}.")

    # COMMENT — provenance block (GenBank spec: 12-space indent for continuation lines)
    if provenance:
        comment_text = _format_provenance_comment(provenance)
        comment_lines = comment_text.splitlines()
        lines.append(f"COMMENT     {comment_lines[0]}")
        for cl in comment_lines[1:]:
            lines.append(f"            {cl}")

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
    df = df.copy()
    if "fragment" in df.columns:
        df["fragment"] = df["fragment"].astype(bool)
    # Drop rows that would produce NaN/inf in pLannotate's rstart/rend calculation
    # (qlen=0 → 0/0=NaN; any NaN in position cols → propagates through trig).
    _pos_cols = [c for c in ("qstart", "qend", "qlen") if c in df.columns]
    if _pos_cols:
        df = df.dropna(subset=_pos_cols)
        if "qlen" in df.columns:
            df = df[df["qlen"] > 0]
        # Verify rstart/rend will be finite for all remaining rows.
        from math import pi as _pi
        import numpy as _np
        _rs = (df["qstart"] / df["qlen"]) * 2 * _pi
        _re = (df["qend"] / df["qlen"]) * 2 * _pi
        df = df[_np.isfinite(_rs) & _np.isfinite(_re)]
    plot = get_bokeh(df, linear=linear)
    plot.plot_width = 600
    plot.plot_height = 200 if linear else 600
    plot.sizing_mode = "stretch_width"
    return json.dumps({"plot": json_item(plot), "linear": linear})


def export_genbank_with_plot(
    sequence: str,
    name: str,
    backbone_name: str = "",
    insert_name: str = "",
    insert_position: int = 0,
    insert_length: int = 0,
    reverse_complement_insert: bool = False,
    linear: bool = False,
    provenance: Optional[list[dict]] = None,
) -> tuple[str, str]:
    """Annotate a sequence, returning both a GenBank string and a Bokeh plot JSON.

    Runs pLannotate once and reuses the result for both outputs.

    Returns:
        (genbank_str, plot_json_str)
    """
    if not _PLANNOTATE_AVAILABLE:
        raise RuntimeError(_PLANNOTATE_MISSING_MSG)
    df = annotate(sequence, linear=linear)
    if _CUSTOM_ANNOTATIONS_AVAILABLE:
        custom_df = query_custom_db(sequence)
        if custom_df is not None:
            df = merge_annotation_results(df, custom_df)

    record = _build_annotated_record(
        sequence, df, name, backbone_name, insert_name,
        insert_position, insert_length, reverse_complement_insert, linear=linear,
        provenance=provenance,
    )
    handle = io.StringIO()
    SeqIO.write(record, handle, "genbank")
    gbk = handle.getvalue()
    plot_json = get_plasmid_plot_json(df, linear=linear)
    return gbk, plot_json
