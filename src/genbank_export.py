#!/usr/bin/env python3
"""
GenBank export — produce an annotated .gb file from a plasmid sequence.

Used to deliver part-in-vector designs built from vendor backbones as
downloadable GenBank files.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


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
