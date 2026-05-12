#!/usr/bin/env python3
"""
Plasmid file intake — parse uploaded GenBank/FASTA files, run plannotate,
and produce a structured chat message for the agent intake workflow.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional


# ── File parsing ───────────────────────────────────────────────────────────────

def parse_upload(content: str, filename: str) -> dict:
    """
    Parse a GenBank or FASTA file uploaded by the user.

    Returns a dict with keys:
        filename, format, locus_name, sequence, size_bp, topology,
        features (existing GenBank annotations), mcs_position
    """
    content = content.strip()

    if re.match(r'^LOCUS', content, re.IGNORECASE):
        return _parse_genbank(content, filename)

    if content.startswith(">"):
        return _parse_fasta(content, filename)

    # Last resort: treat as raw DNA
    raw = re.sub(r'[^ACGTNacgtn\s]', '', content)
    seq = re.sub(r'\s+', '', raw).upper()
    if len(seq) >= 10:
        return {
            "filename": filename,
            "format": "raw",
            "locus_name": Path(filename).stem,
            "sequence": seq,
            "size_bp": len(seq),
            "topology": "unknown",
            "features": [],
            "mcs_position": None,
        }

    raise ValueError(
        f"Unrecognized file format for '{filename}'. "
        "Supported formats: GenBank (.gb, .gbk, .genbank), FASTA (.fasta, .fa), or raw DNA."
    )


def _parse_genbank(content: str, filename: str) -> dict:
    try:
        from .genbank_utils import parse_genbank
    except ImportError:
        from genbank_utils import parse_genbank

    parsed = parse_genbank(content)
    if not parsed:
        raise ValueError("Could not parse GenBank file — no valid sequence found.")

    m = re.search(r'^LOCUS\s+\S+.+?(circular|linear)', content, re.MULTILINE | re.IGNORECASE)
    topology = m.group(1).lower() if m else "unknown"

    return {
        "filename": filename,
        "format": "genbank",
        "locus_name": parsed.get("locus_name") or Path(filename).stem,
        "sequence": parsed["sequence"],
        "size_bp": parsed["size_bp"],
        "topology": topology,
        "features": parsed.get("features", []),
        "mcs_position": parsed.get("mcs_position"),
    }


def _parse_fasta(content: str, filename: str) -> dict:
    lines = content.splitlines()
    header = lines[0][1:].strip() if lines else ""
    seq = "".join(l.strip() for l in lines[1:] if l.strip() and not l.startswith(">"))
    seq = re.sub(r'[^ACGTNacgtn]', '', seq).upper()
    if not seq:
        raise ValueError("Could not parse FASTA file — no sequence data found.")
    return {
        "filename": filename,
        "format": "fasta",
        "locus_name": header.split()[0] if header else Path(filename).stem,
        "sequence": seq,
        "size_bp": len(seq),
        "topology": "unknown",
        "features": [],
        "mcs_position": None,
    }


# ── Placeholder region detection ──────────────────────────────────────────────

_PLACEHOLDER_KEYWORDS = [
    "your gene", "your insert", "insert here", "gene of interest",
    "cds of interest", "insert site", "stuffer", "gap", "cloning site",
    "multiple cloning site", "insert sequence",
]


def find_placeholder_region(
    sequence: str,
    features: list[dict],
    min_n_run: int = 10,
) -> Optional[dict]:
    """
    Locate the placeholder region where an insert should replace backbone sequence.

    Checks in priority order:
      1. Feature annotations with gap-like names/types (most reliable)
      2. Runs of ≥min_n_run consecutive N characters (vendor convention)

    Returns {"start": int, "end": int, "type": str, "description": str} or None.
    start is inclusive, end is exclusive (Python slice semantics).
    """
    # Priority 1: annotated features
    for feat in features:
        name = (feat.get("name") or "").lower()
        ftype = (feat.get("type") or "").lower()
        desc = (feat.get("description") or "").lower()

        is_placeholder = ftype == "gap" or any(
            kw in name or kw in desc for kw in _PLACEHOLDER_KEYWORDS
        )
        if is_placeholder:
            start = feat.get("start", 0)
            end = feat.get("end", 0)
            if end > start:
                return {
                    "start": start,
                    "end": end,
                    "type": "annotation",
                    "description": feat.get("name") or "annotated placeholder",
                }

    # Priority 2: longest N-run
    seq_upper = sequence.upper()
    best: Optional[dict] = None
    best_len = min_n_run - 1  # must beat this to qualify

    i = 0
    while i < len(seq_upper):
        if seq_upper[i] == "N":
            j = i + 1
            while j < len(seq_upper) and seq_upper[j] == "N":
                j += 1
            run_len = j - i
            if run_len > best_len:
                best_len = run_len
                best = {
                    "start": i,
                    "end": j,
                    "type": "n_run",
                    "description": f"{run_len} consecutive N placeholder",
                }
            i = j
        else:
            i += 1

    return best


# ── plannotate ─────────────────────────────────────────────────────────────────

def run_plannotate(sequence: str, min_pident: float = 90.0) -> list[dict]:
    """
    Run plannotate on a sequence. Returns a list of feature dicts.
    Returns an empty list if plannotate is unavailable or finds nothing.

    Each feature: {name, description, type, start, end, strand, match_pct}
    """
    try:
        from plannotate.annotate import annotate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = annotate(sequence, linear=False)

        if df is None or df.empty:
            return []

        features: list[dict] = []
        seen: set[str] = set()
        for _, row in df.iterrows():
            name = str(row.get("Feature", "")).strip()
            if not name:
                continue
            pident = float(row.get("percmatch", 0))
            if pident < min_pident:
                continue
            # Deduplicate by name — keep highest match
            if name in seen:
                continue
            seen.add(name)
            features.append({
                "name": name,
                "description": str(row.get("Description", "")).strip(),
                "type": str(row.get("Type", "misc_feature")).strip(),
                "start": int(row.get("qstart", 0)),
                "end": int(row.get("qend", 0)),
                "strand": int(row.get("sframe", 1)),
                "match_pct": round(pident, 1),
            })
        return features

    except Exception:
        return []


# ── Plasmid type inference ─────────────────────────────────────────────────────

def infer_plasmid_type(
    plannotate_features: list[dict],
    existing_features: list[dict],
) -> str:
    """
    Infer the functional category of the plasmid from its features.
    Returns: "backbone" | "part_in_vector" | "expression_plasmid" | "unknown"
    """
    all_features = plannotate_features + existing_features
    all_names = [f.get("name", "").lower() for f in all_features]
    all_types = [f.get("type", "").lower() for f in all_features]

    gg_enzymes = ["bsai", "bbsi", "paqci", "esp3i", "bsmbi", "type iis", "ggtctc", "gaagac", "cacctgc", "cgtctc"]
    has_gg = any(any(e in n for e in gg_enzymes) for n in all_names)

    # CDS sequences intrinsic to backbone vectors (not payload inserts)
    _backbone_cds = [
        "ampr", "kanr", "specr", "hygr", "puror", "chlorr", "bla", "resistance", "npt",  # antibiotic resistance
        "lacz", "laczα", "lacza", "ccdb", "cat", "rop",  # backbone selection/screening markers
    ]
    has_resistance = any(any(r in n for r in _backbone_cds) for n in all_names)
    has_origin = any(any(o in n for o in ["ori", "origin", "cole1", "puc", "f1 ori"]) for n in all_names)
    has_promoter = any("promoter" in t for t in all_types) or any("promoter" in n for n in all_names)
    has_polya = any(any(p in n for p in ["polya", "poly_a", "bgh", "sv40 pa", "terminator"]) for n in all_names)

    # Payload CDS = a coding sequence that is NOT a backbone-intrinsic gene
    has_payload_cds = any(
        "cds" in all_types[i]
        and not any(r in all_names[i] for r in _backbone_cds)
        for i in range(len(all_features))
    )

    if has_gg:
        return "part_in_vector"
    if has_promoter and has_payload_cds and has_polya:
        return "expression_plasmid"
    if (has_resistance or has_origin) and not has_payload_cds:
        return "backbone"
    return "unknown"


# ── Message builder ────────────────────────────────────────────────────────────

_TYPE_LABELS = {
    "backbone": "backbone vector (cloning chassis)",
    "part_in_vector": "part-in-vector (Golden Gate ready insert in carrier)",
    "expression_plasmid": "complete expression plasmid",
    "unknown": "unknown / novel",
}


def build_intake_message(
    filename: str,
    parsed: dict,
    plannotate_features: list[dict],
    placeholder: Optional[dict] = None,
) -> str:
    """
    Build the chat message that is auto-sent when a plasmid file is uploaded.
    Includes sequence, plannotate results, and a request for intake.
    """
    inferred = infer_plasmid_type(plannotate_features, parsed.get("features", []))

    # Auto-detect placeholder if not provided
    if placeholder is None:
        placeholder = find_placeholder_region(
            parsed.get("sequence", ""),
            plannotate_features + parsed.get("features", []),
        )

    lines = [
        f"I've uploaded a plasmid file: **{filename}**",
        "",
        f"- **Size:** {parsed['size_bp']:,} bp ({parsed['topology']})",
        f"- **Inferred type:** {_TYPE_LABELS.get(inferred, inferred)}",
    ]

    if placeholder:
        lines.append(
            f"- **Insertion site detected:** {placeholder['description']} "
            f"(pos {placeholder['start']}–{placeholder['end']}, "
            f"type: {placeholder['type']}) — "
            "the insert will replace this region automatically, no manual position needed."
        )

    existing = parsed.get("features", [])
    if existing:
        names = ", ".join(f["name"] for f in existing[:8])
        suffix = f" (+ {len(existing) - 8} more)" if len(existing) > 8 else ""
        lines.append(f"- **Existing annotations:** {names}{suffix}")

    if plannotate_features:
        lines.append(f"- **plannotate identified {len(plannotate_features)} feature(s):**")
        for f in plannotate_features[:12]:
            strand = "+" if f["strand"] >= 0 else "−"
            lines.append(
                f"    • {f['name']} ({f['type']}, pos {f['start']}–{f['end']}, "
                f"strand {strand}, {f['match_pct']}% match)"
            )
        if len(plannotate_features) > 12:
            lines.append(f"    • … and {len(plannotate_features) - 12} more")
    else:
        lines.append("- **plannotate:** No standard features recognized.")

    lines += [
        "",
        f"Full sequence ({parsed['size_bp']} bp):",
        parsed["sequence"],
        "",
        "Please help me add this to my library.",
    ]

    return "\n".join(lines)
