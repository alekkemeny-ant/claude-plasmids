"""GenBank file parsing utilities.

Module-level functions for parsing GenBank flat files.
Used by user_library.py (BYOL support) and any other module
that needs to read GenBank content.
"""

import re
from typing import Optional, Dict, List, Any


def parse_genbank_sequence(content: str) -> Optional[str]:
    """Extract DNA sequence from GenBank format content (ORIGIN section)."""
    origin_match = re.search(r'ORIGIN\s*\n(.*?)(?://|\Z)', content, re.DOTALL)
    if not origin_match:
        return None
    origin_section = origin_match.group(1)
    sequence = re.sub(r'[^atcgATCGnN]', '', origin_section).upper()
    return sequence if len(sequence) > 100 else None


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
