#!/usr/bin/env python3
"""
Vendor Backbone Library — persist backbone sequences supplied by synthesis
companies (Ansa, Twist, etc.) so they can be used as carrier vectors.

Entries are stored in library/vendor_backbones.json with a `vendor:` ID prefix
and are automatically picked up by get_backbone_by_id() in library.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

VENDOR_BACKBONES_PATH = Path(__file__).parent.parent / "library" / "vendor_backbones.json"


def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug.strip("-")


def _load_raw() -> dict:
    if VENDOR_BACKBONES_PATH.exists():
        with open(VENDOR_BACKBONES_PATH) as f:
            return json.load(f)
    return {"backbones": []}


def _save_raw(data: dict) -> None:
    VENDOR_BACKBONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VENDOR_BACKBONES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_vendor_backbones() -> list[dict]:
    """Return all saved vendor backbone entries."""
    return _load_raw()["backbones"]


def save_vendor_backbone(
    name: str,
    sequence: str,
    company: Optional[str] = None,
    description: Optional[str] = None,
    enzyme_name: Optional[str] = None,
) -> dict:
    """
    Save a vendor-supplied backbone sequence to the local library.

    If an entry with the same ID already exists it is updated in place.
    Returns the saved entry (including its `vendor:` ID).
    """
    try:
        from .assembler import clean_sequence
    except ImportError:
        from assembler import clean_sequence

    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        raise ValueError("Backbone sequence is empty or contains invalid characters.")

    company_slug = _slugify(company) if company else "vendor"
    name_slug = _slugify(name)
    backbone_id = f"vendor:{company_slug}-{name_slug}"

    # Auto-detect placeholder region (N-runs or gap annotations)
    try:
        from .plasmid_intake import find_placeholder_region
    except ImportError:
        from plasmid_intake import find_placeholder_region

    placeholder = find_placeholder_region(clean_seq, [])

    entry: dict = {
        "id": backbone_id,
        "aliases": [name_slug],
        "name": name,
        "company": company or "Unknown vendor",
        "description": description or f"{name} backbone from {company or 'synthesis vendor'}.",
        "size_bp": len(clean_seq),
        "source": "vendor",
        "sequence": clean_seq,
        "mcs_position": {"start": placeholder["start"]} if placeholder else None,
        "placeholder_region": placeholder,
    }
    if enzyme_name:
        entry["assembly_enzyme"] = enzyme_name

    data = _load_raw()
    # Update existing entry or append
    for i, b in enumerate(data["backbones"]):
        if b["id"] == backbone_id:
            data["backbones"][i] = entry
            _save_raw(data)
            return entry

    data["backbones"].append(entry)
    _save_raw(data)
    return entry


def update_vendor_backbone_mcs(backbone_id: str, insertion_point: int) -> dict:
    """
    Persist the MCS insertion point for a vendor backbone.

    Sets mcs_position = {"start": insertion_point} so that find_mcs_insertion_point()
    can return it on subsequent calls. The update is written back to vendor_backbones.json.

    Raises ValueError if backbone_id is not found.
    """
    data = _load_raw()
    for b in data["backbones"]:
        if b["id"] == backbone_id:
            b["mcs_position"] = {"start": insertion_point}
            # A manually confirmed position overrides any auto-detected N-run placeholder
            b["placeholder_region"] = None
            _save_raw(data)
            return b
    raise ValueError(f"Vendor backbone {backbone_id!r} not found in local library.")


def get_vendor_backbone_by_id(backbone_id: str) -> Optional[dict]:
    """Look up a vendor backbone by its full ID, name slug, or alias."""
    normalized = backbone_id.lower().replace("vendor:", "")
    for b in load_vendor_backbones():
        if b["id"] == backbone_id:
            return b
        if _slugify(b.get("name", "")) == normalized:
            return b
        if any(a == normalized for a in b.get("aliases", [])):
            return b
    return None
