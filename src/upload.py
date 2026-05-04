"""File upload parsing and BYOL library saving.

Parses GenBank and FASTA files via Biopython, extracts metadata, and
optionally saves to the user's local library directory with path-traversal
hardening.
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

from Bio import SeqIO


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
VALID_KINDS = frozenset({"backbones", "inserts", "annotations"})


def safe_filename(name: str) -> str:
    """Sanitize a filename to ``[A-Za-z0-9._-]+``, capped at 100 chars."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    sanitized = sanitized.strip("._-") or "unnamed"
    return sanitized[:100]


def parse_sequence_file(data: bytes, hint_filename: str = "") -> dict[str, Any]:
    """Parse GenBank or FASTA from raw bytes. Returns metadata dict.

    Raises ``ValueError`` on unparseable input.
    """
    text = data.decode("utf-8", errors="replace")

    fmt = _detect_format(text)
    if fmt is None:
        raise ValueError(
            "Unrecognized file format. Expected GenBank (starts with LOCUS) "
            "or FASTA (starts with >)."
        )

    try:
        record = SeqIO.read(io.StringIO(text), fmt)
    except Exception as e:
        raise ValueError(f"Failed to parse {fmt} file: {e}") from e

    features = []
    for f in record.features:
        features.append({
            "type": f.type,
            "location": str(f.location),
            "qualifiers": {k: v for k, v in f.qualifiers.items()},
        })

    result: dict[str, Any] = {
        "name": record.name or Path(hint_filename).stem or "unnamed",
        "description": record.description or "",
        "sequence": str(record.seq),
        "length": len(record.seq),
        "format": fmt,
        "features": features,
        "organism": record.annotations.get("organism", ""),
        "accession": record.id if record.id != record.name else "",
        "topology": record.annotations.get("topology", ""),
        "molecule_type": record.annotations.get("molecule_type", ""),
    }
    return result


def get_library_status() -> dict[str, Any]:
    """Return the current state of $PLASMID_USER_LIBRARY."""
    raw = os.environ.get("PLASMID_USER_LIBRARY")
    if not raw:
        return {
            "configured": False,
            "path": None,
            "exists": False,
            "writable": False,
            "subdirs": {},
            "setup_hint": (
                "Set PLASMID_USER_LIBRARY to a directory path, e.g.:\n"
                "  export PLASMID_USER_LIBRARY=~/plasmid-library\n"
                "Then create subdirectories: backbones/, inserts/, annotations/"
            ),
        }
    path = Path(raw).expanduser()
    exists = path.is_dir()
    writable = exists and os.access(path, os.W_OK)
    subdirs = {}
    for kind in VALID_KINDS:
        sub = path / kind
        subdirs[kind] = {"exists": sub.is_dir(), "file_count": len(list(sub.glob("*.gb*"))) if sub.is_dir() else 0}
    return {
        "configured": True,
        "path": str(path),
        "exists": exists,
        "writable": writable,
        "subdirs": subdirs,
    }


def save_to_library(
    parsed: dict[str, Any],
    kind: str,
    raw: bytes,
) -> dict[str, str]:
    """Write a file to the user library with path-traversal protection.

    Returns ``{"saved_path": ..., "id": "user:<stem>"}``.
    Raises ``ValueError`` on invalid kind or path-traversal attempt.
    Raises ``EnvironmentError`` if the library is not configured/writable.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind {kind!r}; must be one of {sorted(VALID_KINDS)}")

    lib_dir = os.environ.get("PLASMID_USER_LIBRARY")
    if not lib_dir:
        raise EnvironmentError(
            "PLASMID_USER_LIBRARY is not set. "
            "Export it to a directory path to enable the user library."
        )

    base = Path(lib_dir).expanduser().resolve()
    if not base.is_dir():
        raise EnvironmentError(f"PLASMID_USER_LIBRARY={lib_dir} is not a directory")

    target_dir = base / kind
    target_dir.mkdir(parents=False, exist_ok=True)

    stem = safe_filename(parsed.get("name", "unnamed"))
    ext = ".gb" if parsed.get("format") == "genbank" else ".fasta"
    filename = stem + ext

    target_path = (target_dir / filename).resolve()
    if not target_path.is_relative_to(base):
        raise ValueError("Path traversal detected; refusing to write outside library directory")

    target_path.write_bytes(raw)
    return {
        "saved_path": str(target_path),
        "id": f"user:{stem}",
    }


def _detect_format(text: str) -> str | None:
    """Sniff GenBank vs FASTA from content (not extension)."""
    stripped = text.lstrip()
    if stripped.startswith("LOCUS"):
        return "genbank"
    if stripped.startswith(">"):
        return "fasta"
    return None
