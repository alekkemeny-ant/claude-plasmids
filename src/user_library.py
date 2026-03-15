#!/usr/bin/env python3
"""
User Library — Bring Your Own Library (BYOL)

Loads user-provided GenBank files from $PLASMID_USER_LIBRARY and converts
them to the same dict format as the built-in library, with a `user:` ID
prefix to namespace them away from built-in and Addgene-cached entries.

Directory layout expected:
    $PLASMID_USER_LIBRARY/
        backbones/*.gb   (or .gbk, .genbank)
        inserts/*.gb

Entries are read-only from our side; the built-in library cache-write
path in library.py never touches user entries (enforced by the `user:`
prefix — Addgene fallback won't match, and cache-write re-reads built-in
from disk before appending).
"""

import logging
import os
from pathlib import Path
from typing import Any

# Support both package import (`from .addgene_integration import ...`) and
# bare import (`from addgene_integration import ...`) — app/app.py uses the
# latter via sys.path manipulation.
try:
    from .addgene_integration import parse_genbank
except ImportError:
    from addgene_integration import parse_genbank

logger = logging.getLogger(__name__)

USER_PREFIX = "user:"
GENBANK_EXTENSIONS = (".gb", ".gbk", ".genbank")


def _user_library_dir() -> Path | None:
    """Return the user library directory if configured and exists, else None."""
    raw = os.environ.get("PLASMID_USER_LIBRARY")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_dir():
        logger.warning(f"PLASMID_USER_LIBRARY={raw} is not a directory; ignoring")
        return None
    return path


def _parse_file_to_entry(path: Path) -> dict[str, Any] | None:
    """Parse a single GenBank file into a library entry dict.

    ID is `user:<LOCUS name>` with filename stem as fallback.
    """
    try:
        content = path.read_text(errors="replace")
    except OSError as e:
        logger.warning(f"Cannot read {path}: {e}")
        return None

    parsed = parse_genbank(content)
    if not parsed:
        logger.warning(f"No valid sequence in {path}; skipping")
        return None

    base_id = parsed["locus_name"] or path.stem
    entry_id = f"{USER_PREFIX}{base_id}"

    entry: dict[str, Any] = {
        "id": entry_id,
        "name": base_id,
        "aliases": [f"{USER_PREFIX}{path.stem}"] if path.stem != base_id else [],
        "description": f"User-provided GenBank file: {path.name}",
        "size_bp": parsed["size_bp"],
        "source": "user_library",
        "sequence": parsed["sequence"],
        "features": parsed["features"],
    }
    if parsed["mcs_position"]:
        entry["mcs_position"] = parsed["mcs_position"]
    return entry


def _scan_subdir(subdir_name: str) -> list[dict[str, Any]]:
    """Scan $PLASMID_USER_LIBRARY/<subdir_name>/ for GenBank files."""
    root = _user_library_dir()
    if not root:
        return []
    subdir = root / subdir_name
    if not subdir.is_dir():
        return []

    entries: list[dict[str, Any]] = []
    for path in sorted(subdir.iterdir()):
        if path.suffix.lower() not in GENBANK_EXTENSIONS:
            continue
        entry = _parse_file_to_entry(path)
        if entry:
            entries.append(entry)

    if entries:
        logger.info(f"Loaded {len(entries)} user {subdir_name} from {subdir}")
    return entries


def load_user_backbones() -> list[dict[str, Any]]:
    """Load all user backbones from $PLASMID_USER_LIBRARY/backbones/."""
    return _scan_subdir("backbones")


def load_user_inserts() -> list[dict[str, Any]]:
    """Load all user inserts from $PLASMID_USER_LIBRARY/inserts/."""
    return _scan_subdir("inserts")
