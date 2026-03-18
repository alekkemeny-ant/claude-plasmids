#!/usr/bin/env python3
"""
User Library — Bring Your Own Library (BYOL)

Loads user-provided GenBank files from $PLASMID_USER_LIBRARY and converts
them to the same dict format as the built-in library, with a `user:` ID
prefix to namespace them away from built-in and Addgene-cached entries.

Directory layout expected:
    $PLASMID_USER_LIBRARY/
        backbones/*.gb              (or .gbk, .genbank)
        inserts/*.gb
        backbones_description.csv   (optional metadata overlay)
        inserts_description.csv     (optional metadata overlay)

CSV formats
-----------
inserts_description.csv columns (tab- or comma-separated):
    id, Description, TypeIIS cutsite, Overhang L, Overhang R,
    Size, Selection, Category

backbones_description.csv columns (tab- or comma-separated):
    ID, E coli strain, Antibiotic resistance, Neg selection marker,
    Assembly enzyme, Overhang pair 1, Next step enzyme, Overhang pair 2,
    Downstream, Mammalian selection marker

The CSV `id`/`ID` column is matched against the GenBank LOCUS name (without
the `user:` prefix).  Rows with no matching .gb file are logged and ignored.
.gb files with no CSV row load with inferred-only metadata.

Entries are read-only from our side; the built-in library cache-write
path in library.py never touches user entries (enforced by the `user:`
prefix — Addgene fallback won't match, and cache-write re-reads built-in
from disk before appending).
"""

import csv
import logging
import os
import re
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


def _is_circular(content: str) -> bool:
    """Return True if the GenBank LOCUS line declares the topology as circular."""
    m = re.search(r'^LOCUS\s+\S.*?(circular|linear)', content, re.MULTILINE | re.IGNORECASE)
    return bool(m and m.group(1).lower() == "circular")


def _load_insert_csv(path: Path) -> dict[str, dict[str, str]]:
    """Parse inserts_description.csv → dict keyed by part id.

    Accepts both comma- and tab-separated files (sniffed automatically).
    Returns an empty dict if the file is missing or unparseable.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t")
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            # Normalise key — strip whitespace from all field names and values
            row = {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            part_id = row.get("id") or row.get("ID") or ""
            if part_id:
                result[part_id] = row
        logger.info(f"Loaded {len(result)} rows from {path.name}")
        return result
    except Exception as e:
        logger.warning(f"Could not parse {path.name}: {e}")
        return {}


def _load_backbone_csv(path: Path) -> dict[str, dict[str, str]]:
    """Parse backbones_description.csv → dict keyed by backbone ID.

    Same dialect-sniffing as _load_insert_csv.
    """
    return _load_insert_csv(path)   # identical parsing logic; key normalisation handles "ID"


def _aliases_from_id(base_id: str) -> list[str]:
    """Extract aliases from a compound locus/ID string.

    For IDs like 'AICS_SynP0002_Cleavage-FLAG-GS-Aph4', the split point is the
    first underscore-delimited token that contains a hyphen (start of the
    human-readable description portion):
        → ['AICS_SynP0002', 'Cleavage-FLAG-GS-Aph4']

    Simple IDs with no hyphens (e.g. 'myGene', 'pTestVector') return [].
    """
    tokens = base_id.split("_")
    split_idx = next((i for i, t in enumerate(tokens) if "-" in t), None)
    if split_idx is None or split_idx == 0:
        return []
    code_part = "_".join(tokens[:split_idx])
    desc_part = "_".join(tokens[split_idx:])
    return [code_part, desc_part]


def _apply_insert_csv_meta(entry: dict[str, Any], row: dict[str, str]) -> None:
    """Overlay CSV metadata onto a parsed insert entry (in-place)."""
    description = row.get("Description", "")
    if description:
        entry["name"] = description
        # Aliases: code part + description part extracted from the locus ID,
        # e.g. 'AICS_SynP0002_Cleavage-FLAG-GS-Aph4' → ['AICS_SynP0002', 'Cleavage-FLAG-GS-Aph4']
        base_id = entry["id"].removeprefix(USER_PREFIX)
        id_aliases = _aliases_from_id(base_id)
        existing = entry.get("aliases", [])
        entry["aliases"] = list(dict.fromkeys(existing + id_aliases))  # dedupe, preserve order

    enzyme = row.get("TypeIIS cutsite", "")
    if enzyme:
        entry["assembly_enzyme"] = enzyme

    overhang_l = row.get("Overhang L", "")
    if overhang_l:
        entry["overhang_left"] = overhang_l

    overhang_r = row.get("Overhang R", "")
    if overhang_r:
        entry["overhang_right"] = overhang_r

    # CSV Size = insert size after excision (NOT the full carrier vector size).
    # size_bp (from GenBank) stays as the full plasmid size.
    size_str = row.get("Size", "")
    if size_str:
        try:
            entry["insert_size_bp"] = int(size_str)
        except ValueError:
            logger.warning(f"Non-integer Size '{size_str}' for {entry['id']}; skipping")

    selection = row.get("Selection", "")
    if selection:
        entry["bacterial_resistance"] = selection

    category = row.get("Category", "")
    if category:
        entry["category"] = category


def _apply_backbone_csv_meta(entry: dict[str, Any], row: dict[str, str]) -> None:
    """Overlay CSV metadata onto a parsed backbone entry (in-place)."""
    ecoli_strain = row.get("E coli strain", "")
    if ecoli_strain:
        entry["ecoli_strain"] = ecoli_strain

    resistance = row.get("Antibiotic resistance", "")
    if resistance:
        entry["bacterial_resistance"] = resistance

    mammalian = row.get("Mammalian selection marker", "")
    if mammalian:
        entry["mammalian_selection"] = mammalian

    enzyme = row.get("Assembly enzyme", "")
    if enzyme:
        entry["assembly_enzyme"] = enzyme

    next_enzyme = row.get("Next step enzyme", "")
    if next_enzyme:
        entry["next_step_enzyme"] = next_enzyme

    # "Overhang pair 1" → "ACCG-GTTT" split into left/right
    pair1 = row.get("Overhang pair 1", "")
    if pair1 and "-" in pair1:
        left, right = pair1.split("-", 1)
        entry["overhang_left"] = left.strip()
        entry["overhang_right"] = right.strip()

    pair2 = row.get("Overhang pair 2", "")
    if pair2 and "-" in pair2:
        left2, right2 = pair2.split("-", 1)
        entry["overhang_left_2"] = left2.strip()
        entry["overhang_right_2"] = right2.strip()

    # Auto-build a human-readable description from available fields
    neg_sel = row.get("Neg selection marker", "")
    downstream = row.get("Downstream", "")
    desc_parts = []
    if resistance:
        desc_parts.append(f"{resistance} backbone")
    if enzyme:
        desc_parts.append(enzyme)
    if neg_sel:
        desc_parts.append(f"neg-sel {neg_sel}")
    if downstream:
        desc_parts.append(f"for {downstream}")
    if desc_parts:
        entry["description"] = ", ".join(desc_parts)


def _parse_file_to_entry(
    path: Path,
    is_insert: bool = False,
    csv_meta: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Parse a single GenBank file into a library entry dict.

    ID is `user:<LOCUS name>` with filename stem as fallback.

    For insert entries, if the GenBank LOCUS line declares the sequence as
    circular the file is treated as a part-in-vector (carrier plasmid).  The
    full sequence is stored under `plasmid_sequence` (required by
    assemble_golden_gate) and `category` is set to `"part_in_vector"`.
    Linear insert files are stored as plain `sequence` entries.

    If `csv_meta` is provided (a single CSV row dict), it is overlaid after
    the GenBank parse to enrich name, aliases, enzyme, overhangs, etc.
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
    circular = _is_circular(content)

    entry: dict[str, Any] = {
        "id": entry_id,
        "name": base_id,
        "aliases": [f"{USER_PREFIX}{path.stem}"] if path.stem != base_id else [],
        "description": f"User-provided GenBank file: {path.name}",
        "size_bp": parsed["size_bp"],
        "source": "user_library",
        "features": parsed["features"],
    }

    if is_insert and circular:
        # Part-in-vector: full circular carrier sequence stored as plasmid_sequence.
        # assemble_golden_gate() requires this key to excise the insert via Type IIS sites.
        entry["plasmid_sequence"] = parsed["sequence"]
        entry["category"] = "part_in_vector"
    else:
        entry["sequence"] = parsed["sequence"]

    if parsed["mcs_position"]:
        entry["mcs_position"] = parsed["mcs_position"]

    # Overlay CSV metadata if provided
    if csv_meta:
        if is_insert:
            _apply_insert_csv_meta(entry, csv_meta)
        else:
            _apply_backbone_csv_meta(entry, csv_meta)

    return entry


def _scan_subdir(subdir_name: str) -> list[dict[str, Any]]:
    """Scan $PLASMID_USER_LIBRARY/<subdir_name>/ for GenBank files.

    Also loads the corresponding *_description.csv from the library root
    and overlays its metadata onto matching entries.
    """
    root = _user_library_dir()
    if not root:
        return []
    subdir = root / subdir_name
    if not subdir.is_dir():
        return []

    is_insert = subdir_name == "inserts"

    # Load CSV metadata if present
    csv_filename = "inserts_description.csv" if is_insert else "backbones_description.csv"
    csv_loader = _load_insert_csv if is_insert else _load_backbone_csv
    csv_data = csv_loader(root / csv_filename)

    # Warn about CSV rows with no matching .gb file
    gb_stems = {
        p.stem for p in subdir.iterdir()
        if p.suffix.lower() in GENBANK_EXTENSIONS
    }
    for csv_id in csv_data:
        if csv_id not in gb_stems:
            logger.warning(
                f"{csv_filename}: id '{csv_id}' has no matching GenBank file in {subdir.name}/; skipping"
            )

    entries: list[dict[str, Any]] = []
    for path in sorted(subdir.iterdir()):
        if path.suffix.lower() not in GENBANK_EXTENSIONS:
            continue
        csv_meta = csv_data.get(path.stem)
        entry = _parse_file_to_entry(path, is_insert=is_insert, csv_meta=csv_meta)
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
