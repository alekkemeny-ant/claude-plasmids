"""
SQLite persistence layer for the plasmid library.

Tables:
  constructs          — one row per saved construct
  construct_parts     — one row per backbone / insert in a construct
  construct_validations — one row per validation check result
"""

import io
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Add project root so src/ is importable (mirrors app.py pattern).
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from Bio import SeqIO as _SeqIO
    _BIOPYTHON = True
except ImportError:
    _BIOPYTHON = False

try:
    from src.assembler import clean_sequence, validate_dna, reverse_complement
    _ASSEMBLER_OK = True
except Exception:
    _ASSEMBLER_OK = False


# ── Schema ──────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS constructs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_name    TEXT NOT NULL,
    user_name         TEXT,
    notes             TEXT,
    genbank_content   TEXT,
    total_size_bp     INTEGER,
    session_id        TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sequence_verified BOOLEAN DEFAULT 0,
    verified_sequence TEXT,
    backbone_name     TEXT,
    insert_names      TEXT
);

CREATE TABLE IF NOT EXISTS construct_parts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id      INTEGER NOT NULL REFERENCES constructs(id) ON DELETE CASCADE,
    part_type         TEXT NOT NULL,
    part_name         TEXT NOT NULL,
    part_region       TEXT,
    position_start    INTEGER,
    position_end      INTEGER,
    source_system     TEXT,
    source_url        TEXT,
    source_doi        TEXT,
    source_pubmed_id  TEXT,
    genbank_accession TEXT,
    addgene_id        TEXT
);

CREATE TABLE IF NOT EXISTS construct_validations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES constructs(id) ON DELETE CASCADE,
    check_section TEXT,
    check_name    TEXT,
    severity      TEXT,
    passed        BOOLEAN,
    detail        TEXT
);
"""

_EDITABLE_FIELDS = {"user_name", "notes", "sequence_verified", "verified_sequence", "local_path"}


# ── Connection ───────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


# ── Public API ───────────────────────────────────────────────────────────────

def init_db(db_path: Path) -> None:
    with _connect(db_path) as con:
        con.executescript(_DDL)
        # Safe migrations for columns added after initial schema
        existing = {row[1] for row in con.execute("PRAGMA table_info(constructs)")}
        if "origin" not in existing:
            con.execute("ALTER TABLE constructs ADD COLUMN origin TEXT DEFAULT 'designer'")
        if "local_path" not in existing:
            con.execute("ALTER TABLE constructs ADD COLUMN local_path TEXT")
        if "part_type" not in existing:
            con.execute("ALTER TABLE constructs ADD COLUMN part_type TEXT")
        if "metadata" not in existing:
            con.execute("ALTER TABLE constructs ADD COLUMN metadata TEXT")


def save_construct(
    db_path: Path,
    *,
    construct_name: str,
    genbank_content: str,
    total_size_bp: Optional[int],
    session_id: Optional[str],
    backbone_name: str,
    insert_names: list[str],
    parts: list[dict],
    validations: list[dict],
    origin: str = "designer",
    local_path: Optional[str] = None,
    part_type: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    with _connect(db_path) as con:
        cur = con.execute(
            """INSERT INTO constructs
               (construct_name, genbank_content, total_size_bp, session_id,
                backbone_name, insert_names, origin, local_path, part_type, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                construct_name,
                genbank_content,
                total_size_bp,
                session_id,
                backbone_name,
                json.dumps(insert_names),
                origin,
                local_path,
                part_type,
                json.dumps(metadata) if metadata else None,
            ),
        )
        construct_id = cur.lastrowid

        for p in parts:
            con.execute(
                """INSERT INTO construct_parts
                   (construct_id, part_type, part_name, part_region,
                    position_start, position_end, source_system, source_url,
                    source_doi, source_pubmed_id, genbank_accession, addgene_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    construct_id,
                    p.get("part_type"),
                    p.get("part_name"),
                    p.get("part_region"),
                    p.get("position_start"),
                    p.get("position_end"),
                    p.get("source_system"),
                    p.get("source_url"),
                    p.get("source_doi"),
                    p.get("source_pubmed_id"),
                    p.get("genbank_accession"),
                    p.get("addgene_id"),
                ),
            )

        for v in validations:
            con.execute(
                """INSERT INTO construct_validations
                   (construct_id, check_section, check_name, severity, passed, detail)
                   VALUES (?,?,?,?,?,?)""",
                (
                    construct_id,
                    v.get("check_section"),
                    v.get("check_name"),
                    v.get("severity"),
                    v.get("passed"),
                    v.get("detail"),
                ),
            )

        return construct_id


def list_constructs(db_path: Path) -> list[dict]:
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT * FROM constructs ORDER BY created_at DESC"
        ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["accession"] = f"PLM-{d['id']:05d}"
            d["insert_names"] = json.loads(d["insert_names"] or "[]")
            parts = con.execute(
                "SELECT * FROM construct_parts WHERE construct_id=?", (d["id"],)
            ).fetchall()
            d["parts"] = [dict(p) for p in parts]
            validations = con.execute(
                "SELECT * FROM construct_validations WHERE construct_id=?", (d["id"],)
            ).fetchall()
            d["validations"] = [
                {**dict(v), "passed": bool(v["passed"])} for v in validations
            ]
            d["sequence_verified"] = bool(d["sequence_verified"])
            d["metadata"] = json.loads(d["metadata"]) if d.get("metadata") else {}
            result.append(d)
        return result


def delete_construct(db_path: Path, construct_id: int) -> bool:
    with _connect(db_path) as con:
        cur = con.execute("DELETE FROM constructs WHERE id=?", (construct_id,))
        return cur.rowcount > 0


def update_construct(db_path: Path, construct_id: int, fields: dict) -> bool:
    safe = {k: v for k, v in fields.items() if k in _EDITABLE_FIELDS}
    if not safe:
        return False
    set_clause = ", ".join(f"{k}=?" for k in safe)
    values = list(safe.values()) + [construct_id]
    with _connect(db_path) as con:
        cur = con.execute(
            f"UPDATE constructs SET {set_clause} WHERE id=?", values
        )
        return cur.rowcount > 0


def get_construct_genbank(db_path: Path, construct_id: int) -> Optional[tuple[str, str]]:
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT construct_name, genbank_content FROM constructs WHERE id=?",
            (construct_id,),
        ).fetchone()
        if row is None:
            return None
        return row["construct_name"], row["genbank_content"]


def get_construct_by_local_path(db_path: Path, local_path: str) -> Optional[dict]:
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT id, construct_name FROM constructs WHERE local_path=?",
            (local_path,),
        ).fetchone()
        return dict(row) if row else None


def get_graph_data(db_path: Path) -> dict:
    with _connect(db_path) as con:
        constructs = con.execute(
            """SELECT id, construct_name, total_size_bp, created_at,
                      backbone_name, insert_names, sequence_verified, user_name,
                      origin
               FROM constructs"""
        ).fetchall()
        parts = con.execute(
            """SELECT construct_id, part_type, part_name,
                      source_system, source_url, source_doi, genbank_accession, addgene_id
               FROM construct_parts"""
        ).fetchall()

    # Count how many constructs use each part
    part_usage: dict[str, int] = {}
    for p in parts:
        key = p["part_name"]
        part_usage[key] = part_usage.get(key, 0) + 1

    nodes = []
    edges = []
    seen_parts: set[str] = set()

    for c in constructs:
        accession = f"PLM-{c['id']:05d}"
        insert_names = json.loads(c["insert_names"] or "[]")
        nodes.append({
            "data": {
                "id": f"c_{c['id']}",
                "label": c["construct_name"],
                "nodeType": "construct",
                "size_bp": c["total_size_bp"],
                "accession": accession,
                "created_at": (c["created_at"] or "")[:16],
                "backbone_name": c["backbone_name"] or "",
                "insert_names": insert_names,
                "sequence_verified": bool(c["sequence_verified"]),
                "user_name": c["user_name"] or "",
                "origin": c["origin"] or "designer",
            }
        })

    for p in parts:
        part_id = f"p_{p['part_name']}"
        node_type = "backbone" if p["part_type"] == "backbone" else "insert"
        if part_id not in seen_parts:
            seen_parts.add(part_id)
            nodes.append({
                "data": {
                    "id": part_id,
                    "label": p["part_name"],
                    "nodeType": node_type,
                    "source_system": p["source_system"] or "",
                    "source_url": p["source_url"] or "",
                    "source_doi": p["source_doi"] or "",
                    "genbank_accession": p["genbank_accession"] or "",
                    "addgene_id": str(p["addgene_id"]) if p["addgene_id"] else "",
                    "usage_count": part_usage.get(p["part_name"], 1),
                }
            })
        edges.append({
            "data": {
                "id": f"e_{p['construct_id']}_{p['part_name'].replace(' ', '_')}",
                "source": f"c_{p['construct_id']}",
                "target": part_id,
                "edgeType": p["part_type"],
            }
        })

    return {"nodes": nodes, "edges": edges}


# ── Validation helper ────────────────────────────────────────────────────────

def _extract_sequence_from_genbank(genbank_content: str) -> Optional[str]:
    if not _BIOPYTHON or not genbank_content:
        return None
    try:
        record = next(_SeqIO.parse(io.StringIO(genbank_content), "genbank"))
        return str(record.seq)
    except Exception:
        return None


def run_validation_structured(
    genbank_content: str,
    backbone_name: str,
    insert_name: str,
) -> list[dict]:
    if not _ASSEMBLER_OK:
        return []

    sequence = _extract_sequence_from_genbank(genbank_content)
    if not sequence:
        return [{
            "check_section": "Output Verification",
            "check_name": "GenBank Parseable",
            "severity": "Critical",
            "passed": False,
            "detail": "Could not extract sequence from GenBank content",
        }]

    results = []

    seq = clean_sequence(sequence)
    valid, errors = validate_dna(seq)
    results.append({
        "check_section": "Input Validation",
        "check_name": "Valid DNA Characters",
        "severity": "Critical",
        "passed": valid,
        "detail": "; ".join(errors) if errors else "OK",
    })

    if valid and len(seq) > 0:
        results.append({
            "check_section": "Size Check",
            "check_name": "Sequence Non-Empty",
            "severity": "Major",
            "passed": True,
            "detail": f"{len(seq)} bp",
        })

        # Basic ORF checks on the full sequence
        has_start = "ATG" in seq.upper()
        results.append({
            "check_section": "Biological Sanity",
            "check_name": "Start Codon Present",
            "severity": "Major",
            "passed": has_start,
            "detail": "ATG found" if has_start else "No ATG in sequence",
        })

        stop_codons = {"TAA", "TAG", "TGA"}
        has_stop = any(seq.upper()[i:i+3] in stop_codons for i in range(0, len(seq)-2, 3))
        results.append({
            "check_section": "Biological Sanity",
            "check_name": "Stop Codon Present",
            "severity": "Minor",
            "passed": has_stop,
            "detail": "Stop codon found" if has_stop else "No in-frame stop codon",
        })

    results.append({
        "check_section": "Output Verification",
        "check_name": "GenBank Parseable",
        "severity": "Critical",
        "passed": True,
        "detail": "Parsed successfully",
    })

    return results


# ── Provenance helpers ───────────────────────────────────────────────────────

def build_parts_from_library(
    backbone_name: str,
    insert_names: list[str],
) -> list[dict]:
    try:
        from src.library import get_backbone_by_id, get_insert_by_id
    except Exception:
        return _build_parts_minimal(backbone_name, insert_names)

    parts = []

    bb = None
    if backbone_name:
        try:
            bb = get_backbone_by_id(backbone_name)
        except Exception:
            bb = None

    if bb and not bb.get("needs_disambiguation"):
        addgene_id = bb.get("addgene_id")
        url = bb.get("url") or (
            f"https://www.addgene.org/{addgene_id}/" if addgene_id else None
        )
        parts.append({
            "part_type": "backbone",
            "part_name": bb.get("name") or backbone_name,
            "part_region": "Vector backbone",
            "source_system": bb.get("source") or bb.get("sequence_source"),
            "source_url": url,
            "source_doi": bb.get("article_doi"),
            "source_pubmed_id": bb.get("article_pubmed_id") or bb.get("pubmed_id"),
            "genbank_accession": bb.get("genbank_accession"),
            "addgene_id": str(addgene_id) if addgene_id is not None else None,
        })
    elif backbone_name:
        parts.append({
            "part_type": "backbone",
            "part_name": backbone_name,
            "part_region": "Vector backbone",
            "source_system": None,
            "source_url": None,
            "source_doi": None,
            "source_pubmed_id": None,
            "genbank_accession": None,
            "addgene_id": None,
        })

    regions = _assign_insert_regions(insert_names)
    for name, region in zip(insert_names, regions):
        ins = None
        if name:
            try:
                ins = get_insert_by_id(name)
            except Exception:
                ins = None

        if ins and not ins.get("needs_disambiguation"):
            parts.append({
                "part_type": "insert",
                "part_name": ins.get("name") or name,
                "part_region": region,
                "source_system": ins.get("source"),
                "source_url": _ncbi_url(ins.get("genbank_accession")),
                "source_doi": None,
                "source_pubmed_id": None,
                "genbank_accession": ins.get("genbank_accession"),
                "addgene_id": None,
            })
        else:
            parts.append({
                "part_type": "insert",
                "part_name": name,
                "part_region": region,
                "source_system": None,
                "source_url": None,
                "source_doi": None,
                "source_pubmed_id": None,
                "genbank_accession": None,
                "addgene_id": None,
            })

    return parts


def _build_parts_minimal(backbone_name: str, insert_names: list[str]) -> list[dict]:
    parts = []
    if backbone_name:
        parts.append({
            "part_type": "backbone", "part_name": backbone_name,
            "part_region": "Vector backbone",
        })
    regions = _assign_insert_regions(insert_names)
    for name, region in zip(insert_names, regions):
        parts.append({"part_type": "insert", "part_name": name, "part_region": region})
    return parts


def _assign_insert_regions(insert_names: list[str]) -> list[str]:
    n = len(insert_names)
    if n == 0:
        return []
    if n == 1:
        return ["MCS region"]
    regions = []
    for i in range(n):
        if i == 0:
            regions.append("N-terminus")
        elif i == n - 1:
            regions.append("C-terminus")
        else:
            regions.append("Linker region")
    return regions


def _ncbi_url(accession: Optional[str]) -> Optional[str]:
    if not accession:
        return None
    return f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}"
