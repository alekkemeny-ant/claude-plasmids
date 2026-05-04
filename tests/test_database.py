"""Tests for app/database.py — SQLite plasmid library."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root so app/database.py can import src/ modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

def _load_database():
    spec = importlib.util.spec_from_file_location(
        "plasmid_database", PROJECT_ROOT / "app" / "database.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

db = _load_database()


@pytest.fixture()
def tmp_db(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


def _minimal_construct(name="pTest"):
    return dict(
        construct_name=name,
        genbank_content="LOCUS test 100 bp",
        total_size_bp=100,
        session_id="sess-1",
        backbone_name="pcDNA3.1(+)",
        insert_names=["EGFP"],
        parts=[{
            "part_type": "backbone",
            "part_name": "pcDNA3.1(+)",
            "part_region": "Vector backbone",
            "source_system": "Thermo Fisher",
            "source_url": None,
            "source_doi": None,
            "source_pubmed_id": None,
            "genbank_accession": None,
            "addgene_id": None,
        }],
        validations=[{
            "check_section": "Output Verification",
            "check_name": "Parseable",
            "severity": "Critical",
            "passed": True,
            "detail": "OK",
        }],
    )


# ── Schema ────────────────────────────────────────────────────────────────────

def test_init_db_creates_tables(tmp_db):
    import sqlite3
    con = sqlite3.connect(str(tmp_db))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "constructs" in tables
    assert "construct_parts" in tables
    assert "construct_validations" in tables
    con.close()


# ── CRUD ──────────────────────────────────────────────────────────────────────

def test_save_and_list_construct(tmp_db):
    cid = db.save_construct(tmp_db, **_minimal_construct())
    assert isinstance(cid, int) and cid > 0
    rows = db.list_constructs(tmp_db)
    assert len(rows) == 1
    assert rows[0]["construct_name"] == "pTest"
    assert rows[0]["total_size_bp"] == 100


def test_list_includes_nested_parts(tmp_db):
    db.save_construct(tmp_db, **_minimal_construct())
    rows = db.list_constructs(tmp_db)
    assert len(rows[0]["parts"]) == 1
    assert rows[0]["parts"][0]["part_type"] == "backbone"


def test_list_includes_nested_validations(tmp_db):
    db.save_construct(tmp_db, **_minimal_construct())
    rows = db.list_constructs(tmp_db)
    assert len(rows[0]["validations"]) == 1
    assert rows[0]["validations"][0]["passed"] is True


def test_update_editable_fields(tmp_db):
    cid = db.save_construct(tmp_db, **_minimal_construct())
    ok = db.update_construct(tmp_db, cid, {"user_name": "My Construct", "notes": "good notes"})
    assert ok is True
    rows = db.list_constructs(tmp_db)
    assert rows[0]["user_name"] == "My Construct"
    assert rows[0]["notes"] == "good notes"


def test_update_sequence_verified(tmp_db):
    cid = db.save_construct(tmp_db, **_minimal_construct())
    ok = db.update_construct(tmp_db, cid, {"sequence_verified": True, "verified_sequence": "ATGC"})
    assert ok is True
    rows = db.list_constructs(tmp_db)
    assert rows[0]["sequence_verified"] is True
    assert rows[0]["verified_sequence"] == "ATGC"


def test_protected_fields_not_updateable(tmp_db):
    cid = db.save_construct(tmp_db, **_minimal_construct())
    ok = db.update_construct(tmp_db, cid, {
        "construct_name": "HACKED",
        "genbank_content": "HACKED",
        "user_name": "legitimate",
    })
    assert ok is True  # returns True because user_name was updated
    rows = db.list_constructs(tmp_db)
    # Protected fields unchanged
    assert rows[0]["construct_name"] == "pTest"
    assert rows[0]["genbank_content"] == "LOCUS test 100 bp"
    # Editable field did update
    assert rows[0]["user_name"] == "legitimate"


def test_update_returns_false_for_nonexistent(tmp_db):
    ok = db.update_construct(tmp_db, 999, {"user_name": "x"})
    assert ok is False


def test_update_returns_false_for_empty_safe_fields(tmp_db):
    cid = db.save_construct(tmp_db, **_minimal_construct())
    ok = db.update_construct(tmp_db, cid, {"construct_name": "evil"})
    assert ok is False


def test_get_construct_genbank(tmp_db):
    cid = db.save_construct(tmp_db, **_minimal_construct())
    result = db.get_construct_genbank(tmp_db, cid)
    assert result is not None
    name, content = result
    assert name == "pTest"
    assert "LOCUS" in content


def test_get_construct_genbank_missing(tmp_db):
    assert db.get_construct_genbank(tmp_db, 999) is None


# ── Graph ─────────────────────────────────────────────────────────────────────

def test_graph_empty(tmp_db):
    g = db.get_graph_data(tmp_db)
    assert g["nodes"] == []
    assert g["edges"] == []


def test_graph_bipartite_structure(tmp_db):
    db.save_construct(tmp_db, **_minimal_construct("pTest1"))
    db.save_construct(tmp_db, **_minimal_construct("pTest2"))
    g = db.get_graph_data(tmp_db)

    node_types = {n["data"]["nodeType"] for n in g["nodes"]}
    assert "construct" in node_types
    assert "backbone" in node_types

    construct_nodes = [n for n in g["nodes"] if n["data"]["nodeType"] == "construct"]
    assert len(construct_nodes) == 2

    # Two constructs sharing the same backbone → one backbone node
    backbone_nodes = [n for n in g["nodes"] if n["data"]["nodeType"] == "backbone"]
    assert len(backbone_nodes) == 1

    # Two edges (one per construct → backbone)
    assert len(g["edges"]) == 2


def test_graph_no_duplicate_part_nodes(tmp_db):
    # Three constructs all using the same backbone
    for i in range(3):
        db.save_construct(tmp_db, **_minimal_construct(f"pTest{i}"))
    g = db.get_graph_data(tmp_db)
    backbone_nodes = [n for n in g["nodes"] if n["data"]["nodeType"] == "backbone"]
    assert len(backbone_nodes) == 1


def test_graph_with_inserts(tmp_db):
    kwargs = _minimal_construct("pEGFP")
    kwargs["parts"].append({
        "part_type": "insert",
        "part_name": "EGFP",
        "part_region": "MCS region",
        "source_system": "FPbase",
        "source_url": None,
        "source_doi": None,
        "source_pubmed_id": None,
        "genbank_accession": "U55762",
        "addgene_id": None,
    })
    db.save_construct(tmp_db, **kwargs)
    g = db.get_graph_data(tmp_db)
    insert_nodes = [n for n in g["nodes"] if n["data"]["nodeType"] == "insert"]
    assert len(insert_nodes) == 1
    assert insert_nodes[0]["data"]["label"] == "EGFP"


# ── Part region assignment ────────────────────────────────────────────────────

def test_assign_insert_regions_single(tmp_db):
    regions = db._assign_insert_regions(["EGFP"])
    assert regions == ["MCS region"]


def test_assign_insert_regions_fusion(tmp_db):
    regions = db._assign_insert_regions(["EGFP", "mCherry"])
    assert regions == ["N-terminus", "C-terminus"]


def test_assign_insert_regions_triple(tmp_db):
    regions = db._assign_insert_regions(["EGFP", "Linker", "mCherry"])
    assert regions[0] == "N-terminus"
    assert regions[1] == "Linker region"
    assert regions[2] == "C-terminus"


# ── build_parts_from_library ──────────────────────────────────────────────────

def test_build_parts_fallback_unknown_names(tmp_db):
    parts = db.build_parts_from_library("UnknownBackbone", ["UnknownInsert"])
    assert len(parts) == 2
    bb = next(p for p in parts if p["part_type"] == "backbone")
    assert bb["part_name"] == "UnknownBackbone"
    ins = next(p for p in parts if p["part_type"] == "insert")
    assert ins["part_name"] == "UnknownInsert"


def test_build_parts_empty_names(tmp_db):
    parts = db.build_parts_from_library("", [])
    assert parts == []
