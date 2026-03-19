#!/usr/bin/env python3
"""Tests for bring-your-own-library (BYOL) support."""

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "user_library"


@pytest.fixture
def user_library_env(monkeypatch):
    """Point PLASMID_USER_LIBRARY at the test fixture directory."""
    monkeypatch.setenv("PLASMID_USER_LIBRARY", str(FIXTURE_DIR))
    # Force re-evaluation of the cached dir lookup on each call by
    # importing fresh — the module reads the env var at call time,
    # not import time, so this is safe.
    yield FIXTURE_DIR


@pytest.fixture
def no_user_library(monkeypatch):
    """Ensure PLASMID_USER_LIBRARY is unset."""
    monkeypatch.delenv("PLASMID_USER_LIBRARY", raising=False)


# ── user_library module ──

def test_load_user_backbones_finds_fixture(user_library_env):
    from src.user_library import load_user_backbones
    entries = load_user_backbones()
    assert len(entries) == 1
    bb = entries[0]
    assert bb["id"] == "user:pTestVector"
    assert bb["source"] == "user_library"
    assert bb["size_bp"] == 300
    assert bb["sequence"].startswith("ATCGATCGAT")
    assert bb["mcs_position"]["start"] == 59  # 60 in 1-based GenBank → 59 in 0-based
    assert bb["mcs_position"]["end"] == 120


def test_load_user_inserts_finds_fixture(user_library_env):
    from src.user_library import load_user_inserts
    entries = load_user_inserts()
    assert len(entries) == 1
    ins = entries[0]
    assert ins["id"] == "user:myGene"
    assert ins["size_bp"] == 180
    assert ins["sequence"].startswith("ATGAAAGCGT")
    assert "mcs_position" not in ins  # inserts don't have MCS


def test_load_user_backbones_empty_when_unset(no_user_library):
    from src.user_library import load_user_backbones
    assert load_user_backbones() == []


def test_load_user_backbones_empty_when_dir_missing(monkeypatch):
    monkeypatch.setenv("PLASMID_USER_LIBRARY", "/nonexistent/path/xyz")
    from src.user_library import load_user_backbones
    assert load_user_backbones() == []


def test_filename_stem_fallback_when_no_locus(user_library_env, tmp_path, monkeypatch):
    """If LOCUS line is missing, fall back to filename stem for the ID."""
    subdir = tmp_path / "backbones"
    subdir.mkdir()
    # GenBank with no LOCUS line but valid ORIGIN
    (subdir / "noLocus.gb").write_text(
        "FEATURES             Location/Qualifiers\n"
        "ORIGIN\n"
        "        1 " + ("atcg" * 30) + "\n"
        "//\n"
    )
    monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
    from src.user_library import load_user_backbones
    entries = load_user_backbones()
    assert len(entries) == 1
    assert entries[0]["id"] == "user:noLocus"


def test_non_genbank_files_ignored(tmp_path, monkeypatch):
    subdir = tmp_path / "backbones"
    subdir.mkdir()
    (subdir / "readme.txt").write_text("not a genbank file")
    (subdir / "data.json").write_text("{}")
    monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
    from src.user_library import load_user_backbones
    assert load_user_backbones() == []


# ── library.py merge integration ──

def test_library_load_backbones_merges_user(user_library_env):
    from src.library import load_backbones
    data = load_backbones()
    user_ids = [b["id"] for b in data["backbones"] if b["id"].startswith("user:")]
    assert "user:pTestVector" in user_ids


def test_library_load_backbones_no_merge_when_unset(no_user_library):
    from src.library import load_backbones
    data = load_backbones()
    user_ids = [b["id"] for b in data["backbones"] if b["id"].startswith("user:")]
    assert user_ids == []


def test_builtin_loader_never_includes_user(user_library_env):
    """Critical cache-isolation invariant: _load_builtin_* must NOT merge."""
    from src.library import _load_builtin_backbones, _load_builtin_inserts
    bb = _load_builtin_backbones()
    ins = _load_builtin_inserts()
    assert not any(b["id"].startswith("user:") for b in bb["backbones"])
    assert not any(i["id"].startswith("user:") for i in ins["inserts"])


def test_get_backbone_by_id_finds_user_entry(user_library_env):
    from src.library import get_backbone_by_id
    bb = get_backbone_by_id("user:pTestVector")
    assert bb is not None
    assert bb["source"] == "user_library"


def test_search_backbones_finds_user_entry(user_library_env):
    from src.library import search_backbones
    results = search_backbones("pTestVector")
    user_results = [r for r in results if r["id"].startswith("user:")]
    assert len(user_results) >= 1


# ── CSV metadata overlay ──

def test_insert_csv_enriches_name_and_aliases(user_library_env):
    from src.user_library import load_user_inserts
    ins = load_user_inserts()[0]
    # name comes from CSV Description column
    assert ins["name"] == "MyGene-Test-Insert"
    # aliases come from the locus ID: 'myGene' has no hyphens so no ID-based split
    # (simple IDs produce no aliases beyond the existing filename-stem alias)


def test_insert_csv_aliases_from_compound_id(tmp_path, monkeypatch):
    """Compound locus IDs like LAB_P001_MyPromoter-Kozak produce two aliases."""
    subdir = tmp_path / "inserts"
    subdir.mkdir()
    (subdir / "LAB_P001_MyPromoter-Kozak.gbk").write_text(
        "LOCUS       LAB_P001_MyPromoter-Kozak  180 bp    DNA     linear   SYN 01-JAN-2026\n"
        "DEFINITION  Compound ID test insert.\n"
        "FEATURES             Location/Qualifiers\n"
        "     CDS             1..180\n"
        "                     /label=\"test CDS\"\n"
        "ORIGIN\n"
        "        1 atgaaagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtta\n"
        "       61 gcgttagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtta\n"
        "      121 gcgttagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtaa\n"
        "//\n"
    )
    csv_text = "id\tDescription\nLAB_P001_MyPromoter-Kozak\tMyPromoter-Kozak Description\n"
    (tmp_path / "inserts_description.csv").write_text(csv_text)
    monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
    from src.user_library import load_user_inserts
    entries = load_user_inserts()
    assert len(entries) == 1
    aliases = entries[0]["aliases"]
    assert "LAB_P001" in aliases
    assert "MyPromoter-Kozak" in aliases


def test_insert_csv_sets_enzyme_and_overhangs(user_library_env):
    from src.user_library import load_user_inserts
    ins = load_user_inserts()[0]
    assert ins["assembly_enzyme"] == "Esp3I"
    assert ins["overhang_l"] == "CACC"
    assert ins["overhang_r"] == "CTGG"


def test_insert_csv_size_stored_as_insert_size_bp(user_library_env):
    from src.user_library import load_user_inserts
    ins = load_user_inserts()[0]
    # insert_size_bp (from CSV) is the excised insert size — distinct from
    # size_bp (from GenBank), which is the full file sequence length.
    assert ins["insert_size_bp"] == 120
    assert ins["size_bp"] == 180          # full GenBank sequence untouched
    assert ins["insert_size_bp"] != ins["size_bp"]


def test_insert_csv_sets_selection_and_category(user_library_env):
    from src.user_library import load_user_inserts
    ins = load_user_inserts()[0]
    assert ins["bacterial_resistance"] == "AmpR"
    assert ins["category"] == "insert"


def test_backbone_csv_enriches_metadata(user_library_env):
    from src.user_library import load_user_backbones
    bb = load_user_backbones()[0]
    assert bb["bacterial_resistance"] == "KanR"
    assert bb["mammalian_selection"] == "PuroR"
    assert bb["assembly_enzyme"] == "Esp3I"
    assert bb["ecoli_strain"] == "DH5alpha"
    assert "next_step_enzyme" not in bb   # empty in fixture — not stored
    assert bb["overhang_left"] == "CACC"
    assert bb["overhang_right"] == "CTGG"
    assert bb["overhang_left_2"] == "AACG"
    assert bb["overhang_right_2"] == "GTTT"


def test_backbone_csv_builds_description(user_library_env):
    from src.user_library import load_user_backbones
    bb = load_user_backbones()[0]
    desc = bb["description"]
    assert "KanR" in desc
    assert "Esp3I" in desc
    assert "mCherry" in desc
    assert "pTarget" in desc


_MINIMAL_INSERT_GB = (
    "LOCUS       myGene                   180 bp    DNA     linear   SYN 01-JAN-2026\n"
    "DEFINITION  Minimal test insert.\n"
    "FEATURES             Location/Qualifiers\n"
    "     CDS             1..180\n"
    "                     /label=\"myGene CDS\"\n"
    "ORIGIN\n"
    "        1 atgaaagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtta\n"
    "       61 gcgttagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtta\n"
    "      121 gcgttagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtaa\n"
    "//\n"
)


def test_csv_absent_loads_genbank_only(tmp_path, monkeypatch):
    """Entries load fine when no CSV is present."""
    subdir = tmp_path / "inserts"
    subdir.mkdir()
    (subdir / "myGene.gbk").write_text(_MINIMAL_INSERT_GB)
    monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
    from src.user_library import load_user_inserts
    entries = load_user_inserts()
    assert len(entries) == 1
    assert entries[0]["name"] == "myGene"      # falls back to locus name
    assert "assembly_enzyme" not in entries[0]
    assert "insert_size_bp" not in entries[0]


def test_csv_row_with_no_matching_gb_is_skipped(tmp_path, monkeypatch, caplog):
    """A CSV row whose id has no .gb file is warned about and ignored."""
    import logging
    subdir = tmp_path / "inserts"
    subdir.mkdir()
    (subdir / "realPart.gbk").write_text(
        "LOCUS       realPart                 180 bp    DNA     linear   SYN 01-JAN-2026\n"
        "DEFINITION  Minimal test insert.\n"
        "FEATURES             Location/Qualifiers\n"
        "     CDS             1..180\n"
        "                     /label=\"realPart CDS\"\n"
        "ORIGIN\n"
        "        1 atgaaagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtta\n"
        "       61 gcgttagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtta\n"
        "      121 gcgttagcgt tagcgttagc gttagcgtta gcgttagcgt tagcgttagc gttagcgtaa\n"
        "//\n"
    )
    csv_text = "id\tDescription\nrealPart\tReal Part\nghostPart\tGhost Part\n"
    (tmp_path / "inserts_description.csv").write_text(csv_text)
    monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="src.user_library"):
        from src.user_library import load_user_inserts
        entries = load_user_inserts()
    assert len(entries) == 1
    assert any("ghostPart" in r.message for r in caplog.records)
