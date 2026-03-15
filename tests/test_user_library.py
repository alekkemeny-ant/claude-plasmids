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
