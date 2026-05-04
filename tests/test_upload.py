"""Tests for src/upload.py — file upload parsing and BYOL saving."""
import os
import pytest
from pathlib import Path

from src.upload import (
    safe_filename,
    parse_sequence_file,
    save_to_library,
    get_library_status,
)


MINIMAL_GENBANK = b"""\
LOCUS       TestPlasmid              30 bp    DNA     circular   UNK
DEFINITION  Test plasmid.
ACCESSION   unknown
FEATURES             Location/Qualifiers
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat
//
"""

MINIMAL_FASTA = b"""\
>TestSeq a test sequence
ATGCATGCATGCATGCATGCATGCATGCAT
"""


class TestSafeFilename:
    def test_normal(self):
        assert safe_filename("my_plasmid.gb") == "my_plasmid.gb"

    def test_traversal(self):
        result = safe_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_unicode(self):
        result = safe_filename("plàsmíd_über.gb")
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for c in result)

    def test_empty(self):
        assert safe_filename("") == "unnamed"

    def test_only_dots(self):
        assert safe_filename("...") == "unnamed"

    def test_length_cap(self):
        assert len(safe_filename("a" * 200)) <= 100


class TestParseSequenceFile:
    def test_genbank(self):
        result = parse_sequence_file(MINIMAL_GENBANK, "test.gb")
        assert result["format"] == "genbank"
        assert result["name"] == "TestPlasmid"
        assert result["length"] == 30
        assert "ATGCATGCAT" in result["sequence"].upper()

    def test_fasta(self):
        result = parse_sequence_file(MINIMAL_FASTA, "test.fasta")
        assert result["format"] == "fasta"
        assert result["name"] == "TestSeq"
        assert result["length"] == 30

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            parse_sequence_file(b"this is not a sequence file", "garbage.txt")

    def test_genbank_metadata_extraction(self):
        result = parse_sequence_file(MINIMAL_GENBANK, "test.gb")
        assert "Test plasmid" in result["description"]
        assert result["format"] == "genbank"


class TestGetLibraryStatus:
    def test_not_configured(self, monkeypatch):
        monkeypatch.delenv("PLASMID_USER_LIBRARY", raising=False)
        status = get_library_status()
        assert status["configured"] is False
        assert "setup_hint" in status

    def test_configured_exists(self, tmp_path, monkeypatch):
        (tmp_path / "backbones").mkdir()
        (tmp_path / "inserts").mkdir()
        monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
        status = get_library_status()
        assert status["configured"] is True
        assert status["exists"] is True
        assert status["writable"] is True
        assert status["subdirs"]["backbones"]["exists"] is True


class TestSaveToLibrary:
    def test_save_genbank(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
        parsed = parse_sequence_file(MINIMAL_GENBANK, "test.gb")
        result = save_to_library(parsed, "backbones", MINIMAL_GENBANK)
        assert result["id"] == "user:TestPlasmid"
        assert Path(result["saved_path"]).exists()
        assert Path(result["saved_path"]).read_bytes() == MINIMAL_GENBANK

    def test_invalid_kind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
        parsed = parse_sequence_file(MINIMAL_GENBANK, "test.gb")
        with pytest.raises(ValueError, match="Invalid kind"):
            save_to_library(parsed, "../../etc", MINIMAL_GENBANK)

    def test_traversal_in_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
        parsed = parse_sequence_file(MINIMAL_GENBANK, "test.gb")
        parsed["name"] = "../../etc/passwd"
        result = save_to_library(parsed, "backbones", MINIMAL_GENBANK)
        saved = Path(result["saved_path"])
        assert str(saved).startswith(str(tmp_path))
        assert saved.exists()

    def test_not_configured_raises(self, monkeypatch):
        monkeypatch.delenv("PLASMID_USER_LIBRARY", raising=False)
        parsed = parse_sequence_file(MINIMAL_GENBANK, "test.gb")
        with pytest.raises(EnvironmentError):
            save_to_library(parsed, "backbones", MINIMAL_GENBANK)

    def test_creates_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLASMID_USER_LIBRARY", str(tmp_path))
        parsed = parse_sequence_file(MINIMAL_GENBANK, "test.gb")
        save_to_library(parsed, "annotations", MINIMAL_GENBANK)
        assert (tmp_path / "annotations").is_dir()
