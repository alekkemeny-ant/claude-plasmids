#!/usr/bin/env python3
"""Tests for AddgeneClient._parse_api_response edge cases."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from addgene_integration import AddgeneClient


@pytest.fixture
def client():
    return AddgeneClient()


def _base_api_payload(**overrides):
    """Minimal valid API payload with sensible defaults."""
    payload = {
        "id": 12345,
        "name": "pTest",
        "description": "A test plasmid",
        "depositor_comments": None,
        "cloning": {},
        "promoter": None,
        "bacterial_resistance": "Ampicillin",
        "resistance_markers": None,
        "growth_strain": None,
        "inserts": [],
        "article": {},
        "sequences": {
            "public_addgene_full_sequences": [
                {"sequence": "ATCGATCG"}
            ]
        },
    }
    payload.update(overrides)
    return payload


def test_parse_api_response_normal(client):
    """Sequence is extracted when public_addgene_full_sequences is populated."""
    data = _base_api_payload()
    result = client._parse_api_response(data)
    assert result.sequence == "ATCGATCG"
    assert result.addgene_id == "12345"
    assert result.name == "pTest"


def test_parse_api_response_empty_sequences_list(client):
    """No IndexError when public_addgene_full_sequences is an empty list (e.g. #73032)."""
    data = _base_api_payload(sequences={"public_addgene_full_sequences": []})
    result = client._parse_api_response(data)
    assert result.sequence is None


def test_parse_api_response_missing_sequences_key(client):
    """No error when the sequences key is absent entirely."""
    data = _base_api_payload(sequences={})
    result = client._parse_api_response(data)
    assert result.sequence is None


def test_parse_api_response_no_inserts(client):
    """gene_insert is None when inserts list is empty."""
    data = _base_api_payload(inserts=[])
    result = client._parse_api_response(data)
    assert result.gene_insert is None


def test_parse_api_response_with_insert(client):
    """gene_insert is populated from the first insert's name."""
    data = _base_api_payload(inserts=[{"name": "EGFP"}, {"name": "mCherry"}])
    result = client._parse_api_response(data)
    assert result.gene_insert == "EGFP"
