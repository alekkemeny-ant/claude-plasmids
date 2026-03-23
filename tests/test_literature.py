#!/usr/bin/env python3
"""Tests for Unpaywall literature lookup."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def unpaywall_email(monkeypatch):
    monkeypatch.setenv("UNPAYWALL_EMAIL", "test@example.com")


@pytest.fixture
def no_unpaywall_email(monkeypatch):
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def test_missing_email_returns_error(no_unpaywall_email):
    from src.literature import fetch_oa_fulltext
    result = fetch_oa_fulltext("10.1038/nature12373")
    assert result["found"] is False
    assert "UNPAYWALL_EMAIL" in result["error"]


def test_oa_paper_returns_pdf_url(unpaywall_email):
    from src.literature import fetch_oa_fulltext
    mock_json = {
        "is_oa": True,
        "title": "CRISPR-Cas9 genome editing",
        "journal_name": "Nature",
        "year": 2013,
        "best_oa_location": {
            "url": "https://www.nature.com/articles/nature12373",
            "url_for_pdf": "https://www.nature.com/articles/nature12373.pdf",
            "host_type": "publisher",
            "license": "cc-by",
        },
    }
    with patch("src.literature.requests.get", return_value=_mock_response(json_data=mock_json)) as mock_get:
        result = fetch_oa_fulltext("10.1038/nature12373")
    mock_get.assert_called_once()
    call_url = mock_get.call_args[0][0]
    assert "10.1038/nature12373" in call_url
    assert result["found"] is True
    assert result["is_oa"] is True
    assert result["pdf_url"] == "https://www.nature.com/articles/nature12373.pdf"
    assert result["license"] == "cc-by"


def test_non_oa_paper(unpaywall_email):
    from src.literature import fetch_oa_fulltext
    mock_json = {"is_oa": False, "title": "Closed paper", "journal_name": "Cell", "year": 2020}
    with patch("src.literature.requests.get", return_value=_mock_response(json_data=mock_json)):
        result = fetch_oa_fulltext("10.1016/j.cell.2020.01.001")
    assert result["found"] is True
    assert result["is_oa"] is False
    assert "pdf_url" not in result


def test_doi_not_found(unpaywall_email):
    from src.literature import fetch_oa_fulltext
    with patch("src.literature.requests.get", return_value=_mock_response(status_code=404)):
        result = fetch_oa_fulltext("10.9999/fake")
    assert result["found"] is False
    assert "not found" in result["error"].lower()


def test_doi_url_prefix_stripped(unpaywall_email):
    from src.literature import fetch_oa_fulltext
    with patch("src.literature.requests.get", return_value=_mock_response(json_data={"is_oa": False})) as mock_get:
        fetch_oa_fulltext("https://doi.org/10.1038/nature12373")
    call_url = mock_get.call_args[0][0]
    # Should have stripped the prefix — URL should be .../v2/10.1038/..., not .../v2/https://doi.org/...
    assert call_url.endswith("/10.1038/nature12373")
    assert "doi.org" not in call_url.split("/v2/")[1]


def test_network_error_handled(unpaywall_email):
    import requests
    from src.literature import fetch_oa_fulltext
    with patch("src.literature.requests.get", side_effect=requests.ConnectionError("dns fail")):
        result = fetch_oa_fulltext("10.1038/nature12373")
    assert result["found"] is False
    assert "request failed" in result["error"]
