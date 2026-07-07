#!/usr/bin/env python3
"""
Literature — Open Access Full-Text Lookup via Unpaywall

Complements the PubMed MCP server: PubMed covers PMC-indexed papers, but
~30% of open-access literature lives outside PMC (journal-hosted OA,
preprints, institutional repositories). Unpaywall indexes those.

Unpaywall API: https://unpaywall.org/products/api
  GET /v2/{doi}?email={email}  →  {is_oa, best_oa_location: {url, url_for_pdf, ...}, ...}

Requires UNPAYWALL_EMAIL env var (Unpaywall requires a real email but no
API key or registration).
"""

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

UNPAYWALL_API = "https://api.unpaywall.org/v2"
_TIMEOUT_S = 10


def _unpaywall_email() -> str | None:
    return os.environ.get("UNPAYWALL_EMAIL")


def fetch_oa_fulltext(doi: str) -> dict[str, Any]:
    """Look up a DOI on Unpaywall and return OA status + full-text URLs.

    Returns a dict with:
      - found: bool — DOI resolved on Unpaywall
      - is_oa: bool — paper has an open-access copy
      - title, journal, year: metadata if available
      - pdf_url: direct PDF URL if available
      - landing_url: OA landing page URL if no direct PDF
      - error: str — only present on failure

    The agent should try PubMed MCP's get_full_text_article first (covers
    PMC); use this for papers PubMed can't fetch full text for.
    """
    email = _unpaywall_email()
    if not email:
        return {"found": False, "error": "UNPAYWALL_EMAIL not set"}

    doi = doi.strip()
    # Strip common DOI URL prefixes
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break

    try:
        resp = requests.get(
            f"{UNPAYWALL_API}/{doi}",
            params={"email": email},
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException as e:
        logger.warning(f"Unpaywall request failed for {doi}: {e}")
        return {"found": False, "error": f"request failed: {e}"}

    if resp.status_code == 404:
        return {"found": False, "error": f"DOI not found on Unpaywall: {doi}"}
    if resp.status_code != 200:
        return {"found": False, "error": f"Unpaywall returned {resp.status_code}"}

    data = resp.json()
    result: dict[str, Any] = {
        "found": True,
        "is_oa": bool(data.get("is_oa")),
        "title": data.get("title"),
        "journal": data.get("journal_name"),
        "year": data.get("year"),
    }

    best = data.get("best_oa_location") or {}
    if best:
        result["pdf_url"] = best.get("url_for_pdf")
        result["landing_url"] = best.get("url")
        result["host_type"] = best.get("host_type")  # "publisher" or "repository"
        result["license"] = best.get("license")

    return result
