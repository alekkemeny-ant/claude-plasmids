#!/usr/bin/env python3
"""Tests for NCBI integration — promoter detection + genomic upstream fetch."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from library import is_known_promoter, KNOWN_PROMOTERS


class TestKnownPromoters:
    def test_cmv(self):
        assert is_known_promoter("CMV")

    def test_ef1a_normalized(self):
        assert is_known_promoter("EF-1a")
        assert is_known_promoter("EF1α")  # Greek alpha normalized away

    def test_case_insensitive(self):
        assert is_known_promoter("cmv")
        assert is_known_promoter("Cmv")

    def test_u6_pol3(self):
        assert is_known_promoter("U6")

    def test_bespoke_rejected(self):
        assert not is_known_promoter("p65")
        assert not is_known_promoter("IFNβ")
        assert not is_known_promoter("custom-promoter")

    def test_set_contents(self):
        # Sanity — a reasonable number of standard promoters
        assert len(KNOWN_PROMOTERS) >= 20


# ── Network-dependent tests (marked slow) ────────────────────────────────

try:
    from ncbi_integration import fetch_genomic_upstream, BIOPYTHON_AVAILABLE
except ImportError:
    BIOPYTHON_AVAILABLE = False


@pytest.mark.skipif(not BIOPYTHON_AVAILABLE, reason="Biopython not installed")
@pytest.mark.slow
class TestFetchGenomicUpstream:
    def test_tp53_upstream(self):
        """Fetch 1000bp upstream of human TP53 (gene_id=7157, minus strand)."""
        result = fetch_genomic_upstream(gene_id="7157", bp_upstream=1000)
        assert result is not None
        assert result["gene_id"] == "7157"
        assert "TP53" in result["gene_symbol"].upper() or result["gene_symbol"].startswith("GeneID")
        assert 900 <= len(result["sequence"]) <= 1000  # allow slight NCBI boundary slop
        # Valid DNA
        assert set(result["sequence"]) <= set("ACGTN")
        assert result["warning"]  # warning must be present

    def test_invalid_bp_range(self):
        with pytest.raises(ValueError):
            fetch_genomic_upstream(gene_id="7157", bp_upstream=50)
        with pytest.raises(ValueError):
            fetch_genomic_upstream(gene_id="7157", bp_upstream=50000)
