"""Tests for GenBank provenance COMMENT generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from references import ReferenceTracker
from assembler import format_as_genbank


class TestReferenceTrackerComment:
    def _tracker_with_refs(self) -> ReferenceTracker:
        t = ReferenceTracker()
        t.add_backbone({
            "name": "pcDNA3.1(+)",
            "addgene_id": "V79020",
            "genbank_accession": "V79020",
            "organism": "Mammalian",
        })
        t.add_ncbi_gene({
            "symbol": "EGFP",
            "accession": "U55762",
            "gene_id": "112345",
            "organism": "Aequorea victoria",
        })
        t.add_addgene_plasmid({
            "addgene_id": "17448",
            "name": "pLenti-CMV-GFP-Puro",
            "url": "https://www.addgene.org/17448/",
            "pubmed_id": "19657394",
            "article_title": "High-efficiency transduction",
            "depositor": "Eric Bhatt",
        })
        return t

    def test_comment_contains_backbone(self):
        t = self._tracker_with_refs()
        c = t.format_genbank_comment()
        assert "pcDNA3.1(+)" in c
        assert "V79020" in c

    def test_comment_contains_insert(self):
        t = self._tracker_with_refs()
        c = t.format_genbank_comment()
        assert "EGFP" in c
        assert "U55762" in c

    def test_comment_contains_pubmed(self):
        t = self._tracker_with_refs()
        c = t.format_genbank_comment()
        assert "19657394" in c
        assert "pubmed.ncbi.nlm.nih.gov" in c

    def test_comment_contains_doi_and_depositor(self):
        t = self._tracker_with_refs()
        c = t.format_genbank_comment()
        assert "Eric Bhatt" in c
        assert "High-efficiency transduction" in c

    def test_empty_tracker_returns_empty(self):
        t = ReferenceTracker()
        assert t.format_genbank_comment() == ""

    def test_comment_in_genbank_output(self):
        t = self._tracker_with_refs()
        comment = t.format_genbank_comment()
        gbk = format_as_genbank(
            sequence="ATGCATGC" * 100,
            name="test_construct",
            backbone_name="pcDNA3.1(+)",
            insert_name="EGFP",
            insert_position=10,
            insert_length=50,
            comment=comment,
        )
        assert "COMMENT" in gbk
        assert "pcDNA3.1(+)" in gbk
        assert "EGFP" in gbk
        assert "19657394" in gbk
        assert "LOCUS" in gbk
        assert "FEATURES" in gbk
