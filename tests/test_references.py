#!/usr/bin/env python3
"""Tests for src/references.py — Reference and ReferenceTracker."""

import pytest

from src.references import Reference, ReferenceTracker


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _backbone_library(**overrides) -> dict:
    """Minimal backbone dict matching library/backbones.json shape."""
    base = {
        "id": "pcDNA3.1(+)",
        "name": "pcDNA3.1(+)",
        "source": "Thermo Fisher (Invitrogen)",
        "organism": "mammalian",
        "genbank_accession": None,
        "addgene_id": None,
    }
    base.update(overrides)
    return base


def _insert_library(**overrides) -> dict:
    """Minimal insert dict matching library/inserts.json shape."""
    base = {
        "id": "EGFP",
        "name": "Enhanced Green Fluorescent Protein (EGFP)",
        "genbank_accession": "U55762",
        "organism_source": "Aequorea victoria (synthetic codon-optimized)",
        "category": "fluorescent_protein",
    }
    base.update(overrides)
    return base


def _ncbi_gene_result(**overrides) -> dict:
    """Minimal dict returned by ncbi_integration.fetch_gene_sequence."""
    base = {
        "sequence": "ATGCCC",
        "symbol": "TP53",
        "organism": "Homo sapiens",
        "accession": "NM_000546.6",
        "length": 1182,
        "full_name": "tumor protein p53",
        "gene_id": "7157",
    }
    base.update(overrides)
    return base


def _addgene_plasmid(**overrides) -> dict:
    """Minimal dict representing an AddgenePlasmid (or its asdict form)."""
    base = {
        "addgene_id": "50005",
        "name": "pSpCas9(BB)-2A-Puro (PX459)",
        "depositor": "Feng Zhang",
        "pubmed_id": "24157548",
        "article_title": "Multiplex genome engineering using CRISPR/Cas systems",
        "url": "https://www.addgene.org/50005/",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# add_backbone
# ---------------------------------------------------------------------------

class TestAddBackbone:
    def test_library_backbone_no_accession_no_addgene(self):
        tracker = ReferenceTracker()
        tracker.add_backbone(_backbone_library())
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "library"
        assert refs[0]["identifier"] == "pcDNA3.1(+)"
        assert refs[0]["component_type"] == "backbone"
        assert refs[0]["url"] is None

    def test_backbone_with_addgene_id(self):
        tracker = ReferenceTracker()
        tracker.add_backbone(_backbone_library(addgene_id="52535", name="pLenti-EF1a"))
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "addgene"
        assert refs[0]["identifier"] == "52535"
        assert refs[0]["url"] == "https://www.addgene.org/52535/"

    def test_backbone_with_genbank_accession(self):
        tracker = ReferenceTracker()
        tracker.add_backbone(_backbone_library(genbank_accession="U55762"))
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "ncbi"
        assert refs[0]["accession"] == "U55762"
        assert refs[0]["url"] == "https://www.ncbi.nlm.nih.gov/nuccore/U55762"

    def test_addgene_id_takes_priority_over_genbank(self):
        """When both addgene_id and genbank_accession exist, addgene wins."""
        tracker = ReferenceTracker()
        tracker.add_backbone(_backbone_library(
            addgene_id="12345", genbank_accession="U55762",
        ))
        refs = tracker.to_list()
        assert refs[0]["source"] == "addgene"


# ---------------------------------------------------------------------------
# add_insert
# ---------------------------------------------------------------------------

class TestAddInsert:
    def test_insert_with_genbank(self):
        tracker = ReferenceTracker()
        tracker.add_insert(_insert_library())
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "ncbi"
        assert refs[0]["accession"] == "U55762"
        assert refs[0]["url"] == "https://www.ncbi.nlm.nih.gov/nuccore/U55762"
        assert refs[0]["organism"] == "Aequorea victoria (synthetic codon-optimized)"

    def test_insert_without_genbank(self):
        tracker = ReferenceTracker()
        tracker.add_insert(_insert_library(genbank_accession=None))
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "library"
        assert refs[0]["identifier"] == "EGFP"


# ---------------------------------------------------------------------------
# add_ncbi_gene
# ---------------------------------------------------------------------------

class TestAddNcbiGene:
    def test_with_gene_id(self):
        tracker = ReferenceTracker()
        tracker.add_ncbi_gene(_ncbi_gene_result())
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "ncbi"
        assert refs[0]["identifier"] == "7157"
        assert refs[0]["url"] == "https://www.ncbi.nlm.nih.gov/gene/7157"
        assert refs[0]["accession"] == "NM_000546.6"
        assert refs[0]["organism"] == "Homo sapiens"

    def test_without_gene_id_uses_accession(self):
        tracker = ReferenceTracker()
        tracker.add_ncbi_gene(_ncbi_gene_result(gene_id=None))
        refs = tracker.to_list()
        assert refs[0]["identifier"] == "NM_000546.6"
        assert refs[0]["url"] == "https://www.ncbi.nlm.nih.gov/nuccore/NM_000546.6"

    def test_without_gene_id_or_accession(self):
        tracker = ReferenceTracker()
        tracker.add_ncbi_gene(_ncbi_gene_result(gene_id=None, accession=None))
        refs = tracker.to_list()
        assert refs[0]["identifier"] == "TP53"
        assert refs[0]["url"] is None


# ---------------------------------------------------------------------------
# add_addgene_plasmid
# ---------------------------------------------------------------------------

class TestAddAddgenePlasmid:
    def test_full_plasmid(self):
        tracker = ReferenceTracker()
        tracker.add_addgene_plasmid(_addgene_plasmid())
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "addgene"
        assert refs[0]["identifier"] == "50005"
        assert refs[0]["depositor"] == "Feng Zhang"
        assert refs[0]["pubmed_id"] == "24157548"
        assert refs[0]["article_title"] is not None
        assert refs[0]["url"] == "https://www.addgene.org/50005/"

    def test_plasmid_without_url_constructs_one(self):
        tracker = ReferenceTracker()
        tracker.add_addgene_plasmid(_addgene_plasmid(url=None))
        refs = tracker.to_list()
        assert refs[0]["url"] == "https://www.addgene.org/50005/"

    def test_plasmid_without_optional_fields(self):
        tracker = ReferenceTracker()
        tracker.add_addgene_plasmid({
            "addgene_id": "99999",
            "name": "pTest",
        })
        refs = tracker.to_list()
        assert refs[0]["depositor"] is None
        assert refs[0]["pubmed_id"] is None


# ---------------------------------------------------------------------------
# add_custom
# ---------------------------------------------------------------------------

class TestAddCustom:
    def test_user_provided(self):
        tracker = ReferenceTracker()
        tracker.add_custom("MyInsert", "A custom CDS provided by the user")
        refs = tracker.to_list()
        assert len(refs) == 1
        assert refs[0]["source"] == "user_provided"
        assert refs[0]["identifier"] == "MyInsert"
        assert refs[0]["name"] == "MyInsert"
        assert refs[0]["component_type"] == "insert"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_source_and_identifier_not_duplicated(self):
        tracker = ReferenceTracker()
        tracker.add_ncbi_gene(_ncbi_gene_result())
        tracker.add_ncbi_gene(_ncbi_gene_result())
        assert len(tracker.to_list()) == 1

    def test_different_sources_same_identifier_kept(self):
        tracker = ReferenceTracker()
        tracker.add_insert(_insert_library())  # source=ncbi, identifier=U55762
        tracker.add_backbone(_backbone_library(
            genbank_accession="U55762",
        ))  # source=ncbi, identifier=U55762 — same key, deduped
        assert len(tracker.to_list()) == 1

    def test_same_source_different_identifiers_kept(self):
        tracker = ReferenceTracker()
        tracker.add_ncbi_gene(_ncbi_gene_result(gene_id="7157"))
        tracker.add_ncbi_gene(_ncbi_gene_result(gene_id="672", symbol="BRCA1"))
        assert len(tracker.to_list()) == 2


# ---------------------------------------------------------------------------
# format_references
# ---------------------------------------------------------------------------

class TestFormatReferences:
    def test_empty_tracker(self):
        tracker = ReferenceTracker()
        assert tracker.format_references() == ""

    def test_contains_header(self):
        tracker = ReferenceTracker()
        tracker.add_backbone(_backbone_library())
        output = tracker.format_references()
        assert output.startswith("## References")

    def test_library_backbone_format(self):
        tracker = ReferenceTracker()
        tracker.add_backbone(_backbone_library())
        output = tracker.format_references()
        assert "**Library:**" in output
        assert "- pcDNA3.1(+)" in output

    def test_ncbi_gene_format(self):
        tracker = ReferenceTracker()
        tracker.add_ncbi_gene(_ncbi_gene_result())
        output = tracker.format_references()
        assert "**NCBI:**" in output
        assert "TP53" in output
        assert "NM_000546.6" in output
        assert "https://www.ncbi.nlm.nih.gov/gene/7157" in output

    def test_addgene_format(self):
        tracker = ReferenceTracker()
        tracker.add_addgene_plasmid(_addgene_plasmid())
        output = tracker.format_references()
        assert "**Addgene:**" in output
        assert "Addgene #50005" in output
        assert "Feng Zhang" in output
        assert "PMID: 24157548" in output
        assert "https://www.addgene.org/50005/" in output

    def test_user_provided_format(self):
        tracker = ReferenceTracker()
        tracker.add_custom("MySeq", "custom")
        output = tracker.format_references()
        assert "**User-Provided:**" in output
        assert "User-provided sequence" in output

    def test_multiple_sources_grouped(self):
        tracker = ReferenceTracker()
        tracker.add_backbone(_backbone_library())  # library
        tracker.add_ncbi_gene(_ncbi_gene_result())  # ncbi
        tracker.add_addgene_plasmid(_addgene_plasmid())  # addgene
        tracker.add_custom("MySeq", "custom")  # user_provided
        output = tracker.format_references()

        # Check ordering: library before ncbi before addgene before user_provided
        lib_pos = output.index("**Library:**")
        ncbi_pos = output.index("**NCBI:**")
        addgene_pos = output.index("**Addgene:**")
        user_pos = output.index("**User-Provided:**")
        assert lib_pos < ncbi_pos < addgene_pos < user_pos

    def test_addgene_pubmed_only(self):
        """Addgene ref with pubmed_id but no article_title."""
        tracker = ReferenceTracker()
        tracker.add_addgene_plasmid(_addgene_plasmid(article_title=None))
        output = tracker.format_references()
        assert "PMID: 24157548" in output
        assert "Publication:" not in output


# ---------------------------------------------------------------------------
# to_list
# ---------------------------------------------------------------------------

class TestToList:
    def test_returns_plain_dicts(self):
        tracker = ReferenceTracker()
        tracker.add_ncbi_gene(_ncbi_gene_result())
        result = tracker.to_list()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert result[0]["source"] == "ncbi"

    def test_empty_tracker_returns_empty_list(self):
        tracker = ReferenceTracker()
        assert tracker.to_list() == []

    def test_all_fields_present(self):
        tracker = ReferenceTracker()
        tracker.add_addgene_plasmid(_addgene_plasmid())
        d = tracker.to_list()[0]
        expected_keys = {
            "source", "identifier", "name", "component_type", "url",
            "organism", "accession", "pubmed_id", "article_title", "depositor",
        }
        assert set(d.keys()) == expected_keys
