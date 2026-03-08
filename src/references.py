#!/usr/bin/env python3
"""
Reference Tracker Module

Accumulates source references as the plasmid design agent retrieves
sequences from external sources (NCBI, Addgene) or the local curated
library.  Produces a formatted "References" section for the final
construct summary.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Reference:
    """A single source reference for a plasmid component."""

    source: str  # "library", "ncbi", "addgene", "user_provided"
    identifier: str  # GenBank accession, Addgene ID, gene ID, or insert/backbone ID
    name: str  # Human-readable name (e.g., "EGFP", "pcDNA3.1(+)")
    component_type: str  # "backbone" or "insert"
    url: Optional[str] = None
    organism: Optional[str] = None
    accession: Optional[str] = None  # GenBank/RefSeq accession
    pubmed_id: Optional[str] = None
    article_title: Optional[str] = None
    depositor: Optional[str] = None


class ReferenceTracker:
    """Accumulates and formats source references for plasmid components."""

    def __init__(self) -> None:
        self._references: list[Reference] = []
        self._seen: set[tuple[str, str]] = set()  # (source, identifier)

    def _add(self, ref: Reference) -> None:
        """Add a reference, deduplicating by (source, identifier)."""
        key = (ref.source, ref.identifier)
        if key in self._seen:
            return
        self._seen.add(key)
        self._references.append(ref)

    # ------------------------------------------------------------------
    # Public add helpers
    # ------------------------------------------------------------------

    def add_backbone(self, backbone: dict) -> None:
        """Extract reference info from a backbone dict (library or Addgene-sourced)."""
        addgene_id = backbone.get("addgene_id")
        genbank_acc = backbone.get("genbank_accession")

        if addgene_id:
            source = "addgene"
            identifier = str(addgene_id)
            url = f"https://www.addgene.org/{addgene_id}/"
        elif genbank_acc:
            source = "ncbi"
            identifier = genbank_acc
            url = f"https://www.ncbi.nlm.nih.gov/nuccore/{genbank_acc}"
        else:
            source = "library"
            identifier = backbone.get("id", backbone.get("name", "unknown"))
            url = None

        self._add(Reference(
            source=source,
            identifier=identifier,
            name=backbone.get("name", identifier),
            component_type="backbone",
            url=url,
            organism=backbone.get("organism"),
            accession=genbank_acc,
        ))

    def add_insert(self, insert: dict) -> None:
        """Extract reference info from an insert dict (library entry)."""
        genbank_acc = insert.get("genbank_accession")

        if genbank_acc:
            source = "ncbi"
            identifier = genbank_acc
            url = f"https://www.ncbi.nlm.nih.gov/nuccore/{genbank_acc}"
        else:
            source = "library"
            identifier = insert.get("id", insert.get("name", "unknown"))
            url = None

        self._add(Reference(
            source=source,
            identifier=identifier,
            name=insert.get("name", identifier),
            component_type="insert",
            url=url,
            organism=insert.get("organism_source"),
            accession=genbank_acc,
        ))

    def add_ncbi_gene(self, gene_result: dict) -> None:
        """Extract reference info from a fetch_gene_sequence result."""
        gene_id = gene_result.get("gene_id")
        accession = gene_result.get("accession")

        if gene_id:
            url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"
            identifier = str(gene_id)
        elif accession:
            url = f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}"
            identifier = accession
        else:
            url = None
            identifier = gene_result.get("symbol", "unknown")

        self._add(Reference(
            source="ncbi",
            identifier=identifier,
            name=gene_result.get("symbol") or gene_result.get("full_name", identifier),
            component_type="insert",
            url=url,
            organism=gene_result.get("organism"),
            accession=accession,
        ))

    def add_addgene_plasmid(self, plasmid: dict) -> None:
        """Extract reference from an AddgenePlasmid (or its dict representation)."""
        addgene_id = str(plasmid.get("addgene_id", ""))
        url = plasmid.get("url") or f"https://www.addgene.org/{addgene_id}/"

        self._add(Reference(
            source="addgene",
            identifier=addgene_id,
            name=plasmid.get("name", f"Addgene #{addgene_id}"),
            component_type="backbone",
            url=url,
            depositor=plasmid.get("depositor"),
            pubmed_id=plasmid.get("pubmed_id"),
            article_title=plasmid.get("article_title"),
        ))

    def add_custom(self, name: str, description: str) -> None:
        """Record a user-provided sequence."""
        self._add(Reference(
            source="user_provided",
            identifier=name,
            name=name,
            component_type="insert",
        ))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def format_references(self) -> str:
        """Return a formatted text block grouped by source type."""
        if not self._references:
            return ""

        groups: dict[str, list[Reference]] = {}
        for ref in self._references:
            groups.setdefault(ref.source, []).append(ref)

        # Ordered display
        section_order = [
            ("library", "Library"),
            ("ncbi", "NCBI"),
            ("addgene", "Addgene"),
            ("user_provided", "User-Provided"),
        ]

        lines: list[str] = ["## References", ""]
        for key, heading in section_order:
            refs = groups.get(key)
            if not refs:
                continue
            lines.append(f"**{heading}:**")
            for ref in refs:
                lines.append(self._format_single(ref))
            lines.append("")

        return "\n".join(lines).rstrip()

    @staticmethod
    def _format_single(ref: Reference) -> str:
        """Format a single reference entry."""
        parts: list[str] = [f"- {ref.name}"]

        if ref.source == "library":
            pass  # name is sufficient

        elif ref.source == "ncbi":
            if ref.accession:
                parts[0] += f" ({ref.accession})"
            if ref.organism:
                parts[0] += f" — {ref.organism}"
            if ref.url:
                parts.append(f"  {ref.url}")

        elif ref.source == "addgene":
            parts[0] += f" (Addgene #{ref.identifier})"
            if ref.depositor:
                parts[0] += f" — Depositor: {ref.depositor}"
            if ref.article_title and ref.pubmed_id:
                parts.append(
                    f'  Publication: "{ref.article_title}" (PMID: {ref.pubmed_id})'
                )
            elif ref.pubmed_id:
                parts.append(f"  PMID: {ref.pubmed_id}")
            if ref.url:
                parts.append(f"  {ref.url}")

        elif ref.source == "user_provided":
            parts[0] += " — User-provided sequence"

        return "\n".join(parts)

    def to_list(self) -> list[dict]:
        """Return references as a list of plain dicts."""
        return [asdict(ref) for ref in self._references]
