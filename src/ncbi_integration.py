#!/usr/bin/env python3
"""
NCBI Gene Integration Module

Provides gene search and CDS (coding sequence) retrieval from NCBI
using the Biopython Entrez module. Email-only access, no API key required.

Key functions:
- search_gene: Search NCBI Gene DB by symbol/name
- fetch_gene_sequence: Get CDS DNA sequence from RefSeq mRNA
- fetch_sequence_by_accession: Direct fetch by NM_/NR_ accession
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Biopython Entrez
try:
    from Bio import Entrez, SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Configure Entrez email (required by NCBI)
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "plasmid-designer@example.com")
if BIOPYTHON_AVAILABLE:
    Entrez.email = NCBI_EMAIL


def search_gene(
    query: str,
    organism: Optional[str] = None,
) -> list[dict]:
    """Search NCBI Gene database by gene symbol or name.

    Args:
        query: Gene symbol or name (e.g., "TP53", "EGFP", "MyD88")
        organism: Optional organism filter (e.g., "human", "mouse", "Homo sapiens")

    Returns:
        List of dicts with gene_id, symbol, full_name, organism, aliases
    """
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError("Biopython is required for NCBI integration. Install with: pip install biopython")

    # Build search term
    search_term = query
    if organism:
        # Normalize common organism names
        org_map = {
            "human": "Homo sapiens",
            "mouse": "Mus musculus",
            "rat": "Rattus norvegicus",
            "zebrafish": "Danio rerio",
            "fly": "Drosophila melanogaster",
            "worm": "Caenorhabditis elegans",
            "yeast": "Saccharomyces cerevisiae",
            "chicken": "Gallus gallus",
            "dog": "Canis lupus familiaris",
            "pig": "Sus scrofa",
        }
        org_name = org_map.get(organism.lower(), organism)
        search_term = f"{query}[Gene Name] AND {org_name}[Organism]"
    else:
        search_term = f"{query}[Gene Name]"

    try:
        handle = Entrez.esearch(db="gene", term=search_term, retmax=10, sort="relevance")
        record = Entrez.read(handle)
        handle.close()

        gene_ids = record.get("IdList", [])
        if not gene_ids:
            # Try broader search without [Gene Name] qualifier
            handle = Entrez.esearch(db="gene", term=query if not organism else f"{query} AND {org_map.get(organism.lower(), organism) if organism else ''}[Organism]", retmax=10)
            record = Entrez.read(handle)
            handle.close()
            gene_ids = record.get("IdList", [])

        if not gene_ids:
            return []

        # Fetch gene summaries
        handle = Entrez.esummary(db="gene", id=",".join(gene_ids))
        summaries = Entrez.read(handle)
        handle.close()

        results = []
        doc_sums = summaries.get("DocumentSummarySet", {}).get("DocumentSummary", [])
        for doc in doc_sums:
            gene_id = str(doc.attributes.get("uid", "")) if hasattr(doc, "attributes") else ""
            result = {
                "gene_id": gene_id,
                "symbol": str(doc.get("Name", "")),
                "full_name": str(doc.get("Description", "")),
                "organism": str(doc.get("Organism", {}).get("ScientificName", "")) if isinstance(doc.get("Organism"), dict) else str(doc.get("Organism", "")),
                "aliases": str(doc.get("OtherAliases", "")),
                "summary": str(doc.get("Summary", ""))[:200],
            }
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"NCBI gene search error for '{query}': {e}")
        return []


def fetch_gene_sequence(
    gene_id: Optional[str] = None,
    gene_symbol: Optional[str] = None,
    organism: Optional[str] = None,
) -> Optional[dict]:
    """Fetch the CDS (coding sequence) for a gene from NCBI RefSeq.

    Retrieves the mRNA transcript and extracts the CDS region.

    Args:
        gene_id: NCBI Gene ID (e.g., "7157" for human TP53)
        gene_symbol: Gene symbol (e.g., "TP53") — used with organism to find the gene
        organism: Organism name (e.g., "human", "Homo sapiens")

    Returns:
        Dict with sequence, symbol, organism, accession, length, full_name
        or None if not found.
    """
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError("Biopython is required for NCBI integration. Install with: pip install biopython")

    # If we have a symbol but no gene_id, search first
    if not gene_id and gene_symbol:
        results = search_gene(gene_symbol, organism)
        if not results:
            return None
        gene_id = results[0]["gene_id"]

    if not gene_id:
        return None

    try:
        # Fetch gene record to get RefSeq mRNA accession
        handle = Entrez.efetch(db="gene", id=gene_id, rettype="gene_table", retmode="text")
        gene_text = handle.read()
        handle.close()

        # Find RefSeq mRNA accession (NM_ preferred)
        nm_accessions = re.findall(r'(NM_\d+(?:\.\d+)?)', gene_text)

        if not nm_accessions:
            # Try linking gene to nucleotide
            handle = Entrez.elink(dbfrom="gene", db="nucleotide", id=gene_id, linkname="gene_nuccore_refseqrna")
            link_results = Entrez.read(handle)
            handle.close()

            nuc_ids = []
            for linkset in link_results:
                for link_db in linkset.get("LinkSetDb", []):
                    for link in link_db.get("Link", []):
                        nuc_ids.append(link["Id"])

            if nuc_ids:
                # Fetch the first nucleotide record to get the accession
                handle = Entrez.esummary(db="nucleotide", id=nuc_ids[0])
                nuc_summary = Entrez.read(handle)
                handle.close()
                if nuc_summary:
                    acc = str(nuc_summary[0].get("AccessionVersion", ""))
                    if acc.startswith("NM_"):
                        nm_accessions = [acc]

        if not nm_accessions:
            logger.warning(f"No RefSeq mRNA found for gene ID {gene_id}")
            return None

        # Fetch the mRNA sequence and extract CDS
        return fetch_sequence_by_accession(nm_accessions[0])

    except Exception as e:
        logger.error(f"NCBI gene fetch error for gene_id={gene_id}: {e}")
        return None


def fetch_sequence_by_accession(accession: str) -> Optional[dict]:
    """Fetch a sequence by its NCBI accession number and extract the CDS.

    Args:
        accession: RefSeq accession (e.g., "NM_000546.6" for human TP53)

    Returns:
        Dict with sequence (CDS only), symbol, organism, accession, length, full_name
        or None if not found.
    """
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError("Biopython is required for NCBI integration. Install with: pip install biopython")

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        # Extract gene symbol and organism
        symbol = ""
        full_name = ""
        organism = str(record.annotations.get("organism", ""))

        # Find the CDS feature
        cds_sequence = None
        for feature in record.features:
            if feature.type == "gene" and not symbol:
                symbol = feature.qualifiers.get("gene", [""])[0]
            if feature.type == "CDS":
                # Extract CDS sequence
                cds_sequence = str(feature.extract(record.seq)).upper()
                if not symbol:
                    symbol = feature.qualifiers.get("gene", [""])[0]
                full_name = feature.qualifiers.get("product", [""])[0]
                break

        if not cds_sequence:
            # No CDS found, return the full mRNA sequence
            cds_sequence = str(record.seq).upper()
            logger.warning(f"No CDS feature found in {accession}, returning full sequence")

        return {
            "sequence": cds_sequence,
            "symbol": symbol,
            "organism": organism,
            "accession": accession,
            "length": len(cds_sequence),
            "full_name": full_name or str(record.description),
        }

    except Exception as e:
        logger.error(f"NCBI accession fetch error for {accession}: {e}")
        return None
