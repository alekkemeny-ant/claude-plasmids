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
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Biopython Entrez
try:
    from Bio import Entrez, SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Configure Entrez email + optional API key (required by NCBI)
# Without an API key: max 3 req/s. With NCBI_API_KEY: max 10 req/s.
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "plasmid-designer@example.com")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
if BIOPYTHON_AVAILABLE:
    Entrez.email = NCBI_EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY

# ── Rate limiting ──
# NCBI policy: 3 req/s without API key, 10 req/s with.
# Simple token-bucket: track last request time, sleep if needed.
_RATE_LIMIT_INTERVAL = 0.11 if NCBI_API_KEY else 0.34  # seconds between requests
_last_request_time = 0.0
_rate_lock = threading.Lock()


def _rate_limit():
    """Enforce NCBI rate limit. Call before each Entrez request."""
    global _last_request_time
    with _rate_lock:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < _RATE_LIMIT_INTERVAL:
            time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
        _last_request_time = time.monotonic()


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
        _rate_limit()
        handle = Entrez.esearch(db="gene", term=search_term, retmax=10, sort="relevance")
        record = Entrez.read(handle)
        handle.close()

        gene_ids = record.get("IdList", [])
        if not gene_ids:
            # Try broader search without [Gene Name] qualifier
            _rate_limit()
            handle = Entrez.esearch(db="gene", term=query if not organism else f"{query} AND {org_map.get(organism.lower(), organism) if organism else ''}[Organism]", retmax=10)
            record = Entrez.read(handle)
            handle.close()
            gene_ids = record.get("IdList", [])

        if not gene_ids:
            return []

        # Fetch gene summaries
        _rate_limit()
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

        # Ambiguity check: if no organism was specified and the search
        # returned multiple distinct species, DO NOT auto-pick the first.
        # Return a disambiguation signal so the caller can ask the user.
        if not organism and len(results) > 1:
            organisms_seen = {r.get("organism", "") for r in results if r.get("organism")}
            if len(organisms_seen) > 1:
                return {
                    "needs_disambiguation": True,
                    "reason": "multiple_species",
                    "query": gene_symbol,
                    "options": [
                        {
                            "gene_id": r["gene_id"],
                            "symbol": r["symbol"],
                            "organism": r.get("organism", ""),
                            "full_name": r.get("full_name", ""),
                        }
                        for r in results[:10]
                    ],
                }

        gene_id = results[0]["gene_id"]

    if not gene_id:
        return None

    try:
        # Fetch gene record to get RefSeq mRNA accession
        _rate_limit()
        handle = Entrez.efetch(db="gene", id=gene_id, rettype="gene_table", retmode="text")
        gene_text = handle.read()
        handle.close()

        # Find RefSeq mRNA accession (NM_ preferred)
        nm_accessions = re.findall(r'(NM_\d+(?:\.\d+)?)', gene_text)

        if not nm_accessions:
            # Try linking gene to nucleotide
            _rate_limit()
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
                _rate_limit()
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
        _rate_limit()
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


def fetch_genomic_upstream(
    gene_id: str,
    bp_upstream: int = 2000,
) -> Optional[dict]:
    """Fetch the genomic sequence upstream of a gene's transcription start.

    Retrieves the ~bp_upstream nucleotides 5' of the gene from NCBI
    genomic records. This is the NATIVE regulatory region — may include
    enhancers, silencers, and other elements beyond the core promoter.

    Args:
        gene_id: NCBI Gene ID (e.g., "7157" for human TP53)
        bp_upstream: How many bp upstream of TSS to fetch (default 2000)

    Returns:
        {
            "sequence": str,           # upstream genomic DNA (sense strand relative to gene)
            "gene_symbol": str,
            "gene_id": str,
            "organism": str,
            "chromosome_accession": str,
            "strand": str,             # "+" or "-" (gene's strand in genomic)
            "length": int,
            "source": "NCBI genomic (native upstream region)",
            "warning": (
                "This is the native regulatory region, not a validated "
                "minimal promoter. It may include enhancers, silencers, "
                "or other elements. Expression strength/specificity is "
                "unpredictable. Recommend literature search for a "
                "characterized minimal promoter if available."
            ),
        }
        or None if the gene cannot be found or genomic coordinates
        cannot be resolved.
    """
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError("Biopython is required for NCBI integration.")

    if bp_upstream < 100 or bp_upstream > 10000:
        raise ValueError(f"bp_upstream must be between 100 and 10000, got {bp_upstream}")

    try:
        # ── Step 1: Get gene's genomic coordinates via gene_table ──
        _rate_limit()
        handle = Entrez.efetch(db="gene", id=gene_id, rettype="gene_table", retmode="text")
        gene_table = handle.read()
        handle.close()

        # gene_table format includes lines like:
        # "Gene: TP53, Homo sapiens"
        # and a header row with the genomic accession + coordinates, e.g.:
        # "Reference TP53 assembly ... NC_000017.11 ... complement"
        # followed by exon coordinate rows.

        # Extract gene symbol and organism
        symbol = ""
        organism = ""
        m_gene = re.search(r"^Gene:\s+(\S+)(?:,\s+(.+))?$", gene_table, re.MULTILINE)
        if m_gene:
            symbol = m_gene.group(1)
            organism = (m_gene.group(2) or "").strip()

        # Find the chromosome accession (NC_ for assembled chromosome)
        m_nc = re.search(r"(NC_\d+(?:\.\d+)?)", gene_table)
        if not m_nc:
            logger.warning(f"No NC_ genomic accession found for gene {gene_id}")
            return None
        chrom_accession = m_nc.group(1)

        # Detect strand — gene_table says "complement" if on minus strand
        minus_strand = "complement" in gene_table.lower()

        # Extract coordinates. gene_table has lines like:
        #   "mRNA    NM_000546.6: ..."
        #   followed by coordinate blocks. The cleanest way is to look for
        #   all numbers in coordinate lines and take the smallest (plus
        #   strand TSS) or largest (minus strand TSS). But gene_table
        #   format varies — use a permissive extraction.
        #
        # Look for coordinate pairs in lines with tab-separated numbers.
        # Pattern: whitespace + digits + whitespace + digits (exon bounds)
        coord_nums = re.findall(r"^\s*(\d{3,})\s+(\d{3,})\s*$", gene_table, re.MULTILINE)
        if not coord_nums:
            # Try alternative pattern — sometimes coordinates are in
            # different format. Look for any large numbers (>1000, typical
            # genomic coords) after the NC_ accession line.
            after_nc = gene_table[gene_table.index(chrom_accession):]
            all_nums = [int(n) for n in re.findall(r"\b(\d{4,})\b", after_nc)]
            if not all_nums:
                logger.warning(f"No genomic coordinates found for gene {gene_id}")
                return None
            coords_flat = all_nums
        else:
            coords_flat = [int(n) for pair in coord_nums for n in pair]

        # TSS proxy: smallest coordinate for +strand, largest for -strand
        if minus_strand:
            tss = max(coords_flat)
            # Upstream on minus strand = HIGHER genomic coords
            seq_start = tss + 1
            seq_stop = tss + bp_upstream
        else:
            tss = min(coords_flat)
            # Upstream on plus strand = LOWER genomic coords
            seq_start = max(1, tss - bp_upstream)
            seq_stop = tss - 1

        # ── Step 2: Fetch the upstream region from the chromosome ──
        _rate_limit()
        handle = Entrez.efetch(
            db="nucleotide",
            id=chrom_accession,
            rettype="fasta",
            retmode="text",
            seq_start=seq_start,
            seq_stop=seq_stop,
        )
        record = SeqIO.read(handle, "fasta")
        handle.close()

        upstream_seq = str(record.seq).upper()

        # For minus-strand genes, reverse-complement so the sequence is
        # oriented 5'→3' relative to the gene (i.e., promoter is 5' of ATG).
        if minus_strand:
            complement = str.maketrans("ACGT", "TGCA")
            upstream_seq = upstream_seq.translate(complement)[::-1]

        return {
            "sequence": upstream_seq,
            "gene_symbol": symbol or f"GeneID_{gene_id}",
            "gene_id": gene_id,
            "organism": organism,
            "chromosome_accession": chrom_accession,
            "strand": "-" if minus_strand else "+",
            "length": len(upstream_seq),
            "source": "NCBI genomic (native upstream region)",
            "warning": (
                "This is the native regulatory region, not a validated "
                "minimal promoter. It may include enhancers, silencers, "
                "or other elements. Expression strength/specificity is "
                "unpredictable. Recommend literature search for a "
                "characterized minimal promoter if available."
            ),
        }

    except Exception as e:
        logger.error(f"NCBI genomic upstream fetch error for gene_id={gene_id}: {e}")
        return None
