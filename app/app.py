#!/usr/bin/env python3
"""
Plasmid Design Agent — Web UI

A chat interface for the Claude-powered plasmid design agent.
Uses the Anthropic API with tool-use to orchestrate the MCP tools locally.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python app.py
    # Open http://localhost:8000 in your browser
"""

import csv
import io
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
import threading
import uuid
import time

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env") 
except ImportError:
    pass  # dotenv not installed; rely on environment variables

import anthropic

# Add src/ to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assembler import (
    assemble_construct as _assemble_construct,
    fuse_sequences as _fuse_sequences,
    assemble_golden_gate as _assemble_golden_gate,
    GG_ENZYMES,
    reverse_complement,
    find_mcs_insertion_point,
    resolve_insertion_point,
    clean_sequence,
    validate_dna,
    format_as_fasta,
    format_as_genbank,
    export_genbank_with_plot,
)

# Stores plot JSON from the most recent genbank export, read by the SSE handler.
# Uses threading.local() so concurrent SSE requests don't see each other's plots.
_thread_local = threading.local()


def _get_last_plot_json() -> Optional[str]:
    return getattr(_thread_local, "last_plot_json", None)


def _set_last_plot_json(val: Optional[str]):
    _thread_local.last_plot_json = val

try:
    from ncbi_integration import (
        search_gene as _search_gene_fn,
        fetch_gene_sequence as _fetch_gene_fn,
    )
    NCBI_AVAILABLE = True
except ImportError:
    NCBI_AVAILABLE = False
from library import (
    get_backbone_by_id,
    get_insert_by_id,
    search_backbones,
    search_inserts,
    search_all_sources as _search_all_sources,
    get_all_backbones,
    get_all_inserts,
    validate_dna_sequence,
    format_backbone_summary,
    format_insert_summary,
    infer_species_from_cell_line,
)

try:
    from addgene_integration import (
        search_addgene as _search_addgene,
        get_addgene_plasmid as _get_addgene_plasmid,
        AddgeneLibraryIntegration,
    )
    ADDGENE_AVAILABLE = True
except ImportError:
    ADDGENE_AVAILABLE = False

try:
    from fpbase_integration import (
        search_fpbase as _search_fpbase,
        fetch_fpbase_sequence as _fetch_fpbase,
    )
    FPBASE_AVAILABLE = True
except ImportError:
    FPBASE_AVAILABLE = False

# ── Phase-2 advanced design modules (Design Confidence, Protein Analysis,
#    Mutations, Bespoke Promoters). These are new modules that may not exist
#    in every deployment — guard each import so the app still loads.

try:
    from confidence import compute_confidence, format_confidence_report
    CONFIDENCE_AVAILABLE = True
except ImportError:
    CONFIDENCE_AVAILABLE = False

try:
    from protein_analysis import translate as _translate_dna, find_fusion_sites as _find_fusion_sites
    PROTEIN_ANALYSIS_AVAILABLE = True
except ImportError:
    PROTEIN_ANALYSIS_AVAILABLE = False

try:
    from mutations import (
        lookup_known_mutations as _lookup_known_mutations,
        apply_point_mutation as _apply_point_mutation,
        design_premature_stop as _design_premature_stop,
        parse_mutation_notation as _parse_mutation_notation,
    )
    MUTATIONS_AVAILABLE = True
except ImportError:
    MUTATIONS_AVAILABLE = False

try:
    from ncbi_integration import fetch_genomic_upstream as _fetch_genomic_upstream
    GENOMIC_UPSTREAM_AVAILABLE = True
except ImportError:
    GENOMIC_UPSTREAM_AVAILABLE = False

try:
    from library import is_known_promoter as _is_known_promoter
    PROMOTER_DETECTION_AVAILABLE = True
except ImportError:
    PROMOTER_DETECTION_AVAILABLE = False

from references import ReferenceTracker

logger = logging.getLogger(__name__)

LIBRARY_PATH = PROJECT_ROOT / "library"

# ── Load system prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"  # lives in app/
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""

# ── Tool definitions for the Anthropic API ──────────────────────────────

TOOLS = [
    {
        "name": "search_backbones",
        "description": "Search for plasmid backbone vectors by name, features, or organism.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "organism": {"type": "string", "description": "Filter by organism type", "enum": ["mammalian", "bacterial", "lentiviral_packaging"]},
                "promoter": {"type": "string", "description": "Filter by promoter type"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_backbone",
        "description": "Get complete information about a specific plasmid backbone, including sequence if requested.",
        "input_schema": {
            "type": "object",
            "properties": {
                "backbone_id": {"type": "string", "description": "Backbone ID or name"},
                "include_sequence": {"type": "boolean", "description": "Include full DNA sequence", "default": False},
            },
            "required": ["backbone_id"],
        },
    },
    {
        "name": "search_inserts",
        "description": "Search for insert sequences (fluorescent proteins, tags, reporters) by name or category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "category": {"type": "string", "description": "Filter by category", "enum": ["fluorescent_protein", "reporter", "epitope_tag"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_insert",
        "description": "Get complete information about a specific insert including its DNA sequence. Fallback chain: local library → FPbase (fluorescent proteins) → NCBI Gene. If the gene query is ambiguous across species, returns a disambiguation list instead of guessing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "insert_id": {"type": "string", "description": "Insert ID, gene symbol, or fluorescent protein name"},
                "organism": {"type": "string", "description": "Species for NCBI fallback (e.g., 'human', 'mouse'). Required when the gene exists in multiple species."},
            },
            "required": ["insert_id"],
        },
    },
    {
        "name": "list_all_backbones",
        "description": "List all available backbone plasmids in the library.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_all_inserts",
        "description": "List all available insert sequences in the library.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_insertion_site",
        "description": "Get MCS (multiple cloning site) position info for a backbone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "backbone_id": {"type": "string", "description": "Backbone ID or name"},
            },
            "required": ["backbone_id"],
        },
    },
    {
        "name": "validate_sequence",
        "description": "Validate a DNA sequence and get basic statistics (length, GC content, start/stop codons).",
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA sequence to validate"},
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "assemble_construct",
        "description": (
            "Assemble an expression construct by splicing an insert into a backbone. "
            "Use library IDs to auto-resolve sequences and MCS positions, or provide raw sequences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "backbone_id": {"type": "string", "description": "Backbone ID from library"},
                "insert_id": {"type": "string", "description": "Insert ID from library"},
                "backbone_sequence": {"type": "string", "description": "Raw backbone DNA sequence"},
                "insert_sequence": {"type": "string", "description": "Raw insert DNA sequence"},
                "insertion_position": {"type": "integer", "description": "0-based insertion position"},
                "replace_region_end": {"type": "integer", "description": "End of region to replace"},
                "reverse_complement_insert": {"type": "boolean", "description": "Reverse-complement insert before insertion", "default": False},
            },
        },
    },
    {
        "name": "export_construct",
        "description": "Export an assembled construct sequence in raw, FASTA, or GenBank format.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "Assembled construct DNA sequence"},
                "output_format": {"type": "string", "description": "Output format", "enum": ["raw", "fasta", "genbank"]},
                "construct_name": {"type": "string", "description": "Name for the construct", "default": "construct"},
                "backbone_name": {"type": "string", "description": "Backbone name for annotation", "default": ""},
                "insert_name": {"type": "string", "description": "Insert name for annotation", "default": ""},
                "insert_position": {"type": "integer", "description": "Insert start position", "default": 0},
                "insert_length": {"type": "integer", "description": "Insert length in bp", "default": 0},
                "reverse_complement_insert": {"type": "boolean", "description": "True if insert was inserted in reverse complement orientation", "default": False},
            },
            "required": ["sequence", "output_format"],
        },
    },
    {
        "name": "validate_construct",
        "description": "Validate an assembled construct against ground-truth sequences from the library/NCBI. Checks backbone preservation, insert presence/position/orientation, size, and codons. Prefer passing backbone_id/insert_id (the tool will fetch canonical sequences and verify identity). Only pass insert_sequence directly for custom/fused inserts — in that case identity cannot be verified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "construct_sequence": {"type": "string", "description": "Assembled construct to validate"},
                "backbone_id": {"type": "string", "description": "Backbone library ID (preferred — resolved to canonical sequence)"},
                "insert_id": {"type": "string", "description": "Insert library ID or gene symbol (preferred — resolved to canonical sequence for ground-truth identity check)"},
                "backbone_sequence": {"type": "string", "description": "Raw backbone sequence (only if backbone_id unavailable)"},
                "insert_sequence": {"type": "string", "description": "Raw insert sequence (for custom/fused inserts — identity will NOT be verified)"},
                "expected_insert_position": {"type": "integer", "description": "Expected 0-indexed insert position in the construct"},
            },
            "required": ["construct_sequence"],
        },
    },
    {
        "name": "search_addgene",
        "description": "Search Addgene's plasmid repository. Use when a plasmid is not in the local library.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_addgene_plasmid",
        "description": "Fetch detailed info about a specific Addgene plasmid by catalog number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "addgene_id": {"type": "string", "description": "Addgene catalog number"},
                "include_sequence": {"type": "boolean", "description": "Return full DNA sequence text in the response (default False — sequence is large). Set True only when you need the raw sequence.", "default": False},
            },
            "required": ["addgene_id"],
        },
    },
    {
        "name": "import_addgene_to_library",
        "description": "Import a plasmid from Addgene into the local library.",
        "input_schema": {
            "type": "object",
            "properties": {
                "addgene_id": {"type": "string", "description": "Addgene catalog number"},
                "include_sequence": {"type": "boolean", "description": "Fetch and store sequence", "default": True},
            },
            "required": ["addgene_id"],
        },
    },
    {
        "name": "search_all",
        "description": (
            "Search local library, NCBI Gene, and Addgene concurrently in a single call. "
            "Returns combined results from all sources. Use this as the first search step "
            "when you don't know whether the query is a local insert, an NCBI gene, or an "
            "Addgene plasmid. Faster than calling search_inserts, search_gene, and search_addgene separately."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Gene symbol, plasmid name, or search term"},
                "organism": {"type": "string", "description": "Organism filter (e.g., 'human', 'mouse')"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_gene",
        "description": "Search NCBI Gene database by gene symbol or name. Returns matching genes with IDs, symbols, organisms, and aliases. Use when a user mentions a gene not in the local insert library.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Gene symbol or name (e.g., 'TP53', 'MyD88')"},
                "organism": {"type": "string", "description": "Organism filter (e.g., 'human', 'mouse')"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_fpbase",
        "description": "Search FPbase (fpbase.org) for fluorescent proteins by name. FPbase is the canonical reference for engineered FPs like mRuby, mScarlet, mNeonGreen — these are NOT natural genes and won't be found in NCBI Gene. Use when the user wants an FP not in the local library.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Fluorescent protein name (e.g., 'mRuby', 'mScarlet')"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_cell_line_info",
        "description": "Look up the species for a common cell line name (e.g., HEK293 → human, RAW 264.7 → mouse). Use when the user mentions a cell line but not a species, to infer the likely organism for gene retrieval. IMPORTANT: this infers the cell line's species — the user might still want a gene from a DIFFERENT species. Confirm with the user when the species matters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_line": {"type": "string", "description": "Cell line name (e.g., 'HEK293', 'RAW 264.7', 'NIH3T3')"},
            },
            "required": ["cell_line"],
        },
    },
    {
        "name": "fetch_gene",
        "description": "Fetch the coding DNA sequence (CDS) for a gene from NCBI RefSeq. Returns the CDS sequence, accession, organism, and metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_id": {"type": "string", "description": "NCBI Gene ID (e.g., '7157' for human TP53)"},
                "gene_symbol": {"type": "string", "description": "Gene symbol (e.g., 'TP53')"},
                "organism": {"type": "string", "description": "Organism (e.g., 'human', 'mouse')"},
            },
        },
    },
    {
        "name": "fuse_inserts",
        "description": "Fuse multiple coding sequences into a single CDS for protein tagging or fusion proteins. Handles start/stop codon management at junctions. For protein fusions (H2B-EGFP), the ATG is automatically removed from non-first 'protein' parts — set type='tag' to preserve ATG for small epitope tags (FLAG, HA, Myc). Use for N-terminal tags (FLAG-GeneX), C-terminal tags (GeneX-FLAG), or multi-domain fusions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "inserts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "insert_id": {"type": "string", "description": "Insert ID from library (e.g., 'FLAG_tag', 'EGFP')"},
                            "sequence": {"type": "string", "description": "Raw DNA sequence (if not using library ID)"},
                            "name": {"type": "string", "description": "Name for this sequence"},
                            "type": {
                                "type": "string",
                                "enum": ["protein", "tag"],
                                "description": "Sequence type: 'protein' (default) removes ATG from non-first positions to keep the reading frame in a fusion; 'tag' preserves ATG for small epitope tags (FLAG, HA, Myc, His) that either lack ATG or need it kept intact.",
                            },
                        },
                    },
                    "description": "Ordered list of sequences to fuse (N-terminal first, C-terminal last)",
                },
                "linker": {
                    "type": "string",
                    "description": "Linker DNA sequence between fusion partners. Omit for default (GGGGS)x4 flexible linker. Pass empty string '' for direct concatenation (epitope tags).",
                },
            },
            "required": ["inserts"],
        },
    },
    {
        "name": "score_construct_confidence",
        "description": (
            "Compute a Design Confidence Score (0-100) for an insert/CDS. "
            "Checks for cryptic polyA/splice signals, codon adaptation index (CAI), "
            "Kozak context, GC content, fusion linker adequacy, repeat runs, and "
            "promoter count. Use this before presenting a final design to flag "
            "potential expression problems."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "insert_sequence": {"type": "string", "description": "Insert/CDS DNA sequence to analyze"},
                "backbone_id": {"type": "string", "description": "Optional backbone ID (for promoter-count check)"},
                "fusion_parts": {
                    "type": "array",
                    "description": "Optional fusion part metadata for linker adequacy check",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "aa_length": {"type": "integer"},
                            "is_linker": {"type": "boolean"},
                        },
                    },
                },
            },
            "required": ["insert_sequence"],
        },
    },
    {
        "name": "predict_fusion_sites",
        "description": (
            "Predict disordered regions in a protein as candidate fusion-insertion sites. "
            "Use when designing an internal (loop) fusion rather than terminal fusion, "
            "or when troubleshooting a terminal fusion that failed. Accepts either an "
            "amino-acid sequence OR a DNA CDS (which will be translated). "
            "Returns ranked disordered windows (longest + most disordered first). "
            "NOTE: This is a sequence-based heuristic, not a full structure prediction. "
            "For high-stakes designs, verify against AlphaFold2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protein_sequence": {"type": "string", "description": "Amino-acid sequence (single-letter code)"},
                "dna_sequence": {"type": "string", "description": "Alternative: DNA CDS (will be translated in frame 0)"},
                "min_window": {"type": "integer", "description": "Minimum disordered-window length in residues (default 10)", "default": 10},
            },
        },
    },
    {
        "name": "lookup_known_mutations",
        "description": (
            "Look up curated gain-of-function (GoF) or loss-of-function (LoF) mutations "
            "for common oncogenes and tumor suppressors (BRAF, KRAS, TP53, EGFR, PTEN, "
            "PIK3CA, IDH1/2, etc.). Returns mutation notation, phenotype, and PMID. "
            "Use when the user wants a constitutively active, dominant-negative, or "
            "kinase-dead version of a gene."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "Gene symbol (e.g., 'BRAF', 'TP53')"},
                "mutation_type": {"type": "string", "description": "Filter: 'GoF' or 'LoF' (optional)", "enum": ["GoF", "LoF"]},
            },
            "required": ["gene_symbol"],
        },
    },
    {
        "name": "apply_mutation",
        "description": (
            "Apply a deterministic point mutation or premature stop codon to a CDS. "
            "For point mutations: swaps ONE codon at the specified AA position for the "
            "preferred human codon for the target AA. For premature stop: introduces an "
            "in-frame TGA at ~position_fraction through the CDS. The rest of the sequence "
            "is preserved exactly. Returns the modified sequence plus change details. "
            "SAFETY: This is targeted single-codon editing only — no sequence is invented."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dna_sequence": {"type": "string", "description": "Input CDS DNA sequence (in-frame from position 0)"},
                "method": {
                    "type": "string",
                    "enum": ["point_mutation", "premature_stop"],
                    "description": "Mutation method (default: point_mutation)",
                    "default": "point_mutation",
                },
                "mutation": {"type": "string", "description": "Standard notation like 'V600E' (for point_mutation)"},
                "aa_position": {"type": "integer", "description": "1-indexed AA position (alternative to 'mutation' param)"},
                "new_aa": {"type": "string", "description": "Target amino acid single-letter code (with aa_position)"},
                "position_fraction": {"type": "number", "description": "For premature_stop: where to place the stop (0-1, default 0.1)", "default": 0.1},
            },
            "required": ["dna_sequence"],
        },
    },
    {
        "name": "fetch_promoter_region",
        "description": (
            "Fetch the native upstream genomic region of a gene from NCBI (~2kb 5' of "
            "the TSS). Use this ONLY for bespoke promoter requests when the user "
            "explicitly chooses option (c) — fetch native upstream region — after you've "
            "offered the three options (Addgene search / paste sequence / native upstream). "
            "IMPORTANT: This is the endogenous regulatory region, NOT a validated minimal "
            "promoter. Warn the user about this in your design summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_id": {"type": "string", "description": "NCBI Gene ID (e.g., '7157' for human TP53). Required if gene_symbol not given."},
                "gene_symbol": {"type": "string", "description": "Gene symbol (e.g., 'TP53'). Will be resolved to gene_id via search."},
                "organism": {"type": "string", "description": "Organism for symbol→ID resolution (e.g., 'human')"},
                "bp_upstream": {"type": "integer", "description": "How many bp upstream to fetch (100-10000, default 2000)", "default": 2000},
            },
        },
    },
    {
        "name": "assemble_golden_gate",
        "description": (
            "Perform in-silico Golden Gate assembly. "
            "Digests the backbone vector at its Type IIS restriction enzyme sites "
            "to open the cloning window (discarding the dropout cassette). "
            "Each part is excised from its carrier vector using the same enzyme. "
            "Parts are ligated in the order dictated by complementary 4-nt overhangs. "
            "Use this for Allen Institute modular expression system parts or any "
            "standard Golden Gate workflow. "
            "Parts must have category='part_in_vector' and a 'plasmid_sequence' field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "backbone_id": {
                    "type": "string",
                    "description": "Library ID of the backbone vector (must contain Type IIS sites).",
                },
                "part_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Library IDs of the parts to assemble, in approximate order. "
                        "Exact order is inferred from overhang matching."
                    ),
                },
                "enzyme_name": {
                    "type": "string",
                    "enum": list(GG_ENZYMES.keys()),
                    "description": (
                        "Type IIS restriction enzyme used for the assembly "
                        "(Esp3I, BsmBI, BsaI, or BbsI). Defaults to Esp3I."
                    ),
                },
            },
            "required": ["backbone_id", "part_ids"],
        },
    },
    {
        "name": "log_experimental_outcome",
        "description": (
            "Record a wet-lab outcome for the current design session. The outcome is "
            "stored in session memory and will be injected into future turns' context "
            "for troubleshooting mode. Use when the user reports that a construct "
            "worked, didn't express, had wrong size, was toxic, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "failed", "partial"], "description": "Experimental outcome"},
                "observation": {"type": "string", "description": "What was observed (e.g., 'no fluorescence', 'wrong size on Western', 'low yield')"},
                "construct_name": {"type": "string", "description": "Name/description of the construct tested (optional)"},
            },
            "required": ["status", "observation"],
        },
    },
]


# ── Sequence truncation for agent-visible output ──
# Large sequences (e.g., BRCA1 CDS is ~5.6 kb) dumped verbatim into tool
# results cause token bloat and rate limiting. The system prompt instructs
# the agent to use insert_id rather than copying sequence text, so we can
# safely truncate. Full sequences are always available via the library
# (assemble_construct resolves by ID, not by pasted text).
_SEQ_TRUNC_THRESHOLD = 4000  # bp — show full seq below this

def _fmt_seq_for_agent(seq: str, label: str = "Sequence") -> str:
    """Format a DNA sequence for agent output, truncating if large.

    Below threshold: full sequence. Above: head + tail with note that
    the full sequence is used internally when referenced by ID.
    """
    n = len(seq)
    if n <= _SEQ_TRUNC_THRESHOLD:
        return f"{label} ({n} bp):\n{seq}"
    head, tail = seq[:200], seq[-200:]
    return (
        f"{label} ({n} bp, truncated for context):\n"
        f"{head}\n... [{n-400} bp omitted] ...\n{tail}\n"
        f"[Full sequence is stored and used internally. Reference by ID in "
        f"assemble_construct/validate_construct rather than copying this text.]"
    )


# ── Tool execution ──────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, tracker: "ReferenceTracker | None" = None) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if name == "search_backbones":
            results = search_backbones(args["query"], args.get("organism"), args.get("promoter"))
            if not results:
                return f"No backbones found matching '{args['query']}'"
            return "\n\n---\n\n".join(format_backbone_summary(bb) for bb in results)

        elif name == "get_backbone":
            bb = get_backbone_by_id(args["backbone_id"])
            if not bb:
                return f"Backbone '{args['backbone_id']}' not found in library or on Addgene."
            if tracker:
                tracker.add_backbone(bb)
            out = format_backbone_summary(bb)
            if bb.get("unconfirmed"):
                out += (
                    "\n\n⚠️ **Unconfirmed Addgene match** — this backbone was "
                    "fuzzy-matched from Addgene search results and has NOT been "
                    "cached. Please confirm this is the intended plasmid before "
                    "proceeding.\n"
                )
                alts = bb.get("addgene_search_alternatives", [])
                if alts:
                    out += "\nOther Addgene search results for this query:\n"
                    for a in alts:
                        out += f"  - {a.get('name')} (Addgene #{a.get('addgene_id')})\n"
                out += (
                    "\nIf correct, call `import_addgene_to_library` with the "
                    "confirmed addgene_id to cache it. If wrong, ask the user "
                    "for the exact plasmid name or Addgene catalog number."
                )
            if args.get("include_sequence") and bb.get("sequence"):
                out += f"\n\nDNA Sequence ({len(bb['sequence'])} bp):\n{bb['sequence'][:200]}... [{len(bb['sequence'])} bp total]"
            return out

        elif name == "search_inserts":
            results = search_inserts(args["query"], args.get("category"))
            if not results:
                return f"No inserts found matching '{args['query']}'"
            return "\n\n---\n\n".join(format_insert_summary(ins) for ins in results)

        elif name == "get_insert":
            ins = get_insert_by_id(args["insert_id"], organism=args.get("organism"))
            if not ins:
                return (
                    f"Insert '{args['insert_id']}' not found in local library, "
                    f"FPbase, or NCBI Gene. Please provide the DNA sequence "
                    f"directly, or check the spelling/species."
                )
            # ── Disambiguation signals ──
            if ins.get("needs_disambiguation"):
                reason = ins.get("reason", "")
                if reason == "gene_family":
                    out = (
                        f"⚠️ **Ambiguous gene family**: '{args['insert_id']}' is "
                        f"a family name, not a specific gene. I cannot "
                        f"auto-select. Please specify which family member:\n\n"
                    )
                    for m in ins.get("members", []):
                        out += f"  - {m}\n"
                    out += "\nAsk the user which one they want, then retry with the specific name."
                    return out
                if reason == "fpbase_no_dna":
                    out = (
                        f"✅ Found on FPbase: **{ins.get('fpbase_name')}**\n"
                        f"  URL: {ins.get('fpbase_url')}\n"
                        f"  Protein: {ins.get('aa_length', '?')} amino acids"
                    )
                    if ins.get("ex_max") and ins.get("em_max"):
                        out += f", Ex/Em {ins['ex_max']}/{ins['em_max']} nm"
                    out += (
                        f"\n\n❌ **No DNA sequence available on FPbase** — only "
                        f"the amino-acid sequence is stored.\n\n"
                        f"I cannot synthesize a DNA sequence (every nucleotide "
                        f"must come from a verified source). Please ask the user "
                        f"to provide the DNA coding sequence for "
                        f"{ins.get('fpbase_name')} — they can find it in:\n"
                        f"  - The original publication\n"
                        f"  - Addgene (many FP plasmids have depositor sequences)\n"
                        f"  - A codon-optimized version they have on hand\n\n"
                        f"Then pass it as insert_sequence to assemble_construct."
                    )
                    return out
                # Multiple species (NCBI fallback)
                out = (
                    f"⚠️ **Ambiguous gene query**: '{args['insert_id']}' matched "
                    f"multiple entries across different species. I cannot "
                    f"auto-select. Please specify which one:\n\n"
                )
                for opt in ins.get("options", []):
                    out += (
                        f"  - {opt.get('symbol', '?')} ({opt.get('organism', '?')}) — "
                        f"{opt.get('full_name', '')}\n"
                        f"    gene_id: {opt.get('gene_id', '?')}\n"
                    )
                out += (
                    "\nEither tell me which species you want, or call "
                    "fetch_gene with the specific gene_id."
                )
                return out
            if tracker:
                tracker.add_insert(ins)
            out = format_insert_summary(ins)
            if ins.get("sequence"):
                out += f"\n\n{_fmt_seq_for_agent(ins['sequence'], 'DNA Sequence')}"
            return out

        elif name == "list_all_backbones":
            bbs = get_all_backbones()
            lines = [f"Available Backbones ({len(bbs)} total):\n"]
            for bb in bbs:
                has_seq = "seq" if bb.get("sequence") else "no seq"
                lines.append(f"- {bb['id']} ({bb['size_bp']} bp, {bb.get('organism','?')}, {bb.get('promoter','?')}, {has_seq})")
            return "\n".join(lines)

        elif name == "list_all_inserts":
            inserts = get_all_inserts()
            lines = [f"Available Inserts ({len(inserts)} total):\n"]
            for ins in inserts:
                lines.append(f"- {ins['id']} ({ins['size_bp']} bp, {ins.get('category','?')})")
            return "\n".join(lines)

        elif name == "get_insertion_site":
            bb = get_backbone_by_id(args["backbone_id"])
            if not bb:
                return f"Backbone '{args['backbone_id']}' not found."
            mcs = bb.get("mcs_position")
            if not mcs:
                return f"No MCS info for {bb['id']}."
            return f"MCS for {bb['id']}: position {mcs['start']}-{mcs['end']}. {mcs.get('description','')}"

        elif name == "validate_sequence":
            result = validate_dna_sequence(args["sequence"])
            return json.dumps(result, indent=2)

        elif name == "assemble_construct":
            # Resolve backbone
            backbone_seq = args.get("backbone_sequence")
            backbone_data = None
            if not backbone_seq and args.get("backbone_id"):
                backbone_data = get_backbone_by_id(args["backbone_id"])
                if backbone_data:
                    backbone_seq = backbone_data.get("sequence")
            if not backbone_seq:
                return "Error: No backbone sequence available. Provide backbone_id (with sequence in library) or backbone_sequence."
            if backbone_data and tracker:
                tracker.add_backbone(backbone_data)

            # Resolve insert
            insert_seq = args.get("insert_sequence")
            insert_data = None
            if not insert_seq and args.get("insert_id"):
                insert_data = get_insert_by_id(args["insert_id"])
                if insert_data:
                    if insert_data.get("needs_disambiguation"):
                        return (
                            f"Error: insert '{args['insert_id']}' is ambiguous "
                            f"(matched multiple species on NCBI). Resolve with "
                            f"get_insert(insert_id=..., organism=...) first, "
                            f"then pass the resolved insert_id."
                        )
                    insert_seq = insert_data.get("sequence")
            if not insert_seq:
                return "Error: No insert sequence available. Provide insert_id or insert_sequence."
            if insert_data and tracker:
                tracker.add_insert(insert_data)

            # Resolve position
            pos = args.get("insertion_position")
            auto_rc = False
            if pos is None and backbone_data:
                pos, auto_rc = resolve_insertion_point(backbone_data, backbone_seq)
            if pos is None:
                return "Error: No insertion position. Provide insertion_position or use a backbone with MCS data."

            result = _assemble_construct(
                backbone_seq=backbone_seq,
                insert_seq=insert_seq,
                insertion_position=pos,
                replace_region_end=args.get("replace_region_end"),
                reverse_complement_insert=args.get("reverse_complement_insert", False) or auto_rc,
                backbone=backbone_data,
            )

            if not result.success:
                return "Assembly FAILED:\n" + "\n".join(f"- {e}" for e in result.errors)

            bb_name = backbone_data["name"] if backbone_data else "custom"
            ins_name = insert_data["name"] if insert_data else "custom"

            out = f"Assembly Successful: {ins_name} in {bb_name}\n"
            out += f"Total size: {result.total_size_bp} bp\n"
            out += f"Insert position: {result.insert_position}\n"
            out += f"Backbone preserved: Yes\n"
            out += f"Insert preserved: Yes\n"
            out += f"Start codon (ATG): {'Yes' if result.insert_has_start_codon else 'No'}\n"
            out += f"Stop codon: {'Yes' if result.insert_has_stop_codon else 'No'}\n"
            out += f"Reading frame ok: {'Yes' if result.insert_length_valid else 'No'}\n"
            if result.warnings:
                out += "Warnings:\n" + "\n".join(f"- {w}" for w in result.warnings) + "\n"
            # NOTE: full sequence is required here — the agent passes it to
            # validate_construct and export_construct. Do not truncate.
            out += f"\nAssembled sequence ({result.total_size_bp} bp):\n{result.sequence}"
            return out

        elif name == "export_construct":
            _set_last_plot_json(None)
            seq = clean_sequence(args["sequence"])
            fmt = args["output_format"]
            cname = args.get("construct_name", "construct")
            bname = args.get("backbone_name", "")
            iname = args.get("insert_name", "")
            ipos = args.get("insert_position", 0)
            ilen = args.get("insert_length", 0)
            rc_insert = args.get("reverse_complement_insert", False)

            if fmt == "raw":
                return seq
            elif fmt == "fasta":
                desc = f"{iname} in {bname}, {len(seq)} bp" if bname else f"{len(seq)} bp"
                return format_as_fasta(seq, cname, desc)
            elif fmt in ("genbank", "gb"):
                # export_genbank_with_plot requires pLannotate (conda-only).
                # In pip environments it raises RuntimeError — fall back to
                # format_as_genbank (no plot) rather than failing the export.
                try:
                    gbk, plot_json = export_genbank_with_plot(
                        sequence=seq, name=cname, backbone_name=bname,
                        insert_name=iname, insert_position=ipos, insert_length=ilen,
                        reverse_complement_insert=rc_insert,
                    )
                    _set_last_plot_json(plot_json)
                    return gbk
                except RuntimeError as e:
                    logger.info(
                        f"pLannotate unavailable ({e}); falling back to "
                        f"basic GenBank export without plasmid map."
                    )
                    return format_as_genbank(
                        sequence=seq, name=cname, backbone_name=bname,
                        insert_name=iname, insert_position=ipos,
                        insert_length=ilen,
                        reverse_complement_insert=rc_insert,
                    )
            else:
                return f"Unknown format: {fmt}"

        elif name == "validate_construct":
            construct_seq = clean_sequence(args["construct_sequence"])

            # ── Resolve backbone (ground truth) ──
            backbone_data = None
            backbone_seq = args.get("backbone_sequence")
            backbone_source = "agent-supplied"
            if args.get("backbone_id"):
                backbone_data = get_backbone_by_id(args["backbone_id"])
                if backbone_data and backbone_data.get("sequence"):
                    backbone_seq = backbone_data.get("sequence")
                    backbone_source = (
                        f"library/Addgene ({backbone_data.get('id', args['backbone_id'])})"
                    )

            # ── Resolve insert (ground truth) ──
            insert_seq = args.get("insert_sequence")
            insert_source = "agent-supplied"
            if args.get("insert_id"):
                ins = get_insert_by_id(args["insert_id"])
                if ins and ins.get("sequence"):
                    insert_seq = ins.get("sequence")
                    insert_source = f"library/NCBI ({ins.get('id', args['insert_id'])})"
                    if ins.get("needs_disambiguation"):
                        insert_source += " ⚠️ AMBIGUOUS"

            checks = []
            warnings = []

            # ── Valid DNA ──
            ok, _errs = validate_dna(construct_seq)
            checks.append(f"Valid DNA: {'PASS' if ok else 'FAIL (CRITICAL)'}")
            checks.append(f"Size: {len(construct_seq)} bp")

            # ── Source provenance ──
            if insert_seq:
                checks.append(f"Insert sequence source: {insert_source}")
                if insert_source == "agent-supplied":
                    warnings.append(
                        "Insert identity UNVERIFIED — validation only confirms the "
                        "supplied sequence is present, not that it is the correct "
                        "gene. Pass insert_id for ground-truth identity check."
                    )
            if backbone_seq:
                checks.append(f"Backbone sequence source: {backbone_source}")

            # ── Find insert in construct (try 8 variants) ──
            found_seq = None
            found_desc = ""
            if insert_seq:
                insert_seq = clean_sequence(insert_seq)
                _has_atg = insert_seq[:3] == "ATG"
                _has_stop = insert_seq[-3:] in ("TAA", "TAG", "TGA")
                _candidates = [
                    (insert_seq, ""),
                    (reverse_complement(insert_seq), "reverse complement"),
                ]
                if _has_atg:
                    _no_atg = insert_seq[3:]
                    _candidates += [
                        (_no_atg, "ATG removed"),
                        (reverse_complement(_no_atg), "ATG removed, reverse complement"),
                    ]
                if _has_stop:
                    _no_stop = insert_seq[:-3]
                    _candidates += [
                        (_no_stop, "stop removed"),
                        (reverse_complement(_no_stop), "stop removed, reverse complement"),
                    ]
                if _has_atg and _has_stop:
                    _no_both = insert_seq[3:-3]
                    _candidates += [
                        (_no_both, "ATG and stop removed"),
                        (reverse_complement(_no_both), "ATG and stop removed, reverse complement"),
                    ]

                for _seq, _desc in _candidates:
                    if len(_seq) >= 9 and _seq in construct_seq:
                        found_seq, found_desc = _seq, _desc
                        break

                found = found_seq is not None
                _suffix = f" ({found_desc})" if found_desc else ""
                checks.append(
                    f"Insert found in construct: {'PASS' + _suffix if found else 'FAIL (CRITICAL)'}"
                )

                # ── Orientation check against backbone MCS direction ──
                if found and backbone_data and backbone_seq:
                    try:
                        _, auto_rc = resolve_insertion_point(backbone_data, backbone_seq)
                        is_rc = "reverse complement" in found_desc
                        if auto_rc != is_rc:
                            checks.append(
                                f"Orientation: FAIL (CRITICAL) — backbone MCS expects "
                                f"{'RC' if auto_rc else 'forward'} insert but found "
                                f"{'RC' if is_rc else 'forward'}"
                            )
                        else:
                            checks.append(
                                f"Orientation: PASS ({'RC' if is_rc else 'forward'}, "
                                f"matches backbone MCS direction)"
                            )
                    except Exception as e:
                        checks.append(f"Orientation: could not determine ({e})")

                if found:
                    pos = construct_seq.index(found_seq)
                    checks.append(f"Insert position: {pos}")
                    exp = args.get("expected_insert_position")
                    if exp is not None:
                        checks.append(
                            f"Position correct: {'PASS' if pos == exp else 'FAIL — expected ' + str(exp)}"
                        )
                    # Codon checks on the expressed (sense) orientation
                    expressed = (
                        reverse_complement(found_seq)
                        if "reverse complement" in found_desc
                        else found_seq
                    )
                    start_ok = expressed[:3] == "ATG"
                    stop_ok = expressed[-3:] in ("TAA", "TAG", "TGA")
                    checks.append(
                        f"Start codon: {'PASS' if start_ok else 'Note — ATG absent (expected for non-N-terminal fusion parts)'}"
                    )
                    checks.append(
                        f"Stop codon: {'PASS' if stop_ok else 'Note — stop absent (expected for non-C-terminal fusion parts)'}"
                    )

            # ── Backbone preservation ──
            if backbone_seq and found_seq:
                backbone_seq = clean_sequence(backbone_seq)
                ipos = construct_seq.index(found_seq)
                up_ok = construct_seq[:ipos] == backbone_seq[:ipos]
                dn_ok = (
                    construct_seq[ipos + len(found_seq):] == backbone_seq[ipos:]
                )
                checks.append(
                    f"Backbone upstream preserved: {'PASS' if up_ok else 'FAIL (CRITICAL)'}"
                )
                checks.append(
                    f"Backbone downstream preserved: {'PASS' if dn_ok else 'FAIL (CRITICAL)'}"
                )
                exp_size = len(backbone_seq) + len(found_seq)
                checks.append(
                    f"Expected size {exp_size} bp: {'PASS' if len(construct_seq) == exp_size else 'FAIL'}"
                )

            out = "Validation Report:\n" + "\n".join(f"  {c}" for c in checks)
            if warnings:
                out += "\n\n⚠️ Warnings:\n" + "\n".join(f"  - {w}" for w in warnings)
            return out

        elif name == "search_addgene":
            if not ADDGENE_AVAILABLE:
                return "Addgene integration not available."
            results = _search_addgene(args["query"], args.get("limit", 10))
            if not results:
                return f"No Addgene results for '{args['query']}'"
            lines = [f"Addgene results for '{args['query']}':"]
            for r in results:
                lines.append(f"- {r.get('name','?')} (Addgene #{r.get('addgene_id','?')})")
            return "\n".join(lines)

        elif name == "get_addgene_plasmid":
            if not ADDGENE_AVAILABLE:
                return "Addgene integration not available."
            plasmid = _get_addgene_plasmid(args["addgene_id"])
            if not plasmid:
                return f"Could not fetch Addgene #{args['addgene_id']}"
            if tracker:
                tracker.add_addgene_plasmid(plasmid.__dict__)
            out = f"Addgene #{args['addgene_id']}: {plasmid.name}\n"
            out += f"Size: {plasmid.size_bp} bp\n"
            out += f"Resistance: {plasmid.bacterial_resistance}\n"
            if plasmid.sequence:
                out += f"Sequence: {len(plasmid.sequence)} bp available\n"
                # If explicitly requested, include the full sequence text so the
                # agent can pass it directly to assemble_construct.
                if args.get("include_sequence", False):
                    out += f"\nSequence ({len(plasmid.sequence)} bp):\n{plasmid.sequence}"
                else:
                    out += "(Use get_backbone or import_addgene_to_library to make it available by ID, or set include_sequence=true to retrieve raw text.)"
            else:
                out += "Sequence: not available (Addgene may require login for this plasmid's depositor sequence)"
            return out

        elif name == "import_addgene_to_library":
            if not ADDGENE_AVAILABLE:
                return "Addgene integration not available."
            integration = AddgeneLibraryIntegration(LIBRARY_PATH)
            bb = integration.import_plasmid(args["addgene_id"], args.get("include_sequence", True))
            if not bb:
                return f"Failed to import Addgene #{args['addgene_id']}"
            if tracker:
                tracker.add_backbone(bb)
            out = f"Imported: {bb['id']} ({bb['size_bp']} bp)"
            if bb.get("sequence"):
                out += f", sequence: {len(bb['sequence'])} bp"
            return out

        elif name == "search_all":
            results = _search_all_sources(args["query"], args.get("organism"))
            lines = [f"Concurrent search results for '{args['query']}':"]
            if results["local_inserts"]:
                lines.append(f"\n--- Local Inserts ({len(results['local_inserts'])} found) ---")
                for ins in results["local_inserts"]:
                    lines.append(f"  - {ins['id']} ({ins['size_bp']} bp, {ins.get('category', '?')})")
            if results["local_backbones"]:
                lines.append(f"\n--- Local Backbones ({len(results['local_backbones'])} found) ---")
                for bb in results["local_backbones"]:
                    lines.append(f"  - {bb['id']} ({bb['size_bp']} bp, {bb.get('organism', '?')}, {bb.get('promoter', '?')})")
            if results["ncbi_genes"]:
                lines.append(f"\n--- NCBI Gene ({len(results['ncbi_genes'])} found) ---")
                for g in results["ncbi_genes"]:
                    aliases = f" (aliases: {g['aliases']})" if g.get("aliases") else ""
                    lines.append(f"  - {g['symbol']} (ID: {g['gene_id']}) — {g['full_name']} [{g['organism']}]{aliases}")
            if results["addgene_plasmids"]:
                lines.append(f"\n--- Addgene ({len(results['addgene_plasmids'])} found) ---")
                for p in results["addgene_plasmids"]:
                    lines.append(f"  - {p.get('name', '?')} (Addgene #{p.get('addgene_id', '?')})")
            if results["errors"]:
                lines.append(f"\n--- Errors ---")
                for src, err in results["errors"].items():
                    lines.append(f"  - {src}: {err}")
            total = (len(results["local_inserts"]) + len(results["local_backbones"]) +
                     len(results["ncbi_genes"]) + len(results["addgene_plasmids"]))
            if total == 0:
                lines.append("\nNo results found in any source.")
            lines.append(f"\nSources searched: {', '.join(results['sources_searched'])}")
            return "\n".join(lines)

        elif name == "search_gene":
            if not NCBI_AVAILABLE:
                return "NCBI integration not available. Install biopython: pip install biopython"
            results = _search_gene_fn(args["query"], args.get("organism"))
            if not results:
                return f"No genes found matching '{args['query']}'"
            lines = [f"NCBI Gene results for '{args['query']}':"]
            for r in results:
                aliases = f" (aliases: {r['aliases']})" if r.get("aliases") else ""
                lines.append(f"- {r['symbol']} (Gene ID: {r['gene_id']}) — {r['full_name']} [{r['organism']}]{aliases}")
            return "\n".join(lines)

        elif name == "fetch_gene":
            if not NCBI_AVAILABLE:
                return "NCBI integration not available. Install biopython: pip install biopython"
            result = _fetch_gene_fn(
                gene_id=args.get("gene_id"),
                gene_symbol=args.get("gene_symbol"),
                organism=args.get("organism"),
            )
            if not result:
                return "Could not fetch gene sequence from NCBI."
            # Disambiguation signal — multiple species matched, no organism given
            if result.get("needs_disambiguation"):
                out = (
                    f"⚠️ **Ambiguous gene**: '{args.get('gene_symbol', '?')}' "
                    f"matched {len(result.get('options', []))} entries across "
                    f"multiple species. Please specify organism:\n\n"
                )
                for opt in result.get("options", []):
                    out += (
                        f"  - {opt.get('symbol')} ({opt.get('organism')}) — "
                        f"{opt.get('full_name')}\n"
                        f"    gene_id: {opt.get('gene_id')}\n"
                    )
                out += "\nRetry with organism set, or pass gene_id directly."
                return out
            if tracker:
                tracker.add_ncbi_gene(result)
            out = f"Gene: {result['symbol']} ({result['organism']})\n"
            out += f"Accession: {result['accession']}\n"
            out += f"Full name: {result['full_name']}\n"
            out += f"CDS length: {result['length']} bp\n"
            out += f"\n{_fmt_seq_for_agent(result['sequence'], 'CDS Sequence')}"
            return out

        elif name == "search_fpbase":
            if not FPBASE_AVAILABLE:
                return "FPbase integration not available."
            results = _search_fpbase(args["name"], limit=5)
            if not results:
                return (
                    f"No fluorescent proteins found on FPbase matching "
                    f"'{args['name']}'. Try a different spelling or check "
                    f"https://www.fpbase.org/"
                )
            lines = [f"FPbase results for '{args['name']}':"]
            for r in results:
                ex_em = ""
                if r.get("ex_max") and r.get("em_max"):
                    ex_em = f" — Ex/Em {r['ex_max']}/{r['em_max']} nm"
                lines.append(
                    f"  - {r['name']} (slug: {r['slug']}){ex_em}"
                )
            lines.append(
                "\nUse get_insert with the FP name to retrieve the DNA sequence."
            )
            return "\n".join(lines)

        elif name == "get_cell_line_info":
            cl = args["cell_line"]
            species = infer_species_from_cell_line(cl)
            if species:
                return (
                    f"Cell line '{cl}' is from species: **{species}**\n"
                    f"Note: confirm with the user before assuming the gene of "
                    f"interest is also {species} — they may want a different "
                    f"species' gene expressed in {cl} cells."
                )
            return (
                f"Cell line '{cl}' not found in the known cell lines database. "
                f"Ask the user what species it is."
            )

        elif name == "fuse_inserts":
            sequences = []
            atg_removals = []
            for i, item in enumerate(args["inserts"]):
                seq = item.get("sequence")
                seq_name = item.get("name", "")
                seq_type = item.get("type", "protein")
                if not seq and item.get("insert_id"):
                    ins = get_insert_by_id(item["insert_id"])
                    if not ins:
                        return f"Insert '{item['insert_id']}' not found in library, FPbase, or NCBI."
                    if ins.get("needs_disambiguation"):
                        return (
                            f"Cannot fuse: insert '{item['insert_id']}' is ambiguous "
                            f"(matched multiple species). Call get_insert first "
                            f"with an organism, or specify gene_id directly."
                        )
                    seq = ins.get("sequence")
                    seq_name = seq_name or ins.get("name", item["insert_id"])
                    if tracker:
                        tracker.add_insert(ins)
                if not seq:
                    return f"No sequence available for '{seq_name or 'unknown'}'."
                sequences.append({"sequence": seq, "name": seq_name, "type": seq_type})
                if i > 0 and seq_type == "protein":
                    from assembler import clean_sequence as _clean_seq
                    if _clean_seq(seq)[:3] == "ATG":
                        atg_removals.append(seq_name or f"sequence_{i}")

            try:
                fused = _fuse_sequences(sequences, args.get("linker"))
            except ValueError as e:
                return f"Fusion error: {e}"

            names = [s["name"] for s in sequences]
            out = f"Fused CDS: {'-'.join(names)}\n"
            out += f"Length: {len(fused)} bp\n"
            out += f"Start codon: {'Yes' if fused[:3] == 'ATG' else 'No'}\n"
            out += f"Stop codon: {'Yes' if fused[-3:] in ('TAA', 'TAG', 'TGA') else 'No'}\n"
            out += f"In frame: {'Yes' if len(fused) % 3 == 0 else 'No'}\n"
            if atg_removals:
                out += f"\nNote: Start codon (ATG) removed from: {', '.join(atg_removals)}\n"
                out += "This is correct for a protein fusion — translation initiates from the first ATG only.\n"
            out += f"\nFused sequence ({len(fused)} bp):\n{fused}"
            return out

        elif name == "score_construct_confidence":
            if not CONFIDENCE_AVAILABLE:
                return "Design Confidence module not available in this deployment."
            insert_seq = clean_sequence(args["insert_sequence"])
            backbone = None
            if args.get("backbone_id"):
                backbone = get_backbone_by_id(args["backbone_id"])
            report = compute_confidence(
                insert_seq=insert_seq,
                backbone=backbone,
                fusion_parts=args.get("fusion_parts"),
            )
            return format_confidence_report(report)

        elif name == "predict_fusion_sites":
            if not PROTEIN_ANALYSIS_AVAILABLE:
                return "Protein Analysis module not available in this deployment."
            # Accept either AA or DNA input
            aa_seq = args.get("protein_sequence")
            if not aa_seq and args.get("dna_sequence"):
                dna = clean_sequence(args["dna_sequence"])
                aa_seq = _translate_dna(dna)
            if not aa_seq:
                return "Error: provide either protein_sequence (AA) or dna_sequence (CDS)."
            aa_seq = aa_seq.upper().strip()
            min_window = args.get("min_window", 10)
            sites = _find_fusion_sites(aa_seq, min_window=min_window)
            if not sites:
                return (
                    f"No disordered regions ≥{min_window} residues found in this "
                    f"protein ({len(aa_seq)} aa). The protein may be highly "
                    f"structured throughout — terminal fusion is likely the only "
                    f"option. Consider a longer flexible linker if terminal fusion "
                    f"has failed."
                )
            lines = [
                f"Found {len(sites)} candidate fusion site(s) in protein "
                f"({len(aa_seq)} aa), ranked by suitability:\n"
            ]
            for i, s in enumerate(sites[:5], 1):
                lines.append(
                    f"  {i}. Residues {s['start']+1}-{s['end']} "
                    f"({s['length']} aa, mean disorder {s['mean_disorder']:.2f}) "
                    f"— context: ...{s['context']}..."
                )
            lines.append(
                "\nNote: Disorder prediction is a sequence-based heuristic. "
                "For high-stakes designs, verify against AlphaFold2 pLDDT "
                "or published domain boundaries."
            )
            return "\n".join(lines)

        elif name == "lookup_known_mutations":
            if not MUTATIONS_AVAILABLE:
                return "Mutation Design module not available in this deployment."
            muts = _lookup_known_mutations(
                args["gene_symbol"], args.get("mutation_type")
            )
            if not muts:
                filter_txt = f" ({args['mutation_type']})" if args.get("mutation_type") else ""
                return (
                    f"No curated{filter_txt} mutations found for "
                    f"'{args['gene_symbol']}'. The curated database covers "
                    f"common oncogenes (BRAF, KRAS, EGFR, PIK3CA, IDH1/2, "
                    f"NRAS, CTNNB1, AKT1, MYC) and tumor suppressors (TP53, "
                    f"PTEN, RB1, FBXW7). For other genes, ask the user for "
                    f"the specific mutation they want, or offer a premature-stop "
                    f"LoF design."
                )
            lines = [
                f"Curated mutations for {args['gene_symbol'].upper()}"
                f"{' (' + args['mutation_type'] + ')' if args.get('mutation_type') else ''}:\n"
            ]
            for m in muts:
                ref = f" [{m['reference']}]" if m.get("reference") else ""
                lines.append(
                    f"  • {m['mutation']} ({m['type']}): {m['phenotype']}{ref}"
                )
                if m.get("codon_change"):
                    lines.append(f"    Codon change: {m['codon_change']}")
            return "\n".join(lines)

        elif name == "apply_mutation":
            if not MUTATIONS_AVAILABLE:
                return "Mutation Design module not available in this deployment."
            dna = clean_sequence(args["dna_sequence"])
            method = args.get("method", "point_mutation")

            if method == "premature_stop":
                frac = args.get("position_fraction", 0.1)
                result = _design_premature_stop(dna, position_fraction=frac)
                out = (
                    f"Premature stop introduced:\n"
                    f"  AA position: {result['stop_position_aa']}\n"
                    f"  DNA position: {result['stop_position_dna']}\n"
                    f"  Original codon: {result['original_codon']} "
                    f"({result['original_aa']})\n"
                    f"  New codon: TGA (*)\n"
                    f"  Sequence length preserved: "
                    f"{len(result['sequence'])} bp\n\n"
                    f"Mutated sequence:\n{result['sequence']}"
                )
                return out

            # point_mutation — accept either 'mutation' (V600E notation)
            # or aa_position + new_aa
            aa_pos = args.get("aa_position")
            new_aa = args.get("new_aa")
            if args.get("mutation"):
                parsed = _parse_mutation_notation(args["mutation"])
                if not parsed:
                    return (
                        f"Could not parse mutation notation '{args['mutation']}'. "
                        f"Use standard format like 'V600E' or pass aa_position "
                        f"+ new_aa directly."
                    )
                aa_pos = parsed["position"]
                new_aa = parsed["new_aa"]
                expected_original = parsed["original_aa"]
            else:
                expected_original = None

            if aa_pos is None or not new_aa:
                return (
                    "Error: for point_mutation, provide either 'mutation' "
                    "(e.g., 'V600E') or both 'aa_position' and 'new_aa'."
                )

            result = _apply_point_mutation(dna, aa_position=aa_pos, new_aa=new_aa)

            # Sanity check: did the original AA match what the notation said?
            warning = ""
            if expected_original and result["original_aa"] != expected_original:
                warning = (
                    f"\n\n⚠️ WARNING: Mutation notation '{args['mutation']}' "
                    f"expects original AA '{expected_original}' at position "
                    f"{aa_pos}, but the sequence has '{result['original_aa']}'. "
                    f"This may be the wrong transcript/isoform, or the position "
                    f"is off by one. Please verify the sequence and position."
                )

            out = (
                f"Point mutation applied: {result['original_aa']}{aa_pos}"
                f"{result['new_aa']}\n"
                f"  DNA position: {result['dna_position']}\n"
                f"  Original codon: {result['original_codon']} → "
                f"New codon: {result['new_codon']}\n"
                f"  Sequence length preserved: "
                f"{len(result['sequence'])} bp\n"
                f"{warning}\n\n"
                f"Mutated sequence:\n{result['sequence']}"
            )
            return out

        elif name == "fetch_promoter_region":
            if not GENOMIC_UPSTREAM_AVAILABLE:
                return (
                    "Genomic upstream fetch not available (requires Biopython "
                    "+ NCBI access)."
                )
            gene_id = args.get("gene_id")
            bp_upstream = args.get("bp_upstream", 2000)

            # Resolve symbol → gene_id if needed
            if not gene_id and args.get("gene_symbol"):
                if not NCBI_AVAILABLE:
                    return "NCBI gene search not available."
                genes = _search_gene_fn(args["gene_symbol"], args.get("organism"))
                if not genes:
                    return f"Gene '{args['gene_symbol']}' not found on NCBI."
                if len(genes) > 1 and not args.get("organism"):
                    organisms = {g.get("organism", "") for g in genes}
                    if len(organisms) > 1:
                        return (
                            f"Gene '{args['gene_symbol']}' is ambiguous across "
                            f"species: {', '.join(sorted(organisms))}. "
                            f"Specify organism and retry."
                        )
                gene_id = genes[0]["gene_id"]

            if not gene_id:
                return "Error: provide gene_id or gene_symbol."

            result = _fetch_genomic_upstream(gene_id=gene_id, bp_upstream=bp_upstream)
            if not result:
                return (
                    f"Could not fetch upstream region for gene_id={gene_id}. "
                    f"The gene may not have annotated genomic coordinates, "
                    f"or NCBI is temporarily unavailable."
                )
            out = (
                f"Native upstream region for {result['gene_symbol']} "
                f"(gene_id={result['gene_id']}):\n"
                f"  Organism: {result.get('organism', '?')}\n"
                f"  Chromosome: {result.get('chromosome_accession', '?')}\n"
                f"  Strand: {result.get('strand', '?')}\n"
                f"  Length: {result['length']} bp\n\n"
                f"⚠️ {result['warning']}\n\n"
                f"Upstream sequence ({result['length']} bp):\n"
                f"{result['sequence']}"
            )
            return out

        elif name == "assemble_golden_gate":
            backbone_id = args["backbone_id"]
            part_ids = args["part_ids"]
            enzyme_name = args.get("enzyme_name", "Esp3I")

            backbone = get_backbone_by_id(backbone_id)
            if not backbone:
                return f"Backbone {backbone_id!r} not found in library."
            bb_seq = backbone.get("plasmid_sequence") or backbone.get("sequence", "")
            if not bb_seq:
                return f"Backbone {backbone_id!r} has no plasmid_sequence."

            parts = []
            for pid in part_ids:
                part = get_insert_by_id(pid)
                if not part:
                    return f"Part {pid!r} not found in library."
                ps = part.get("plasmid_sequence") or part.get("sequence", "")
                if not ps:
                    return (
                        f"Part {pid!r} has no plasmid_sequence. "
                        "Golden Gate requires the full carrier vector sequence."
                    )
                parts.append({
                    "name": part.get("name", pid),
                    "plasmid_sequence": ps,
                    "overhang_l": part.get("overhang_l"),
                    "overhang_r": part.get("overhang_r"),
                })

            result = _assemble_golden_gate(
                backbone_plasmid_seq=bb_seq,
                parts=parts,
                enzyme_name=enzyme_name,
            )

            if not result.success:
                return "Golden Gate assembly failed:\n" + "\n".join(
                    f"  • {e}" for e in result.errors
                )

            warnings_block = ""
            if result.warnings:
                warnings_block = "\n\nWarnings:\n" + "\n".join(
                    f"  ⚠ {w}" for w in result.warnings
                )

            junctions = " → ".join(result.junction_overhangs)
            order_str = " → ".join(result.assembly_order) if result.assembly_order else "(backbone only)"

            return (
                f"Golden Gate assembly successful ({enzyme_name}).\n\n"
                f"Assembly order : {order_str}\n"
                f"Junctions (4-nt): {junctions}\n"
                f"Total size     : {result.total_size_bp} bp\n\n"
                # NOTE: full sequence must be returned here so the agent can pass it
                # directly to validate_construct and export_construct. Do NOT truncate.
                f"Assembled sequence ({result.total_size_bp} bp):\n{result.sequence}"
                + warnings_block
            )

        elif name == "log_experimental_outcome":
            # This tool needs session context to store the outcome. The
            # execute_tool dispatcher doesn't have session access, so we
            # return a marker the agent-loop can intercept (or, simpler:
            # just format for display and rely on the caller to persist).
            # For now, return a special prefix the agent loop can detect.
            status = args["status"]
            observation = args["observation"]
            cname = args.get("construct_name", "")
            return (
                f"[OUTCOME_LOGGED] status={status} "
                f"construct={cname!r} observation={observation!r}\n\n"
                f"Outcome recorded for this session: **{status}** — "
                f"{observation}. Future troubleshooting turns will see this "
                f"context."
            )

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {str(e)}"



# ── Session management ──────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_cancelled_sessions: set[str] = set()
_sessions_lock = threading.Lock()

# ── Batch job state ─────────────────────────────────────────────────────
_batch_jobs: dict[str, dict] = {}
SESSIONS_FILE = Path(__file__).parent / ".sessions.json"

MODEL = "claude-opus-4-6"


def _serialize_content(content):
    """Convert Anthropic SDK content blocks to JSON-serializable format.

    Filters out thinking blocks and non-API-compatible fields so the
    serialized history can be safely replayed to the Anthropic API.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        serialized = []
        for b in content:
            if hasattr(b, "model_dump"):
                d = b.model_dump()
            elif isinstance(b, dict):
                d = b
            else:
                continue
            # Skip thinking blocks — they cause Error 400 on replay
            if isinstance(d, dict) and d.get("type") == "thinking":
                continue
            serialized.append(d)
        return serialized
    return content


def _save_sessions():
    """Persist sessions to disk so they survive server restarts.

    Uses atomic write (write tmp -> copy backup -> replace) to avoid
    race conditions where the sessions file disappears mid-write.
    Thread-safe via _sessions_lock.
    """
    import shutil

    with _sessions_lock:
        try:
            serializable = {}
            for sid, data in _sessions.items():
                # Serialize history message-by-message so one bad message
                # doesn't drop the entire session (which is what caused
                # users to see their chat history vanish on reload).
                safe_history = []
                for m in data.get("history", []):
                    try:
                        sm = {"role": m["role"], "content": _serialize_content(m["content"])}
                        json.dumps(sm)
                        safe_history.append(sm)
                    except (TypeError, ValueError) as e:
                        logger.warning(
                            f"Dropping unserializable message in session "
                            f"{sid[:8]} (role={m.get('role','?')}): {e}"
                        )
                        # Preserve turn structure so replay doesn't break
                        safe_history.append({
                            "role": m.get("role", "user"),
                            "content": "[message serialization failed]",
                        })
                # Base fields (always serializable — primitive types only)
                base_fields = {
                    "created_at": data.get("created_at", time.time()),
                    "first_message": data.get("first_message"),
                    "history": safe_history,
                    # Phase-2 troubleshooting/project-memory fields — default
                    # to empty for sessions created before these were added.
                    "project_name": data.get("project_name"),
                    "experimental_outcomes": data.get("experimental_outcomes", []),
                }
                try:
                    s = {"display_messages": data.get("display_messages", []), **base_fields}
                    json.dumps(s)
                    serializable[sid] = s
                except (TypeError, ValueError) as e:
                    # Fall back to saving session metadata + history only
                    # (display_messages may contain the bad block)
                    logger.warning(
                        f"Session {sid[:8]} display_messages unserializable, "
                        f"saving with empty display: {e}"
                    )
                    serializable[sid] = {"display_messages": [], **base_fields}

            tmp_file = SESSIONS_FILE.with_suffix(".json.tmp")
            with open(tmp_file, "w") as f:
                json.dump(serializable, f)

            if SESSIONS_FILE.exists():
                bak_file = SESSIONS_FILE.with_suffix(".json.bak")
                try:
                    shutil.copy2(str(SESSIONS_FILE), str(bak_file))
                except OSError:
                    pass

            os.replace(str(tmp_file), str(SESSIONS_FILE))
        except Exception as e:
            logger.debug(f"Failed to save sessions: {e}")
            bak_file = SESSIONS_FILE.with_suffix(".json.bak")
            if not SESSIONS_FILE.exists() and bak_file.exists():
                try:
                    shutil.copy2(str(bak_file), str(SESSIONS_FILE))
                except OSError:
                    pass


def _load_sessions():
    """Load sessions from disk on startup. Falls back to .bak if main file is corrupt."""
    global _sessions
    for filepath in [SESSIONS_FILE, SESSIONS_FILE.with_suffix(".json.bak")]:
        try:
            if filepath.exists():
                with open(filepath) as f:
                    _sessions = json.load(f)
                if _sessions:
                    return
        except Exception as e:
            logger.debug(f"Failed to load sessions from {filepath}: {e}")
    _sessions = {}


# Load persisted sessions at import time
_load_sessions()


def create_session() -> str:
    """Create a new conversation session."""
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "history": [],
        "display_messages": [],
        "created_at": time.time(),
        "first_message": None,
        # Troubleshooting / project-memory fields (Phase 2)
        "project_name": None,            # user-assigned project label (optional)
        "experimental_outcomes": [],     # list of {status, observation, construct_name, timestamp}
    }
    _save_sessions()
    return sid


def _build_system_prompt(session: dict) -> str:
    """Build the system prompt for a turn, injecting per-session context.

    Starts with the static SYSTEM_PROMPT and appends troubleshooting
    context if the session has prior experimental outcomes. This enables
    "project memory" — the agent can see what the user already tried.
    """
    prompt = SYSTEM_PROMPT
    outcomes = session.get("experimental_outcomes") or []
    if outcomes:
        prompt += "\n\n---\n\n## Troubleshooting Context — Prior Experimental Outcomes\n\n"
        prompt += (
            "This session has recorded wet-lab outcomes for constructs the "
            "user previously tried. Use this history to diagnose failures "
            "and propose revised designs (see Troubleshooting Mode section "
            "above).\n\n"
        )
        for i, o in enumerate(outcomes, 1):
            cname = o.get("construct_name") or "unnamed construct"
            prompt += (
                f"**Prior attempt {i}** ({cname}):\n"
                f"  Status: {o.get('status', '?')}\n"
                f"  Observation: {o.get('observation', '?')}\n\n"
            )
    return prompt


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def delete_session_by_id(session_id: str) -> bool:
    deleted = _sessions.pop(session_id, None) is not None
    if deleted:
        _save_sessions()
    return deleted


def list_sessions() -> list[dict]:
    result = []
    for sid, data in sorted(
        _sessions.items(), key=lambda x: x[1]["created_at"], reverse=True
    ):
        result.append({
            "session_id": sid,
            "first_message": data["first_message"],
            "created_at": data["created_at"],
            "project_name": data.get("project_name"),
            "outcomes_count": len(data.get("experimental_outcomes") or []),
        })
    return result


def cancel_session(session_id: str):
    _cancelled_sessions.add(session_id)


# ── Agent loop ──────────────────────────────────────────────────────────


def run_agent_turn_streaming(user_message: str, session_id: str, write_event, model: str = MODEL):
    """Run one agent turn with streaming, scoped to a session."""
    _cancelled_sessions.discard(session_id)

    session = get_session(session_id)
    if not session:
        write_event({"type": "error", "content": "Session not found"})
        return

    tracker = ReferenceTracker()
    export_called = False
    # Build the system prompt once per turn (not per retry) so that
    # prompt caching works. The prompt is dynamic because it includes
    # per-session troubleshooting context (experimental_outcomes).
    turn_system_prompt = _build_system_prompt(session)
    client = anthropic.Anthropic()
    history = session["history"]
    history.append({"role": "user", "content": user_message})
    session["display_messages"].append({"role": "user", "content": user_message})

    if session["first_message"] is None:
        session["first_message"] = user_message[:80]

    disconnected = False
    is_cancelled = lambda: session_id in _cancelled_sessions

    def safe_write(data: dict):
        nonlocal disconnected
        if disconnected or is_cancelled():
            return
        try:
            write_event(data)
        except (BrokenPipeError, ConnectionResetError):
            disconnected = True

    max_iterations = 15
    max_retries = 3
    assistant_text = ""
    assistant_blocks = []
    current_thinking_text = ""
    current_text_content = ""

    for _ in range(max_iterations):
        if is_cancelled() or disconnected:
            break

        stop_reason = None
        final_message = None
        tool_results: list = []  # also reset inside retry loop; init here for static analysis

        # Retry loop for rate limits
        for retry_attempt in range(max_retries + 1):
            # Reset per-API-call state on each retry. If a stream partially
            # succeeded before rate-limiting, any tool_results accumulated
            # reference tool_use_ids from the aborted stream — replaying them
            # alongside the retry's fresh tool_use_ids causes a 400 error
            # (tool_use/tool_result ID mismatch).
            current_block_type = None
            current_tool_name = None
            current_tool_id = None
            current_tool_input_json = ""
            tool_results = []
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=16000,
                    system=turn_system_prompt,
                    tools=TOOLS,
                    messages=history,
                    thinking={"type": "enabled", "budget_tokens": 5000},
                ) as stream:
                    for event in stream:
                        if is_cancelled() or disconnected:
                            stream.close()
                            break

                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "thinking":
                                current_block_type = "thinking"
                                current_thinking_text = ""
                                safe_write({"type": "thinking_start"})
                            elif block.type == "text":
                                current_block_type = "text"
                                current_text_content = ""
                                safe_write({"type": "text_start"})
                            elif block.type == "tool_use":
                                current_block_type = "tool_use"
                                current_tool_name = block.name
                                current_tool_id = block.id
                                current_tool_input_json = ""
                                safe_write({"type": "tool_use_start", "tool": block.name})

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "thinking_delta":
                                current_thinking_text += delta.thinking
                                safe_write({"type": "thinking_delta", "content": delta.thinking})
                            elif delta.type == "text_delta":
                                assistant_text += delta.text
                                current_text_content += delta.text
                                safe_write({"type": "text_delta", "content": delta.text})
                            elif delta.type == "input_json_delta":
                                current_tool_input_json += delta.partial_json

                        elif event.type == "content_block_stop":
                            if current_block_type == "thinking":
                                assistant_blocks.append({"type": "thinking", "content": current_thinking_text})
                                safe_write({"type": "thinking_end"})
                            elif current_block_type == "text":
                                assistant_blocks.append({"type": "text", "content": current_text_content})
                                safe_write({"type": "text_end"})
                            elif current_block_type == "tool_use":
                                if is_cancelled() or disconnected:
                                    break
                                tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                                result_str = execute_tool(current_tool_name, tool_input, tracker)
                                if current_tool_name == "export_construct":
                                    export_called = True
                                # Intercept outcome-log marker and persist to session
                                if (
                                    current_tool_name == "log_experimental_outcome"
                                    and result_str.startswith("[OUTCOME_LOGGED]")
                                ):
                                    session.setdefault("experimental_outcomes", []).append({
                                        "status": tool_input.get("status"),
                                        "observation": tool_input.get("observation"),
                                        "construct_name": tool_input.get("construct_name", ""),
                                        "timestamp": time.time(),
                                    })
                                    _save_sessions()
                                display_result = result_str[:2000] + "..." if len(result_str) > 2000 else result_str
                                event_data = {
                                    "type": "tool_result",
                                    "tool": current_tool_name,
                                    "input": tool_input,
                                    "content": display_result,
                                }
                                # For export_construct, include full content for download
                                if current_tool_name == "export_construct":
                                    event_data["download_content"] = result_str
                                    fmt = tool_input.get("output_format", "raw")
                                    cname = tool_input.get("construct_name", "construct")
                                    ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
                                    event_data["download_filename"] = cname + ext
                                safe_write(event_data)
                                # Emit plasmid plot after genbank export
                                _plot = _get_last_plot_json()
                                if current_tool_name == "export_construct" and _plot:
                                    safe_write({"type": "plot_data", "plot_json": json.loads(_plot)})
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": current_tool_id,
                                    "content": result_str,
                                })
                                assistant_blocks.append({
                                    "type": "tool_use",
                                    "name": current_tool_name,
                                    "input": tool_input,
                                    "result": display_result,
                                    "download_content": event_data.get("download_content"),
                                    "download_filename": event_data.get("download_filename"),
                                })
                            current_block_type = None

                        elif event.type == "message_delta":
                            stop_reason = event.delta.stop_reason

                    if is_cancelled() or disconnected:
                        break

                    final_message = stream.get_final_message()
                # Stream succeeded, break out of retry loop
                break

            except anthropic.RateLimitError:
                if retry_attempt < max_retries:
                    wait_time = 2 ** retry_attempt  # 1s, 2s, 4s
                    safe_write({"type": "text_delta", "content": f"\n[Rate limited, retrying in {wait_time}s...]\n"})
                    time.sleep(wait_time)
                    continue
                else:
                    safe_write({"type": "error", "content": "Rate limit exceeded after retries. Please try again later."})
                    break
            except Exception:
                if is_cancelled() or disconnected:
                    break
                raise

        if is_cancelled() or disconnected:
            break

        # Guard: if all retries were exhausted (rate limit) or the stream
        # broke before get_final_message, final_message is None. Don't try
        # to append history — just exit the agent loop.
        if final_message is None:
            break

        # Convert content blocks to plain dicts to strip extra SDK fields
        # (e.g. parsed_output) that cause 400 errors on replay.
        # Unknown block types are DROPPED — passing them through can cause
        # 400 errors on the next API call when the SDK emits a new block type
        # we don't handle (redacted_thinking, server_tool_use, etc.).
        filtered_content = []
        for b in final_message.content:
            btype = getattr(b, 'type', None)
            if btype == 'thinking':
                continue
            elif btype == 'text':
                filtered_content.append({"type": "text", "text": b.text})
            elif btype == 'tool_use':
                filtered_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            else:
                logger.warning(f"Dropping unknown content block type from history: {btype or type(b).__name__}")
                continue
        history.append({"role": "assistant", "content": filtered_content})

        if tool_results:
            history.append({"role": "user", "content": tool_results})
        else:
            break

        if stop_reason == "end_turn":
            break

    # Flush any in-progress block that was interrupted mid-stream
    if current_text_content and not any(
        b.get("type") == "text" and b.get("content") == current_text_content
        for b in assistant_blocks
    ):
        assistant_blocks.append({"type": "text", "content": current_text_content})
    if current_thinking_text and not any(
        b.get("type") == "thinking" and b.get("content") == current_thinking_text
        for b in assistant_blocks
    ):
        assistant_blocks.append({"type": "thinking", "content": current_thinking_text})

    # Append formatted references only when a sequence file was exported this turn
    if export_called and not (is_cancelled() or disconnected):
        refs_text = tracker.format_references()
        if refs_text:
            ref_block = f"\n\n{refs_text}"
            assistant_text += ref_block
            assistant_blocks.append({"type": "text", "content": ref_block})
            safe_write({"type": "text_start"})
            safe_write({"type": "text_delta", "content": ref_block})
            safe_write({"type": "text_end"})

    # Save assistant text and structured blocks to display messages
    if assistant_text or assistant_blocks:
        session["display_messages"].append({
            "role": "assistant",
            "content": assistant_text,
            "blocks": assistant_blocks,
        })
    elif is_cancelled() or disconnected:
        # Remove dangling user message if no response was generated
        if history and history[-1]["role"] == "user" and isinstance(history[-1].get("content"), str):
            history.pop()
            if session["display_messages"] and session["display_messages"][-1]["role"] == "user":
                session["display_messages"].pop()

    # Persist updated session to disk
    _save_sessions()

    if not disconnected:
        try:
            write_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            pass


# ── HTML UI ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plasmid Designer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.1.min.js"></script>
<style>
  :root {
    --brand-fig: #D97757;
    --brand-fig-hover: #B5603F;
    --brand-fig-10: rgba(217,119,87,0.1);
    --brand-fig-30: rgba(217,119,87,0.3);
    --brand-aqua: #24B283;
    --brand-aqua-dark: #0E6B54;
    --brand-aqua-10: rgba(36,178,131,0.1);
    --brand-aqua-20: rgba(36,178,131,0.2);
    --brand-orange: #E86235;
    --brand-orange-100: #FAEFEB;
    --brand-coral: #F5E0D8;
    --brand-coral-30: rgba(245,224,216,0.3);
    --sand-50: #FAF9F7;
    --sand-100: #F5F3EF;
    --sand-200: #E8E6DC;
    --sand-300: #D4D1C7;
    --sand-400: #ADAAA0;
    --sand-500: #87867F;
    --sand-600: #5C5B56;
    --sand-700: #3D3D3A;
    --sand-800: #2A2A28;
    --sand-900: #1A1A19;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: white;
    color: var(--sand-900);
    height: 100vh;
    display: flex;
    flex-direction: column;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Header ── */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    border-bottom: 1px solid var(--sand-200);
    background: white;
    flex-shrink: 0;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .header-logo {
    width: 32px; height: 32px; border-radius: 8px;
    background: var(--brand-fig-10);
    display: flex; align-items: center; justify-content: center;
  }
  .header-logo svg { width: 20px; height: 20px; stroke: var(--brand-fig); fill: none; }
  .header-title h1 { font-size: 16px; font-weight: 600; color: var(--sand-800); line-height: 1.2; }
  .header-title p { font-size: 12px; color: var(--sand-400); line-height: 1.2; }
  .health-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 500; border: 1px solid; transition: all 0.2s;
  }
  .health-badge.online {
    background: var(--brand-aqua-10); color: var(--brand-aqua-dark); border-color: var(--brand-aqua-20);
  }
  .health-badge.offline {
    background: var(--sand-100); color: var(--sand-500); border-color: var(--sand-200);
  }
  .health-dot { width: 6px; height: 6px; border-radius: 50%; }
  .health-badge.online .health-dot { background: var(--brand-aqua); }
  .health-badge.offline .health-dot { background: var(--sand-400); }

  /* ── Layout ── */
  .main { flex: 1; display: flex; overflow: hidden; }

  /* ── Sidebar ── */
  .sidebar {
    width: 240px; background: var(--sand-50);
    border-right: 1px solid var(--sand-200);
    display: flex; flex-direction: column; flex-shrink: 0;
    transition: width 0.3s ease; overflow: hidden;
  }
  .sidebar.collapsed { width: 0; border-right: none; }
  .sidebar-toolbar {
    padding: 12px; display: flex; align-items: center; gap: 8px;
  }
  .sidebar-toggle-btn {
    width: 32px; height: 32px; border: none; background: none;
    border-radius: 8px; color: var(--sand-400); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: all 0.15s;
  }
  .sidebar-toggle-btn:hover { color: var(--sand-600); background: var(--sand-100); }
  .new-chat-btn {
    flex: 1; display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border: 1px solid var(--sand-200);
    background: none; border-radius: 8px; color: var(--sand-600);
    font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.15s;
    font-family: inherit;
  }
  .new-chat-btn:hover { background: var(--sand-100); }
  .sessions-list {
    flex: 1; overflow-y: auto; padding: 0 8px 12px;
    scrollbar-width: thin; scrollbar-color: transparent transparent;
  }
  .sessions-list:hover { scrollbar-color: var(--sand-300) transparent; }
  .session-item {
    width: 100%; text-align: left; padding: 10px 12px;
    border-radius: 8px; border: none; background: none;
    color: var(--sand-500); font-size: 14px; cursor: pointer;
    transition: all 0.15s; display: flex; align-items: center;
    justify-content: space-between; gap: 8px; margin-bottom: 2px;
    font-family: inherit;
  }
  .session-item:hover { background: var(--sand-100); color: var(--sand-700); }
  .session-item.active { background: var(--sand-200); color: var(--sand-800); }
  .session-name {
    overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; font-weight: 500; flex: 1;
  }
  .delete-btn {
    opacity: 0; border: none; background: none;
    color: var(--sand-300); cursor: pointer; padding: 2px;
    flex-shrink: 0; transition: opacity 0.15s, color 0.15s;
  }
  .session-item:hover .delete-btn { opacity: 1; }
  .delete-btn:hover { color: var(--brand-orange); }
  .no-sessions { text-align: center; font-size: 12px; color: var(--sand-400); margin-top: 16px; }
  .sidebar-reopen-btn {
    position: absolute; top: 68px; left: 12px; z-index: 10;
    width: 32px; height: 32px; border: none; background: none;
    border-radius: 8px; color: var(--sand-400); cursor: pointer;
    display: none; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  .sidebar-reopen-btn:hover { color: var(--sand-600); background: var(--sand-100); }
  .sidebar-reopen-btn.visible { display: flex; }

  /* ── Chat Panel ── */
  .chat-panel { flex: 1; display: flex; flex-direction: column; background: white; min-width: 0; position: relative; }
  .messages {
    flex: 1; overflow-y: auto; padding: 24px;
    scrollbar-width: thin; scrollbar-color: transparent transparent;
  }
  .messages:hover { scrollbar-color: var(--sand-300) transparent; }
  .messages::-webkit-scrollbar { width: 6px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb { background: transparent; border-radius: 3px; }
  .messages:hover::-webkit-scrollbar-thumb { background: var(--sand-300); }
  .messages-inner { max-width: 768px; margin: 0 auto; }

  /* ── Welcome ── */
  .welcome {
    display: flex; align-items: center; justify-content: center;
    padding-top: 128px; text-align: center;
  }
  .welcome-icon {
    width: 48px; height: 48px; border-radius: 50%;
    background: var(--brand-coral-30);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 12px;
  }
  .welcome-icon svg { width: 24px; height: 24px; stroke: var(--brand-fig); fill: none; }
  .welcome h2 {
    font-size: 16px; font-weight: 500; color: var(--sand-600); margin-bottom: 8px;
  }
  .welcome p { font-size: 13px; color: var(--sand-400); line-height: 1.5; margin-bottom: 4px; }
  .examples {
    display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px;
  }
  .examples button {
    background: white; border: 1px solid var(--sand-200);
    border-radius: 12px; padding: 8px 14px; color: var(--sand-600);
    font-size: 13px; cursor: pointer; transition: border-color 0.15s, color 0.15s;
    font-family: inherit;
  }
  .examples button:hover { border-color: var(--brand-fig); color: var(--sand-800); }

  /* ── Messages ── */
  .msg { margin-bottom: 24px; display: flex; }
  .msg.user { justify-content: flex-end; }
  .msg.assistant { justify-content: flex-start; }
  .msg-bubble-user {
    background: var(--sand-100); color: var(--sand-800);
    border-radius: 16px 16px 4px 16px;
    padding: 10px 16px; max-width: 80%;
    font-size: 14px; line-height: 1.6;
    white-space: pre-wrap; word-wrap: break-word;
  }
  .msg-bubble-assistant {
    color: var(--sand-800); max-width: 80%;
    font-size: 14px; line-height: 1.6;
  }
  /* Batch cards fill the full message column width */
  .msg.assistant:has(.batch-card) { width: 100%; }
  .msg.assistant:has(.batch-card) > .batch-card { max-width: none; }

  /* ── Streaming cursor ── */
  .streaming-cursor::after {
    content: '\25CF';
    animation: blink 1s step-end infinite;
    color: var(--brand-fig);
    font-size: 0.5em; vertical-align: middle; margin-left: 2px;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── Collapsible blocks (thinking + tool) ── */
  .thinking-block, .tool-block { margin: 6px 0; }
  .block-card {
    border: 1px solid var(--sand-200); border-radius: 8px; overflow: hidden;
  }
  .block-header {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px; background: var(--sand-50);
    cursor: pointer; user-select: none; transition: background 0.15s;
  }
  .block-header:hover { background: var(--sand-100); }
  .block-header svg { width: 14px; height: 14px; flex-shrink: 0; }
  .block-label { font-size: 12px; font-weight: 500; color: var(--sand-600); }
  .block-meta { margin-left: auto; font-size: 11px; color: var(--sand-400); }
  .block-chevron {
    width: 14px; height: 14px; color: var(--sand-400);
    transition: transform 0.2s; flex-shrink: 0;
  }
  .block-chevron.open { transform: rotate(90deg); }
  .block-body {
    display: none; padding: 8px 12px;
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
    font-size: 12px; color: var(--sand-600);
    white-space: pre-wrap; word-break: break-word;
    max-height: 256px; overflow-y: auto; line-height: 1.5;
    border-top: 1px solid var(--sand-100); background: var(--sand-50);
  }
  .block-body.open { display: block; }
  .block-body .section { margin-bottom: 8px; }
  .block-body .label {
    font-size: 11px; color: var(--sand-400);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
  }
  .pulse-dot {
    display: inline-block; width: 6px; height: 6px;
    background: var(--brand-fig); border-radius: 50%;
    animation: pulse-tool 1.5s ease-in-out infinite;
    margin-left: 4px;
  }
  @keyframes pulse-tool { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }

  /* ── Code blocks & tables ── */
  .msg-bubble-assistant pre, .msg-bubble-assistant code {
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    background: var(--sand-50); border-radius: 4px;
  }
  .msg-bubble-assistant code { padding: 2px 5px; font-size: 13px; }
  .msg-bubble-assistant pre {
    padding: 12px; overflow-x: auto; margin: 8px 0;
    border: 1px solid var(--sand-200); font-size: 12px; line-height: 1.5;
  }
  .msg-bubble-assistant pre code { padding: 0; background: none; }
  .msg-bubble-assistant table {
    width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12px;
  }
  .msg-bubble-assistant th, .msg-bubble-assistant td {
    border: 1px solid var(--sand-200); padding: 6px 12px; text-align: left;
  }
  .msg-bubble-assistant th { background: var(--sand-50); font-weight: 600; }
  .msg-bubble-assistant tr:nth-child(even) { background: var(--sand-50); }

  /* ── Input area ── */
  .input-area { padding: 8px 24px 24px; flex-shrink: 0; }
  .input-wrapper {
    max-width: 768px; margin: 0 auto; position: relative;
  }
  .input-wrapper textarea {
    width: 100%; resize: none;
    border: 1px solid var(--sand-200); border-radius: 16px;
    padding: 12px 16px 48px; font-family: inherit; font-size: 14px;
    color: var(--sand-900); outline: none;
    min-height: 52px; max-height: 200px; line-height: 1.4;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .input-wrapper textarea:focus {
    border-color: rgba(217,119,87,0.5);
    box-shadow: 0 0 0 3px rgba(217,119,87,0.1);
  }
  .input-wrapper textarea::placeholder { color: var(--sand-400); }
  .input-wrapper textarea:disabled { background: var(--sand-50); color: var(--sand-400); }
  .input-meta { position: absolute; left: 12px; bottom: 12px; display: flex; align-items: center; }
  .model-select {
    appearance: none; -webkit-appearance: none;
    background: var(--sand-50); border: 1px solid var(--sand-200); border-radius: 8px;
    padding: 4px 24px 4px 8px; font-size: 11px; font-family: inherit;
    color: var(--sand-500); cursor: pointer; outline: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23ADAAA0'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 6px center;
    transition: border-color 0.15s;
  }
  .model-select:hover { border-color: var(--sand-300); }
  .model-select:focus { border-color: var(--brand-fig); }
  .input-buttons { position: absolute; right: 12px; bottom: 12px; }
  .send-btn, .stop-btn {
    width: 36px; height: 36px; border: none; border-radius: 12px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  .send-btn {
    background: var(--brand-fig); color: white;
  }
  .send-btn:hover { background: var(--brand-fig-hover); }
  .send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .send-btn svg, .stop-btn svg { width: 16px; height: 16px; }
  .stop-btn {
    background: white; border: 1px solid var(--sand-300); color: var(--sand-600);
  }
  .stop-btn:hover { background: var(--sand-50); }

  /* ── Download button ── */
  .download-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border: 1px solid var(--brand-aqua-20);
    background: var(--brand-aqua-10); border-radius: 8px;
    color: var(--brand-aqua-dark); font-size: 12px; font-weight: 500;
    cursor: pointer; transition: all 0.15s; font-family: inherit;
  }
  .download-btn:hover { background: var(--brand-aqua-20); border-color: var(--brand-aqua); }
  .download-btn svg { flex-shrink: 0; }

  /* ── Error ── */
  .error-banner {
    background: var(--brand-orange-100); border: 1px solid rgba(232,98,53,0.2);
    color: var(--brand-orange); border-radius: 8px; padding: 12px 16px;
    font-size: 13px; margin-bottom: 24px;
  }

  /* ── Drop overlay (shown when a CSV is dragged over the chat area) ── */
  .drop-overlay {
    display: none; position: absolute; inset: 0; z-index: 50;
    background: rgba(217,119,87,0.06); border: 3px dashed var(--brand-fig);
    border-radius: 0; align-items: center; justify-content: flex-end;
    flex-direction: column; gap: 10px; pointer-events: none; padding-bottom: 144px;
  }
  .drop-overlay.active { display: flex; }
  .drop-overlay-label { font-size: 16px; font-weight: 600; color: var(--brand-fig); }
  .drop-overlay-sub { font-size: 13px; color: var(--brand-fig-hover); }

  /* ── Batch cards (rendered inline in the chat) ── */
  .batch-card {
    border: 1px solid var(--sand-200); border-radius: 10px;
    overflow: hidden; background: white; width: 100%;
  }
  .batch-plot-wrapper { overflow: visible; }
  .batch-row-header {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px; cursor: pointer; user-select: none;
    transition: background 0.12s;
  }
  .batch-row-header:hover { background: var(--sand-50); }
  .batch-row-status { flex-shrink: 0; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; margin-top: 1px; }
  .batch-row-body { flex: 1; min-width: 0; }
  .batch-row-desc { font-size: 13px; color: var(--sand-700); font-weight: 500; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .batch-row-meta { font-size: 12px; color: var(--sand-400); }
  .batch-row-downloads { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
  .batch-row-chevron { flex-shrink: 0; color: var(--sand-300); margin-top: 3px; transition: transform 0.2s; }
  .batch-row-chevron.open { transform: rotate(90deg); }
  .batch-row-log {
    display: none; border-top: 1px solid var(--sand-100);
    padding: 12px 16px; background: var(--sand-50);
  }
  .batch-row-log.open { display: block; }
  .batch-log-entry { margin-bottom: 8px; font-size: 12px; }
  .batch-log-tool {
    border: 1px solid var(--sand-200); border-radius: 6px; overflow: hidden;
  }
  .batch-log-tool-header {
    padding: 4px 8px; background: var(--sand-100);
    font-weight: 600; color: var(--sand-700);
    display: flex; align-items: center; gap: 6px;
  }
  .batch-log-tool-result {
    padding: 6px 8px; color: var(--sand-600);
    white-space: pre-wrap; word-break: break-word;
    max-height: 140px; overflow-y: auto; line-height: 1.5;
  }
  .batch-log-text { color: var(--sand-600); line-height: 1.5; padding: 2px 0; }
  .batch-log-user {
    background: var(--sand-100); border-radius: 8px; padding: 6px 10px;
    color: var(--sand-700); line-height: 1.5;
  }
  .batch-log-error { color: var(--brand-orange); line-height: 1.5; }
  /* Follow-up input inside expanded batch card */
  .batch-followup {
    display: flex; gap: 8px; padding: 10px 14px;
    border-top: 1px solid var(--sand-200); align-items: flex-end;
  }
  .batch-followup-input {
    flex: 1; resize: none; border: 1px solid var(--sand-200); border-radius: 8px;
    padding: 7px 10px; font-size: 13px; font-family: inherit; outline: none;
    line-height: 1.4; min-height: 34px; max-height: 100px; overflow-y: auto;
    background: white;
  }
  .batch-followup-input:focus { border-color: var(--brand-fig); }
  .batch-followup-send {
    width: 32px; height: 32px; flex-shrink: 0; border-radius: 8px;
    background: var(--brand-fig); border: none; cursor: pointer; color: white;
    display: flex; align-items: center; justify-content: center; transition: background 0.15s;
  }
  .batch-followup-send:hover { background: var(--brand-fig-hover); }
  .batch-followup-send:disabled { opacity: 0.35; cursor: not-allowed; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spin { animation: spin 1s linear infinite; transform-origin: center; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="header-logo">
      <svg viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5"/>
      </svg>
    </div>
    <div class="header-title">
      <h1>Plasmid Designer</h1>
      <p>Allen Institute - OCTO AI</p>
    </div>
  </div>
  <div>
    <span class="health-badge offline" id="health-badge">
      <span class="health-dot"></span>
      <span id="health-text">Agent Offline</span>
    </span>
  </div>
</div>

<!-- Main layout -->
<div class="main">
  <!-- Sessions sidebar -->
  <div class="sidebar" id="sidebar">
    <div class="sidebar-toolbar">
      <button class="sidebar-toggle-btn" onclick="toggleSidebar()" title="Hide sidebar">
        <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
          <path d="M3 6h10M3 12h18M3 18h10"/>
        </svg>
      </button>
      <button class="new-chat-btn" onclick="newChat()">
        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
          <path d="M12 4v16m8-8H4"/>
        </svg>
        New Chat
      </button>
    </div>
    <div class="sessions-list" id="sessions-list">
      <p class="no-sessions">No conversations yet</p>
    </div>
  </div>

  <!-- Sidebar reopen button -->
  <button class="sidebar-reopen-btn" id="sidebar-reopen-btn" onclick="toggleSidebar()" title="Show sidebar">
    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
      <path d="M3 6h10M3 12h18M3 18h10"/>
    </svg>
  </button>

  <!-- Chat panel -->
  <div class="chat-panel" id="chat-panel">
    <!-- Drop overlay: shown when a CSV is dragged over the chat area -->
    <div class="drop-overlay" id="drop-overlay">
      <svg width="36" height="36" fill="none" stroke="var(--brand-fig)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <div class="drop-overlay-label">Drop CSV to batch design</div>
      <div class="drop-overlay-sub">Required column: description &nbsp;·&nbsp; Optional: name, output_format</div>
    </div>
    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <div>
          <div class="welcome-icon">
            <svg viewBox="0 0 24 24" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5"/>
            </svg>
          </div>
          <h2>Design an expression plasmid</h2>
          <p>Describe what you want to build. Claude will retrieve verified sequences,<br>
          assemble your construct, validate it, and export the result.</p>
          <p style="font-size:12px;color:var(--sand-300);margin-top:4px;">
            Drag &amp; drop a CSV file here to batch design multiple plasmids at once.
          </p>
          <div class="examples">
            <button onclick="sendExample(this)">Design an EGFP expression plasmid using pcDNA3.1(+)</button>
            <button onclick="sendExample(this)">Put mCherry into a mammalian expression vector</button>
            <button onclick="sendExample(this)">What backbones are available?</button>
            <button onclick="sendExample(this)">Assemble tdTomato in pcDNA3.1(+) and export as GenBank</button>
          </div>
        </div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-wrapper">
        <textarea id="input" placeholder="Describe the plasmid you want to design…" rows="1"
          oninput="autoResize(this)"></textarea>
        <div class="input-meta">
          <select id="model-select" class="model-select">
            <option value="claude-opus-4-6">Opus 4.6</option>
            <option value="claude-sonnet-4-6">Sonnet 4.6</option>
            <option value="claude-haiku-4-5-20251001">Haiku 4.5</option>
          </select>
        </div>
        <div class="input-buttons">
          <button class="send-btn" id="send-btn" onclick="sendMessage()">
            <svg fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
              <path d="M12 19V5M5 12l7-7 7 7"/>
            </svg>
          </button>
          <button class="stop-btn" id="stop-btn" onclick="stopGeneration()" style="display:none">
            <svg fill="currentColor" viewBox="0 0 16 16">
              <rect x="3" y="3" width="10" height="10" rx="1.5"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>

  <input type="file" id="batch-csv-input" accept=".csv" style="display:none" onchange="onBatchFileChosen(this)">
</div>

<script>
// ── State ──
let currentSessionId = sessionStorage.getItem('plasmid_session_id') || null;
let sessions = [];
let isStreaming = false;
let abortController = null;

function saveSessionId(id) {
  currentSessionId = id;
  if (id) {
    sessionStorage.setItem('plasmid_session_id', id);
  } else {
    sessionStorage.removeItem('plasmid_session_id');
  }
}

// ── DOM refs ──
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const modelSelect = document.getElementById('model-select');
const sidebarEl = document.getElementById('sidebar');
const sessionsListEl = document.getElementById('sessions-list');
const reopenBtn = document.getElementById('sidebar-reopen-btn');
const healthBadge = document.getElementById('health-badge');
const healthText = document.getElementById('health-text');

// ── Helpers ──
function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function scrollToBottom() {
  // Only auto-scroll if we're viewing the session that's streaming
  if (streamingSessionId && currentSessionId !== streamingSessionId) return;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Health check ──
async function checkHealth() {
  try {
    const r = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
    const ok = r.ok;
    healthBadge.className = 'health-badge ' + (ok ? 'online' : 'offline');
    healthText.textContent = ok ? 'Agent Online' : 'Agent Offline';
  } catch {
    healthBadge.className = 'health-badge offline';
    healthText.textContent = 'Agent Offline';
  }
}

// ── Sessions ──
async function loadSessions() {
  try {
    const r = await fetch('/api/sessions');
    sessions = await r.json();
    renderSessions();
  } catch {}
}

function renderSessions() {
  if (sessions.length === 0) {
    sessionsListEl.innerHTML = '<p class="no-sessions">No conversations yet</p>';
    return;
  }
  sessionsListEl.innerHTML = sessions.map(function(s) {
    const active = s.session_id === currentSessionId ? ' active' : '';
    const name = escapeHtml((s.first_message || 'New conversation').slice(0, 40));
    return '<div class="session-item' + active + '" onclick="selectSession(\'' + s.session_id + '\')">' +
      '<span class="session-name">' + name + '</span>' +
      '<button class="delete-btn" onclick="event.stopPropagation(); deleteSessionById(\'' + s.session_id + '\')" title="Delete">' +
        '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
          '<path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>' +
        '</svg>' +
      '</button>' +
    '</div>';
  }).join('');
}

async function selectSession(sessionId) {
  // If streaming, stop the current generation before switching
  if (isStreaming) {
    stopGeneration();
    // Reset streaming UI state
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
  }

  saveSessionId(sessionId);
  renderSessions();

  try {
    const r = await fetch('/api/sessions/' + sessionId + '/messages');
    const msgs = await r.json();
    // Guard: if user switched to another session while fetch was in flight, discard
    if (currentSessionId !== sessionId) return;
    renderStoredMessages(msgs);
  } catch {
    // Don't clear messages on fetch failure (e.g., during server reload)
    // — leave the current display intact rather than showing empty state
  }
}

function renderStoredBlock(block, container) {
  const uid = 'stored-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  if (block.type === 'thinking') {
    const wc = (block.content || '').trim().split(/\s+/).length;
    const div = document.createElement('div');
    div.className = 'thinking-block';
    div.innerHTML = '<div class="block-card">' +
      '<div class="block-header" onclick="toggleBlock(\'' + uid + '\')">' +
        '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a1 1 0 01-1 1h-6a1 1 0 01-1-1v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7zM9 21h6M10 21v-1h4v1"/>' +
        '</svg>' +
        '<span class="block-label">Thought process</span>' +
        '<span class="block-meta">' + wc + ' words</span>' +
        '<svg class="block-chevron" id="' + uid + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
      '</div>' +
      '<div class="block-body" id="' + uid + '-body">' + escapeHtml(block.content || '') + '</div>' +
    '</div>';
    container.appendChild(div);
  } else if (block.type === 'tool_use') {
    const div = document.createElement('div');
    div.className = 'tool-block';
    const inputStr = JSON.stringify(block.input || {}, null, 2);
    const bodyHtml = '<div class="section"><div class="label">Input</div>' + escapeHtml(inputStr) + '</div>' +
      '<div class="section"><div class="label">Result</div>' + escapeHtml(block.result || '') + '</div>';
    div.innerHTML = '<div class="block-card">' +
      '<div class="block-header" onclick="toggleBlock(\'' + uid + '\')">' +
        '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>' +
        '</svg>' +
        '<span class="block-label">' + escapeHtml(block.name || 'tool') + '</span>' +
        '<svg class="block-chevron" id="' + uid + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
      '</div>' +
      '<div class="block-body" id="' + uid + '-body">' + bodyHtml + '</div>' +
    '</div>';
    container.appendChild(div);
    if (block.download_content && block.download_filename) {
      addDownloadButton(container, block.download_content, block.download_filename);
    }
  } else if (block.type === 'text') {
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(block.content || '') + '</div>';
    container.appendChild(div);
  }
}

function renderStoredMessages(msgs) {
  if (msgs.length === 0) {
    showWelcome();
    return;
  }
  hideWelcome();
  const inner = document.createElement('div');
  inner.className = 'messages-inner';
  msgs.forEach(function(m) {
    if (m.role === 'user') {
      const div = document.createElement('div');
      div.className = 'msg user';
      div.innerHTML = '<div class="msg-bubble-user">' + escapeHtml(m.content) + '</div>';
      inner.appendChild(div);
    } else if (m.blocks && m.blocks.length > 0) {
      m.blocks.forEach(function(block) { renderStoredBlock(block, inner); });
    } else {
      const div = document.createElement('div');
      div.className = 'msg assistant';
      div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(m.content || '') + '</div>';
      inner.appendChild(div);
    }
  });
  messagesEl.innerHTML = '';
  messagesEl.appendChild(inner);
  scrollToBottom();
}

async function deleteSessionById(sessionId) {
  try {
    await fetch('/api/sessions/' + sessionId, { method: 'DELETE' });
    if (currentSessionId === sessionId) {
      saveSessionId(null);
      showWelcome();
    }
    loadSessions();
  } catch {}
}

function newChat() {
  if (isStreaming) {
    stopGeneration();
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
  }
  saveSessionId(null);
  renderSessions();
  showWelcome();
  inputEl.focus();
}

function showWelcome() {
  messagesEl.innerHTML = '';
  const w = document.createElement('div');
  w.className = 'welcome';
  w.id = 'welcome';
  w.innerHTML = '<div>' +
    '<div class="welcome-icon">' +
      '<svg viewBox="0 0 24 24" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5"/>' +
      '</svg>' +
    '</div>' +
    '<h2>Design an expression plasmid</h2>' +
    '<p>Describe what you want to build. Claude will retrieve verified sequences,<br>' +
    'assemble your construct, validate it, and export the result.</p>' +
    '<p style="font-size:12px;color:var(--sand-300);margin-top:4px;">Drag &amp; drop a CSV file here to batch design multiple plasmids at once.</p>' +
    '<div class="examples">' +
      '<button onclick="sendExample(this)">Design an EGFP expression plasmid using pcDNA3.1(+)</button>' +
      '<button onclick="sendExample(this)">Put mCherry into a mammalian expression vector</button>' +
      '<button onclick="sendExample(this)">What backbones are available?</button>' +
      '<button onclick="sendExample(this)">Assemble tdTomato in pcDNA3.1(+) and export as GenBank</button>' +
    '</div>' +
  '</div>';
  messagesEl.appendChild(w);
}

function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}

// ── Sidebar toggle ──
function toggleSidebar() {
  sidebarEl.classList.toggle('collapsed');
  reopenBtn.classList.toggle('visible', sidebarEl.classList.contains('collapsed'));
}

// ── Markdown rendering ──
function inlineMarkdown(text) {
  let h = escapeHtml(text);
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  return h;
}

function renderContent(text) {
  const codeBlocks = [];
  text = text.replace(/```([\s\S]*?)```/g, function(match, code) {
    codeBlocks.push(code);
    return '%%CODEBLOCK' + (codeBlocks.length - 1) + '%%';
  });

  const lines = text.split('\n');
  const outputParts = [];
  let i = 0;
  while (i < lines.length) {
    if (i + 1 < lines.length &&
        lines[i].trim().startsWith('|') &&
        /^\|[\s:]*-+[\s:]*/.test(lines[i + 1].trim())) {
      const headerCells = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
      i += 2;
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        const cells = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
        bodyRows.push(cells);
        i++;
      }
      let t = '<table><thead><tr>';
      headerCells.forEach(function(c) { t += '<th>' + inlineMarkdown(c) + '</th>'; });
      t += '</tr></thead><tbody>';
      bodyRows.forEach(function(row) {
        t += '<tr>';
        row.forEach(function(c) { t += '<td>' + inlineMarkdown(c) + '</td>'; });
        t += '</tr>';
      });
      t += '</tbody></table>';
      outputParts.push(t);
    } else {
      const trimmed = lines[i].trim();
      // Horizontal rule
      if (/^-{3,}$/.test(trimmed) || /^\*{3,}$/.test(trimmed)) {
        outputParts.push('<hr style="border:none;border-top:1px solid var(--sand-200);margin:12px 0">');
        i++;
        continue;
      }
      let h = escapeHtml(lines[i]);
      h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
      h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      h = h.replace(/^### (.+)$/, '<strong style="font-size:14px">$1</strong>');
      h = h.replace(/^## (.+)$/, '<strong style="font-size:15px">$1</strong>');
      h = h.replace(/^# (.+)$/, '<strong style="font-size:16px">$1</strong>');
      outputParts.push(h);
      i++;
    }
  }

  let html = outputParts.join('<br>\n');
  codeBlocks.forEach(function(code, idx) {
    html = html.replace('%%CODEBLOCK' + idx + '%%', '<pre><code>' + escapeHtml(code) + '</code></pre>');
  });
  return html;
}

// ── Streaming blocks ──
let currentTextDiv = null;
let currentTextRaw = '';
let currentThinkingId = null;
let currentThinkingBody = null;
let currentToolId = null;
// Pinned reference to the .messages-inner container for the active stream.
// Ensures streaming writes go to the correct session even if the user
// clicks a different session in the sidebar mid-stream.
let streamingInner = null;
let streamingSessionId = null;

function getInner() {
  // While streaming, always write to the pinned container
  if (streamingInner) return streamingInner;
  let inner = messagesEl.querySelector('.messages-inner');
  if (!inner) {
    inner = document.createElement('div');
    inner.className = 'messages-inner';
    messagesEl.innerHTML = '';
    messagesEl.appendChild(inner);
  }
  return inner;
}

function toggleBlock(id) {
  const body = document.getElementById(id + '-body');
  const chevron = document.getElementById(id + '-chevron');
  if (body && chevron) {
    body.classList.toggle('open');
    chevron.classList.toggle('open');
  }
}

function startThinkingBlock() {
  currentThinkingId = 'think-' + Date.now();
  const div = document.createElement('div');
  div.className = 'thinking-block';
  div.innerHTML = '<div class="block-card">' +
    '<div class="block-header" onclick="toggleBlock(\'' + currentThinkingId + '\')">' +
      '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a1 1 0 01-1 1h-6a1 1 0 01-1-1v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7zM9 21h6M10 21v-1h4v1"/>' +
      '</svg>' +
      '<span class="block-label">Thinking...</span>' +
      '<span class="block-meta" id="' + currentThinkingId + '-meta"></span>' +
      '<svg class="block-chevron" id="' + currentThinkingId + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
    '</div>' +
    '<div class="block-body" id="' + currentThinkingId + '-body"></div>' +
  '</div>';
  getInner().appendChild(div);
  currentThinkingBody = document.getElementById(currentThinkingId + '-body');
  scrollToBottom();
}

function appendThinkingDelta(text) {
  if (currentThinkingBody) {
    currentThinkingBody.textContent += text;
    if (currentThinkingBody.classList.contains('open')) {
      currentThinkingBody.scrollTop = currentThinkingBody.scrollHeight;
    }
    scrollToBottom();
  }
}

function endThinkingBlock() {
  if (currentThinkingId) {
    const card = currentThinkingBody.closest('.block-card');
    const label = card.querySelector('.block-label');
    if (label) label.textContent = 'Thought process';
    const meta = document.getElementById(currentThinkingId + '-meta');
    if (meta && currentThinkingBody) {
      const wc = currentThinkingBody.textContent.trim().split(/\s+/).length;
      meta.textContent = wc + ' words';
    }
  }
  currentThinkingBody = null;
  currentThinkingId = null;
}

function startTextBlock() {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant"><span class="text-content"></span></div>';
  getInner().appendChild(div);
  currentTextDiv = div.querySelector('.text-content');
  currentTextRaw = '';
  scrollToBottom();
}

function appendTextDelta(text) {
  if (currentTextDiv) {
    currentTextRaw += text;
    currentTextDiv.innerHTML = renderContent(currentTextRaw);
    // Add streaming cursor
    let cursor = currentTextDiv.querySelector('.streaming-cursor');
    if (!cursor) {
      cursor = document.createElement('span');
      cursor.className = 'streaming-cursor';
      currentTextDiv.appendChild(cursor);
    }
    scrollToBottom();
  }
}

function endTextBlock() {
  if (currentTextDiv) {
    const cursor = currentTextDiv.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
  }
  currentTextDiv = null;
  currentTextRaw = '';
}

function startToolBlock(toolName) {
  currentToolId = 'tool-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'tool-block';
  div.innerHTML = '<div class="block-card">' +
    '<div class="block-header" onclick="toggleBlock(\'' + currentToolId + '\')">' +
      '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>' +
      '</svg>' +
      '<span class="block-label">' + escapeHtml(toolName) + '</span>' +
      '<span class="pulse-dot" id="' + currentToolId + '-pulse"></span>' +
      '<svg class="block-chevron" id="' + currentToolId + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
    '</div>' +
    '<div class="block-body" id="' + currentToolId + '-body"><div class="section"><div class="label">Running...</div></div></div>' +
  '</div>';
  getInner().appendChild(div);
  scrollToBottom();
}

function addPlasmidPlot(plotJson) {
  const plotId = 'plot-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant" style="margin-top:8px;padding:12px;width:100%;max-width:640px;">' +
    '<div style="font-size:11px;font-weight:600;color:var(--sand-500);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Plasmid Map</div>' +
    '<div id="' + plotId + '" style="width:100%;"></div>' +
  '</div>';
  getInner().appendChild(div);
  Bokeh.embed.embed_item(plotJson, plotId);
  scrollToBottom();
}

function addDownloadButton(container, content, filename) {
  const dlId = 'dl-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant" style="margin-top:8px">' +
    '<button class="download-btn" id="' + dlId + '">' +
      '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
        '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>' +
      '</svg>' +
      ' Download ' + escapeHtml(filename) +
    '</button></div>';
  container.appendChild(div);
  document.getElementById(dlId).addEventListener('click', function() {
    const blob = new Blob([content], {type: 'application/octet-stream'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

function finishToolBlock(toolName, toolInput, toolResult, downloadContent, downloadFilename) {
  if (currentToolId) {
    const pulse = document.getElementById(currentToolId + '-pulse');
    if (pulse) pulse.remove();
    const body = document.getElementById(currentToolId + '-body');
    if (body) {
      const inputStr = JSON.stringify(toolInput, null, 2);
      let html = '<div class="section"><div class="label">Input</div>' + escapeHtml(inputStr) + '</div>' +
        '<div class="section"><div class="label">Result</div>' + escapeHtml(toolResult) + '</div>';
      body.innerHTML = html;
    }
  }
  // Surface download button in the main chat (not just inside the collapsed tool block)
  if (downloadContent && downloadFilename) {
    addDownloadButton(getInner(), downloadContent, downloadFilename);
  }
  currentToolId = null;
  scrollToBottom();
}

// ── Send / Stop ──
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isStreaming) return;

  isStreaming = true;
  streamingSessionId = currentSessionId;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  inputEl.value = '';
  inputEl.disabled = true;
  autoResize(inputEl);
  hideWelcome();

  const inner = getInner();
  // Pin this container so stream events write here even if user switches sessions
  streamingInner = inner;
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.innerHTML = '<div class="msg-bubble-user">' + escapeHtml(text) + '</div>';
  inner.appendChild(userDiv);
  scrollToBottom();

  abortController = new AbortController();

  try {
    const reqBody = { message: text, model: modelSelect.value };
    if (currentSessionId) reqBody.session_id = currentSessionId;

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(reqBody),
      signal: abortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {stream: true});
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      let streamDone = false;
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const jsonStr = trimmed.slice(6);
        if (!jsonStr) continue;

        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }

        switch (event.type) {
          case 'session_id':
            saveSessionId(event.session_id);
            loadSessions();
            break;
          case 'thinking_start': startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': startTextBlock(); break;
          case 'text_delta': appendTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'error':
            startTextBlock();
            appendTextDelta('Error: ' + event.content);
            endTextBlock();
            break;
          case 'done': streamDone = true; break;
        }
        if (streamDone) break;
      }
      if (streamDone) break;
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      startTextBlock();
      appendTextDelta('Connection error: ' + err.message);
      endTextBlock();
    }
  }

  isStreaming = false;
  abortController = null;
  streamingInner = null;
  streamingSessionId = null;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  inputEl.disabled = false;
  inputEl.focus();
  // Remove any leftover streaming cursor
  const cursor = messagesEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
}

function stopGeneration() {
  if (abortController) abortController.abort();
  if (currentSessionId) {
    fetch('/api/sessions/' + currentSessionId + '/cancel', { method: 'POST' }).catch(function(){});
  }
}

function sendExample(btn) {
  inputEl.value = btn.textContent;
  sendMessage();
}

// ── Keyboard ──
inputEl.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Init ──
checkHealth();
setInterval(checkHealth, 5000);
loadSessions();
setInterval(loadSessions, 5000);
// Restore active session on page load
if (currentSessionId) {
  selectSession(currentSessionId);
}
inputEl.focus();

// ── Batch ──
let batchJobId = null;
let batchPollTimer = null;
const chatPanelEl = document.getElementById('chat-panel');
const dropOverlayEl = document.getElementById('drop-overlay');

// ── Drag & drop CSV onto the chat area ──
var dragCounter = 0;

function isCsvDrag(e) {
  var types = e.dataTransfer && e.dataTransfer.types;
  return types && (Array.from(types).indexOf('Files') !== -1);
}

chatPanelEl.addEventListener('dragenter', function(e) {
  if (!isCsvDrag(e)) return;
  e.preventDefault();
  dragCounter++;
  dropOverlayEl.classList.add('active');
});

chatPanelEl.addEventListener('dragleave', function(e) {
  if (!isCsvDrag(e)) return;
  dragCounter--;
  if (dragCounter <= 0) { dragCounter = 0; dropOverlayEl.classList.remove('active'); }
});

chatPanelEl.addEventListener('dragover', function(e) {
  if (!isCsvDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

chatPanelEl.addEventListener('drop', function(e) {
  e.preventDefault();
  dragCounter = 0;
  dropOverlayEl.classList.remove('active');
  var file = e.dataTransfer.files[0];
  if (!file) return;
  if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
    alert('Please drop a .csv file.');
    return;
  }
  var reader = new FileReader();
  reader.onload = function(ev) { uploadBatchCSV(ev.target.result, file.name); };
  reader.readAsText(file);
});

function onBatchFileChosen(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) { uploadBatchCSV(e.target.result, file.name); };
  reader.readAsText(file);
  input.value = '';
}

function uploadBatchCSV(csvText, filename) {
  var model = modelSelect.value;
  fetch('/api/batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({csv_content: csvText, model: model}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    batchJobId = data.job_id;
    initBatchCards(data.job_id, data.row_count, filename);
    if (batchPollTimer) clearInterval(batchPollTimer);
    batchPollTimer = setInterval(pollBatchStatus, 2000);
    pollBatchStatus();
  })
  .catch(function(e) { alert('Upload failed: ' + e); });
}

function initBatchCards(jobId, count, filename) {
  hideWelcome();
  var inner = getInner();
  // Label
  var label = document.createElement('div');
  label.className = 'msg assistant';
  label.id = 'batch-label-' + jobId;
  label.innerHTML = '<div class="msg-bubble-assistant" style="color:var(--sand-500);font-size:13px;">' +
    'Batch designing <strong>' + count + ' plasmid' + (count === 1 ? '' : 's') + '</strong> from <em>' + escapeHtml(filename) + '</em>. ' +
    'Click any row to expand and see what\u2019s happening, or send a follow-up once it finishes.' +
    '</div>';
  inner.appendChild(label);
  // Placeholder cards
  for (var i = 0; i < count; i++) {
    var card = document.createElement('div');
    card.className = 'msg assistant';
    card.id = 'batch-card-' + jobId + '-' + i;
    card.innerHTML = buildBatchCardHtml(jobId, i, {
      status: 'pending', description: '\u2026', exports: [], error: null, log: []
    }, false);
    inner.appendChild(card);
  }
  scrollToBottom();
}

function pollBatchStatus() {
  if (!batchJobId) return;
  fetch('/api/batch/' + batchJobId)
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) return;
    updateBatchCards(batchJobId, data.rows);
    var anyRunning = data.rows && data.rows.some(function(r) { return r.status === 'running' || r.status === 'pending'; });
    if (data.status === 'done' && !anyRunning) {
      clearInterval(batchPollTimer);
      batchPollTimer = null;
      // Add Download All button to label message
      var labelEl = document.getElementById('batch-label-' + batchJobId);
      if (labelEl && !labelEl.querySelector('.batch-dl-all-btn')) {
        var bubble = labelEl.querySelector('.msg-bubble-assistant');
        if (bubble) {
          var btn = document.createElement('button');
          btn.className = 'download-btn batch-dl-all-btn';
          btn.style.cssText = 'margin-top:8px;display:inline-flex;';
          btn.innerHTML = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download All (.zip)';
          btn.onclick = function() { downloadAllBatch(batchJobId); };
          bubble.appendChild(document.createElement('br'));
          bubble.appendChild(btn);
        }
      }
    }
  })
  .catch(function() {});
}

var STATUS_ICONS = {
  pending: '<svg width="18" height="18" fill="none" stroke="var(--sand-300)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>',
  running: '<svg width="18" height="18" fill="none" stroke="var(--brand-fig)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" class="spin"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>',
  done: '<svg width="18" height="18" fill="none" stroke="var(--brand-aqua)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
  no_export: '<svg width="18" height="18" fill="none" stroke="var(--sand-400)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>',
  error: '<svg width="18" height="18" fill="none" stroke="var(--brand-orange)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg>',
};
var STATUS_LABELS = {pending: 'Pending', running: 'Running\u2026', done: 'Done', no_export: 'No export produced', error: 'Error'};
var CHEV_SVG = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>';

function renderBatchLog(log) {
  if (!log || !log.length) return '<div style="font-size:12px;color:var(--sand-400);padding:4px 0;">No activity yet.</div>';
  return log.map(function(entry) {
    if (entry.type === 'tool') {
      return '<div class="batch-log-entry batch-log-tool">' +
        '<div class="batch-log-tool-header">' +
          '<svg width="11" height="11" fill="none" stroke="var(--brand-fig)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3-3a1 1 0 000-1.4l-1.6-1.6a1 1 0 00-1.4 0l-3 3z"/><path d="M20.26 2.26L9 13.5l-5 1 1-5L16.5 3.74"/></svg>' +
          escapeHtml(entry.name) +
        '</div>' +
        '<div class="batch-log-tool-result">' + escapeHtml(entry.result || '') + '</div>' +
      '</div>';
    } else if (entry.type === 'text') {
      return '<div class="batch-log-entry batch-log-text">' + renderContent(entry.content || '') + '</div>';
    } else if (entry.type === 'user') {
      return '<div class="batch-log-entry batch-log-user">' + escapeHtml(entry.content || '') + '</div>';
    } else if (entry.type === 'error') {
      return '<div class="batch-log-entry batch-log-error">\u26a0 ' + escapeHtml(entry.content || '') + '</div>';
    }
    return '';
  }).join('');
}

function buildDownloadsHtml(jobId, idx, exports) {
  if (!exports || !exports.length) return '';
  var html = '<div class="batch-row-downloads">';
  exports.forEach(function(exp, eidx) {
    html += '<button class="download-btn" onclick="event.stopPropagation();downloadBatchFile(\'' + jobId + '\',' + idx + ',' + eidx + ',\'' + escapeHtml(exp.filename) + '\')">' +
      '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
      escapeHtml(exp.filename) + '</button>';
    if (exp.has_plot) {
      html += '<button class="download-btn" style="border-color:var(--brand-fig-30);color:var(--brand-fig);background:var(--brand-fig-10);" ' +
        'onclick="event.stopPropagation();openBatchPlot(\'' + jobId + '\',' + idx + ',' + eidx + ')">' +
        '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>' +
        'View Map</button>';
    }
  });
  return html + '</div>';
}

function buildFollowupHtml(jobId, idx, status) {
  if (status === 'running' || status === 'pending') return '';
  var fid = 'batch-finput-' + jobId + '-' + idx;
  return '<div class="batch-followup">' +
    '<textarea class="batch-followup-input" id="' + fid + '" rows="1" ' +
      'placeholder="Follow up with the agent about this design\u2026" ' +
      'onkeydown="batchFollowupKey(event,\'' + jobId + '\',' + idx + ')" ' +
      'oninput="this.style.height=\'auto\';this.style.height=Math.min(this.scrollHeight,100)+\'px\'"></textarea>' +
    '<button class="batch-followup-send" onclick="sendBatchFollowup(\'' + jobId + '\',' + idx + ')" title="Send">' +
      '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>' +
    '</button>' +
  '</div>';
}

function buildBatchCardHtml(jobId, idx, row, isOpen) {
  var icon = STATUS_ICONS[row.status] || STATUS_ICONS.pending;
  var label = STATUS_LABELS[row.status] || row.status;
  var desc = escapeHtml((row.description || '').slice(0, 120) + ((row.description || '').length > 120 ? '\u2026' : ''));
  var downloads = buildDownloadsHtml(jobId, idx, row.exports);
  var logId = 'batch-log-' + jobId + '-' + idx;
  var chevId = 'batch-chev-' + jobId + '-' + idx;
  return '<div class="batch-card">' +
    '<div class="batch-row-header" onclick="toggleBatchCard(\'' + jobId + '\',' + idx + ')">' +
      '<div class="batch-row-status">' + icon + '</div>' +
      '<div class="batch-row-body">' +
        '<div class="batch-row-desc">' + desc + '</div>' +
        '<div class="batch-row-meta">' + (idx + 1) + ' \xb7 ' + label + '</div>' +
        downloads +
      '</div>' +
      '<span id="' + chevId + '" class="batch-row-chevron' + (isOpen ? ' open' : '') + '">' + CHEV_SVG + '</span>' +
    '</div>' +
    '<div id="' + logId + '" class="batch-row-log' + (isOpen ? ' open' : '') + '">' +
      renderBatchLog(row.log) +
      buildFollowupHtml(jobId, idx, row.status) +
    '</div>' +
  '</div>';
}

function updateBatchCards(jobId, rows) {
  rows.forEach(function(row, idx) {
    var cardEl = document.getElementById('batch-card-' + jobId + '-' + idx);
    if (!cardEl) return;
    // Preserve expanded state
    var logEl = document.getElementById('batch-log-' + jobId + '-' + idx);
    var isOpen = logEl ? logEl.classList.contains('open') : false;
    cardEl.innerHTML = buildBatchCardHtml(jobId, idx, row, isOpen);
  });
}

function toggleBatchCard(jobId, idx) {
  var log = document.getElementById('batch-log-' + jobId + '-' + idx);
  var chev = document.getElementById('batch-chev-' + jobId + '-' + idx);
  if (!log) return;
  var open = log.classList.toggle('open');
  if (chev) chev.classList.toggle('open', open);
}

function batchFollowupKey(e, jobId, rowIdx) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBatchFollowup(jobId, rowIdx); }
}

function sendBatchFollowup(jobId, rowIdx) {
  var inputEl = document.getElementById('batch-finput-' + jobId + '-' + rowIdx);
  if (!inputEl) return;
  var message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  // Optimistically show the user message in the log
  var logEl = document.getElementById('batch-log-' + jobId + '-' + rowIdx);
  if (logEl) {
    var followup = logEl.querySelector('.batch-followup');
    var userDiv = document.createElement('div');
    userDiv.className = 'batch-log-entry batch-log-user';
    userDiv.textContent = message;
    if (followup) logEl.insertBefore(userDiv, followup);
    else logEl.appendChild(userDiv);
    // Disable input while running
    if (followup) {
      var btn = followup.querySelector('.batch-followup-send');
      if (inputEl) inputEl.disabled = true;
      if (btn) btn.disabled = true;
    }
  }
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/continue', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: message}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    if (!batchPollTimer) batchPollTimer = setInterval(pollBatchStatus, 2000);
  })
  .catch(function(e) { alert('Failed to send: ' + e); });
}

function openBatchPlot(jobId, rowIdx, expIdx) {
  // Expand the card if collapsed
  var log = document.getElementById('batch-log-' + jobId + '-' + rowIdx);
  var chev = document.getElementById('batch-chev-' + jobId + '-' + rowIdx);
  if (log && !log.classList.contains('open')) {
    log.classList.add('open');
    if (chev) chev.classList.add('open');
  }
  // Don't render twice
  var plotWrapperId = 'bplotwrap-' + jobId + '-' + rowIdx + '-' + expIdx;
  if (document.getElementById(plotWrapperId)) return;
  var plotId = 'bplot-' + jobId + '-' + rowIdx + '-' + expIdx;
  // Insert plot container before the follow-up input
  var wrapper = document.createElement('div');
  wrapper.id = plotWrapperId;
  wrapper.className = 'batch-plot-wrapper';
  wrapper.style.cssText = 'padding:12px 16px;border-top:1px solid var(--sand-100);max-width:640px;';
  wrapper.innerHTML =
    '<div style="font-size:11px;font-weight:600;color:var(--sand-500);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;">Plasmid Map</div>' +
    '<div id="' + plotId + '" style="width:600px;height:600px;">Loading\u2026</div>';
  if (log) {
    var followup = log.querySelector('.batch-followup');
    if (followup) log.insertBefore(wrapper, followup);
    else log.appendChild(wrapper);
  }
  // Fetch the plot JSON then wait one animation frame so the browser has
  // laid out the container before Bokeh reads its dimensions.
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/plot/' + expIdx)
  .then(function(r) { return r.json(); })
  .then(function(data) {
    var el = document.getElementById(plotId);
    if (!el) return;
    if (data.error) { el.textContent = 'No map available.'; el.style.minHeight = ''; return; }
    el.innerHTML = '';
    // Double rAF ensures the element is fully painted before Bokeh measures it
    requestAnimationFrame(function() {
      requestAnimationFrame(function() {
        Bokeh.embed.embed_item(data, plotId);
      });
    });
  })
  .catch(function() {
    var el = document.getElementById(plotId);
    if (el) { el.textContent = 'Failed to load map.'; el.style.minHeight = ''; }
  });
}

function downloadAllBatch(jobId) {
  var a = document.createElement('a');
  a.href = '/api/batch/' + jobId + '/download-all';
  a.download = 'batch_designs.zip';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function downloadBatchFile(jobId, rowIdx, expIdx, filename) {
  fetch('/api/batch/' + jobId + '/download/' + rowIdx + '/' + expIdx)
  .then(function(r) { return r.blob(); })
  .then(function(blob) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  })
  .catch(function(e) { alert('Download failed: ' + e); });
}
</script>
</body>
</html>
"""


# ── Batch job runner ────────────────────────────────────────────────────

def _run_batch_row(job_id: str, row_idx: int, row: dict, model: str) -> None:
    """Worker for a single CSV row — runs the agent and stores exports + log in _batch_jobs."""
    job = _batch_jobs.get(job_id)
    if not job:
        return

    row_state = job["rows"][row_idx]
    description = row.get("description", "").strip()
    output_format = (row.get("output_format") or "genbank").strip().lower()

    if output_format == "both":
        prompt = description + "\nPlease export the final construct in both GenBank and FASTA formats."
    elif output_format == "fasta":
        prompt = description + "\nPlease export the final construct in FASTA format."
    else:
        prompt = description + "\nPlease export the final construct in GenBank format."

    row_state["status"] = "running"
    row_state["log"] = []

    def append_log(entry: dict):
        row_state["log"].append(entry)

    try:
        client = anthropic.Anthropic()
        tracker = ReferenceTracker()
        history = [{"role": "user", "content": prompt}]
        exports: list[dict] = []

        for _ in range(15):
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=history,
                thinking={"type": "enabled", "budget_tokens": 5000},
            )

            # Log any text blocks
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    append_log({"type": "text", "content": block.text})

            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = execute_tool(block.name, block.input, tracker)
                # Truncate long results for the log display
                result_preview = result[:600] + ("\u2026" if len(result) > 600 else "")
                append_log({
                    "type": "tool",
                    "name": block.name,
                    "input": block.input,
                    "result": result_preview,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
                if block.name == "export_construct":
                    fmt = block.input.get("output_format", "genbank")
                    cname = block.input.get("construct_name", "construct")
                    ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
                    name = row.get("name", "").strip() or f"plasmid_{row_idx + 1:03d}"
                    _plot_str = _get_last_plot_json()
                    exports.append({
                        "filename": name + ext,
                        "content": result,
                        "plot_json": json.loads(_plot_str) if _plot_str else None,
                    })

            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})

        row_state["exports"] = exports
        row_state["history"] = history  # persist for follow-up turns
        row_state["status"] = "done" if exports else "no_export"

    except Exception as e:
        row_state["status"] = "error"
        row_state["error"] = str(e)
        row_state["log"].append({"type": "error", "content": str(e)})


def _strip_thinking_blocks(history: list) -> list:
    """Remove thinking blocks from assistant messages so follow-ups can run without thinking."""
    clean = []
    for msg in history:
        content = msg.get("content")
        if msg.get("role") == "assistant" and isinstance(content, list):
            filtered = [
                b for b in content
                if not (getattr(b, "type", None) == "thinking" or
                        (isinstance(b, dict) and b.get("type") == "thinking"))
            ]
            if filtered:
                clean.append({"role": "assistant", "content": filtered})
        else:
            clean.append(msg)
    return clean


def _continue_batch_row(job_id: str, row_idx: int, user_message: str) -> None:
    """Continue a finished batch row with a follow-up user message."""
    job = _batch_jobs.get(job_id)
    if not job:
        return
    row_state = job["rows"][row_idx]
    model = job["model"]

    row_state["status"] = "running"
    row_state["log"].append({"type": "user", "content": user_message})

    # Strip thinking blocks so follow-up calls don't require thinking enabled
    history = _strip_thinking_blocks(list(row_state.get("history", [])))
    history.append({"role": "user", "content": user_message})

    try:
        client = anthropic.Anthropic()
        tracker = ReferenceTracker()

        for _ in range(15):
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=history,
            )

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    row_state["log"].append({"type": "text", "content": block.text})

            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = execute_tool(block.name, block.input, tracker)
                result_preview = result[:600] + ("\u2026" if len(result) > 600 else "")
                row_state["log"].append({
                    "type": "tool",
                    "name": block.name,
                    "input": block.input,
                    "result": result_preview,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
                if block.name == "export_construct":
                    fmt = block.input.get("output_format", "genbank")
                    cname = block.input.get("construct_name", "construct")
                    ext = {"genbank": ".gb", "gb": ".gb", "fasta": ".fasta"}.get(fmt, ".txt")
                    name = row_state.get("name", "").strip() or f"plasmid_{row_idx + 1:03d}"
                    _plot_str = _get_last_plot_json()
                    row_state["exports"].append({
                        "filename": name + ext,
                        "content": result,
                        "plot_json": json.loads(_plot_str) if _plot_str else None,
                    })

            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})

        row_state["history"] = history
        row_state["status"] = "done" if row_state["exports"] else "no_export"

    except Exception as e:
        row_state["status"] = "error"
        row_state["error"] = str(e)
        row_state["log"].append({"type": "error", "content": str(e)})


def start_batch_job(rows: list, model: str) -> str:
    """Create a batch job, launch a background thread, return job_id."""
    job_id = str(uuid.uuid4())
    job: dict = {
        "status": "running",
        "model": model,
        "rows": [
            {
                "description": r.get("description", ""),
                "name": r.get("name", ""),
                "output_format": r.get("output_format", "genbank"),
                "status": "pending",
                "exports": [],
                "error": None,
            }
            for r in rows
        ],
    }
    _batch_jobs[job_id] = job

    # Run rows sequentially in one daemon thread to avoid hammering the API
    def worker():
        for idx, row in enumerate(rows):
            _run_batch_row(job_id, idx, row, model)
        job["status"] = "done"

    threading.Thread(target=worker, daemon=True).start()
    return job_id


# ── HTTP Server ─────────────────────────────────────────────────────────

class AgentHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving the UI and API endpoints."""

    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/health":
            self._send_json({"status": "ok"})

        elif path == "/api/sessions":
            self._send_json(list_sessions())

        elif path.startswith("/api/sessions/") and path.endswith("/messages"):
            session_id = path.split("/")[3]
            session = get_session(session_id)
            if session:
                self._send_json(session["display_messages"])
            else:
                self._send_json([], 404)

        elif path.startswith("/api/batch/") and path.endswith("/download-all"):
            # GET /api/batch/{job_id}/download-all — ZIP of all exports
            import zipfile as _zipfile
            job_id = path.split("/")[3]
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            buf = io.BytesIO()
            with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
                for row in job["rows"]:
                    for exp in row.get("exports", []):
                        zf.writestr(exp["filename"], exp["content"])
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="batch_designs.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif path.startswith("/api/batch/") and "/rows/" in path and "/plot/" in path:
            # GET /api/batch/{job_id}/rows/{row_idx}/plot/{export_idx}
            parts = path.split("/")
            try:
                job_id = parts[3]
                row_idx = int(parts[5])
                export_idx = int(parts[7]) if len(parts) > 7 else 0
                export = _batch_jobs[job_id]["rows"][row_idx]["exports"][export_idx]
                plot_json = export.get("plot_json")
                if not plot_json:
                    self._send_json({"error": "No plot available"}, 404)
                    return
                self._send_json(plot_json)
            except (KeyError, IndexError, ValueError):
                self.send_error(404)

        elif path.startswith("/api/batch/") and "/download/" in path:
            # GET /api/batch/{job_id}/download/{row_idx}/{export_idx}
            parts = path.split("/")
            try:
                job_id = parts[3]
                row_idx = int(parts[5])
                export_idx = int(parts[6]) if len(parts) > 6 else 0
                export = _batch_jobs[job_id]["rows"][row_idx]["exports"][export_idx]
                filename = export["filename"]
                content = export["content"]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except (KeyError, IndexError, ValueError):
                self.send_error(404)

        elif path.startswith("/api/batch/"):
            # GET /api/batch/{job_id} — return job status (no full file content)
            job_id = path.split("/")[3]
            job = _batch_jobs.get(job_id)
            if job:
                rows_summary = [
                    {
                        "description": r["description"],
                        "name": r["name"],
                        "status": r["status"],
                        "error": r["error"],
                        "exports": [
                            {"filename": e["filename"], "has_plot": bool(e.get("plot_json"))}
                            for e in r["exports"]
                        ],
                        "log": r.get("log", []),
                    }
                    for r in job["rows"]
                ]
                self._send_json({"status": job["status"], "rows": rows_summary})
            else:
                self._send_json({"error": "Job not found"}, 404)

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            user_message = body.get("message", "")
            request_model = body.get("model", MODEL)

            if not user_message.strip():
                self._send_json({"error": "Empty message"}, 400)
                return

            # Get or create session.
            # If a session_id was provided but doesn't exist, that's an error
            # (stale client state) — don't silently create a fresh one, or the
            # user thinks they're continuing a conversation when they're not.
            session_id = body.get("session_id")
            if session_id and not get_session(session_id):
                self._send_json({
                    "error": (
                        "Session not found. It may have expired or been "
                        "cleared. Please start a new conversation."
                    )
                }, 404)
                return
            if not session_id:
                session_id = create_session()

            # SSE streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def write_event(data: dict):
                try:
                    line = f"data: {json.dumps(data)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            # Send session_id to client
            write_event({"type": "session_id", "session_id": session_id})

            try:
                run_agent_turn_streaming(user_message, session_id, write_event, model=request_model)
            except anthropic.AuthenticationError:
                write_event({"type": "error", "content": "Invalid or missing ANTHROPIC_API_KEY."})
            except Exception as e:
                logger.exception("Agent error")
                write_event({"type": "error", "content": str(e)})

        elif path.startswith("/api/sessions/") and path.endswith("/cancel"):
            session_id = path.split("/")[3]
            cancel_session(session_id)
            self._send_json({"status": "ok"})

        elif path.startswith("/api/sessions/") and path.endswith("/outcome"):
            # POST /api/sessions/{id}/outcome — record experimental result
            session_id = path.split("/")[3]
            session = get_session(session_id)
            if not session:
                self._send_json({"error": "Session not found"}, 404)
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            status = body.get("status")
            observation = body.get("observation")
            if status not in ("success", "failed", "partial"):
                self._send_json({"error": "status must be 'success', 'failed', or 'partial'"}, 400)
                return
            if not observation:
                self._send_json({"error": "observation is required"}, 400)
                return
            session.setdefault("experimental_outcomes", []).append({
                "status": status,
                "observation": observation,
                "construct_name": body.get("construct_name", ""),
                "timestamp": time.time(),
            })
            if body.get("project_name"):
                session["project_name"] = body["project_name"]
            _save_sessions()
            self._send_json({
                "status": "ok",
                "outcomes_count": len(session["experimental_outcomes"]),
            })

        elif path.startswith("/api/batch/") and "/rows/" in path and path.endswith("/continue"):
            # POST /api/batch/{job_id}/rows/{row_idx}/continue
            parts = path.split("/")
            try:
                job_id = parts[3]
                row_idx = int(parts[5])
            except (IndexError, ValueError):
                self._send_json({"error": "Bad request"}, 400)
                return
            job = _batch_jobs.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
                return
            row = job["rows"][row_idx]
            if row["status"] == "running":
                self._send_json({"error": "Row is still running"}, 409)
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            message = body.get("message", "").strip()
            if not message:
                self._send_json({"error": "Empty message"}, 400)
                return
            threading.Thread(
                target=_continue_batch_row,
                args=(job_id, row_idx, message),
                daemon=True,
            ).start()
            self._send_json({"status": "ok"})

        elif path == "/api/batch":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            csv_text = body.get("csv_content", "")
            request_model = body.get("model", MODEL)

            if not csv_text.strip():
                self._send_json({"error": "No CSV content provided"}, 400)
                return

            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)

            if not rows or "description" not in rows[0]:
                self._send_json({"error": "CSV must have a 'description' column"}, 400)
                return

            rows = [r for r in rows if r.get("description", "").strip()]
            if not rows:
                self._send_json({"error": "No non-empty rows found"}, 400)
                return

            job_id = start_batch_job(rows, request_model)
            self._send_json({"job_id": job_id, "row_count": len(rows)})

        elif path == "/api/reset":
            # Legacy endpoint — clear all sessions
            _sessions.clear()
            _save_sessions()
            self._send_json({"status": "ok"})

        else:
            self.send_error(404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/sessions/"):
            session_id = path.split("/")[3]
            deleted = delete_session_by_id(session_id)
            self._send_json({"deleted": deleted})
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def _run_server(port: int):
    """Run the HTTP server."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("=" * 60)
        print("WARNING: ANTHROPIC_API_KEY not set.")
        print("Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("The UI will load but chat will fail without it.")
        print("=" * 60)
        print()

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadingHTTPServer(("0.0.0.0", port), AgentHandler)
    print(f"Plasmid Designer running at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


def _run_with_reload(port: int):
    """Watch for file changes and restart the server automatically."""
    import subprocess

    watch_paths = [Path(__file__).parent, PROJECT_ROOT / "src"]

    def get_mtimes() -> dict[str, float]:
        mtimes = {}
        for d in watch_paths:
            if not d.exists():
                continue
            for f in d.rglob("*.py"):
                try:
                    mtimes[str(f)] = f.stat().st_mtime
                except OSError:
                    pass
        return mtimes

    print(f"Plasmid Designer running at http://localhost:{port} (auto-reload enabled)")
    print("Watching for file changes in app/ and src/...")
    print("Press Ctrl+C to stop.\n")

    while True:
        mtimes = get_mtimes()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--port", str(port)]
        proc = subprocess.Popen(cmd)

        try:
            while True:
                time.sleep(1)
                new_mtimes = get_mtimes()
                if new_mtimes != mtimes:
                    changed = set()
                    for f in set(list(mtimes.keys()) + list(new_mtimes.keys())):
                        if mtimes.get(f) != new_mtimes.get(f):
                            changed.add(Path(f).name)
                    print(f"\nFile changes detected: {', '.join(sorted(changed))}")
                    print("Restarting server...\n")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

                if proc.poll() is not None:
                    print("\nServer process exited.")
                    return
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("\nShutting down.")
            return


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plasmid Designer Web UI")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--reload", action="store_true", help="Auto-reload on file changes")
    args = parser.parse_args()

    if args.reload:
        _run_with_reload(args.port)
    else:
        _run_server(args.port)


if __name__ == "__main__":
    main()
