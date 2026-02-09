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

import json
import os
import sys
import logging
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
import threading
import uuid
import time

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed; rely on environment variables

import anthropic

# Add src/ to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assembler import (
    assemble_construct as _assemble_construct,
    fuse_sequences as _fuse_sequences,
    find_mcs_insertion_point,
    clean_sequence,
    validate_dna,
    format_as_fasta,
    format_as_genbank,
)

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
        "description": "Get complete information about a specific insert, including its DNA sequence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "insert_id": {"type": "string", "description": "Insert ID or name"},
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
            },
            "required": ["sequence", "output_format"],
        },
    },
    {
        "name": "validate_construct",
        "description": "Validate an assembled construct. Checks backbone preservation, insert presence/position/orientation, size, and biology.",
        "input_schema": {
            "type": "object",
            "properties": {
                "construct_sequence": {"type": "string", "description": "Assembled construct to validate"},
                "backbone_id": {"type": "string", "description": "Expected backbone ID"},
                "insert_id": {"type": "string", "description": "Expected insert ID"},
                "backbone_sequence": {"type": "string", "description": "Expected backbone sequence"},
                "insert_sequence": {"type": "string", "description": "Expected insert sequence"},
                "expected_insert_position": {"type": "integer", "description": "Expected insert position"},
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
                "fetch_sequence": {"type": "boolean", "description": "Fetch DNA sequence", "default": True},
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
        "description": "Fuse multiple coding sequences into a single CDS for protein tagging or fusion proteins. Handles start/stop codon management. Use for N-terminal tags (FLAG-GeneX), C-terminal tags (GeneX-FLAG), or fusions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "inserts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "insert_id": {"type": "string", "description": "Insert ID from library"},
                            "sequence": {"type": "string", "description": "Raw DNA sequence"},
                            "name": {"type": "string", "description": "Name for this sequence"},
                        },
                    },
                    "description": "Ordered list of sequences to fuse (N-terminal first, C-terminal last)",
                },
                "linker": {"type": "string", "description": "Optional linker DNA between fusion partners"},
            },
            "required": ["inserts"],
        },
    },
]


# ── Tool execution ──────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
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
                return f"Backbone '{args['backbone_id']}' not found in library."
            out = format_backbone_summary(bb)
            if args.get("include_sequence") and bb.get("sequence"):
                out += f"\n\nDNA Sequence ({len(bb['sequence'])} bp):\n{bb['sequence'][:200]}... [{len(bb['sequence'])} bp total]"
            return out

        elif name == "search_inserts":
            results = search_inserts(args["query"], args.get("category"))
            if not results:
                return f"No inserts found matching '{args['query']}'"
            return "\n\n---\n\n".join(format_insert_summary(ins) for ins in results)

        elif name == "get_insert":
            ins = get_insert_by_id(args["insert_id"])
            if not ins:
                return f"Insert '{args['insert_id']}' not found in library."
            out = format_insert_summary(ins)
            if ins.get("sequence"):
                out += f"\n\nDNA Sequence ({len(ins['sequence'])} bp):\n{ins['sequence']}"
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

            # Resolve insert
            insert_seq = args.get("insert_sequence")
            insert_data = None
            if not insert_seq and args.get("insert_id"):
                insert_data = get_insert_by_id(args["insert_id"])
                if insert_data:
                    insert_seq = insert_data.get("sequence")
            if not insert_seq:
                return "Error: No insert sequence available. Provide insert_id or insert_sequence."

            # Resolve position
            pos = args.get("insertion_position")
            if pos is None and backbone_data:
                pos = find_mcs_insertion_point(backbone_data)
            if pos is None:
                return "Error: No insertion position. Provide insertion_position or use a backbone with MCS data."

            result = _assemble_construct(
                backbone_seq=backbone_seq,
                insert_seq=insert_seq,
                insertion_position=pos,
                replace_region_end=args.get("replace_region_end"),
                reverse_complement_insert=args.get("reverse_complement_insert", False),
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
            out += f"\nAssembled sequence ({result.total_size_bp} bp):\n{result.sequence}"
            return out

        elif name == "export_construct":
            seq = clean_sequence(args["sequence"])
            fmt = args["output_format"]
            cname = args.get("construct_name", "construct")
            bname = args.get("backbone_name", "")
            iname = args.get("insert_name", "")
            ipos = args.get("insert_position", 0)
            ilen = args.get("insert_length", 0)

            if fmt == "raw":
                return seq
            elif fmt == "fasta":
                desc = f"{iname} in {bname}, {len(seq)} bp" if bname else f"{len(seq)} bp"
                return format_as_fasta(seq, cname, desc)
            elif fmt in ("genbank", "gb"):
                return format_as_genbank(
                    sequence=seq, name=cname, backbone_name=bname,
                    insert_name=iname, insert_position=ipos, insert_length=ilen,
                )
            else:
                return f"Unknown format: {fmt}"

        elif name == "validate_construct":
            construct_seq = clean_sequence(args["construct_sequence"])
            backbone_seq = args.get("backbone_sequence")
            if not backbone_seq and args.get("backbone_id"):
                bb = get_backbone_by_id(args["backbone_id"])
                if bb:
                    backbone_seq = bb.get("sequence")
            insert_seq = args.get("insert_sequence")
            if not insert_seq and args.get("insert_id"):
                ins = get_insert_by_id(args["insert_id"])
                if ins:
                    insert_seq = ins.get("sequence")

            checks = []
            # Valid DNA
            ok, errs = validate_dna(construct_seq)
            checks.append(f"Valid DNA: {'PASS' if ok else 'FAIL'}")
            checks.append(f"Size: {len(construct_seq)} bp")

            if insert_seq:
                insert_seq = clean_sequence(insert_seq)
                found = insert_seq in construct_seq
                checks.append(f"Insert found in construct: {'PASS' if found else 'FAIL (CRITICAL)'}")
                if found:
                    pos = construct_seq.index(insert_seq)
                    checks.append(f"Insert position: {pos}")
                    exp = args.get("expected_insert_position")
                    if exp is not None:
                        checks.append(f"Position correct: {'PASS' if pos == exp else 'FAIL — expected ' + str(exp)}")
                    start_ok = insert_seq[:3] == "ATG"
                    stop_ok = insert_seq[-3:] in ("TAA", "TAG", "TGA")
                    checks.append(f"Start codon: {'PASS' if start_ok else 'FAIL (Minor)'}")
                    checks.append(f"Stop codon: {'PASS' if stop_ok else 'FAIL (Minor)'}")

            if backbone_seq and insert_seq:
                backbone_seq = clean_sequence(backbone_seq)
                if insert_seq in construct_seq:
                    ipos = construct_seq.index(insert_seq)
                    up_ok = construct_seq[:ipos] == backbone_seq[:ipos]
                    dn_ok = construct_seq[ipos + len(insert_seq):] == backbone_seq[ipos:]
                    checks.append(f"Backbone upstream preserved: {'PASS' if up_ok else 'FAIL (CRITICAL)'}")
                    checks.append(f"Backbone downstream preserved: {'PASS' if dn_ok else 'FAIL (CRITICAL)'}")
                    exp_size = len(backbone_seq) + len(insert_seq)
                    checks.append(f"Expected size {exp_size} bp: {'PASS' if len(construct_seq) == exp_size else 'FAIL'}")

            return "Validation Report:\n" + "\n".join(f"  {c}" for c in checks)

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
            out = f"Addgene #{args['addgene_id']}: {plasmid.name}\n"
            out += f"Size: {plasmid.size_bp} bp\n"
            out += f"Resistance: {plasmid.bacterial_resistance}\n"
            if plasmid.sequence:
                out += f"Sequence: {len(plasmid.sequence)} bp available"
            else:
                out += "Sequence: not available"
            return out

        elif name == "import_addgene_to_library":
            if not ADDGENE_AVAILABLE:
                return "Addgene integration not available."
            integration = AddgeneLibraryIntegration(LIBRARY_PATH)
            bb = integration.import_plasmid(args["addgene_id"], args.get("include_sequence", True))
            if not bb:
                return f"Failed to import Addgene #{args['addgene_id']}"
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
            out = f"Gene: {result['symbol']} ({result['organism']})\n"
            out += f"Accession: {result['accession']}\n"
            out += f"Full name: {result['full_name']}\n"
            out += f"CDS length: {result['length']} bp\n"
            out += f"\nCDS Sequence ({result['length']} bp):\n{result['sequence']}"
            return out

        elif name == "fuse_inserts":
            sequences = []
            for item in args["inserts"]:
                seq = item.get("sequence")
                seq_name = item.get("name", "")
                if not seq and item.get("insert_id"):
                    ins = get_insert_by_id(item["insert_id"])
                    if not ins:
                        return f"Insert '{item['insert_id']}' not found in library."
                    seq = ins.get("sequence")
                    seq_name = seq_name or ins.get("name", item["insert_id"])
                if not seq:
                    return f"No sequence available for '{seq_name or 'unknown'}'."
                sequences.append({"sequence": seq, "name": seq_name})

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
            out += f"\nFused sequence ({len(fused)} bp):\n{fused}"
            return out

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {str(e)}"



# ── Session management ──────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_cancelled_sessions: set[str] = set()
_sessions_lock = threading.Lock()
SESSIONS_FILE = Path(__file__).parent / ".sessions.json"

MODEL = "claude-opus-4-5-20251101"


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
                try:
                    s = {
                        "display_messages": data["display_messages"],
                        "created_at": data["created_at"],
                        "first_message": data["first_message"],
                        "history": [
                            {"role": m["role"], "content": _serialize_content(m["content"])}
                            for m in data["history"]
                        ],
                    }
                    json.dumps(s)
                    serializable[sid] = s
                except (TypeError, ValueError) as e:
                    logger.debug(f"Skipping session {sid[:8]} (serialization error: {e})")
                    continue

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
    }
    _save_sessions()
    return sid


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

        current_block_type = None
        current_tool_name = None
        current_tool_id = None
        current_tool_input_json = ""
        tool_results = []
        stop_reason = None

        # Retry loop for rate limits
        for retry_attempt in range(max_retries + 1):
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
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
                                result_str = execute_tool(current_tool_name, tool_input)
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

        # Filter out thinking blocks to avoid Error 400 on replay
        filtered_content = [
            b for b in final_message.content
            if getattr(b, 'type', None) != 'thinking'
        ]
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
  .chat-panel { flex: 1; display: flex; flex-direction: column; background: white; min-width: 0; }
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
      <p>Allen Institute for Neural Dynamics</p>
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
  <div class="chat-panel">
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
        <textarea id="input" placeholder="Describe the plasmid you want to design..." rows="1"
          oninput="autoResize(this)"></textarea>
        <div class="input-meta">
          <select id="model-select" class="model-select">
            <option value="claude-opus-4-6">Opus 4.6</option>
            <option value="claude-sonnet-4-5-20250929">Sonnet 4.5</option>
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
    renderStoredMessages(msgs);
  } catch {
    renderStoredMessages([]);
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
</script>
</body>
</html>
"""


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

            # Get or create session
            session_id = body.get("session_id")
            if not session_id or not get_session(session_id):
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

        elif path == "/api/reset":
            # Legacy endpoint — clear all sessions
            _sessions.clear()
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
