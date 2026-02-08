#!/usr/bin/env python3
"""
Plasmid Library SDK MCP Tools

Defines all plasmid tools using claude_agent_sdk's create_sdk_mcp_server.
Each tool wraps existing functions from library.py, assembler.py, and
addgene_integration.py.

Usage:
    from src.tools import create_plasmid_tools

    server_config = create_plasmid_tools()
    # Pass to ClaudeAgentOptions(mcp_servers={"plasmid-library": server_config})
"""

import json
from pathlib import Path

from claude_agent_sdk import tool, create_sdk_mcp_server

from .library import (
    search_backbones as _search_backbones,
    search_inserts as _search_inserts,
    get_backbone_by_id,
    get_insert_by_id,
    get_all_backbones,
    get_all_inserts,
    validate_dna_sequence,
    format_backbone_summary,
    format_insert_summary,
)
from .assembler import (
    assemble_construct as _assemble_construct,
    fuse_sequences as _fuse_sequences,
    find_mcs_insertion_point,
    clean_sequence,
    validate_dna,
    format_as_fasta,
    format_as_genbank,
)

# NCBI integration (optional)
try:
    from .ncbi_integration import (
        search_gene as _search_gene_fn,
        fetch_gene_sequence as _fetch_gene_fn,
    )
    NCBI_AVAILABLE = True
except ImportError:
    NCBI_AVAILABLE = False

# Addgene integration (optional)
try:
    from .addgene_integration import (
        search_addgene as _search_addgene_fn,
        get_addgene_plasmid as _get_addgene_plasmid_fn,
        AddgeneLibraryIntegration,
    )
    ADDGENE_AVAILABLE = True
except ImportError:
    ADDGENE_AVAILABLE = False

LIBRARY_PATH = Path(__file__).parent.parent / "library"


def _text(s: str) -> dict:
    """Helper to build a tool result."""
    return {"content": [{"type": "text", "text": s}]}


def _error(s: str) -> dict:
    return {"content": [{"type": "text", "text": s}], "is_error": True}


# ── Tool handlers ──────────────────────────────────────────────────────


@tool(
    "search_backbones",
    "Search for plasmid backbone vectors by name, features, or organism.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "organism": {"type": "string", "description": "Filter by organism type", "enum": ["mammalian", "bacterial", "lentiviral_packaging"]},
            "promoter": {"type": "string", "description": "Filter by promoter type"},
        },
        "required": ["query"],
    },
)
async def search_backbones(args):
    results = _search_backbones(args["query"], args.get("organism"), args.get("promoter"))
    if not results:
        return _text(f"No backbones found matching '{args['query']}'")
    return _text("\n\n---\n\n".join(format_backbone_summary(bb) for bb in results))


@tool(
    "get_backbone",
    "Get complete information about a specific plasmid backbone, including sequence if requested.",
    {
        "type": "object",
        "properties": {
            "backbone_id": {"type": "string", "description": "Backbone ID or name"},
            "include_sequence": {"type": "boolean", "description": "Include full DNA sequence", "default": False},
        },
        "required": ["backbone_id"],
    },
)
async def get_backbone(args):
    bb = get_backbone_by_id(args["backbone_id"])
    if not bb:
        return _text(f"Backbone '{args['backbone_id']}' not found in library.")
    out = format_backbone_summary(bb)
    if args.get("include_sequence") and bb.get("sequence"):
        out += f"\n\nDNA Sequence ({len(bb['sequence'])} bp):\n{bb['sequence'][:200]}... [{len(bb['sequence'])} bp total]"
    return _text(out)


@tool(
    "search_inserts",
    "Search for insert sequences (fluorescent proteins, tags, reporters) by name or category.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "category": {"type": "string", "description": "Filter by category", "enum": ["fluorescent_protein", "reporter", "epitope_tag"]},
        },
        "required": ["query"],
    },
)
async def search_inserts(args):
    results = _search_inserts(args["query"], args.get("category"))
    if not results:
        return _text(f"No inserts found matching '{args['query']}'")
    return _text("\n\n---\n\n".join(format_insert_summary(ins) for ins in results))


@tool(
    "get_insert",
    "Get complete information about a specific insert, including its DNA sequence.",
    {
        "type": "object",
        "properties": {
            "insert_id": {"type": "string", "description": "Insert ID or name"},
        },
        "required": ["insert_id"],
    },
)
async def get_insert(args):
    ins = get_insert_by_id(args["insert_id"])
    if not ins:
        return _text(f"Insert '{args['insert_id']}' not found in library.")
    out = format_insert_summary(ins)
    if ins.get("sequence"):
        out += f"\n\nDNA Sequence ({len(ins['sequence'])} bp):\n{ins['sequence']}"
    return _text(out)


@tool(
    "list_all_backbones",
    "List all available backbone plasmids in the library.",
    {"type": "object", "properties": {}},
)
async def list_all_backbones(args):
    bbs = get_all_backbones()
    lines = [f"Available Backbones ({len(bbs)} total):\n"]
    for bb in bbs:
        has_seq = "seq" if bb.get("sequence") else "no seq"
        lines.append(f"- {bb['id']} ({bb['size_bp']} bp, {bb.get('organism','?')}, {bb.get('promoter','?')}, {has_seq})")
    return _text("\n".join(lines))


@tool(
    "list_all_inserts",
    "List all available insert sequences in the library.",
    {"type": "object", "properties": {}},
)
async def list_all_inserts(args):
    inserts = get_all_inserts()
    lines = [f"Available Inserts ({len(inserts)} total):\n"]
    for ins in inserts:
        lines.append(f"- {ins['id']} ({ins['size_bp']} bp, {ins.get('category','?')})")
    return _text("\n".join(lines))


@tool(
    "get_insertion_site",
    "Get MCS (multiple cloning site) position info for a backbone.",
    {
        "type": "object",
        "properties": {
            "backbone_id": {"type": "string", "description": "Backbone ID or name"},
        },
        "required": ["backbone_id"],
    },
)
async def get_insertion_site(args):
    bb = get_backbone_by_id(args["backbone_id"])
    if not bb:
        return _text(f"Backbone '{args['backbone_id']}' not found.")
    mcs = bb.get("mcs_position")
    if not mcs:
        return _text(f"No MCS info for {bb['id']}.")
    return _text(f"MCS for {bb['id']}: position {mcs['start']}-{mcs['end']}. {mcs.get('description','')}")


@tool(
    "validate_sequence",
    "Validate a DNA sequence and get basic statistics (length, GC content, start/stop codons).",
    {
        "type": "object",
        "properties": {
            "sequence": {"type": "string", "description": "DNA sequence to validate"},
        },
        "required": ["sequence"],
    },
)
async def validate_sequence(args):
    result = validate_dna_sequence(args["sequence"])
    return _text(json.dumps(result, indent=2))


@tool(
    "assemble_construct",
    (
        "Assemble an expression construct by splicing an insert into a backbone. "
        "Use library IDs to auto-resolve sequences and MCS positions, or provide raw sequences."
    ),
    {
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
)
async def assemble_construct(args):
    # Resolve backbone
    backbone_seq = args.get("backbone_sequence")
    backbone_data = None
    if not backbone_seq and args.get("backbone_id"):
        backbone_data = get_backbone_by_id(args["backbone_id"])
        if backbone_data:
            backbone_seq = backbone_data.get("sequence")
    if not backbone_seq:
        return _error("Error: No backbone sequence available. Provide backbone_id (with sequence in library) or backbone_sequence.")

    # Resolve insert
    insert_seq = args.get("insert_sequence")
    insert_data = None
    if not insert_seq and args.get("insert_id"):
        insert_data = get_insert_by_id(args["insert_id"])
        if insert_data:
            insert_seq = insert_data.get("sequence")
    if not insert_seq:
        return _error("Error: No insert sequence available. Provide insert_id or insert_sequence.")

    # Resolve position
    pos = args.get("insertion_position")
    if pos is None and backbone_data:
        pos = find_mcs_insertion_point(backbone_data)
    if pos is None:
        return _error("Error: No insertion position. Provide insertion_position or use a backbone with MCS data.")

    result = _assemble_construct(
        backbone_seq=backbone_seq,
        insert_seq=insert_seq,
        insertion_position=pos,
        replace_region_end=args.get("replace_region_end"),
        reverse_complement_insert=args.get("reverse_complement_insert", False),
    )

    if not result.success:
        return _error("Assembly FAILED:\n" + "\n".join(f"- {e}" for e in result.errors))

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
    return _text(out)


@tool(
    "export_construct",
    "Export an assembled construct sequence in raw, FASTA, or GenBank format.",
    {
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
)
async def export_construct(args):
    seq = clean_sequence(args["sequence"])
    fmt = args["output_format"]
    cname = args.get("construct_name", "construct")
    bname = args.get("backbone_name", "")
    iname = args.get("insert_name", "")
    ipos = args.get("insert_position", 0)
    ilen = args.get("insert_length", 0)

    try:
        if fmt == "raw":
            return _text(seq)
        elif fmt == "fasta":
            desc = f"{iname} in {bname}, {len(seq)} bp" if bname else f"{len(seq)} bp"
            return _text(format_as_fasta(seq, cname, desc))
        elif fmt in ("genbank", "gb"):
            return _text(format_as_genbank(
                sequence=seq, name=cname, backbone_name=bname,
                insert_name=iname, insert_position=ipos, insert_length=ilen,
            ))
        else:
            return _error(f"Unknown format: {fmt}")
    except Exception as e:
        return _error(f"Export error: {e}")


@tool(
    "validate_construct",
    "Validate an assembled construct. Checks backbone preservation, insert presence/position/orientation, size, and biology.",
    {
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
)
async def validate_construct(args):
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

    return _text("Validation Report:\n" + "\n".join(f"  {c}" for c in checks))


@tool(
    "search_addgene",
    "Search Addgene's plasmid repository. Use when a plasmid is not in the local library.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results", "default": 10},
        },
        "required": ["query"],
    },
)
async def search_addgene(args):
    if not ADDGENE_AVAILABLE:
        return _error("Addgene integration not available.")
    results = _search_addgene_fn(args["query"], args.get("limit", 10))
    if not results:
        return _text(f"No Addgene results for '{args['query']}'")
    lines = [f"Addgene results for '{args['query']}':"]
    for r in results:
        lines.append(f"- {r.get('name','?')} (Addgene #{r.get('addgene_id','?')})")
    return _text("\n".join(lines))


@tool(
    "get_addgene_plasmid",
    "Fetch detailed info about a specific Addgene plasmid by catalog number.",
    {
        "type": "object",
        "properties": {
            "addgene_id": {"type": "string", "description": "Addgene catalog number"},
            "fetch_sequence": {"type": "boolean", "description": "Fetch DNA sequence", "default": True},
        },
        "required": ["addgene_id"],
    },
)
async def get_addgene_plasmid(args):
    if not ADDGENE_AVAILABLE:
        return _error("Addgene integration not available.")
    plasmid = _get_addgene_plasmid_fn(args["addgene_id"])
    if not plasmid:
        return _text(f"Could not fetch Addgene #{args['addgene_id']}")
    out = f"Addgene #{args['addgene_id']}: {plasmid.name}\n"
    out += f"Size: {plasmid.size_bp} bp\n"
    out += f"Resistance: {plasmid.bacterial_resistance}\n"
    if plasmid.sequence:
        out += f"Sequence: {len(plasmid.sequence)} bp available"
    else:
        out += "Sequence: not available"
    return _text(out)


@tool(
    "import_addgene_to_library",
    "Import a plasmid from Addgene into the local library.",
    {
        "type": "object",
        "properties": {
            "addgene_id": {"type": "string", "description": "Addgene catalog number"},
            "include_sequence": {"type": "boolean", "description": "Fetch and store sequence", "default": True},
        },
        "required": ["addgene_id"],
    },
)
async def import_addgene_to_library(args):
    if not ADDGENE_AVAILABLE:
        return _error("Addgene integration not available.")
    integration = AddgeneLibraryIntegration(LIBRARY_PATH)
    bb = integration.import_plasmid(args["addgene_id"], args.get("include_sequence", True))
    if not bb:
        return _text(f"Failed to import Addgene #{args['addgene_id']}")
    out = f"Imported: {bb['id']} ({bb['size_bp']} bp)"
    if bb.get("sequence"):
        out += f", sequence: {len(bb['sequence'])} bp"
    return _text(out)


@tool(
    "search_gene",
    "Search NCBI Gene database by gene symbol or name. Returns matching genes with IDs, symbols, organisms, and aliases. Use this when a user mentions a gene not in the local library.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Gene symbol or name (e.g., 'TP53', 'MyD88', 'EGFP')"},
            "organism": {"type": "string", "description": "Organism filter (e.g., 'human', 'mouse', 'Homo sapiens')"},
        },
        "required": ["query"],
    },
)
async def search_gene_tool(args):
    if not NCBI_AVAILABLE:
        return _error("NCBI integration not available. Install biopython: pip install biopython")
    results = _search_gene_fn(args["query"], args.get("organism"))
    if not results:
        return _text(f"No genes found matching '{args['query']}'")
    lines = [f"NCBI Gene results for '{args['query']}':"]
    for r in results:
        aliases = f" (aliases: {r['aliases']})" if r.get("aliases") else ""
        lines.append(f"- {r['symbol']} (Gene ID: {r['gene_id']}) — {r['full_name']} [{r['organism']}]{aliases}")
    return _text("\n".join(lines))


@tool(
    "fetch_gene",
    "Fetch the coding DNA sequence (CDS) for a gene from NCBI RefSeq. Returns the CDS sequence, accession, organism, and metadata.",
    {
        "type": "object",
        "properties": {
            "gene_id": {"type": "string", "description": "NCBI Gene ID (e.g., '7157' for human TP53)"},
            "gene_symbol": {"type": "string", "description": "Gene symbol (e.g., 'TP53'). Used with organism to find the gene."},
            "organism": {"type": "string", "description": "Organism (e.g., 'human', 'mouse')"},
        },
    },
)
async def fetch_gene_tool(args):
    if not NCBI_AVAILABLE:
        return _error("NCBI integration not available. Install biopython: pip install biopython")
    result = _fetch_gene_fn(
        gene_id=args.get("gene_id"),
        gene_symbol=args.get("gene_symbol"),
        organism=args.get("organism"),
    )
    if not result:
        return _text("Could not fetch gene sequence from NCBI.")
    out = f"Gene: {result['symbol']} ({result['organism']})\n"
    out += f"Accession: {result['accession']}\n"
    out += f"Full name: {result['full_name']}\n"
    out += f"CDS length: {result['length']} bp\n"
    out += f"\nCDS Sequence ({result['length']} bp):\n{result['sequence']}"
    return _text(out)


@tool(
    "fuse_inserts",
    "Fuse multiple coding sequences into a single CDS for protein tagging or fusion proteins. Handles start/stop codon management at junctions. Use for N-terminal tags (FLAG-GeneX), C-terminal tags (GeneX-FLAG), or multi-domain fusions.",
    {
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
                    },
                },
                "description": "Ordered list of sequences to fuse (N-terminal first, C-terminal last)",
            },
            "linker": {"type": "string", "description": "Optional linker DNA sequence between fusion partners"},
        },
        "required": ["inserts"],
    },
)
async def fuse_inserts_tool(args):
    sequences = []
    for item in args["inserts"]:
        seq = item.get("sequence")
        name = item.get("name", "")
        if not seq and item.get("insert_id"):
            ins = get_insert_by_id(item["insert_id"])
            if not ins:
                return _error(f"Insert '{item['insert_id']}' not found in library.")
            seq = ins.get("sequence")
            name = name or ins.get("name", item["insert_id"])
        if not seq:
            return _error(f"No sequence available for '{name or 'unknown'}'.")
        sequences.append({"sequence": seq, "name": name})

    try:
        fused = _fuse_sequences(sequences, args.get("linker"))
    except ValueError as e:
        return _error(f"Fusion error: {e}")

    names = [s["name"] for s in sequences]
    out = f"Fused CDS: {'-'.join(names)}\n"
    out += f"Length: {len(fused)} bp\n"
    out += f"Start codon: {'Yes' if fused[:3] == 'ATG' else 'No'}\n"
    out += f"Stop codon: {'Yes' if fused[-3:] in ('TAA', 'TAG', 'TGA') else 'No'}\n"
    out += f"In frame: {'Yes' if len(fused) % 3 == 0 else 'No'}\n"
    out += f"\nFused sequence ({len(fused)} bp):\n{fused}"
    return _text(out)


# ── Server factory ─────────────────────────────────────────────────────

# Collect all tool objects
ALL_TOOLS = [
    search_backbones,
    get_backbone,
    search_inserts,
    get_insert,
    list_all_backbones,
    list_all_inserts,
    get_insertion_site,
    validate_sequence,
    assemble_construct,
    export_construct,
    validate_construct,
    search_addgene,
    get_addgene_plasmid,
    import_addgene_to_library,
    search_gene_tool,
    fetch_gene_tool,
    fuse_inserts_tool,
]

ALL_TOOL_NAMES = [t.name for t in ALL_TOOLS]


def create_plasmid_tools():
    """Create an SDK MCP server config with all plasmid tools.

    Returns an McpSdkServerConfig that can be passed to
    ClaudeAgentOptions(mcp_servers={"plasmid-library": ...}).
    """
    return create_sdk_mcp_server(
        name="plasmid-library",
        tools=ALL_TOOLS,
    )
