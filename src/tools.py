#!/usr/bin/env python3
"""
Plasmid Library SDK MCP Tools

Defines all plasmid tools using claude_agent_sdk's create_sdk_mcp_server.
Each tool wraps existing functions from library.py, assembler.py, and
addgene_integration.py.

Usage:
    from src.tools import build_mcp_servers

    # Includes plasmid-library + optional Benchling/PubMed (env-gated)
    options = ClaudeAgentOptions(mcp_servers=build_mcp_servers(), ...)
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import tool, create_sdk_mcp_server

from .references import ReferenceTracker
from .library import (
    search_backbones as _search_backbones,
    search_inserts as _search_inserts,
    search_all_sources as _search_all_sources,
    get_backbone_by_id,
    get_insert_by_id,
    get_all_backbones,
    get_all_inserts,
    validate_dna_sequence,
    format_backbone_summary,
    format_insert_summary,
    extract_insert_from_plasmid as _extract_insert_from_plasmid,
    extract_inserts_from_plasmid as _extract_inserts_from_plasmid,
    infer_species_from_cell_line,
)
from .assembler import (
    assemble_construct as _assemble_construct,
    fuse_sequences as _fuse_sequences,
    find_mcs_insertion_point,
    resolve_insertion_point,
    clean_sequence,
    validate_dna,
    reverse_complement,
    format_as_fasta,
    format_as_genbank,
    DEFAULT_FUSION_LINKER as _DEFAULT_FUSION_LINKER,
    assemble_golden_gate as _assemble_golden_gate,
    GG_ENZYMES,
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

# Genomic upstream fetch for bespoke promoters (newer, may not exist)
try:
    from .ncbi_integration import fetch_genomic_upstream as _fetch_genomic_upstream
    GENOMIC_UPSTREAM_AVAILABLE = True
except ImportError:
    GENOMIC_UPSTREAM_AVAILABLE = False

# Bespoke promoter detection
try:
    from .library import is_known_promoter as _is_known_promoter  # noqa: F401
    PROMOTER_DETECTION_AVAILABLE = True
except ImportError:
    PROMOTER_DETECTION_AVAILABLE = False

# ── Phase-2 advanced design modules ──
# Design Confidence Score
try:
    from .confidence import compute_confidence, format_confidence_report
    CONFIDENCE_AVAILABLE = True
except ImportError:
    CONFIDENCE_AVAILABLE = False

# Protein analysis (disorder-based fusion sites)
try:
    from .protein_analysis import (
        translate as _translate_dna,
        find_fusion_sites as _find_fusion_sites,
    )
    PROTEIN_ANALYSIS_AVAILABLE = True
except ImportError:
    PROTEIN_ANALYSIS_AVAILABLE = False

# Smart mutations (curated GoF/LoF + deterministic edits)
try:
    from .mutations import (
        lookup_known_mutations as _lookup_known_mutations,
        apply_point_mutation as _apply_point_mutation,
        design_premature_stop as _design_premature_stop,
        parse_mutation_notation as _parse_mutation_notation,
    )
    MUTATIONS_AVAILABLE = True
except ImportError:
    MUTATIONS_AVAILABLE = False

# Addgene integration (optional)
try:
    from .addgene_integration import (
        search_addgene as _search_addgene_fn,
        fetch_addgene_sequence_with_metadata as _fetch_addgene_sequence_with_metadata_fn,
        AddgeneLibraryIntegration,
    )
    ADDGENE_AVAILABLE = True
except ImportError:
    ADDGENE_AVAILABLE = False

# Literature integration (optional — requires `requests`, which is a core dep,
# but gate anyway for symmetry with other optional integrations)
try:
    from .literature import fetch_oa_fulltext as _fetch_oa_fulltext
    LITERATURE_AVAILABLE = True
except ImportError:
    LITERATURE_AVAILABLE = False

# FPbase integration (optional)
try:
    from .fpbase_integration import search_fpbase as _search_fpbase_fn
    FPBASE_AVAILABLE = True
except ImportError:
    FPBASE_AVAILABLE = False

# GenBank-with-plot export (optional — requires pLannotate, conda-only)
try:
    from .assembler import export_genbank_with_plot as _export_genbank_with_plot
    PLOT_EXPORT_AVAILABLE = True
except ImportError:
    PLOT_EXPORT_AVAILABLE = False

LIBRARY_PATH = Path(__file__).parent.parent / "library"

# ── Per-run reference tracker ──────────────────────────────────────────
# Callers (app/agent.py, evals/run_agent_evals.py) call set_tracker()
# before running the agent and get_tracker() afterwards to retrieve
# the accumulated references.

_tracker: Optional[ReferenceTracker] = None


def set_tracker(tracker: Optional[ReferenceTracker]) -> None:
    global _tracker
    _tracker = tracker


def get_tracker() -> Optional[ReferenceTracker]:
    return _tracker


def _record(method_name: str, *args, **kwargs) -> None:
    """Call a tracker method if a tracker is set, silently ignore otherwise."""
    if _tracker is not None:
        getattr(_tracker, method_name)(*args, **kwargs)


def _text(s: str) -> dict:
    """Helper to build a tool result."""
    return {"content": [{"type": "text", "text": s}]}


# ── Sequence cache ──────────────────────────────────────────────────────
# Stores full sequences fetched during a session so tools can reference
# them by key without the model copying long strings verbatim.
_sequence_cache: dict[str, str] = {}


def _cache_sequence(key: str, sequence: str) -> None:
    _sequence_cache[key] = sequence


def _get_cached_sequence(key: str) -> Optional[str]:
    return _sequence_cache.get(key)


# ── Per-run plot capture ────────────────────────────────────────────────
# export_construct stores the Bokeh plot JSON here so the web UI can emit
# a plot_data SSE event after the export tool result. Same pattern as
# set_tracker/get_tracker — caller clears before run, reads after.
_last_plot_json: Optional[str] = None


def get_last_plot_json() -> Optional[str]:
    return _last_plot_json


def clear_last_plot_json() -> None:
    global _last_plot_json
    _last_plot_json = None


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
        return _text(f"Backbone '{args['backbone_id']}' not found in library or on Addgene.")
    _record("add_backbone", bb)
    out = format_backbone_summary(bb)
    if bb.get("unconfirmed"):
        out += (
            "\n\n⚠️ **Unconfirmed Addgene match** — fuzzy-matched from search, "
            "NOT cached. Confirm with user before proceeding.\n"
        )
        alts = bb.get("addgene_search_alternatives", [])
        if alts:
            out += "\nOther Addgene search results:\n"
            for a in alts:
                out += f"  - {a.get('name')} (Addgene #{a.get('addgene_id')})\n"
        out += "\nIf correct, call import_addgene_to_library with the confirmed addgene_id."
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
    "Get complete information about a specific insert including its DNA sequence. Fallback chain: local library → FPbase (FPs) → NCBI Gene. Returns disambiguation list if the query is ambiguous (gene family or multiple species).",
    {
        "type": "object",
        "properties": {
            "insert_id": {"type": "string", "description": "Insert ID, gene symbol, or FP name"},
            "organism": {"type": "string", "description": "Species for NCBI fallback (e.g., 'human', 'mouse')"},
        },
        "required": ["insert_id"],
    },
)
async def get_insert(args):
    ins = get_insert_by_id(args["insert_id"], organism=args.get("organism"))
    if not ins:
        return _text(
            f"Insert '{args['insert_id']}' not found in local library, "
            f"FPbase, or NCBI Gene. Provide the DNA sequence directly or "
            f"check spelling/species."
        )
    if ins.get("needs_disambiguation"):
        reason = ins.get("reason", "")
        if reason == "gene_family":
            out = (
                f"⚠️ **Ambiguous gene family**: '{args['insert_id']}' is a "
                f"family name, not a specific gene. Specify which member:\n\n"
            )
            for m in ins.get("members", []):
                out += f"  - {m}\n"
            out += "\nAsk the user which one, then retry with the specific name."
            return _text(out)
        if reason == "fpbase_no_dna":
            out = (
                f"✅ Found on FPbase: **{ins.get('fpbase_name')}** "
                f"({ins.get('aa_length', '?')} aa, {ins.get('fpbase_url')})\n\n"
                f"❌ **No DNA sequence available on FPbase** — only the AA "
                f"sequence. I cannot synthesize DNA. Ask the user for the "
                f"coding sequence (from the publication, Addgene, or their "
                f"codon-optimized version), then pass it as insert_sequence."
            )
            return _text(out)
        # multiple_species
        out = (
            f"⚠️ **Ambiguous gene**: '{args['insert_id']}' matched multiple "
            f"species. Specify which:\n\n"
        )
        for opt in ins.get("options", []):
            out += (
                f"  - {opt.get('symbol')} ({opt.get('organism')}) — "
                f"{opt.get('full_name')}\n    gene_id: {opt.get('gene_id')}\n"
            )
        out += "\nRetry with organism set, or pass gene_id directly."
        return _text(out)
    _record("add_insert", ins)
    out = format_insert_summary(ins)
    if ins.get("sequence"):
        out += f"\n\nDNA Sequence ({len(ins['sequence'])} bp):\n{ins['sequence']}"
    return _text(out)


@tool(
    "extract_insert_from_plasmid",
    "Extract a single CDS insert from a full plasmid sequence by name. Uses pLannotate to annotate the plasmid and locate the feature. Use this when an insert cannot be found in the local library or NCBI — for example, when the user provides a plasmid sequence or an Addgene plasmid has been fetched and contains the gene of interest.",
    {
        "type": "object",
        "properties": {
            "plasmid_sequence": {"type": "string", "description": "Full plasmid DNA sequence to search within, OR a sequence_cache_key returned by fetch_addgene_sequence_with_metadata."},
            "insert_name": {"type": "string", "description": "Name of the gene or feature to extract (case-insensitive)."},
            "start": {"type": "integer", "description": "0-based start coordinate. If provided along with end, skips annotation and slices directly. If start >= end the feature is treated as origin-spanning (wraps around position 0)."},
            "end": {"type": "integer", "description": "0-based end coordinate (exclusive). Origin-spanning: if start >= end, extracts seq[start:] + seq[:end]."},
            "strand": {"type": "integer", "description": "1 (forward, default) or -1 (reverse complement the extracted region). Only used with explicit start/end; inferred automatically when using pLannotate annotation.", "enum": [1, -1]},
        },
        "required": ["plasmid_sequence", "insert_name"],
    },
)
async def extract_insert_from_plasmid_tool(args):
    plasmid_seq = args["plasmid_sequence"]
    if plasmid_seq in _sequence_cache:
        plasmid_seq = _sequence_cache[plasmid_seq]
    result = _extract_insert_from_plasmid(
        plasmid_sequence=plasmid_seq,
        insert_name=args["insert_name"],
        start=args.get("start"),
        end=args.get("end"),
        strand=args.get("strand", 1),
    )
    if not result:
        return _text(f"Could not extract '{args['insert_name']}' from the provided plasmid sequence.")
    seq = result["sequence"]
    return _text(
        f"Extracted insert: {result['name']} ({result['size_bp']} bp)\n"
        f"Source: {result['source']}\n\n"
        f"DNA Sequence:\n{seq}"
    )


@tool(
    "extract_inserts_from_plasmid",
    "Extract multiple named CDS inserts from a full plasmid sequence in a single pLannotate annotation pass. Use when you need to pull several genes out of the same plasmid.",
    {
        "type": "object",
        "properties": {
            "plasmid_sequence": {"type": "string", "description": "Full plasmid DNA sequence to search within, OR a sequence_cache_key returned by fetch_addgene_sequence_with_metadata."},
            "insert_names": {"type": "array", "items": {"type": "string"}, "description": "List of gene/feature names to extract (case-insensitive)."},
        },
        "required": ["plasmid_sequence", "insert_names"],
    },
)
async def extract_inserts_from_plasmid_tool(args):
    plasmid_seq = args["plasmid_sequence"]
    if plasmid_seq in _sequence_cache:
        plasmid_seq = _sequence_cache[plasmid_seq]
    result = _extract_inserts_from_plasmid(
        plasmid_sequence=plasmid_seq,
        insert_names=args["insert_names"],
    )
    if not result:
        return _text(f"Could not extract any of {args['insert_names']} from the provided plasmid sequence.")
    return _text(
        f"Extracted region spanning: {result['name']} ({result['size_bp']} bp)\n"
        f"Source: {result['source']}\n\n"
        f"DNA Sequence:\n{result['sequence']}"
    )


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
    if backbone_data:
        _record("add_backbone", backbone_data)

    # Resolve insert
    insert_seq = args.get("insert_sequence")
    insert_data = None
    if not insert_seq and args.get("insert_id"):
        insert_data = get_insert_by_id(args["insert_id"])
        if insert_data:
            if insert_data.get("needs_disambiguation"):
                return _error(
                    f"Insert '{args['insert_id']}' is ambiguous "
                    f"({insert_data.get('reason', 'multiple matches')}). "
                    f"Resolve with get_insert first, then retry with the specific ID."
                )
            insert_seq = insert_data.get("sequence")
    if not insert_seq:
        return _error("Error: No insert sequence available. Provide insert_id or insert_sequence.")
    if insert_data:
        _record("add_insert", insert_data)

    # Resolve position
    pos = args.get("insertion_position")
    auto_rc = False
    if pos is None and backbone_data:
        pos, auto_rc = resolve_insertion_point(backbone_data, backbone_seq)
    if pos is None:
        return _error("Error: No insertion position. Provide insertion_position or use a backbone with MCS data.")

    result = _assemble_construct(
        backbone_seq=backbone_seq,
        insert_seq=insert_seq,
        insertion_position=pos,
        replace_region_end=args.get("replace_region_end"),
        reverse_complement_insert=args.get("reverse_complement_insert", False) or auto_rc,
        backbone=backbone_data,
    )

    if not result.success:
        return _error("Assembly FAILED:\n" + "\n".join(f"- {e}" for e in result.errors))

    bb_name = backbone_data["name"] if backbone_data else "custom"
    ins_name = insert_data["name"] if insert_data else "custom"
    assembly_key = f"assembly:{bb_name}:{ins_name}"
    _cache_sequence(assembly_key, result.sequence)

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
    out += (
        f"\nAssembled sequence ({result.total_size_bp} bp) — cached as \"{assembly_key}\".\n"
        f"To export, call export_construct with sequence_cache_key=\"{assembly_key}\" "
        f"instead of copying the raw sequence."
    )
    return _text(out)


@tool(
    "export_construct",
    "Export an assembled construct sequence in raw, FASTA, or GenBank format.",
    {
        "type": "object",
        "properties": {
            "sequence": {"type": "string", "description": "Assembled construct DNA sequence (omit if using sequence_cache_key)"},
            "sequence_cache_key": {"type": "string", "description": "Cache key returned by fetch_addgene_sequence_with_metadata or other tools — use this instead of copying long sequences verbatim"},
            "output_format": {"type": "string", "description": "Output format", "enum": ["raw", "fasta", "genbank"]},
            "construct_name": {"type": "string", "description": "Name for the construct", "default": "construct"},
            "backbone_name": {"type": "string", "description": "Backbone name for annotation", "default": ""},
            "insert_name": {"type": "string", "description": "Insert name for annotation", "default": ""},
            "insert_position": {"type": "integer", "description": "Insert start position", "default": 0},
            "insert_length": {"type": "integer", "description": "Insert length in bp", "default": 0},
            "reverse_complement_insert": {"type": "boolean", "description": "True if insert was inserted in reverse complement orientation", "default": False},
            "linear": {"type": "boolean", "description": "Export as linear sequence (default: circular/plasmid)", "default": False},
        },
        "required": ["output_format"],
    },
)
async def export_construct(args):
    cache_key = args.get("sequence_cache_key")
    if cache_key:
        cached = _get_cached_sequence(cache_key)
        if not cached:
            return _error(f"No cached sequence found for key '{cache_key}'.")
        seq = clean_sequence(cached)
    elif args.get("sequence"):
        seq = clean_sequence(args["sequence"])
    else:
        return _error("Provide either 'sequence' or 'sequence_cache_key'.")
    fmt = args["output_format"]
    cname = args.get("construct_name", "construct")
    bname = args.get("backbone_name", "")
    iname = args.get("insert_name", "")
    ipos = args.get("insert_position", 0)
    ilen = args.get("insert_length", 0)
    rc_insert = args.get("reverse_complement_insert", False)

    try:
        if fmt == "raw":
            return _text(seq)
        elif fmt == "fasta":
            desc = f"{iname} in {bname}, {len(seq)} bp" if bname else f"{len(seq)} bp"
            return _text(format_as_fasta(seq, cname, desc))
        elif fmt in ("genbank", "gb"):
            global _last_plot_json
            _last_plot_json = None
            linear = bool(args.get("linear", False))
            # export_genbank_with_plot requires pLannotate (conda-only). Fall
            # back to plain format_as_genbank (no plot) if unavailable.
            if PLOT_EXPORT_AVAILABLE:
                try:
                    gbk, plot_json = await asyncio.to_thread(
                        _export_genbank_with_plot,
                        sequence=seq, name=cname, backbone_name=bname,
                        insert_name=iname, insert_position=ipos,
                        insert_length=ilen, linear=linear,
                    )
                    _last_plot_json = plot_json
                    return _text(gbk)
                except Exception:
                    pass  # fall through to non-plot path
            result = await asyncio.to_thread(
                format_as_genbank,
                sequence=seq, name=cname, backbone_name=bname,
                insert_name=iname, insert_position=ipos, insert_length=ilen,
                reverse_complement_insert=rc_insert,
            )
            return _text(result)
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
            "insert_id": {"type": "string", "description": "Expected insert ID — use ONLY for single (non-fusion) inserts. For fusions, use insert_sequence with the full fused sequence instead."},
            "backbone_sequence": {"type": "string", "description": "Expected backbone sequence"},
            "insert_sequence": {"type": "string", "description": "Expected insert sequence. For fusions, pass the complete fused_sequence from fuse_inserts — never a single component ID."},
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

        # Build search candidates: original, RC, and codon-trimmed variants.
        # This handles reverse-complemented inserts (reverse-orientation backbones)
        # as well as fusion parts that had their ATG or stop codon removed.
        _has_atg = insert_seq[:3] == "ATG"
        _has_stop = insert_seq[-3:] in ("TAA", "TAG", "TGA")
        _candidates = [
            (insert_seq, ""),
            (reverse_complement(insert_seq), "reverse complement"),
        ]
        if _has_atg:
            _no_atg = insert_seq[3:]
            _candidates += [(_no_atg, "ATG removed"),
                            (reverse_complement(_no_atg), "ATG removed, reverse complement")]
        if _has_stop:
            _no_stop = insert_seq[:-3]
            _candidates += [(_no_stop, "stop removed"),
                            (reverse_complement(_no_stop), "stop removed, reverse complement")]
        if _has_atg and _has_stop:
            _no_both = insert_seq[3:-3]
            _candidates += [(_no_both, "ATG and stop removed"),
                            (reverse_complement(_no_both), "ATG and stop removed, reverse complement")]

        found_seq = None
        found_desc = ""
        for _seq, _desc in _candidates:
            if len(_seq) >= 9 and _seq in construct_seq:
                found_seq, found_desc = _seq, _desc
                break

        found = found_seq is not None
        _detail_suffix = f" ({found_desc})" if found_desc else ""
        checks.append(f"Insert found in construct: {'PASS' + _detail_suffix if found else 'FAIL (CRITICAL)'}")

        if found:
            pos = construct_seq.index(found_seq)
            checks.append(f"Insert position: {pos}")
            exp = args.get("expected_insert_position")
            if exp is not None:
                checks.append(f"Position correct: {'PASS' if pos == exp else 'FAIL — expected ' + str(exp)}")
            # Codon checks on the expressed (sense) orientation
            expressed = reverse_complement(found_seq) if "reverse complement" in found_desc else found_seq
            start_ok = expressed[:3] == "ATG"
            stop_ok = expressed[-3:] in ("TAA", "TAG", "TGA")
            checks.append(f"Start codon: {'PASS' if start_ok else 'Note — ATG absent (expected for non-N-terminal fusion parts)'}")
            checks.append(f"Stop codon: {'PASS' if stop_ok else 'Note — stop absent (expected for non-C-terminal fusion parts)'}")

    if backbone_seq and insert_seq:
        backbone_seq = clean_sequence(backbone_seq)
        if found_seq and found_seq in construct_seq:
            ipos = construct_seq.index(found_seq)
            up_ok = construct_seq[:ipos] == backbone_seq[:ipos]
            dn_ok = construct_seq[ipos + len(found_seq):] == backbone_seq[ipos:]
            checks.append(f"Backbone upstream preserved: {'PASS' if up_ok else 'FAIL (CRITICAL)'}")
            checks.append(f"Backbone downstream preserved: {'PASS' if dn_ok else 'FAIL (CRITICAL)'}")
            exp_size = len(backbone_seq) + len(found_seq)
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
    "fetch_addgene_sequence_with_metadata",
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
async def fetch_addgene_sequence_with_metadata(args):
    if not ADDGENE_AVAILABLE:
        return _error("Addgene integration not available.")
    plasmid = _fetch_addgene_sequence_with_metadata_fn(args["addgene_id"])
    if not plasmid:
        return _text(f"Could not fetch Addgene #{args['addgene_id']}")
    _record("add_addgene_plasmid", plasmid.__dict__)
    cache_key = f"addgene:{args['addgene_id']}"
    out = f"Addgene #{args['addgene_id']}: {plasmid.name}\n"
    out += f"Size: {plasmid.size_bp} bp\n"
    out += f"Resistance: {plasmid.bacterial_resistance}\n"
    if plasmid.sequence:
        _cache_sequence(cache_key, plasmid.sequence)
        out += (
            f"Sequence: {len(plasmid.sequence)} bp — cached as \"{cache_key}\".\n"
            f"To export, call export_construct with sequence_cache_key=\"{cache_key}\" "
            f"instead of copying the raw sequence."
        )
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
    _record("add_backbone", bb)
    out = f"Imported: {bb['id']} ({bb['size_bp']} bp)"
    if bb.get("sequence"):
        out += f", sequence: {len(bb['sequence'])} bp"
    return _text(out)


@tool(
    "search_all",
    (
        "Search local library, NCBI Gene, and Addgene concurrently in a single call. "
        "Returns combined results from all sources. Use this as the first search step "
        "when you don't know whether the query is a local insert, an NCBI gene, or an "
        "Addgene plasmid — it checks everywhere in parallel and is faster than calling "
        "search_inserts, search_gene, and search_addgene sequentially."
    ),
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Gene symbol, plasmid name, or search term"},
            "organism": {"type": "string", "description": "Organism filter (e.g., 'human', 'mouse')"},
        },
        "required": ["query"],
    },
)
async def search_all_tool(args):
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
    return _text("\n".join(lines))


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
    if result.get("needs_disambiguation"):
        out = (
            f"⚠️ **Ambiguous gene**: '{args.get('gene_symbol', '?')}' matched "
            f"{len(result.get('options', []))} entries across multiple species. "
            f"Specify organism:\n\n"
        )
        for opt in result.get("options", []):
            out += (
                f"  - {opt.get('symbol')} ({opt.get('organism')}) — "
                f"{opt.get('full_name')}\n    gene_id: {opt.get('gene_id')}\n"
            )
        out += "\nRetry with organism set, or pass gene_id directly."
        return _text(out)
    _record("add_ncbi_gene", result)
    out = f"Gene: {result['symbol']} ({result['organism']})\n"
    out += f"Accession: {result['accession']}\n"
    out += f"Full name: {result['full_name']}\n"
    out += f"CDS length: {result['length']} bp\n"
    out += f"\nCDS Sequence ({result['length']} bp):\n{result['sequence']}"
    return _text(out)


@tool(
    "get_cell_line_info",
    "Look up the species for a common cell line name (e.g., HEK293 → human, RAW 264.7 → mouse). Use when the user mentions a cell line but not a species. IMPORTANT: this infers the CELL LINE's species — the user might want a DIFFERENT species' gene.",
    {
        "type": "object",
        "properties": {
            "cell_line": {"type": "string", "description": "Cell line name (e.g., 'HEK293', 'RAW 264.7')"},
        },
        "required": ["cell_line"],
    },
)
async def get_cell_line_info_tool(args):
    cl = args["cell_line"]
    species = infer_species_from_cell_line(cl)
    if species:
        return _text(
            f"Cell line '{cl}' is from species: **{species}**\n"
            f"Note: confirm with user before assuming the gene of interest is "
            f"also {species} — they may want a different species' gene "
            f"expressed in {cl} cells."
        )
    return _text(
        f"Cell line '{cl}' not found in known cell lines database. "
        f"Ask the user what species it is."
    )


@tool(
    "fuse_inserts",
    "Fuse multiple coding sequences into a single CDS for protein tagging or fusion proteins. Handles start/stop codon management at junctions. For protein fusions (EGFP-mCherry), the ATG is automatically removed from non-first protein sequences — set type='tag' to preserve ATG for small epitope tags (FLAG, HA, Myc). Use for N-terminal tags (FLAG-GeneX), C-terminal tags (GeneX-FLAG), or multi-domain fusions.",
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
                        "type": {
                            "type": "string",
                            "enum": ["protein", "tag"],
                            "description": "Sequence type: 'protein' (default) removes ATG from non-first positions to keep the reading frame in a fusion; 'tag' preserves ATG for small epitope tags (FLAG, HA, Myc, His) that either lack ATG or need it kept intact.",
                        },
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
    atg_removals = []  # names of sequences whose ATG will be stripped
    for i, item in enumerate(args["inserts"]):
        seq = item.get("sequence")
        name = item.get("name", "")
        seq_type = item.get("type", "protein")
        if not seq and item.get("insert_id"):
            ins = get_insert_by_id(item["insert_id"])
            if not ins:
                return _error(f"Insert '{item['insert_id']}' not found in library, FPbase, or NCBI.")
            if ins.get("needs_disambiguation"):
                return _error(
                    f"Cannot fuse: insert '{item['insert_id']}' is ambiguous. "
                    f"Resolve with get_insert first, then retry."
                )
            seq = ins.get("sequence")
            name = name or ins.get("name", item["insert_id"])
            _record("add_insert", ins)
        if not seq:
            return _error(f"No sequence available for '{name or 'unknown'}'.")
        sequences.append({"sequence": seq, "name": name, "type": seq_type})
        # Track which non-first protein sequences have an ATG to be removed
        if i > 0 and seq_type == "protein":
            from .assembler import clean_sequence as _clean_seq
            if _clean_seq(seq)[:3] == "ATG":
                atg_removals.append(name or f"sequence_{i}")

    try:
        linker = args.get("linker")
        if linker is None:
            linker = _DEFAULT_FUSION_LINKER
        fused = _fuse_sequences(sequences, linker)
    except ValueError as e:
        return _error(f"Fusion error: {e}")

    names = [s["name"] for s in sequences]
    has_atg = fused[:3] == "ATG"
    has_stop = fused[-3:] in ("TAA", "TAG", "TGA")

    out = f"Fused CDS: {'-'.join(names)}\n"
    out += f"Length: {len(fused)} bp\n"
    out += f"Start codon: {'Yes' if has_atg else 'No — MISSING'}\n"
    out += f"Stop codon: {'Yes' if has_stop else 'No — MISSING'}\n"
    out += f"In frame: {'Yes' if len(fused) % 3 == 0 else 'No'}\n"

    if atg_removals:
        out += f"\nNote: Start codon (ATG) removed from: {', '.join(atg_removals)}\n"
        out += "This is correct for a protein fusion — translation initiates from the first ATG only.\n"

    # Provide ready-to-use sequence with ATG/stop added if missing
    expressible = fused
    modifications = []
    if not has_atg:
        expressible = "ATG" + expressible
        modifications.append("ATG prepended")
    if not has_stop:
        expressible = expressible + "TAA"
        modifications.append("TAA stop appended")

    out += f"\nfused_sequence ({len(fused)} bp):\n{fused}"

    if modifications:
        out += f"\n\nexpressible_sequence ({len(expressible)} bp, {', '.join(modifications)}):\n{expressible}"
        out += "\n\nUse expressible_sequence for assemble_construct to ensure proper translation."

    return _text(out)


# ── Phase-2 Advanced Design tools ──────────────────────────────────────


@tool(
    "score_construct_confidence",
    "Compute a Design Confidence Score (0-100) for an insert/CDS. Checks "
    "cryptic polyA/splice signals, CAI, Kozak, GC, linker adequacy, repeat "
    "runs, promoter count. Use before presenting a final design.",
    {
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
)
async def score_construct_confidence_tool(args):
    if not CONFIDENCE_AVAILABLE:
        return _error("Design Confidence module not available.")
    insert_seq = clean_sequence(args["insert_sequence"])
    backbone = None
    if args.get("backbone_id"):
        backbone = get_backbone_by_id(args["backbone_id"])
    report = compute_confidence(
        insert_seq=insert_seq,
        backbone=backbone,
        fusion_parts=args.get("fusion_parts"),
    )
    return _text(format_confidence_report(report))


@tool(
    "predict_fusion_sites",
    "Predict disordered regions in a protein as candidate fusion-insertion "
    "sites. Use for internal/loop fusions or troubleshooting failed terminal "
    "fusions. Accepts AA sequence or DNA CDS (translated). Returns ranked "
    "disordered windows. Sequence-based heuristic — verify against AlphaFold2 "
    "for high-stakes designs.",
    {
        "type": "object",
        "properties": {
            "protein_sequence": {"type": "string", "description": "Amino-acid sequence (single-letter code)"},
            "dna_sequence": {"type": "string", "description": "Alternative: DNA CDS (translated in frame 0)"},
            "min_window": {"type": "integer", "description": "Minimum window length (residues, default 10)", "default": 10},
        },
    },
)
async def predict_fusion_sites_tool(args):
    if not PROTEIN_ANALYSIS_AVAILABLE:
        return _error("Protein Analysis module not available.")
    aa_seq = args.get("protein_sequence")
    if not aa_seq and args.get("dna_sequence"):
        dna = clean_sequence(args["dna_sequence"])
        aa_seq = _translate_dna(dna)
    if not aa_seq:
        return _error("Provide either protein_sequence (AA) or dna_sequence (CDS).")
    aa_seq = aa_seq.upper().strip()
    min_window = args.get("min_window", 10)
    sites = _find_fusion_sites(aa_seq, min_window=min_window)
    if not sites:
        return _text(
            f"No disordered regions ≥{min_window} residues found "
            f"({len(aa_seq)} aa). Protein may be highly structured; "
            f"terminal fusion is likely the only option."
        )
    lines = [
        f"Found {len(sites)} candidate fusion site(s) in protein "
        f"({len(aa_seq)} aa):\n"
    ]
    for i, s in enumerate(sites[:5], 1):
        lines.append(
            f"  {i}. Residues {s['start']+1}-{s['end']} "
            f"({s['length']} aa, disorder {s['mean_disorder']:.2f}) "
            f"— ...{s['context']}..."
        )
    lines.append(
        "\nNote: Sequence-based heuristic. Verify against AlphaFold2 "
        "for high-stakes designs."
    )
    return _text("\n".join(lines))


@tool(
    "lookup_known_mutations",
    "Look up curated GoF/LoF mutations for common oncogenes and tumor "
    "suppressors (BRAF, KRAS, TP53, EGFR, PTEN, PIK3CA, IDH1/2, etc.). "
    "Returns mutation notation, phenotype, PMID. Use when user wants a "
    "constitutively active / dominant-negative / kinase-dead version.",
    {
        "type": "object",
        "properties": {
            "gene_symbol": {"type": "string", "description": "Gene symbol (e.g., 'BRAF', 'TP53')"},
            "mutation_type": {"type": "string", "description": "Filter: 'GoF' or 'LoF'", "enum": ["GoF", "LoF"]},
        },
        "required": ["gene_symbol"],
    },
)
async def lookup_known_mutations_tool(args):
    if not MUTATIONS_AVAILABLE:
        return _error("Mutation Design module not available.")
    muts = _lookup_known_mutations(args["gene_symbol"], args.get("mutation_type"))
    if not muts:
        ft = f" ({args['mutation_type']})" if args.get("mutation_type") else ""
        return _text(
            f"No curated{ft} mutations for '{args['gene_symbol']}'. "
            f"DB covers BRAF, KRAS, EGFR, PIK3CA, IDH1/2, NRAS, CTNNB1, "
            f"AKT1, MYC, TP53, PTEN, RB1, FBXW7. For other genes, ask "
            f"user for a specific mutation or offer premature-stop LoF."
        )
    lines = [
        f"Curated mutations for {args['gene_symbol'].upper()}"
        f"{' (' + args['mutation_type'] + ')' if args.get('mutation_type') else ''}:\n"
    ]
    for m in muts:
        ref = f" [{m['reference']}]" if m.get("reference") else ""
        lines.append(f"  • {m['mutation']} ({m['type']}): {m['phenotype']}{ref}")
        if m.get("codon_change"):
            lines.append(f"    Codon: {m['codon_change']}")
    return _text("\n".join(lines))


@tool(
    "apply_mutation",
    "Apply a deterministic point mutation or premature stop to a CDS. "
    "Swaps ONE codon at the specified AA position for the preferred human "
    "codon for the target AA. Rest of sequence preserved exactly. "
    "SAFETY: Targeted single-codon editing only — no sequence invented.",
    {
        "type": "object",
        "properties": {
            "dna_sequence": {"type": "string", "description": "Input CDS DNA (in-frame from position 0)"},
            "method": {"type": "string", "enum": ["point_mutation", "premature_stop"], "default": "point_mutation"},
            "mutation": {"type": "string", "description": "Standard notation like 'V600E'"},
            "aa_position": {"type": "integer", "description": "1-indexed AA position (alt to 'mutation')"},
            "new_aa": {"type": "string", "description": "Target AA single-letter code (with aa_position)"},
            "position_fraction": {"type": "number", "description": "For premature_stop: stop position (0-1)", "default": 0.1},
        },
        "required": ["dna_sequence"],
    },
)
async def apply_mutation_tool(args):
    if not MUTATIONS_AVAILABLE:
        return _error("Mutation Design module not available.")
    dna = clean_sequence(args["dna_sequence"])
    method = args.get("method", "point_mutation")

    if method == "premature_stop":
        frac = args.get("position_fraction", 0.1)
        r = _design_premature_stop(dna, position_fraction=frac)
        return _text(
            f"Premature stop introduced at AA position {r['stop_position_aa']} "
            f"(DNA {r['stop_position_dna']}). "
            f"Original: {r['original_codon']} ({r['original_aa']}) → TGA.\n\n"
            f"Mutated sequence ({len(r['sequence'])} bp):\n{r['sequence']}"
        )

    aa_pos = args.get("aa_position")
    new_aa = args.get("new_aa")
    expected_orig = None
    if args.get("mutation"):
        parsed = _parse_mutation_notation(args["mutation"])
        if not parsed:
            return _error(f"Cannot parse mutation notation '{args['mutation']}'.")
        aa_pos = parsed["position"]
        new_aa = parsed["new_aa"]
        expected_orig = parsed["original_aa"]

    if aa_pos is None or not new_aa:
        return _error("Provide 'mutation' (e.g., 'V600E') or aa_position + new_aa.")

    r = _apply_point_mutation(dna, aa_position=aa_pos, new_aa=new_aa)

    warn = ""
    if expected_orig and r["original_aa"] != expected_orig:
        warn = (
            f"\n⚠️ Notation expects '{expected_orig}' at position {aa_pos} "
            f"but sequence has '{r['original_aa']}'. Verify transcript/position."
        )

    return _text(
        f"Point mutation: {r['original_aa']}{aa_pos}{r['new_aa']} "
        f"({r['original_codon']} → {r['new_codon']} at DNA {r['dna_position']}).{warn}\n\n"
        f"Mutated sequence ({len(r['sequence'])} bp):\n{r['sequence']}"
    )


@tool(
    "fetch_promoter_region",
    "Fetch the native upstream genomic region of a gene from NCBI (~2kb 5' "
    "of TSS). Use ONLY for bespoke promoters when user explicitly chose "
    "option (c) — native upstream fetch. Warn user this is endogenous "
    "regulatory region, NOT validated minimal promoter.",
    {
        "type": "object",
        "properties": {
            "gene_id": {"type": "string", "description": "NCBI Gene ID (e.g., '7157' for TP53)"},
            "gene_symbol": {"type": "string", "description": "Gene symbol (resolved to gene_id)"},
            "organism": {"type": "string", "description": "Organism for symbol resolution"},
            "bp_upstream": {"type": "integer", "description": "bp upstream to fetch (100-10000)", "default": 2000},
        },
    },
)
async def fetch_promoter_region_tool(args):
    if not GENOMIC_UPSTREAM_AVAILABLE:
        return _error("Genomic upstream fetch not available (needs Biopython + NCBI).")
    gene_id = args.get("gene_id")
    bp_upstream = args.get("bp_upstream", 2000)

    if not gene_id and args.get("gene_symbol"):
        if not NCBI_AVAILABLE:
            return _error("NCBI gene search not available.")
        genes = _search_gene_fn(args["gene_symbol"], args.get("organism"))
        if not genes:
            return _error(f"Gene '{args['gene_symbol']}' not found on NCBI.")
        if len(genes) > 1 and not args.get("organism"):
            orgs = {g.get("organism", "") for g in genes}
            if len(orgs) > 1:
                return _error(
                    f"'{args['gene_symbol']}' ambiguous across species: "
                    f"{', '.join(sorted(orgs))}. Specify organism."
                )
        gene_id = genes[0]["gene_id"]

    if not gene_id:
        return _error("Provide gene_id or gene_symbol.")

    r = _fetch_genomic_upstream(gene_id=gene_id, bp_upstream=bp_upstream)
    if not r:
        return _error(
            f"Could not fetch upstream region for gene_id={gene_id}. "
            f"Gene may lack annotated genomic coords, or NCBI unavailable."
        )
    return _text(
        f"Native upstream region for {r['gene_symbol']} (gene_id={r['gene_id']}):\n"
        f"  Organism: {r.get('organism', '?')}\n"
        f"  Chromosome: {r.get('chromosome_accession', '?')}\n"
        f"  Strand: {r.get('strand', '?')}\n"
        f"  Length: {r['length']} bp\n\n"
        f"⚠️ {r['warning']}\n\n"
        f"Sequence ({r['length']} bp):\n{r['sequence']}"
    )


# ── Golden Gate Assembly Tool ───────────────────────────────────────────────

@tool(
    "assemble_golden_gate",
    (
        "Perform in-silico Golden Gate assembly. "
        "The backbone plasmid is digested with a Type IIS restriction enzyme "
        "to open the cloning window (removing a dropout cassette if present). "
        "Each part is excised from its carrier vector using the same enzyme. "
        "Parts are ligated into the backbone in the order dictated by "
        "complementary 4-nt overhangs. "
        "Use this for Allen Institute modular expression system parts (Esp3I/BsaI/BbsI) or "
        "any standard Golden Gate workflow. "
        "Parts must be stored with category='part_in_vector' and carry a "
        "'plasmid_sequence' field."
    ),
    {
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
                    "Exact order will be inferred from overhang matching."
                ),
            },
            "enzyme_name": {
                "type": "string",
                "enum": list(GG_ENZYMES.keys()),
                "description": (
                    "Type IIS restriction enzyme used for the assembly. "
                    "Esp3I and BsmBI share the same recognition site (CGTCTC). "
                    "Defaults to Esp3I if not specified."
                ),
            },
        },
        "required": ["backbone_id", "part_ids"],
    },
)
async def assemble_golden_gate_tool(args):
    backbone_id = args["backbone_id"]
    part_ids = args["part_ids"]
    enzyme_name = args.get("enzyme_name", "Esp3I")

    # Fetch backbone
    backbone = get_backbone_by_id(backbone_id)
    if not backbone:
        return _error(f"Backbone {backbone_id!r} not found in library.")

    bb_seq = backbone.get("plasmid_sequence") or backbone.get("sequence", "")
    if not bb_seq:
        return _error(f"Backbone {backbone_id!r} has no plasmid_sequence.")

    # Fetch parts
    parts = []
    for pid in part_ids:
        part = get_insert_by_id(pid)
        if not part:
            return _error(f"Part {pid!r} not found in library.")
        ps = part.get("plasmid_sequence") or part.get("sequence", "")
        if not ps:
            return _error(
                f"Part {pid!r} has no plasmid_sequence. "
                f"Golden Gate requires the full carrier vector sequence."
            )
        parts.append({
            "name": part.get("name", pid),
            "plasmid_sequence": ps,
            "overhang_l": part.get("overhang_l"),
            "overhang_r": part.get("overhang_r"),
        })

    # Run assembly
    result = _assemble_golden_gate(
        backbone_plasmid_seq=bb_seq,
        parts=parts,
        enzyme_name=enzyme_name,
    )

    if not result.success:
        return _error(
            f"Golden Gate assembly failed:\n" + "\n".join(f"  • {e}" for e in result.errors)
        )

    warnings_block = ""
    if result.warnings:
        warnings_block = "\n\nWarnings:\n" + "\n".join(f"  ⚠ {w}" for w in result.warnings)

    junctions = " → ".join(result.junction_overhangs)
    order_str = " → ".join(result.assembly_order) if result.assembly_order else "(backbone only)"

    return _text(
        f"Golden Gate assembly successful ({enzyme_name}).\n\n"
        f"Assembly order : {order_str}\n"
        f"Junctions (4-nt): {junctions}\n"
        f"Total size     : {result.total_size_bp} bp\n\n"
        f"Assembled sequence ({result.total_size_bp} bp):\n{result.sequence}"
        + warnings_block
    )


# ── Server factory ─────────────────────────────────────────────────────

@tool(
    "fetch_oa_fulltext",
    "Look up a paper by DOI on Unpaywall to find open-access full-text URLs. "
    "Use this as a FALLBACK when PubMed's get_full_text_article fails — "
    "Unpaywall covers OA papers outside PubMed Central (journal-hosted OA, "
    "preprint servers, institutional repositories). Returns PDF URL, landing "
    "page URL, and OA status. Requires UNPAYWALL_EMAIL env var.",
    {
        "type": "object",
        "properties": {
            "doi": {
                "type": "string",
                "description": "DOI of the paper (e.g., '10.1038/nature12373' or 'https://doi.org/10.1038/nature12373')",
            },
        },
        "required": ["doi"],
    },
)
async def fetch_oa_fulltext_tool(args):
    if not LITERATURE_AVAILABLE:
        return _error("Literature integration not available (requests not installed)")
    result = _fetch_oa_fulltext(args["doi"])
    if not result.get("found"):
        return _text(f"Unpaywall lookup failed: {result.get('error', 'unknown')}")
    if not result.get("is_oa"):
        return _text(
            f"Paper found but no open-access copy available.\n"
            f"Title: {result.get('title')}\n"
            f"Journal: {result.get('journal')} ({result.get('year')})"
        )
    lines = [
        f"Open-access copy found:",
        f"  Title: {result.get('title')}",
        f"  Journal: {result.get('journal')} ({result.get('year')})",
        f"  Host: {result.get('host_type')} (license: {result.get('license') or 'unspecified'})",
    ]
    if result.get("pdf_url"):
        lines.append(f"  PDF: {result['pdf_url']}")
    if result.get("landing_url"):
        lines.append(f"  Landing page: {result['landing_url']}")
    return _text("\n".join(lines))


@tool(
    "search_fpbase",
    "Search FPbase (fpbase.org) for fluorescent proteins by name. FPbase is the "
    "canonical reference for engineered FPs like mRuby, mScarlet, mNeonGreen — "
    "these are NOT natural genes and won't be found in NCBI Gene. Use when the "
    "user wants an FP not in the local library.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Fluorescent protein name (e.g., 'mRuby', 'mScarlet')"},
        },
        "required": ["name"],
    },
)
async def search_fpbase_tool(args):
    if not FPBASE_AVAILABLE:
        return _error("FPbase integration not available.")
    results = _search_fpbase_fn(args["name"], limit=5)
    if not results:
        return _text(
            f"No fluorescent proteins found on FPbase matching "
            f"'{args['name']}'. Try a different spelling or check "
            f"https://www.fpbase.org/"
        )
    lines = [f"FPbase results for '{args['name']}':"]
    for r in results:
        ex_em = ""
        if r.get("ex_max") and r.get("em_max"):
            ex_em = f" — Ex/Em {r['ex_max']}/{r['em_max']} nm"
        lines.append(f"  - {r['name']} (slug: {r['slug']}){ex_em}")
    lines.append("\nUse get_insert with the FP name to retrieve the DNA sequence.")
    return _text("\n".join(lines))


@tool(
    "log_experimental_outcome",
    "Record a wet-lab experimental outcome for a construct designed in this "
    "session. The web UI persists this to session memory so future "
    "troubleshooting turns can see what the user already tried.",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "failed", "partial"]},
            "observation": {"type": "string", "description": "What was observed in the lab"},
            "construct_name": {"type": "string", "description": "Name of the construct tested", "default": ""},
        },
        "required": ["status", "observation"],
    },
)
async def log_experimental_outcome_tool(args):
    status = args["status"]
    observation = args["observation"]
    cname = args.get("construct_name", "")
    # The web agent loop intercepts the [OUTCOME_LOGGED] marker to persist
    # the outcome to session state. CLI/eval callers ignore the marker.
    return _text(
        f"[OUTCOME_LOGGED] status={status} "
        f"construct={cname!r} observation={observation!r}\n\n"
        f"Outcome recorded for this session: **{status}** — "
        f"{observation}. Future troubleshooting turns will see this context."
    )


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
    fetch_addgene_sequence_with_metadata,
    import_addgene_to_library,
    search_all_tool,
    search_gene_tool,
    fetch_gene_tool,
    get_cell_line_info_tool,
    fuse_inserts_tool,
    extract_insert_from_plasmid_tool,
    extract_inserts_from_plasmid_tool,
    # Phase-2 advanced design tools
    score_construct_confidence_tool,
    predict_fusion_sites_tool,
    lookup_known_mutations_tool,
    apply_mutation_tool,
    fetch_promoter_region_tool,
    # Golden Gate assembly
    assemble_golden_gate_tool,
    # Literature
    fetch_oa_fulltext_tool,
    # FPbase fluorescent protein search
    search_fpbase_tool,
    # Troubleshooting / project memory
    log_experimental_outcome_tool,
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


# ── External MCP server URLs ──
# Benchling: remote MCP, OAuth, subdomain-per-workspace
_BENCHLING_MCP_URL_TMPL = "https://{subdomain}.mcp.benchling.com/2025-06-18/mcp"
# PubMed: hosted on Cloud Run, no auth (source: anthropics/life-sciences/pubmed)
_PUBMED_MCP_URL = "https://pubmed.mcp.claude.com/mcp"


def build_mcp_servers() -> dict[str, Any]:
    """Build the full mcp_servers dict for ClaudeAgentOptions.

    Always includes the in-process plasmid-library server. Conditionally
    adds external HTTP MCP servers based on env vars:
      - BENCHLING_SUBDOMAIN: if set, adds Benchling remote MCP (read+write)
      - PLASMID_ENABLE_PUBMED: if not "0", adds PubMed MCP (default: on)

    NOTE: External MCP servers only work in Agent SDK callsites (app/agent.py,
    evals). The web UI (app/app.py) uses the raw Anthropic client and cannot
    consume McpHttpServerConfig — those tools are unavailable there.

    NOTE: When using this, do NOT set allowed_tools in ClaudeAgentOptions.
    The explicit allowlist only contains in-process tool names and would
    silently block all external MCP tools. Rely on can_use_tool for gating.
    """
    servers: dict[str, Any] = {"plasmid-library": create_plasmid_tools()}

    if subdomain := os.environ.get("BENCHLING_SUBDOMAIN"):
        servers["benchling"] = {
            "type": "http",
            "url": _BENCHLING_MCP_URL_TMPL.format(subdomain=subdomain),
        }

    if os.environ.get("PLASMID_ENABLE_PUBMED", "1") != "0":
        servers["pubmed"] = {
            "type": "http",
            "url": _PUBMED_MCP_URL,
        }

    return servers
