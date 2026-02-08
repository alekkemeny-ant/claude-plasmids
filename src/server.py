#!/usr/bin/env python3
"""
Plasmid Library MCP Server

A Model Context Protocol server providing access to a curated library of 
plasmid backbone and insert sequences for expression vector design.

This server enables AI assistants to:
- Search for plasmid backbones by name, features, or organism
- Retrieve complete sequences for backbones and inserts
- Get metadata about plasmid features (promoters, selection markers, etc.)
- Validate DNA sequences
"""

import json
import logging
from pathlib import Path
from mcp.server import Server
from mcp.types import (
    TextContent,
    Tool,
    Resource,
)
from mcp.server.stdio import stdio_server

# Import Addgene integration (optional - gracefully degrades if not available)
try:
    from .addgene_integration import (
        AddgeneLibraryIntegration,
        search_addgene as _search_addgene,
        get_addgene_plasmid as _get_addgene_plasmid,
    )
    ADDGENE_AVAILABLE = True
except ImportError:
    ADDGENE_AVAILABLE = False

from .assembler import (
    assemble_construct as _assemble_construct,
    fuse_sequences as _fuse_sequences,
    find_mcs_insertion_point,
    export_construct as _export_construct,
    clean_sequence,
)

# NCBI integration (optional)
try:
    from .ncbi_integration import (
        search_gene as _search_gene,
        fetch_gene_sequence as _fetch_gene,
    )
    NCBI_AVAILABLE = True
except ImportError:
    NCBI_AVAILABLE = False

from .library import (
    load_backbones,
    load_inserts,
    search_backbones,
    search_inserts,
    get_backbone_by_id,
    get_insert_by_id,
    validate_dna_sequence,
    format_backbone_summary,
    format_insert_summary,
)

logger = logging.getLogger(__name__)

# Initialize server
server = Server("plasmid-library")

# Load library data
LIBRARY_PATH = Path(__file__).parent.parent / "library"


# Define MCP tools
@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_backbones",
            description="Search for plasmid backbone vectors by name, features, or organism. Returns matching backbones with metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (plasmid name, feature, or keyword)"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Filter by organism type (e.g., 'mammalian', 'bacterial')",
                        "enum": ["mammalian", "bacterial", "lentiviral_packaging"]
                    },
                    "promoter": {
                        "type": "string", 
                        "description": "Filter by promoter type (e.g., 'CMV', 'T7', 'U6')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_backbone",
            description="Get complete information about a specific plasmid backbone, including sequence if available.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backbone_id": {
                        "type": "string",
                        "description": "Backbone ID or name (e.g., 'pcDNA3.1(+)', 'pUC19', 'pET-28a')"
                    },
                    "include_sequence": {
                        "type": "boolean",
                        "description": "Whether to include the full DNA sequence",
                        "default": False
                    }
                },
                "required": ["backbone_id"]
            }
        ),
        Tool(
            name="search_inserts",
            description="Search for insert sequences (fluorescent proteins, tags, reporters) by name or category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (insert name or keyword)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category",
                        "enum": ["fluorescent_protein", "reporter", "epitope_tag"]
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_insert",
            description="Get complete information about a specific insert, including its DNA sequence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "insert_id": {
                        "type": "string",
                        "description": "Insert ID or name (e.g., 'EGFP', 'mCherry', 'FLAG_tag')"
                    }
                },
                "required": ["insert_id"]
            }
        ),
        Tool(
            name="validate_sequence",
            description="Validate a DNA sequence and get basic statistics (length, GC content, start/stop codons).",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "DNA sequence to validate (A, T, C, G, N only)"
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="list_all_backbones",
            description="List all available backbone plasmids in the library with basic info.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_all_inserts",
            description="List all available insert sequences in the library with basic info.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_insertion_site",
            description="Get information about the recommended insertion site (MCS) for a backbone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backbone_id": {
                        "type": "string",
                        "description": "Backbone ID or name"
                    }
                },
                "required": ["backbone_id"]
            }
        ),
        Tool(
            name="design_construct",
            description="Design an expression construct by combining a backbone and an insert. Returns detailed information about the planned construct including estimated size, insertion site, and sequence validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backbone_id": {
                        "type": "string",
                        "description": "Backbone plasmid ID or name (e.g., 'pcDNA3.1(+)')"
                    },
                    "insert_id": {
                        "type": "string",
                        "description": "Insert sequence ID or name (e.g., 'EGFP')"
                    },
                    "include_sequences": {
                        "type": "boolean",
                        "description": "Include full DNA sequences in output",
                        "default": False
                    }
                },
                "required": ["backbone_id", "insert_id"]
            }
        ),
        Tool(
            name="search_addgene",
            description="Search Addgene's plasmid repository for plasmids by name, gene, or features. Returns a list of matching plasmids with Addgene IDs. Use this when a plasmid is not found in the local library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (plasmid name, gene name, or feature)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_addgene_plasmid",
            description="Fetch detailed information about a specific plasmid from Addgene by its catalog number. Use this to get metadata and potentially sequence for plasmids not in the local library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "addgene_id": {
                        "type": "string",
                        "description": "Addgene catalog number (e.g., '50005', '12260')"
                    },
                    "fetch_sequence": {
                        "type": "boolean",
                        "description": "Attempt to fetch the DNA sequence (may not always be available)",
                        "default": True
                    }
                },
                "required": ["addgene_id"]
            }
        ),
        Tool(
            name="import_addgene_to_library",
            description="Import a plasmid from Addgene into the local curated library. This fetches the plasmid data and sequence from Addgene and adds it to the local library for faster future access.",
            inputSchema={
                "type": "object",
                "properties": {
                    "addgene_id": {
                        "type": "string",
                        "description": "Addgene catalog number to import"
                    },
                    "include_sequence": {
                        "type": "boolean",
                        "description": "Fetch and store the DNA sequence",
                        "default": True
                    }
                },
                "required": ["addgene_id"]
            }
        ),
        Tool(
            name="assemble_construct",
            description=(
                "Assemble an expression construct by splicing an insert sequence into a backbone "
                "at a specified position. This performs deterministic sequence assembly and returns "
                "the complete construct DNA sequence. Use library IDs to auto-resolve sequences "
                "and MCS positions, or provide raw sequences and positions directly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "backbone_id": {
                        "type": "string",
                        "description": "Backbone ID from the library (e.g., 'pcDNA3.1(+)'). Mutually exclusive with backbone_sequence."
                    },
                    "insert_id": {
                        "type": "string",
                        "description": "Insert ID from the library (e.g., 'EGFP'). Mutually exclusive with insert_sequence."
                    },
                    "backbone_sequence": {
                        "type": "string",
                        "description": "Raw backbone DNA sequence. Use when the backbone is not in the library."
                    },
                    "insert_sequence": {
                        "type": "string",
                        "description": "Raw insert DNA sequence. Use when the insert is not in the library."
                    },
                    "insertion_position": {
                        "type": "integer",
                        "description": "0-based position in the backbone to insert at. If omitted and a library backbone is used, defaults to the MCS start position."
                    },
                    "replace_region_end": {
                        "type": "integer",
                        "description": "If provided, backbone[insertion_position:replace_region_end] is replaced by the insert instead of a simple insertion."
                    },
                    "reverse_complement_insert": {
                        "type": "boolean",
                        "description": "Reverse-complement the insert before insertion (for reverse-orientation backbones).",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="export_construct",
            description=(
                "Export an assembled construct sequence in a specified format (raw, FASTA, or GenBank). "
                "Provide the assembled sequence directly. For GenBank format, backbone and insert names "
                "are used for annotations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "The assembled construct DNA sequence to export."
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'raw', 'fasta', or 'genbank'.",
                        "enum": ["raw", "fasta", "genbank"]
                    },
                    "construct_name": {
                        "type": "string",
                        "description": "Name for the construct (used in FASTA header and GenBank LOCUS).",
                        "default": "construct"
                    },
                    "backbone_name": {
                        "type": "string",
                        "description": "Backbone name for annotation.",
                        "default": ""
                    },
                    "insert_name": {
                        "type": "string",
                        "description": "Insert name for annotation.",
                        "default": ""
                    },
                    "insert_position": {
                        "type": "integer",
                        "description": "0-based insert start position (for GenBank annotation).",
                        "default": 0
                    },
                    "insert_length": {
                        "type": "integer",
                        "description": "Insert length in bp (for GenBank annotation).",
                        "default": 0
                    }
                },
                "required": ["sequence", "output_format"]
            }
        ),
        Tool(
            name="validate_construct",
            description=(
                "Validate an assembled construct against expected properties. Checks backbone "
                "preservation, insert preservation, correct size, insert orientation, and biology "
                "(start/stop codons, reading frame). Returns a rubric-style pass/fail report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "construct_sequence": {
                        "type": "string",
                        "description": "The assembled construct DNA sequence to validate."
                    },
                    "backbone_id": {
                        "type": "string",
                        "description": "Backbone library ID (to look up expected sequence)."
                    },
                    "insert_id": {
                        "type": "string",
                        "description": "Insert library ID (to look up expected sequence)."
                    },
                    "backbone_sequence": {
                        "type": "string",
                        "description": "Raw backbone sequence (if not using library ID)."
                    },
                    "insert_sequence": {
                        "type": "string",
                        "description": "Raw insert sequence (if not using library ID)."
                    },
                    "expected_insert_position": {
                        "type": "integer",
                        "description": "Expected 0-based position where the insert should start."
                    }
                },
                "required": ["construct_sequence"]
            }
        ),
        Tool(
            name="search_gene",
            description="Search NCBI Gene database by gene symbol or name. Returns matching genes with IDs, symbols, organisms, and aliases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gene symbol or name (e.g., 'TP53', 'MyD88', 'EGFP')"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism filter (e.g., 'human', 'mouse')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="fetch_gene",
            description="Fetch the coding DNA sequence (CDS) for a gene from NCBI RefSeq. Returns the CDS, accession, organism, and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gene_id": {
                        "type": "string",
                        "description": "NCBI Gene ID (e.g., '7157' for human TP53)"
                    },
                    "gene_symbol": {
                        "type": "string",
                        "description": "Gene symbol (e.g., 'TP53')"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism (e.g., 'human', 'mouse')"
                    }
                }
            }
        ),
        Tool(
            name="fuse_inserts",
            description=(
                "Fuse multiple coding sequences into a single CDS for protein tagging or fusion proteins. "
                "Handles start/stop codon management at junctions. Use for N-terminal tags (FLAG-GeneX), "
                "C-terminal tags (GeneX-FLAG), or multi-domain fusions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "inserts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "insert_id": {
                                    "type": "string",
                                    "description": "Insert ID from library"
                                },
                                "sequence": {
                                    "type": "string",
                                    "description": "Raw DNA sequence"
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Name for this sequence"
                                }
                            }
                        },
                        "description": "Ordered list of sequences to fuse (N-terminal first)"
                    },
                    "linker": {
                        "type": "string",
                        "description": "Optional linker DNA between fusion partners"
                    }
                },
                "required": ["inserts"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "search_backbones":
        results = search_backbones(
            arguments["query"],
            arguments.get("organism"),
            arguments.get("promoter")
        )
        if not results:
            return [TextContent(type="text", text=f"No backbones found matching '{arguments['query']}'")]
        
        output = f"Found {len(results)} backbone(s):\n\n"
        for bb in results:
            output += format_backbone_summary(bb) + "\n\n---\n\n"
        return [TextContent(type="text", text=output)]
    
    elif name == "get_backbone":
        backbone = get_backbone_by_id(arguments["backbone_id"])
        if not backbone:
            return [TextContent(type="text", text=f"Backbone '{arguments['backbone_id']}' not found in library.")]
        
        output = format_backbone_summary(backbone)
        
        if arguments.get("include_sequence"):
            if backbone.get("sequence"):
                seq = backbone["sequence"]
                output += f"\n\n**DNA Sequence ({len(seq)} bp):**\n```\n{seq}\n```"
            else:
                output += "\n\n**Note:** Full sequence not yet available for this backbone. Check sequence_file reference or Addgene."
        
        return [TextContent(type="text", text=output)]
    
    elif name == "search_inserts":
        results = search_inserts(
            arguments["query"],
            arguments.get("category")
        )
        if not results:
            return [TextContent(type="text", text=f"No inserts found matching '{arguments['query']}'")]
        
        output = f"Found {len(results)} insert(s):\n\n"
        for ins in results:
            output += format_insert_summary(ins) + "\n\n---\n\n"
        return [TextContent(type="text", text=output)]
    
    elif name == "get_insert":
        insert = get_insert_by_id(arguments["insert_id"])
        if not insert:
            return [TextContent(type="text", text=f"Insert '{arguments['insert_id']}' not found in library.")]
        
        output = format_insert_summary(insert)
        
        if insert.get("sequence"):
            output += f"\n\n**DNA Sequence ({len(insert['sequence'])} bp):**\n```\n{insert['sequence']}\n```"
        
        return [TextContent(type="text", text=output)]
    
    elif name == "validate_sequence":
        result = validate_dna_sequence(arguments["sequence"])
        
        output = "## Sequence Validation Results\n\n"
        output += f"**Valid DNA:** {'Yes' if result['is_valid'] else 'No'}\n"
        output += f"**Length:** {result['length']} bp\n"
        
        if result['gc_content'] is not None:
            output += f"**GC Content:** {result['gc_content']}%\n"
        
        output += f"**Has Start Codon (ATG):** {'Yes' if result['has_start_codon'] else 'No'}\n"
        output += f"**Has Stop Codon:** {'Yes' if result['has_stop_codon'] else 'No'}\n"
        
        if result['invalid_characters']:
            output += f"\n**Invalid Characters Found:** {', '.join(result['invalid_characters'])}"
        
        return [TextContent(type="text", text=output)]
    
    elif name == "list_all_backbones":
        data = load_backbones()
        output = f"## Available Backbone Plasmids ({len(data['backbones'])} total)\n\n"
        output += "| ID | Size (bp) | Organism | Promoter | Resistance |\n"
        output += "|---|---|---|---|---|\n"
        
        for bb in data["backbones"]:
            output += f"| {bb['id']} | {bb['size_bp']} | {bb.get('organism', '-')} | {bb.get('promoter', '-')} | {bb.get('bacterial_resistance', '-')} |\n"
        
        return [TextContent(type="text", text=output)]
    
    elif name == "list_all_inserts":
        data = load_inserts()
        output = f"## Available Insert Sequences ({len(data['inserts'])} total)\n\n"
        output += "| ID | Size (bp) | Category | Description |\n"
        output += "|---|---|---|---|\n"
        
        for ins in data["inserts"]:
            desc = ins.get('description', '')[:50] + '...' if len(ins.get('description', '')) > 50 else ins.get('description', '-')
            output += f"| {ins['id']} | {ins['size_bp']} | {ins.get('category', '-')} | {desc} |\n"
        
        return [TextContent(type="text", text=output)]
    
    elif name == "get_insertion_site":
        backbone = get_backbone_by_id(arguments["backbone_id"])
        if not backbone:
            return [TextContent(type="text", text=f"Backbone '{arguments['backbone_id']}' not found in library.")]
        
        mcs = backbone.get("mcs_position")
        if not mcs:
            return [TextContent(type="text", text=f"No MCS information available for {backbone['id']}.")]
        
        output = f"## Insertion Site for {backbone['name']}\n\n"
        output += f"**MCS Start Position:** {mcs['start']}\n"
        output += f"**MCS End Position:** {mcs['end']}\n"
        output += f"**MCS Length:** {mcs['end'] - mcs['start']} bp\n"
        
        if mcs.get('description'):
            output += f"\n**Description:** {mcs['description']}\n"
        
        # Add feature context
        output += "\n### Nearby Features:\n"
        for feature in backbone.get("features", []):
            if abs(feature["start"] - mcs["start"]) < 500 or abs(feature["end"] - mcs["end"]) < 500:
                output += f"- {feature['name']} ({feature['type']}): {feature['start']}-{feature['end']}\n"
        
        return [TextContent(type="text", text=output)]
    
    elif name == "design_construct":
        backbone = get_backbone_by_id(arguments["backbone_id"])
        if not backbone:
            return [TextContent(type="text", text=f"❌ Backbone '{arguments['backbone_id']}' not found in library.\n\nUse 'list_all_backbones' to see available options.")]
        
        insert = get_insert_by_id(arguments["insert_id"])
        if not insert:
            return [TextContent(type="text", text=f"❌ Insert '{arguments['insert_id']}' not found in library.\n\nUse 'list_all_inserts' to see available options.")]
        
        # Calculate estimated size
        estimated_size = backbone["size_bp"] + insert["size_bp"]
        
        # Validate insert sequence
        insert_validation = None
        if insert.get("sequence"):
            insert_validation = validate_dna_sequence(insert["sequence"])
        
        output = f"## Expression Construct Design\n\n"
        output += f"### Backbone: {backbone['name']}\n"
        output += f"- **Size:** {backbone['size_bp']} bp\n"
        output += f"- **Promoter:** {backbone.get('promoter', 'Unknown')}\n"
        output += f"- **Organism:** {backbone.get('organism', 'Unknown')}\n"
        output += f"- **Selection:** {backbone.get('bacterial_resistance', 'Unknown')} (bacterial)"
        if backbone.get('mammalian_selection'):
            output += f", {backbone['mammalian_selection']} (mammalian)"
        output += "\n"
        
        if backbone.get('mcs_position'):
            mcs = backbone['mcs_position']
            output += f"- **Insertion Site (MCS):** positions {mcs['start']}-{mcs['end']}\n"
        
        output += f"\n### Insert: {insert['name']}\n"
        output += f"- **Size:** {insert['size_bp']} bp\n"
        output += f"- **Category:** {insert.get('category', 'Unknown')}\n"
        
        if insert_validation:
            output += f"- **Sequence Validation:** {'✓ Valid DNA' if insert_validation['is_valid'] else '✗ Invalid'}\n"
            output += f"- **GC Content:** {insert_validation['gc_content']}%\n"
            output += f"- **Start Codon (ATG):** {'✓ Present' if insert_validation['has_start_codon'] else '✗ Missing'}\n"
            output += f"- **Stop Codon:** {'✓ Present' if insert_validation['has_stop_codon'] else '✗ Missing'}\n"
        
        output += f"\n### Construct Summary\n"
        output += f"- **Estimated Total Size:** {estimated_size} bp\n"
        output += f"- **Expression System:** {backbone.get('organism', 'Unknown')}\n"
        
        # Include sequences if requested
        if arguments.get("include_sequences"):
            output += f"\n### Sequences\n"
            if backbone.get("sequence"):
                output += f"\n**Backbone Sequence ({len(backbone['sequence'])} bp):**\n```\n{backbone['sequence']}\n```\n"
            else:
                output += f"\n**Backbone Sequence:** Not available in library\n"
            
            if insert.get("sequence"):
                output += f"\n**Insert Sequence ({len(insert['sequence'])} bp):**\n```\n{insert['sequence']}\n```\n"
        
        return [TextContent(type="text", text=output)]
    
    elif name == "search_addgene":
        if not ADDGENE_AVAILABLE:
            return [TextContent(type="text", text="❌ Addgene integration is not available. Please ensure the addgene_integration module is installed.")]
        
        try:
            results = _search_addgene(arguments["query"], arguments.get("limit", 10))
            
            if not results:
                return [TextContent(type="text", text=f"No plasmids found on Addgene matching '{arguments['query']}'")]
            
            output = f"## Addgene Search Results for '{arguments['query']}'\n\n"
            output += f"Found {len(results)} result(s):\n\n"
            
            for result in results:
                output += f"- **{result.get('name', 'Unknown')}** (Addgene #{result.get('addgene_id', '?')})\n"
                if result.get('url'):
                    output += f"  URL: {result['url']}\n"
            
            output += "\n*Use `get_addgene_plasmid` with the Addgene ID to fetch full details.*"
            
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error searching Addgene: {str(e)}")]
    
    elif name == "get_addgene_plasmid":
        if not ADDGENE_AVAILABLE:
            return [TextContent(type="text", text="❌ Addgene integration is not available.")]
        
        try:
            addgene_id = arguments["addgene_id"]
            plasmid = _get_addgene_plasmid(addgene_id)
            
            if not plasmid:
                return [TextContent(type="text", text=f"❌ Could not fetch plasmid Addgene #{addgene_id}. It may not exist or there was a network error.")]
            
            output = f"## Addgene #{addgene_id}: {plasmid.name or 'Unknown'}\n\n"
            
            if plasmid.description:
                output += f"**Description:** {plasmid.description}\n\n"
            
            output += f"**Size:** {plasmid.size_bp or 'Unknown'} bp\n"
            output += f"**Promoter:** {plasmid.promoter or 'Unknown'}\n"
            output += f"**Bacterial Resistance:** {plasmid.bacterial_resistance or 'Unknown'}\n"
            
            if plasmid.mammalian_selection:
                output += f"**Mammalian Selection:** {plasmid.mammalian_selection}\n"
            
            if plasmid.depositor:
                output += f"**Depositor:** {plasmid.depositor}\n"
            
            if plasmid.url:
                output += f"\n**Addgene URL:** {plasmid.url}\n"
            
            if arguments.get("fetch_sequence", True) and plasmid.sequence:
                output += f"\n**Sequence:** {len(plasmid.sequence)} bp available\n"
            elif arguments.get("fetch_sequence", True):
                output += f"\n**Sequence:** Not available from Addgene page\n"
            
            output += "\n*Use `import_addgene_to_library` to add this plasmid to your local library.*"
            
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error fetching from Addgene: {str(e)}")]
    
    elif name == "import_addgene_to_library":
        if not ADDGENE_AVAILABLE:
            return [TextContent(type="text", text="❌ Addgene integration is not available.")]
        
        try:
            addgene_id = arguments["addgene_id"]
            include_sequence = arguments.get("include_sequence", True)
            
            integration = AddgeneLibraryIntegration(LIBRARY_PATH)
            backbone = integration.import_plasmid(addgene_id, include_sequence)
            
            if not backbone:
                return [TextContent(type="text", text=f"❌ Could not import plasmid Addgene #{addgene_id}")]
            
            output = f"## ✓ Imported Addgene #{addgene_id}\n\n"
            output += f"**ID:** {backbone['id']}\n"
            output += f"**Size:** {backbone['size_bp']} bp\n"
            output += f"**Organism:** {backbone.get('organism', 'Unknown')}\n"
            output += f"**Promoter:** {backbone.get('promoter', 'Unknown')}\n"
            
            if backbone.get('sequence'):
                output += f"**Sequence:** ✓ {len(backbone['sequence'])} bp stored\n"
            else:
                output += f"**Sequence:** ✗ Not available\n"
            
            output += f"\nThis plasmid is now available in your local library as '{backbone['id']}'."
            
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error importing from Addgene: {str(e)}")]
    
    elif name == "assemble_construct":
        # Resolve backbone sequence
        backbone_seq = arguments.get("backbone_sequence")
        backbone_data = None
        if not backbone_seq:
            bb_id = arguments.get("backbone_id")
            if not bb_id:
                return [TextContent(type="text", text="Error: Provide either backbone_id or backbone_sequence.")]
            backbone_data = get_backbone_by_id(bb_id)
            if not backbone_data:
                return [TextContent(type="text", text=f"Backbone '{bb_id}' not found in library.")]
            backbone_seq = backbone_data.get("sequence")
            if not backbone_seq:
                return [TextContent(type="text", text=f"Backbone '{bb_id}' has no sequence in the library. Import it from Addgene or provide backbone_sequence directly.")]

        # Resolve insert sequence
        insert_seq = arguments.get("insert_sequence")
        insert_data = None
        if not insert_seq:
            ins_id = arguments.get("insert_id")
            if not ins_id:
                return [TextContent(type="text", text="Error: Provide either insert_id or insert_sequence.")]
            insert_data = get_insert_by_id(ins_id)
            if not insert_data:
                return [TextContent(type="text", text=f"Insert '{ins_id}' not found in library.")]
            insert_seq = insert_data.get("sequence")
            if not insert_seq:
                return [TextContent(type="text", text=f"Insert '{ins_id}' has no sequence in the library.")]

        # Resolve insertion position
        insertion_pos = arguments.get("insertion_position")
        if insertion_pos is None:
            if backbone_data:
                insertion_pos = find_mcs_insertion_point(backbone_data)
            if insertion_pos is None:
                return [TextContent(type="text", text="Error: No insertion_position provided and backbone has no MCS position data. Specify insertion_position explicitly.")]

        result = _assemble_construct(
            backbone_seq=backbone_seq,
            insert_seq=insert_seq,
            insertion_position=insertion_pos,
            replace_region_end=arguments.get("replace_region_end"),
            reverse_complement_insert=arguments.get("reverse_complement_insert", False),
        )

        if not result.success:
            output = "## Assembly Failed\n\n"
            for err in result.errors:
                output += f"- {err}\n"
            return [TextContent(type="text", text=output)]

        bb_name = backbone_data["name"] if backbone_data else "custom backbone"
        ins_name = insert_data["name"] if insert_data else "custom insert"

        output = "## Assembly Successful\n\n"
        output += f"**Construct:** {ins_name} in {bb_name}\n"
        output += f"**Total Size:** {result.total_size_bp} bp\n"
        output += f"**Insert Position:** {result.insert_position}\n"
        output += f"**Backbone Preserved:** Yes\n"
        output += f"**Insert Preserved:** Yes\n"
        output += f"**Start Codon (ATG):** {'Yes' if result.insert_has_start_codon else 'No'}\n"
        output += f"**Stop Codon:** {'Yes' if result.insert_has_stop_codon else 'No'}\n"
        output += f"**Reading Frame (len % 3 == 0):** {'Yes' if result.insert_length_valid else 'No'}\n"

        if result.warnings:
            output += "\n### Warnings\n"
            for w in result.warnings:
                output += f"- {w}\n"

        output += f"\n### Assembled Sequence ({result.total_size_bp} bp)\n```\n{result.sequence}\n```\n"

        return [TextContent(type="text", text=output)]

    elif name == "export_construct":
        from .assembler import (
            format_as_fasta,
            format_as_genbank,
        )

        sequence = clean_sequence(arguments["sequence"])
        fmt = arguments["output_format"]
        construct_name = arguments.get("construct_name", "construct")
        backbone_name = arguments.get("backbone_name", "")
        insert_name = arguments.get("insert_name", "")
        insert_position = arguments.get("insert_position", 0)
        insert_length = arguments.get("insert_length", 0)

        try:
            if fmt == "raw":
                exported = sequence
            elif fmt == "fasta":
                desc = f"{insert_name} in {backbone_name}, {len(sequence)} bp" if backbone_name else f"{len(sequence)} bp"
                exported = format_as_fasta(sequence, construct_name, desc)
            elif fmt in ("genbank", "gb"):
                exported = format_as_genbank(
                    sequence=sequence,
                    name=construct_name,
                    backbone_name=backbone_name,
                    insert_name=insert_name,
                    insert_position=insert_position,
                    insert_length=insert_length,
                )
            else:
                return [TextContent(type="text", text=f"Unknown format: {fmt}. Use 'raw', 'fasta', or 'genbank'.")]

            output = f"## Exported Construct ({fmt})\n\n```\n{exported}\n```"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Export error: {str(e)}")]

    elif name == "validate_construct":
        construct_seq = clean_sequence(arguments["construct_sequence"])

        # Resolve backbone
        backbone_seq = arguments.get("backbone_sequence")
        backbone_data = None
        if not backbone_seq and arguments.get("backbone_id"):
            backbone_data = get_backbone_by_id(arguments["backbone_id"])
            if backbone_data:
                backbone_seq = backbone_data.get("sequence")
        if backbone_seq:
            backbone_seq = clean_sequence(backbone_seq)

        # Resolve insert
        insert_seq = arguments.get("insert_sequence")
        insert_data = None
        if not insert_seq and arguments.get("insert_id"):
            insert_data = get_insert_by_id(arguments["insert_id"])
            if insert_data:
                insert_seq = insert_data.get("sequence")
        if insert_seq:
            insert_seq = clean_sequence(insert_seq)

        expected_pos = arguments.get("expected_insert_position")

        # Build rubric-style report
        checks = []
        overall_pass = True
        critical_fail = False

        # 1. Valid DNA
        from .assembler import validate_dna
        dna_ok, dna_errs = validate_dna(construct_seq)
        checks.append(("Construct is valid DNA", "Critical", dna_ok, "; ".join(dna_errs) if dna_errs else ""))
        if not dna_ok:
            critical_fail = True

        # 2. Construct size
        checks.append(("Construct length", "Info", True, f"{len(construct_seq)} bp"))

        # 3. Insert found in construct
        if insert_seq:
            insert_found = insert_seq in construct_seq
            checks.append(("Insert sequence found in construct", "Critical", insert_found, ""))
            if not insert_found:
                critical_fail = True
            else:
                # Find position
                found_pos = construct_seq.index(insert_seq)
                checks.append(("Insert position", "Info", True, f"Found at position {found_pos}"))

                if expected_pos is not None:
                    pos_match = found_pos == expected_pos
                    checks.append(("Insert at expected position", "Critical", pos_match,
                                   f"Expected {expected_pos}, found {found_pos}" if not pos_match else ""))
                    if not pos_match:
                        critical_fail = True

                # Insert biology
                has_atg = insert_seq[:3] == "ATG"
                has_stop = insert_seq[-3:] in ("TAA", "TAG", "TGA")
                frame_ok = len(insert_seq) % 3 == 0
                checks.append(("Insert has start codon (ATG)", "Minor", has_atg, ""))
                checks.append(("Insert has stop codon", "Minor", has_stop, ""))
                checks.append(("Insert length multiple of 3", "Minor", frame_ok, f"{len(insert_seq)} bp"))

        # 4. Backbone preservation
        if backbone_seq and insert_seq:
            insert_pos_in_construct = construct_seq.find(insert_seq) if insert_seq in construct_seq else None
            if insert_pos_in_construct is not None:
                upstream_ok = construct_seq[:insert_pos_in_construct] == backbone_seq[:insert_pos_in_construct]
                downstream_ok = construct_seq[insert_pos_in_construct + len(insert_seq):] == backbone_seq[insert_pos_in_construct:]
                backbone_ok = upstream_ok and downstream_ok
                checks.append(("Backbone sequence preserved", "Critical", backbone_ok, ""))
                if not backbone_ok:
                    critical_fail = True

                # Size check
                expected_size = len(backbone_seq) + len(insert_seq)
                size_ok = len(construct_seq) == expected_size
                checks.append(("Total size correct", "Minor", size_ok,
                               f"Expected {expected_size}, got {len(construct_seq)}" if not size_ok else f"{len(construct_seq)} bp"))

        # Build output
        output = "## Construct Validation Report\n\n"
        output += "| Check | Severity | Result | Details |\n"
        output += "|-------|----------|--------|---------|\n"
        for check_name, severity, passed, details in checks:
            status = "PASS" if passed else "FAIL"
            output += f"| {check_name} | {severity} | {status} | {details} |\n"

        output += "\n"
        if critical_fail:
            output += "### Result: FAIL (critical check failed)\n"
        else:
            # Count passes
            total = len([c for c in checks if c[1] != "Info"])
            passed_count = len([c for c in checks if c[1] != "Info" and c[2]])
            score = round(passed_count / total * 100) if total > 0 else 100
            output += f"### Result: PASS ({score}% — {passed_count}/{total} checks passed)\n"

        return [TextContent(type="text", text=output)]

    elif name == "search_gene":
        if not NCBI_AVAILABLE:
            return [TextContent(type="text", text="NCBI integration not available. Install biopython.")]
        try:
            results = _search_gene(arguments["query"], arguments.get("organism"))
            if not results:
                return [TextContent(type="text", text=f"No genes found matching '{arguments['query']}'")]
            output = f"NCBI Gene results for '{arguments['query']}':\n\n"
            for r in results:
                aliases = f" (aliases: {r['aliases']})" if r.get("aliases") else ""
                output += f"- **{r['symbol']}** (Gene ID: {r['gene_id']}) — {r['full_name']} [{r['organism']}]{aliases}\n"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"NCBI search error: {str(e)}")]

    elif name == "fetch_gene":
        if not NCBI_AVAILABLE:
            return [TextContent(type="text", text="NCBI integration not available. Install biopython.")]
        try:
            result = _fetch_gene(
                gene_id=arguments.get("gene_id"),
                gene_symbol=arguments.get("gene_symbol"),
                organism=arguments.get("organism"),
            )
            if not result:
                return [TextContent(type="text", text="Could not fetch gene sequence from NCBI.")]
            output = f"## {result['symbol']} ({result['organism']})\n\n"
            output += f"**Accession:** {result['accession']}\n"
            output += f"**Full name:** {result['full_name']}\n"
            output += f"**CDS length:** {result['length']} bp\n"
            output += f"\n**CDS Sequence ({result['length']} bp):**\n```\n{result['sequence']}\n```"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"NCBI fetch error: {str(e)}")]

    elif name == "fuse_inserts":
        try:
            sequences = []
            for item in arguments["inserts"]:
                seq = item.get("sequence")
                seq_name = item.get("name", "")
                if not seq and item.get("insert_id"):
                    ins = get_insert_by_id(item["insert_id"])
                    if not ins:
                        return [TextContent(type="text", text=f"Insert '{item['insert_id']}' not found in library.")]
                    seq = ins.get("sequence")
                    seq_name = seq_name or ins.get("name", item["insert_id"])
                if not seq:
                    return [TextContent(type="text", text=f"No sequence available for '{seq_name or 'unknown'}'.")]
                sequences.append({"sequence": seq, "name": seq_name})

            fused = _fuse_sequences(sequences, arguments.get("linker"))
            names = [s["name"] for s in sequences]
            output = f"## Fused CDS: {'-'.join(names)}\n\n"
            output += f"**Length:** {len(fused)} bp\n"
            output += f"**Start codon:** {'Yes' if fused[:3] == 'ATG' else 'No'}\n"
            output += f"**Stop codon:** {'Yes' if fused[-3:] in ('TAA', 'TAG', 'TGA') else 'No'}\n"
            output += f"**In frame:** {'Yes' if len(fused) % 3 == 0 else 'No'}\n"
            output += f"\n**Fused sequence ({len(fused)} bp):**\n```\n{fused}\n```"
            return [TextContent(type="text", text=output)]
        except ValueError as e:
            return [TextContent(type="text", text=f"Fusion error: {str(e)}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# Define resources
@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="plasmid://backbones",
            name="Backbone Library",
            description="Complete list of available plasmid backbones",
            mimeType="application/json"
        ),
        Resource(
            uri="plasmid://inserts", 
            name="Insert Library",
            description="Complete list of available insert sequences",
            mimeType="application/json"
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    if uri == "plasmid://backbones":
        return json.dumps(load_backbones(), indent=2)
    elif uri == "plasmid://inserts":
        return json.dumps(load_inserts(), indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
