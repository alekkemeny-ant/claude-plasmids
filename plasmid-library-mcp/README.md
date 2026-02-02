# Plasmid Library MCP Server

A Model Context Protocol (MCP) server providing access to a curated library of plasmid backbone and insert sequences for expression vector design.

## Overview

This MCP server enables AI assistants to:

- **Search for plasmid backbones** by name, features, organism, or promoter type
- **Retrieve complete sequences** for backbones and common inserts (fluorescent proteins, tags, reporters)
- **Get metadata** about plasmid features (promoters, selection markers, origins of replication, etc.)
- **Validate DNA sequences** (check for valid nucleotides, start/stop codons, GC content)
- **Identify insertion sites** (MCS position and context)
- **Design expression constructs** by combining backbones and inserts
- **Search Addgene** for plasmids not in the local library
- **Import from Addgene** to expand the local library

## Use Case

This server was developed as part of a collaboration between Anthropic and the Allen Institute to support AI-assisted plasmid design workflows. The goal is to provide reliable, curated sequence data to avoid hallucination issues when designing expression vectors.

## Architecture

The server uses a **hybrid architecture**:

1. **Local Curated Library** - High-reliability, verified sequences for common vectors
2. **Addgene Integration** - Access to 100,000+ plasmids when local library doesn't have what you need
3. **Automatic Fallback** - If a plasmid isn't found locally, can search/import from Addgene

## Available Backbones (Local Library)

| ID | Size (bp) | Organism | Promoter | Use Case |
|---|---|---|---|---|
| pcDNA3.1(+) | 5428 | mammalian | CMV | Transient/stable mammalian expression |
| pcDNA3.1(-) | 5427 | mammalian | CMV | Transient/stable mammalian expression (reverse MCS) |
| pUC19 | 2686 | bacterial | lac | General cloning, blue/white screening |
| pET-28a(+) | 5369 | bacterial | T7 | Bacterial protein expression with His-tag |
| pEGFP-N1 | 4733 | mammalian | CMV | C-terminal EGFP fusions |
| pLKO.1-puro | 7032 | mammalian | U6 | Lentiviral shRNA expression |
| pSpCas9(BB)-2A-Puro | 9175 | mammalian | CBh/U6 | CRISPR/Cas9 genome editing |
| psPAX2 | 10703 | packaging | multiple | Lentiviral packaging (2nd gen) |
| pMD2.G | 5824 | packaging | CMV | VSV-G envelope for lentivirus |

## Available Inserts

### Fluorescent Proteins
- **EGFP** (720 bp) - Enhanced green fluorescent protein
- **mCherry** (711 bp) - Red fluorescent protein
- **mNeonGreen** (714 bp) - Brightest monomeric green FP
- **tdTomato** (1431 bp) - Bright tandem dimer red FP
- **mTagBFP2** (714 bp) - Blue fluorescent protein

### Reporters
- **Firefly Luciferase** (1653 bp) - Bioluminescence reporter
- **Renilla Luciferase** (936 bp) - Dual-reporter assays

### Epitope Tags
- **FLAG** (24 bp) - DYKDDDDK epitope
- **HA** (27 bp) - Hemagglutinin epitope
- **6xHis** (18 bp) - Polyhistidine tag
- **Myc** (30 bp) - c-Myc epitope

## MCP Tools

### `search_backbones`
Search for plasmid backbones by name, features, or organism.

```json
{
  "query": "pcDNA",
  "organism": "mammalian",
  "promoter": "CMV"
}
```

### `get_backbone`
Get complete information about a specific backbone.

```json
{
  "backbone_id": "pcDNA3.1(+)",
  "include_sequence": false
}
```

### `search_inserts`
Search for insert sequences by name or category.

```json
{
  "query": "GFP",
  "category": "fluorescent_protein"
}
```

### `get_insert`
Get complete information about an insert, including DNA sequence.

```json
{
  "insert_id": "EGFP"
}
```

### `validate_sequence`
Validate a DNA sequence and get statistics.

```json
{
  "sequence": "ATGGTGAGCAAGGGCGAGGAG..."
}
```

### `get_insertion_site`
Get MCS information for a backbone.

```json
{
  "backbone_id": "pcDNA3.1(+)"
}
```

### `design_construct`
Design an expression construct by combining a backbone and insert.

```json
{
  "backbone_id": "pcDNA3.1(+)",
  "insert_id": "EGFP",
  "include_sequences": true
}
```

### `list_all_backbones` / `list_all_inserts`
List all available entries in the library.

## Addgene Integration Tools

### `search_addgene`
Search Addgene's repository for plasmids not in the local library.

```json
{
  "query": "pLKO shRNA",
  "limit": 10
}
```

### `get_addgene_plasmid`
Fetch detailed information about a specific Addgene plasmid.

```json
{
  "addgene_id": "8453",
  "fetch_sequence": true
}
```

### `import_addgene_to_library`
Import an Addgene plasmid into the local curated library.

```json
{
  "addgene_id": "8453",
  "include_sequence": true
}
```

## Addgene API Configuration

The Addgene integration supports two modes:

### Web Scraping Mode (Default)
Works immediately without configuration. Scrapes publicly available data from Addgene's website.

### Official API Mode (Recommended for Production)
For production use, apply for API access at https://developers.addgene.org/

Set the API token via environment variable:
```bash
export ADDGENE_API_TOKEN="your-token-here"
```

## Installation

```bash
# Install with pip
pip install -e .

# Or with uv
uv pip install -e .
```

## Running the Server

```bash
# As a module
python -m src.server

# Or using the entry point
plasmid-library-mcp
```

## MCP Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "plasmid-library": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/plasmid-library-mcp"
    }
  }
}
```

## Data Sources

Sequences in this library are compiled from:
- [Addgene](https://www.addgene.org/) - Nonprofit plasmid repository
- [SnapGene](https://www.snapgene.com/plasmids) - Curated plasmid database
- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/) - Reference sequences
- Vendor documentation (Thermo Fisher, Novagen, Clontech, etc.)

## Extending the Library

To add new backbones or inserts, edit the JSON files in `library/`:

- `library/backbones.json` - Plasmid backbone definitions
- `library/inserts.json` - Insert sequence definitions

Each entry should include:
- Unique ID and common aliases
- Size in base pairs
- Key features and their positions
- DNA sequence (for inserts) or sequence file reference (for backbones)

## License

MIT License - See LICENSE file for details.

## Contributing

This is part of an ongoing collaboration. For contributions or issues, please contact the project maintainers.
