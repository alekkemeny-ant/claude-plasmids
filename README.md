# Plasmid Designer -- Nautilex 

## Hackathon Brief

**Plasmid Designer** is a Claude-native agentic tool for designing expression plasmids, built as a collaboration between Anthropic and the Allen Institute. You'll be hacking on the `nautilex` branch of the `claude-plasmids` repo.

---

### What It Does

Tell it what you want in plain English — the agent figures out the rest:

```
"Put EGFP into pcDNA3.1(+)"
"Design a plasmid to express human TP53 in HEK293 cells"
"Add an N-terminal FLAG tag to EGFP in pcDNA3.1(+)"
```

Claude handles orchestration: understanding intent, selecting tools, retrieving sequences, and validating results. All sequence operations are **deterministic** — no LLM ever generates DNA. Every nucleotide in the output comes from a verified source (curated library, Addgene, or NCBI GenBank).

---

### Requirements
- [conda](https://docs.conda.io/en/latest/miniconda.html)
- Anthropic API key
  - Contact Mialy DeFelice for a Nautilex Anthropic API key. This key will be available during the hackathon and disabled afterward.
- Laptop
- You can develop from Terminal, VSCode or your preferred IDE.
  
See below for full setup instructions.
---

### How a Request Flows

1. **User input** — natural language description of the desired construct
2. **Sequence retrieval** — checks local library first, then auto-fetches from Addgene or NCBI if needed
3. **Assembly** — deterministic string splicing into the backbone's MCS
4. **Validation** — rubric-based checks (insert intact, correct orientation, reading frame, start/stop codons, promoter/polyA placement)
5. **Export** — raw sequence, FASTA, or annotated GenBank; downloadable from the browser

---

### Key Capabilities

| Feature | Description |
|---|---|
| **Library** | 21 curated backbones (pcDNA3.1, pUC19, pEGFP-N1, pAAV-CMV, and more) + fluorescent proteins, reporters, epitope tags |
| **User Library**  | Ability to upload your own user library and metadata, that loads at runtime and is not saved to the main library|
| **Addgene integration** | Auto-fetches and caches any plasmid not in the local library |
| **NCBI gene retrieval** | Fetches CDS sequences by gene name, with species and family disambiguation |
| **Protein fusions** | N- and C-terminal tag and fusions (e.g. FLAG-EGFP, mCherry-HA) with automatic start/stop codon management, default linkers.|
| **Plasmid Assembly**  | Agents can assembly vectors to users specifications. The agents can determine the backbone direction and update the insert orientation as needed.  |
| **Multiple Cloning Site Detection**  | If the location of the MCS is not specified in the data, the program can locate the MCS and the MCS orientation  |
| **Golden Gate Aseembly**  | Can assemble backbones and parts for golden gate assembly, taking care to extract inserts from the vectors they are contained in, maintain overhangs, and extract dropouts from the intended backbone vectors  |
| **Natural language backbone selection** | Agent infers backbone from context (organism, promoter type, selection marker) |
| **MCP server** | 18 tools exposed for Claude to call, usable standalone or via the agent loop |
| **

---

### Where to Start Hacking

We are looking to determine how users _want_ to interact with a Plasmid Designer tool. 
Start asking the designer questions and looking at the output. Note any issues with plasmids designs. 


If you are computationally inclined, you can also start hacking into the code to fix the outputs.


**Good Luck, and Have Fun!!**



-------------------
# About
A Claude-native agentic tool for designing expression plasmids. Built as a collaboration between Anthropic and the Allen Institute.

The tool takes a backbone vector, insert gene, and optional parameters as input, and outputs a complete, validated plasmid construct sequence. Claude handles orchestration (understanding user intent, selecting tools, validating results) while all sequence operations are deterministic — no LLM ever generates DNA.

## Setup

pLannotate (used for GenBank annotation) is only available via conda/bioconda and requires Python <3.13. The project uses a conda environment instead of a plain venv.

### 1. Create the conda environment

This installs all Python dependencies (including pLannotate and everything in `requirements.txt`) in one step:

```bash
cd claude-plasmids
conda env create -f environment.yml
conda activate claude-plasmids
```

### 2. Download pLannotate annotation databases

One-time download (~1–2 GB) required for GenBank annotation:

```bash
plannotate setupdb
```

### 3. Configure your API key

Create a `.env` file in the `app/` directory:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > app/.env
export "ANTHROPIC_API_KEY=sk-ant-..." > app/.env
```

You can get an API key at https://console.anthropic.com.

### Optional capabilities

Additional env vars enable optional data sources:

| Env var | Effect | Availability |
|---|---|---|
| `ADDGENE_API_TOKEN` | Addgene developer API token. Required for automatic plasmid sequence retrieval from Addgene. Without this, sequences cannot be fetched automatically — users must manually upload GenBank files for any plasmid not already in the local library. See [Addgene API key setup](#addgene-api-key) below. | CLI + Web UI |
| `PLASMID_USER_LIBRARY` | Path to a directory of user-provided GenBank files (`backbones/*.gb`, `inserts/*.gb`, `annotations/*.gb`). Backbone/insert entries appear with `user:` ID prefix. Annotation files extend pLannotate with custom feature recognition. | CLI + Web UI |
| `BENCHLING_SUBDOMAIN` | Your Benchling workspace subdomain. Enables read+write access via Benchling's remote MCP. | CLI only¹ |
| `PLASMID_ENABLE_PUBMED` | Default `1`. Set `0` to disable PubMed MCP (literature search + PMC full text). | CLI only¹ |
| `UNPAYWALL_EMAIL` | Your email. Enables `fetch_oa_fulltext` for open-access papers outside PMC. | CLI + Web UI |

### Addgene API key

The Addgene API token enables automatic retrieval of plasmid sequences directly from [Addgene's developer API](https://api.developers.addgene.org). Without it, the tool cannot fetch sequences for plasmids not already in the local library — you will need to manually download GenBank files from Addgene and upload them via the `PLASMID_USER_LIBRARY` mechanism.

**To set up:**

1. Obtain an API token from the [Addgene developer portal](https://www.addgene.org/tools/api/).
2. Add it to your `.env` file:

```bash
ADDGENE_API_TOKEN=your-token-here
```

The token is automatically picked up at startup. No restart is required if you add it while the server is not running.

¹ The web UI uses the raw Anthropic API (not the Agent SDK) and cannot attach external MCP servers. Benchling and PubMed tools are only available via `python app/agent.py` or the evals harness.

### 4. Start the web UI

```bash
python app/app.py --reload
# Open http://localhost:8000
```

The `--reload` flag watches for file changes and automatically restarts the server, so edits to source files, the system prompt, or library JSON take effect immediately.

To run without auto-reload (e.g., in production):

```bash
python app/app.py
```

## Architecture

```
src/
├── assembler.py           # Deterministic sequence assembly engine + fusion support
├── library.py             # Backbone/insert library search + Addgene/NCBI auto-fallback
├── ncbi_integration.py    # NCBI Entrez gene search + CDS retrieval (Biopython)
├── server.py              # MCP server with 18 tools (imports from library.py)
├── tools.py               # Standalone tool definitions for agent loop
└── addgene_integration.py # Addgene web scraping, GenBank feature parsing, API client

app/
├── app.py                 # Web UI + SSE streaming server
├── agent.py               # Claude Agent SDK agent loop
└── system_prompt.md       # Agent system prompt (5-step workflow)

library/
├── backbones.json         # Curated backbones + auto-cached Addgene fetches (grows over time)
└── inserts.json           # Inserts: fluorescent proteins, reporters, epitope tags, NCBI genes

evals/
├── rubric.py              # Allen Institute verification rubric (~32 weighted checks, 6 sections)
├── test_cases.py          # 27 benchmark cases across 3 tiers
├── simulated_user.py      # Simulated user for multi-turn disambiguation evals
├── llm_judge.py           # LLM-as-judge grading for transcript quality
└── run_agent_evals.py     # End-to-end agent eval runner (39 cases, Claude Agent SDK)

tests/
├── test_assembler.py      # Assembly engine tests (22 tests)
├── test_library.py        # Library function tests (7 tests)
└── test_pipeline.py       # Pipeline integration tests (27 cases, rubric-scored)

requirements.txt             # Python dependencies
```

## How It Works

1. **User describes what they want** — e.g., "Put EGFP into pcDNA3.1(+)"
2. **Claude retrieves sequences** — from the curated library first, auto-fetching from Addgene (with GenBank feature parsing) if not found locally
3. **Deterministic assembly** — insert spliced into backbone at MCS position (string operations, not LLM generation)
4. **Validation** — rubric-based checks (backbone preserved, insert intact, correct orientation, reading frame, start/stop codons)
5. **Export + Download** — raw sequence, FASTA, or GenBank format with annotations; browser download button for exported files

## Key Design Principle

Every nucleotide in the output comes from a verified source (library JSON, Addgene, GenBank, or user input). The assembly engine is deterministic string splicing. Claude never generates DNA sequences.

## Running Tests

```bash
# All tests (unit + integration + pipeline), excluding slow pLannotate BLAST tests
python -m pytest tests/ -v -m "not slow"

# Include slow tests (requires plannotate setupdb)
python -m pytest tests/ -v

# Annotation tests only
python -m pytest tests/test_annotation.py -v -m "not slow"

# Pipeline tests only (rubric-scored assembly cases)
python -m pytest tests/test_pipeline.py -v

# Single pipeline case
python -m pytest tests/test_pipeline.py -v -k "T1_001"

# Pipeline tests by tier
python -m pytest tests/test_pipeline.py -v -k "tier1"
python -m pytest tests/test_pipeline.py -v -k "tier2"
python -m pytest tests/test_pipeline.py -v -k "tier3"
```

## Running Evals

Evals send natural language prompts through the full Claude agent loop and score the output. Requires `ANTHROPIC_API_KEY` in `app/.env`.

```bash
python -m evals.run_agent_evals
python -m evals.run_agent_evals --case A1-001 -v
python -m evals.run_agent_evals --model sonnet
```

## Test Tiers

### Pipeline tests

Pipeline tests (`tests/test_pipeline.py`) run the assembly engine directly against 27 benchmark cases:

| Tier | Cases | Description |
|------|-------|-------------|
| 1 | 16 | Library sequences provided directly (baseline correctness) |
| 2 | 7 | Backbone/insert resolved by alias from library (name resolution) |
| 3 | 4 | Addgene ground truth comparison (end-to-end validation) |

### Agent evals

Agent evals (`evals/run_agent_evals.py`) send natural language prompts through the full Claude agent loop and score output with the Allen Institute rubric. 39 cases across 8 categories:

| Category | ID Prefix | Cases | Description |
|----------|-----------|-------|-------------|
| Explicit backbone + insert | A1 | 9 | Both backbone and insert named directly (e.g., "Put EGFP into pcDNA3.1(+)"). Baseline correctness across multiple backbones. |
| Alias / name resolution | A2 | 5 | Common aliases and variant spellings (e.g., "eGFP", "GFP", "pGEX", "pcDNA3.1-"). Tests fuzzy matching. |
| Natural language | A3 | 3 | Underspecified requests where the agent must infer backbone and insert (e.g., "I want a green fluorescent protein in mammalian cells"). May have multiple valid answers via `alternative_expected`. |
| Specific insert types | A4 | 4 | Non-standard inserts: large reporters (luciferase, 1653 bp), small epitope tags (FLAG 24 bp, HA 27 bp), tandem dimers (tdTomato). |
| Multi-step workflow | A5 | 3 | Full 5-step workflow: retrieve, assemble, validate, export. Agent must call multiple tools in sequence. |
| NCBI gene retrieval | A6 | 7 | Genes not in the local library — agent must use NCBI Entrez. Includes species disambiguation, gene family disambiguation, alternative name resolution (PAI-1 → SERPINE1), and natural language backbone selection. Multi-turn cases use `user_persona` with a simulated user. |
| Protein tagging / fusions | A7 | 5 | N-terminal and C-terminal tag fusions (FLAG-EGFP, mCherry-HA), NCBI + fusion (H2B-EGFP with default and custom linkers). Tests `fuse_inserts` tool, start/stop codon management, and Kozak sequence handling. Uses `expected_insert_sequence` for ground truth. |
| Negative / balanced | A8 | 3 | Tests that the agent does NOT over-trigger tools. E.g., EGFP (in local library) should not call NCBI; plain EGFP should not call `fuse_inserts`. Uses `tools_should_not_use` assertions. |

## Verification Rubric

The rubric implements the Allen Institute's weighted scoring system across 6 sections:

| Section | Checks | What it validates |
|---------|--------|-------------------|
| Input Validation | Backbone/insert valid DNA, start/stop codons, reading frame | Source sequences are correct |
| Construct Assembly | Insert found, correct position, correct orientation, backbone preserved | Assembly engine correctness |
| Construct Integrity | Full-length output, total size, key features preserved | Structural completeness |
| Biological Sanity | Promoter upstream, polyA downstream, markers intact, origins intact, Kozak context | Biological validity |
| Output Verification | GenBank format, parseable, sequence match, LOCUS size, annotations | Export quality |
| Output Quality | Ground truth comparison (Addgene) | End-to-end accuracy |

Severity weights: **Critical** = 2 pts, **Major** = 1 pt, **Minor** = 0.5 pts, **Info** = 0 pts. A case passes if there are no Critical failures and the weighted score is >= 90%.

## Current Results

- **64 tests passing** (31 unit/integration + 33 pipeline), 1 skipped (pET-28a has no sequence)
- **22/22** Tier 1+2 pipeline tests at 100%
- **4/4** Tier 3 Addgene ground truth tests at 100%
- Primary benchmark (pcDNA3.1(+) + EGFP): **30.0/30.0 pts** across 25 scored checks

## Phased Development Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Single plasmid design for mammalian cells: assembly engine, validation rubric, Addgene integration, NCBI gene retrieval, protein tagging/fusions, web UI, evals | In progress |
| **Phase 2** | Multi-plasmid systems, lentiviral packaging vectors, CRISPR guide RNA design, codon optimization | Planned |
| **Phase 3** | Advanced workflows: restriction enzyme cloning simulation, primer design, gateway cloning, Gibson assembly | Planned |

## Completed Sprint Goals

- **NCBI Gene Retrieval** — Users say "human TP53" and get the correct CDS sequence from NCBI RefSeq
- **Protein Tagging / Fusions** — N-terminal/C-terminal tag fusions (e.g., FLAG-EGFP) with automatic start/stop codon management
- **Natural Language Backbone Selection** — Autonomous backbone selection based on expression context (organism, promoter, selection)
- **Gene Name Disambiguation** — Ambiguous gene names (TRAF -> TRAF1-7), alternative names (SERPINE1 = PAI-1), species disambiguation
- **Multi-turn Disambiguation Evals** — Simulated user for testing agent clarification workflows

## Sample Prompts

```
"Put EGFP into pcDNA3.1(+)"
"Design a plasmid to express human TP53 in HEK293 cells"
"Add an N-terminal FLAG tag to EGFP in pcDNA3.1(+)"
"I want to express a green fluorescent protein in mammalian cells"
"Design a vector to allow expression of MyD88 in RAW 264 cells"
"Express a C-terminal HA-tagged mCherry in pcDNA3.1(+)"
```

## MCP Server

The MCP server (`src/server.py`) exposes 18 tools for Claude to use:

- `search_backbones` / `get_backbone` — search and retrieve backbone vectors
- `search_inserts` / `get_insert` — search and retrieve insert sequences
- `search_addgene` / `fetch_addgene_sequence_with_metadata` / `import_addgene_to_library` — Addgene integration
- `search_gene` / `fetch_gene` — NCBI gene search + CDS retrieval
- `fuse_inserts` — protein tagging / fusion CDS assembly
- `validate_sequence` — DNA validation (valid chars, GC content, codons)
- `assemble_construct` — deterministic sequence assembly
- `export_construct` — format as raw/FASTA/GenBank
- `validate_construct` — rubric-style validation report
- `get_insertion_site` — MCS position info
- `design_construct` — metadata and size estimates

To use as a standalone MCP server:

```bash
python -m src.server
```

## Backbone Library

21 curated backbones including pcDNA3.1(+/-), pUC19, pEGFP-N1, pGEX-4T-1, pBABE-puro, pAAV-CMV, pLKO.1-puro, pCDNA3, and more. When a backbone isn't found locally, it is automatically fetched from Addgene — the GenBank file is parsed for sequence, feature annotations (promoters, resistance genes, origins, polyA signals, MCS), and cached in `backbones.json` for future fast lookups. Backbones with feature annotations get full biological sanity checks in the rubric.

## Custom Annotations

The custom annotation system lets you extend pLannotate's feature recognition with your own sequences — useful for lab-private constructs or recently-published sequences not yet in any public database.

### Setup

Place annotated GenBank files in an `annotations/` subdirectory of your `PLASMID_USER_LIBRARY`:

```
$PLASMID_USER_LIBRARY/
    backbones/          ← existing BYOL backbones
    inserts/            ← existing BYOL inserts
    annotations/        ← custom annotation GenBank files (NEW)
        my_promoter.gb
        new_fluorophore.gb
        ...
```

Each GenBank file can contain one or more annotated features. Any feature with a `/label`, `/gene`, or `/product` qualifier is extracted and becomes a BLAST target. The feature type (CDS, promoter, misc_feature, etc.) and label are preserved in the annotation output.

### How it works

On startup, the app automatically:
1. Scans `annotations/*.gb` for annotated features
2. Builds a local BLAST database from those features (stored in `annotations/.blast_db/`)
3. Rebuilds only when the source files change (MD5 manifest cache)

When you call `extract_insert_from_plasmid` or `extract_inserts_from_plasmid`, results from your custom database are merged with pLannotate's output. Custom annotations take priority when they cover the same region at equal or higher identity.

### Privacy

All annotation sequences stay on your local machine. The BLAST database is built and queried entirely via local subprocesses — no sequences are transmitted to any external service.

### Requirements

BLAST+ must be available (it is installed automatically with the conda environment):

```bash
conda activate claude-plasmids
which makeblastdb   # should resolve to the conda env bin
```

If BLAST is not found, custom annotations are silently disabled and pLannotate-only behaviour is preserved.

### Example

Given a GenBank file `annotations/mCerulean3.gb` with a CDS feature labelled `mCerulean3`, the agent can then extract it by name from any plasmid that contains it:

```
"Extract mCerulean3 from the sequence I uploaded"
```

The feature will be found even if pLannotate's built-in databases don't include it.

## Batch Design

Design multiple plasmids at once by uploading a CSV of descriptions.

### Web UI

Drag and drop your design CSV (see below for expected format) into the chat pane to upload your file. Each row runs through the full agent loop independently. A live progress panel shows the status of every row and provides per-file download buttons as results come in.

### CLI

```bash
python app/batch.py designs.csv
python app/batch.py designs.csv --output ./outputs/
python app/batch.py designs.csv --output ./outputs/ --model claude-sonnet-4-6
```

The CSV must have a `description` column. Two optional columns control output:

| column | required | description |
|---|---|---|
| `description` | yes | free-text design prompt |
| `name` | no | output filename prefix (default: `plasmid_001`, `plasmid_002`, …) |
| `output_format` | no | `genbank` / `fasta` / `both` (default: `genbank`) |

Example CSV:

```csv
description,name,output_format
"Express EGFP in HEK293 cells using pcDNA3.1(+)",egfp_hek293,genbank
"Put mCherry into a lentiviral backbone",mcherry_lenti,both
"Tag GAPDH with FLAG at the C-terminus",gapdh_flag,fasta
```

Rows run sequentially to avoid API rate limits. Output files are saved to the output directory; if no export is produced the agent's text response is saved as `<name>_output.txt` for inspection.
