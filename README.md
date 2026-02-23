# Plasmid Designer

A Claude-native agentic tool for designing expression plasmids. Built as a collaboration between Anthropic and the Allen Institute.

The tool takes a backbone vector, insert gene, and optional parameters as input, and outputs a complete, validated plasmid construct sequence. Claude handles orchestration (understanding user intent, selecting tools, validating results) while all sequence operations are deterministic — no LLM ever generates DNA.

## Setup

### 1. Create a virtual environment

```bash
cd claude-plasmids

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the `app/` directory:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > app/.env
```

You can get an API key at https://console.anthropic.com.

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
# All tests (unit + integration + pipeline)
python -m pytest tests/ -v

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
- `search_addgene` / `get_addgene_plasmid` / `import_addgene_to_library` — Addgene integration
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
