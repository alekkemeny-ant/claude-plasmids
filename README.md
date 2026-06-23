# Plasmid Designer

A Claude-native agentic tool for designing expression plasmids. Takes a natural language description of what you want to build and returns a complete, annotated, validated plasmid construct in GenBank format.

Claude handles orchestration — understanding intent, selecting tools, validating results — while all sequence operations are deterministic. No LLM ever generates DNA.

---

## Table of Contents

- [Setup](#setup)
- [Running the App](#running-the-app)
- [How to Use It Effectively](#how-to-use-it-effectively)
- [Modes and Features](#modes-and-features)
- [Architecture](#architecture)
- [Backbone and Insert Library](#backbone-and-insert-library)
- [Bring Your Own Library (BYOL)](#bring-your-own-library-byol)
- [Custom Annotations](#custom-annotations)
- [Batch Design](#batch-design)
- [Tests and Evals](#tests-and-evals)

---

## Setup

### Prerequisites

pLannotate (used for GenBank annotation) is only available via conda/bioconda and requires Python <3.13. The project uses a conda environment.

### 1. Create the conda environment

```bash
cd claude-plasmids
conda env create -f environment.yml
conda activate claude-plasmids
```

This installs all Python dependencies — including pLannotate and everything in `requirements.txt` — in one step.

### 2. Download pLannotate annotation databases

One-time download (~1–2 GB). Required for GenBank export and feature annotation:

```bash
plannotate setupdb
```

### 3. Configure your API key

**Option A — settings menu (easiest):** Start the app and open **Settings** (gear icon, top-right). Paste your key in the *Anthropic API Key* field and click Save. The key is written to `app/.env` automatically.

**Option B — `.env` file:** Create `app/.env` before starting:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > app/.env
```

Get an API key at https://console.anthropic.com.

> The app will open without a key, but sending a message will prompt you to add one via Settings.

### 4. Optional capabilities

Additional environment variables enable optional data sources. Set them in the **Settings** menu or in `app/.env`:

| Variable | Effect |
|---|---|
| `ADDGENE_API_TOKEN` | Addgene developer API token. Required for automatic sequence retrieval from Addgene. Without it, you must manually upload GenBank files for plasmids not in the local library. Tokens available at [addgene.org/tools/api](https://www.addgene.org/tools/api/). |
| `PLASMID_USER_LIBRARY` | Path to a directory of user-provided GenBank files (`backbones/*.gb`, `inserts/*.gb`, `annotations/*.gb`). Entries appear with a `user:` ID prefix. |
| `UNPAYWALL_EMAIL` | Your email. Enables `fetch_oa_fulltext` for open-access papers via Unpaywall. |
| `BENCHLING_SUBDOMAIN` | Your Benchling workspace subdomain. Enables Benchling read/write via MCP. CLI agent only. |
| `PLASMID_ENABLE_PUBMED` | Default `1`. Set `0` to disable PubMed MCP (literature search + PMC full text). CLI agent only. |

> The web UI uses the Anthropic API directly and cannot attach external MCP servers. Benchling and PubMed tools are only available via `python app/agent.py` or the evals harness.

---

## Running the App

### Web UI

```bash
python app/app.py --reload
# Open http://localhost:8000
```

The `--reload` flag watches for file changes and restarts automatically — useful during development. Drop it for production.

### CLI agent

```bash
python app/agent.py
```

The CLI agent uses the Claude Agent SDK and supports external MCP servers (Benchling, PubMed).

---

## How to Use It Effectively

### Basic prompts

The simplest prompts work best. You don't need to specify technical details — the agent will ask for what it needs.

```
Put EGFP into pcDNA3.1(+)
Design a plasmid to express human TP53 in HEK293 cells
Add an N-terminal FLAG tag to EGFP in pcDNA3.1(+)
I want to express a green fluorescent protein in mammalian cells
Express a C-terminal HA-tagged mCherry in pcDNA3.1(+)
Make a kinase-dead BRAF construct
```

### What the agent always does

Every design runs a five-step workflow:
1. **Clarify** — resolve backbone, insert, and any special requirements
2. **Retrieve sequences** — from the curated library, Addgene, NCBI, or your files
3. **Assemble** — deterministic string splicing at the MCS position
4. **Validate** — backbone preservation, insert integrity, reading frame, start/stop codons, biological sanity
5. **Export** — annotated GenBank file with provenance in the COMMENT field

### Specifying backbones

Name the backbone directly (`pcDNA3.1(+)`, `pLKO.1`, `pAAV-CMV`) or describe what you need:
- `"mammalian expression, constitutive, strong"` → selects CMV-driven backbone
- `"lentiviral, stable integration"` → selects a lentiviral transfer vector
- `"bacterial, IPTG-inducible"` → selects pET-series or similar

### Specifying inserts

- **Fluorescent proteins**: use standard names (`EGFP`, `mCherry`, `mNeonGreen`, `tdTomato`). The agent routes FP-like names through FPbase automatically.
- **Genes from NCBI**: say `human TP53`, `mouse Myc`, or just `TP53` and the agent will ask for species if ambiguous.
- **Gene family disambiguation**: if you say `TRAF` or `H2B`, the agent will present all members and ask which one.
- **Alternative gene names**: `PAI-1` resolves to `SERPINE1`, `eIF4e` resolves to the standard symbol, etc.

### Tips for complex requests

- **Fusion tags**: be explicit about terminus — "N-terminal FLAG on EGFP" or "mCherry with a C-terminal HA tag"
- **Linkers**: for protein-protein fusions (not epitope tags), the agent uses `(GGGGS)×4` by default and will ask if you prefer something else
- **Replacements**: "replace the CMV promoter with EF1α" triggers a feature-swap workflow, not a new assembly
- **Uploading files**: drag a `.gb`, `.gbk`, or `.fasta` file into the chat — the agent annotates it and asks what you'd like to do with it

---

## Modes and Features

### Standard plasmid design (MCS cloning)

The default mode. Insert a gene into a backbone at the multiple cloning site. The agent resolves sequences from the curated library, Addgene, or NCBI, then splices them deterministically.

Works for: fluorescent protein reporters, epitope-tagged proteins, any CDS from NCBI Gene.

---

### Protein tagging and fusions

Handles N-terminal tags, C-terminal tags, and multi-domain fusions with automatic codon management at every junction:

- Stop codons are removed from non-terminal parts
- ATG is retained for small epitope tags (FLAG, HA, Myc, His6, V5)
- ATG is removed from non-first protein partners in a fusion
- Kozak sequences are inserted at the start

Default linker for protein-protein fusions: `(GGGGS)×4`. Custom linkers accepted.

```
Add a FLAG tag to the N-terminus of EGFP in pcDNA3.1(+)
Express H2B-mNeonGreen with a 7x GGGGS linker in pcDNA3.1(+)
Make a C-terminally HA-tagged mCherry
```

---

### Combinatorial fluorescent fusion design

When you ask for a fluorescent fusion, the agent runs `design_fusion_variants` before assembling. This analyses:

- **FP suitability** — pKa compatibility with target compartment (e.g., EGFP fails in lysosomes), oligomerization state, brightness
- **Protein topology** — signal peptides, mitochondrial targeting sequences, transmembrane helices, GPI anchors; each feature may block one terminus
- **Internal loop sites** — disordered regions in the target protein as candidate internal insertion sites
- **Alternative FP suggestions** — if your requested FP has a known issue with the target compartment

Returns ~5 ranked designs for you to choose from before assembly begins.

```
I want to tag CHCHD4 with mCherry to see where it localizes
Design a fusion to label the ER membrane with EGFP
```

---

### Golden Gate assembly

For Type IIS enzyme-based cloning (Golden Gate, MoClo). The agent:

1. Identifies the enzyme (BsaI, Esp3I/BsmBI, BbsI, PaqCI)
2. Checks every insert for enzyme recognition sites — and if found in a CDS, offers to design a silent mutation to remove it
3. Assembles parts from their carrier vectors, discarding the dropout cassette (mCherry/ccdB)
4. Validates and exports

```
Assemble EGFP into my pDONR backbone using Golden Gate with BsaI
```

---

### Golden Gate oligo design (de novo)

When you have raw sequences and want to design a Golden Gate assembly from scratch, the agent designs the primer/oligo set. Supported output formats:

| Format | Description |
|---|---|
| PCR primers | Forward/reverse primers per fragment with enzyme sites in tails |
| Annealing oligos | Top/bottom oligo pairs — no digestion step; best for ≤500 bp fragments |
| gBlocks | Synthesis sequences with flanking enzyme sites |
| Insert cassette | Complete insert for submission to a synthesis vendor (Ansa, Twist) |
| Part-in-vector | Full circular plasmid per fragment for whole-plasmid synthesis (Azenta/Genewiz) |

```
Design Golden Gate oligos for a 3-fragment assembly using BsaI
I need to clone EGFP and mCherry into pDONR using PaqCI — give me annealing oligos
```

---

### Vendor backbone import

Import and save backbones from synthesis vendors (Ansa, Twist, Azenta, etc.) by uploading the GenBank file or pasting the sequence. The agent:

1. Detects the enzyme and insertion site automatically where possible
2. Saves the backbone to the library under a `vendor:` ID
3. Optionally builds the complete part-in-vector plasmids and exports them

---

### Feature swapping (promoter and terminator replacements)

Replace any annotated feature in a plasmid — promoter, terminator, CDS, or regulatory element — by name. The agent uses pLannotate to locate the feature and handles orientation automatically. For promoter swaps, it checks for adjacent enhancers (CMV enhancer + CMV promoter) and swaps both together.

```
Replace the CMV promoter in pcDNA3.1(+) with EF1α
Swap the BGH polyA for SV40 polyA
```

---

### Internal loop insertion

When both termini of a target protein are blocked by signal peptides, transmembrane helices, or targeting sequences, the agent predicts disordered internal loops as candidate insertion sites and offers the top candidates for selection.

---

### Smart mutation design

Introduces GoF or LoF point mutations into any CDS. Has a curated database of well-validated mutations for common oncogenes and tumor suppressors (BRAF, KRAS, TP53, EGFR, PIK3CA, IDH1/2, PTEN, AKT1, MYC, RB1, FBXW7, CTNNB1, NRAS).

- **Point mutation**: swaps exactly one codon using the Kazusa human codon-usage table
- **Premature stop (LoF)**: inserts a stop codon at a specified fractional position

```
Make BRAF V600E in pcDNA3.1(+)
Design a kinase-dead version of EGFR
Give me a TP53 dominant-negative (R175H) construct
```

---

### Bespoke promoter design

When you request a promoter that is not in the standard set, the agent asks which approach you prefer:

1. Search Addgene for published constructs
2. Paste the promoter sequence directly
3. Fetch the native upstream region (~2 kb upstream of TSS) from NCBI

---

### Plasmid file upload intake

Drag a `.gb`, `.gbk`, or `.fasta` file into the chat. The agent:

1. Annotates the plasmid with pLannotate
2. Infers the plasmid type (backbone, part-in-vector, expression plasmid)
3. Walks through a short Q&A to collect metadata (vendor, enzyme, insertion point, name)
4. Saves it to the appropriate library location

After saving, you can immediately use the plasmid in a new design.

---

### Bulk design

Design multiple plasmids at once by uploading a CSV of descriptions. The bulk workflow builds a preview of the first construct for your approval, then processes the remaining rows automatically.

See the [Batch Design](#batch-design) section for CSV format and CLI instructions.

---

### Literature-assisted design

When you reference a paper, the agent searches PubMed for the citation, scans the Methods section for plasmid names and Addgene IDs, and routes them through the normal design workflow. Requires `PLASMID_ENABLE_PUBMED=1` (default) in the CLI agent.

```
The plasmid I want is described in PMID 31819157 — can you find and export it?
```

---

### Troubleshooting mode

When you describe a failed experiment, the agent:

1. Acknowledges the prior attempt and observed outcome
2. Diagnoses the likely cause (no expression, wrong size, mislocalization, toxicity)
3. Re-scores the insert with the Design Confidence Score
4. Proposes 1–3 concrete changes
5. Logs the new outcome when you report results

```
I assembled EGFP in pcDNA3.1(+) last week but got no fluorescence. The cells survived and the plasmid was present on Western.
```

---

### Design Confidence Score

Before finalizing any construct, the agent can compute a 0–100 confidence score for the insert that checks:

- Cryptic polyadenylation signals
- Cryptic splice sites
- Codon Adaptation Index (CAI) for human expression
- Kozak context
- GC content extremes
- Homopolymer runs
- Promoter count in the backbone

Scores below 70 are flagged with a recommendation; scores below 50 prompt a redesign suggestion.

---

## Architecture

```
src/                          # Core modules
├── assembler.py              # Deterministic sequence assembly engine
├── library.py                # Library search + Addgene/NCBI auto-fallback
├── user_library.py           # BYOL — load user-provided GenBank files
├── ncbi_integration.py       # NCBI Entrez gene search + CDS retrieval
├── addgene_integration.py    # Addgene API client + GenBank parsing
├── fpbase_integration.py     # FPbase fluorescent protein database
├── tools.py                  # All tool definitions + build_mcp_servers()
├── server.py                 # MCP server (wraps tools.py)
├── fusion_designer.py        # Combinatorial FP fusion ranking
├── protein_analysis.py       # Disorder prediction, topology analysis
├── confidence.py             # Design Confidence Score engine
├── mutations.py              # Curated GoF/LoF DB + codon-swap engine
├── restriction_utils.py      # Enzyme site checking + silent mutations
├── gg_denovo.py              # Golden Gate oligo design (de novo)
├── vendor_backbone.py        # Vendor backbone import + library saving
├── genbank_export.py         # GenBank with pLannotate plot
├── genbank_utils.py          # GenBank parsing utilities
├── custom_annotations.py     # BYOL custom annotation BLAST integration
├── literature.py             # Unpaywall open-access full-text lookup
├── plasmid_intake.py         # File upload parsing + pLannotate intake
├── references.py             # Provenance tracking for export
└── codon_tables.py           # Kazusa human codon-usage table

app/                          # Web UI + agent
├── app.py                    # HTTP server (SSE streaming, file upload)
├── agent.py                  # Claude Agent SDK agent loop
├── sessions.py               # In-memory session and job state
├── streaming.py              # Anthropic streaming agent loop
├── batch_worker.py           # Background batch job workers
├── bulk_planner.py           # Bulk design planning + cost estimation
├── database.py               # Constructs SQLite database
└── system_prompt.md          # Agent system prompt

library/                      # JSON data (curated + auto-cached)
├── backbones.json            # Curated backbones + Addgene cache
├── inserts.json              # Fluorescent proteins, reporters, epitope tags
└── vendor_backbones.json     # Vendor backbone registry

evals/                        # Evaluation infrastructure
├── rubric.py                 # Weighted scoring rubric (~32 checks, 6 sections)
├── test_cases.py             # Benchmark cases across 3 pipeline tiers
├── run_agent_evals.py        # End-to-end agent eval runner
├── simulated_user.py         # Simulated user for multi-turn evals
└── llm_judge.py              # LLM-as-judge for transcript quality

tests/                        # Test suite
├── test_assembler.py         # Assembly engine
├── test_library.py           # Library functions
├── test_pipeline.py          # Pipeline integration (rubric-scored)
├── test_user_library.py      # BYOL tests
├── test_annotation.py        # pLannotate annotation tests
├── test_gg_denovo.py         # Golden Gate oligo design
├── test_golden_gate.py       # GG assembly
├── test_mutations.py         # Mutation design
└── ...                       # Additional unit tests
```

### Key design principle

Every nucleotide in the output comes from a verified source: the curated library JSON, Addgene (via API), NCBI RefSeq (via Biopython/Entrez), or a sequence the user provides directly. The assembly engine is deterministic string splicing. Claude never generates, guesses, or hallucinates DNA sequences.

---

## Backbone and Insert Library

### Backbone library

21+ curated backbones including:
- **Mammalian**: pcDNA3.1(+), pcDNA3.1(−), pCMV, pEGFP-N1, pCAGGS
- **Lentiviral**: pLKO.1-puro, pBABE-puro, pHAGE
- **AAV**: pAAV-CMV
- **Bacterial**: pUC19, pGEX-4T-1, pET-28a, pBR322
- **Yeast/insect**: pPICZ, pFastBac

When a backbone is not found locally, it is auto-fetched from Addgene (requires `ADDGENE_API_TOKEN`), its GenBank file is parsed for sequence and features (promoters, resistance genes, origins, polyA signals, MCS), and the result is cached in `backbones.json` for all future lookups.

### Insert library

Curated inserts in `library/inserts.json`:
- **Fluorescent proteins**: EGFP, mCherry, mNeonGreen, mTurquoise2, tdTomato, mRuby3, iRFP713, and more
- **Reporters**: firefly luciferase, Renilla luciferase, NanoLuc, β-galactosidase, Cre recombinase
- **Epitope tags**: FLAG, HA, Myc, His6, V5, Strep-II, AVI, SNAP, HALO

Any gene not in the curated library is automatically retrieved from NCBI Gene + RefSeq via Biopython.

---

## Bring Your Own Library (BYOL)

Point `PLASMID_USER_LIBRARY` at a directory of GenBank files:

```
$PLASMID_USER_LIBRARY/
    backbones/        ← backbone vectors (.gb or .gbk)
    inserts/          ← insert sequences (.gb or .gbk)
    annotations/      ← custom annotation files for pLannotate
```

User library entries appear in search results with a `user:` ID prefix and are treated identically to curated entries. The BYOL path is read-only at startup — add files to the directory and restart to make them available.

---

## Custom Annotations

Extend pLannotate's feature recognition with your own sequences — useful for lab-private constructs or recently-published sequences not yet in any public database.

Place annotated GenBank files in `$PLASMID_USER_LIBRARY/annotations/`. Each file can contain one or more annotated features. Any feature with a `/label`, `/gene`, or `/product` qualifier is extracted and becomes a local BLAST target.

On startup the app:
1. Scans `annotations/*.gb` for annotated features
2. Builds a local BLAST database (stored in `annotations/.blast_db/`)
3. Rebuilds only when source files change (MD5 manifest cache)

When you call `extract_insert_from_plasmid` or `extract_inserts_from_plasmid`, custom annotation results are merged with pLannotate output. Custom annotations take priority when they cover the same region.

All sequences stay local — the BLAST database is built and queried via local subprocesses. BLAST+ is installed automatically with the conda environment.

---

## Batch Design

Design multiple plasmids at once by uploading a CSV of descriptions.

### Web UI

Drag and drop your design CSV into the chat pane. A live progress panel shows the status of every row and provides per-file download buttons as results arrive.

### CSV format

The CSV must have a `description` column:

| Column | Required | Description |
|---|---|---|
| `description` | yes | Free-text design prompt |
| `name` | no | Output filename prefix (default: `plasmid_001`, `plasmid_002`, …) |
| `output_format` | no | `genbank` / `fasta` / `both` (default: `genbank`) |

Example:

```csv
description,name,output_format
"Express EGFP in HEK293 cells using pcDNA3.1(+)",egfp_hek293,genbank
"Put mCherry into a lentiviral backbone",mcherry_lenti,both
"Tag GAPDH with FLAG at the C-terminus",gapdh_flag,fasta
```

---

## Tests and Evals

### Running tests

```bash
# All tests (unit + integration + pipeline), excluding slow pLannotate BLAST tests
pytest tests/ -v -m "not slow"

# Include slow tests (requires plannotate setupdb)
pytest tests/ -v

# Pipeline tests only (rubric-scored assembly cases)
pytest tests/test_pipeline.py -v

# Single pipeline case
pytest tests/test_pipeline.py -v -k "T1_001"

# By tier
pytest tests/test_pipeline.py -v -k "tier1"
pytest tests/test_pipeline.py -v -k "tier2"
```

### Pipeline test tiers

Pipeline tests (`tests/test_pipeline.py`) run the assembly engine directly against 27 benchmark cases:

| Tier | Cases | Description |
|---|---|---|
| 1 | 16 | Library sequences provided directly (baseline correctness) |
| 2 | 7 | Backbone/insert resolved by alias (name resolution) |
| 3 | 4 | Addgene ground truth comparison (end-to-end) |

### Running agent evals

Agent evals send natural language prompts through the full Claude agent loop and score output with the Allen Institute rubric. Requires `ANTHROPIC_API_KEY`.

```bash
python -m evals.run_agent_evals
python -m evals.run_agent_evals --case A1-001 -v
python -m evals.run_agent_evals --model sonnet
```

39 cases across 8 categories:

| Category | Prefix | Description |
|---|---|---|
| Explicit backbone + insert | A1 | Both named directly — baseline correctness |
| Alias / name resolution | A2 | Common aliases and variant spellings |
| Natural language | A3 | Underspecified requests; agent must infer |
| Specific insert types | A4 | Luciferase, epitope tags, tdTomato |
| Multi-step workflow | A5 | Full 5-step: retrieve, assemble, validate, export |
| NCBI gene retrieval | A6 | Species disambiguation, family disambiguation, alt names |
| Protein tagging / fusions | A7 | N/C-terminal tags, NCBI + fusion, custom linkers |
| Negative / balanced | A8 | Verifies agent does NOT over-trigger tools |

### Verification rubric

The rubric implements a weighted scoring system across 6 sections:

| Section | What it validates |
|---|---|
| Input Validation | Backbone/insert valid DNA, start/stop codons, reading frame |
| Construct Assembly | Insert found, correct position/orientation, backbone preserved |
| Construct Integrity | Full-length output, total size, key features present |
| Biological Sanity | Promoter upstream, polyA downstream, markers intact, origins intact, Kozak context |
| Output Verification | GenBank format, parseable, sequence match, LOCUS size, annotations |
| Output Quality | Ground truth comparison (Addgene) |

Severity weights: Critical = 2 pts, Major = 1 pt, Minor = 0.5 pts, Info = 0 pts. A case passes if there are no Critical failures and weighted score ≥ 90%.

---

## Development Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Single plasmid design for mammalian cells: assembly engine, validation rubric, Addgene + NCBI + FPbase integration, protein tagging/fusions, Golden Gate, smart mutations, web UI, evals | In progress |
| **Phase 2** | Multi-plasmid systems, lentiviral packaging vectors, CRISPR guide RNA design | Planned |
| **Phase 3** | Advanced workflows: gateway cloning, Gibson assembly simulation, primer design, codon optimization | Planned |
