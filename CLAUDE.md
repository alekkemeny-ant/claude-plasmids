# Claude-Plasmids

Claude-powered plasmid designer built as a collaboration between Anthropic and the Allen Institute.

## Architecture

Imports use a single convention: `from src.X import ...` (absolute, `src`
is a package). No bare `from X import` and `src/` is not on `sys.path`.

```
src/                        # Core modules (a package — src/__init__.py)
├── assembler.py            # Deterministic assembly: insertion, fusion, MCS, export
├── references.py           # Reference tracker (source provenance)
├── server.py               # MCP server (imports from src.library)
├── tools.py                # Tool definitions + build_mcp_servers() for Agent SDK
├── cloning/                # Cloning strategies
│   └── golden_gate/        # Golden Gate (Type IIS) assembly
│       ├── assembly.py     # GG_ENZYMES, find_gg_sites, assemble_golden_gate (split from assembler.py)
│       └── denovo.py       # De novo overhang/primer/oligo/gBlock design
├── integrations/           # Third-party API connectors
│   ├── addgene_integration.py  # Addgene web scraping, GenBank parsing, API client
│   ├── ncbi_integration.py     # NCBI Entrez gene search + CDS retrieval
│   ├── fpbase_integration.py   # FPbase fluorescent-protein sequence lookup
│   └── literature.py           # Unpaywall open-access full-text lookup
├── utils/                  # Shared helpers
│   ├── genbank_utils.py    # GenBank parse/format/export + annotation
│   ├── restriction_utils.py    # Type IIS site checks + silent-mutation design
│   └── codon_tables.py     # Human codon-usage tables (CAI)
├── analysis/               # Sequence/protein analysis + intake
│   ├── confidence.py       # Design confidence scoring
│   ├── mutations.py        # GoF/LoF mutation design
│   ├── protein_analysis.py # Translation, disorder prediction
│   ├── fusion_designer.py  # Fusion protein design advisor
│   ├── plasmid_intake.py   # User upload parser (GenBank/FASTA)
│   └── custom_annotations.py   # BYOA custom BLAST annotation DB
└── library/                # Library package (facade re-exports its public API)
    ├── __init__.py         # Public API: from src.library import load_backbones, ...
    ├── core.py             # Built-in library search/get + Addgene/NCBI/FPbase fallback
    ├── user.py             # BYOL — user GenBank files ($PLASMID_USER_LIBRARY)
    └── vendor.py           # Vendor backbones (Ansa, Twist, ...) -> vendor_backbones.json

app/                        # Web UI + agent
├── app.py                  # Web UI + SSE streaming server + agent loop
├── agent.py                # Claude Agent SDK agent loop
└── system_prompt.md        # Agent system prompt (5-step workflow)

library/                    # JSON DATA (distinct from the src/library/ code package)
├── backbones.json          # Curated backbones + auto-cached Addgene fetches
├── inserts.json            # Inserts: fluorescent proteins, reporters, epitope tags, NCBI genes
└── vendor_backbones.json   # Vendor-supplied backbones saved via save_vendor_backbone

evals/                      # Evaluation infrastructure
├── rubric.py               # Allen Institute verification rubric (~32 weighted checks)
├── test_cases.py           # 27 benchmark cases across 3 tiers
├── run_agent_evals.py      # End-to-end agent eval runner (39 cases, Agent SDK)
├── simulated_user.py       # Simulated user for multi-turn disambiguation evals
└── llm_judge.py            # LLM-as-judge grading for transcript quality

tests/                      # Test suite
├── test_assembler.py       # Assembly engine tests
├── test_library.py         # Library function tests
├── test_user_library.py    # BYOL tests (incl. cache-isolation invariant)
├── test_literature.py      # Unpaywall lookup tests (mocked)
└── test_pipeline.py        # Pipeline integration tests (rubric-scored)
```

## How to Run

```bash
# Web UI (with auto-reload on file changes)
python app/app.py --reload

# Tests
pytest tests/ -v

# Agent evals (requires ANTHROPIC_API_KEY)
python -m evals.run_agent_evals
```

## Key Design Principle

Every nucleotide in the output comes from a verified source (library JSON, Addgene, NCBI, or user input). The assembly engine is deterministic string splicing. Claude never generates DNA sequences.

## Model

`claude-opus-4-7` — used in both the web UI agent loop and agent evals.

## Dependencies

- `anthropic` — Claude API client
- `python-dotenv` — environment variable loading
- `requests` — HTTP client for Addgene/NCBI
- `pytest` — test runner
- `biopython` — NCBI Entrez gene retrieval

## Testing Conventions

- Pipeline tests use rubric scoring (>=90%, no critical failures)
- Agent evals use the Claude Agent SDK with the same tools/system prompt as production
- All tests: `pytest tests/ -v`
- Single pipeline case: `pytest tests/test_pipeline.py -v -k "T1_001"`

## Phase Roadmap

- **Phase 1** (current): Single plasmid design for mammalian cells — assembly engine, validation rubric, Addgene integration, NCBI gene retrieval, protein tagging/fusions, web UI, evals
- **Phase 2**: Multi-plasmid systems, lentiviral packaging, CRISPR guide RNA design
- **Phase 3**: Advanced workflows — codon optimization, restriction enzyme cloning simulation, primer design
