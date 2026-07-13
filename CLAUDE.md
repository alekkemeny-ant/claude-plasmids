# Claude-Plasmids

Claude-powered plasmid designer built as a collaboration between Anthropic and the Allen Institute.

## Architecture

Imports use a single convention: `from src.X import ...` (absolute, `src`
is a package). No bare `from X import` and `src/` is not on `sys.path`.

```
src/                        # Core modules (a package — src/__init__.py)
├── references.py           # Reference tracker (source provenance)
├── config.py               # Shared constants (default fusion linker, Kozak)
├── export.py               # Construct export entrypoint (export_construct)
├── server.py               # MCP server (imports from src.library)
├── tools.py                # Tool definitions + build_mcp_servers() for Agent SDK
├── cloning/                # Cloning strategies
│   ├── assembler.py        # Deterministic assembly: insertion, fusion, MCS, export
│   ├── multiple_cloning_site_handler.py  # MCS detection/handling
│   └── golden_gate/        # Golden Gate (Type IIS) assembly
│       ├── assembly.py     # GG_ENZYMES, find_gg_sites, assemble_golden_gate (split from cloning/assembler.py)
│       └── denovo.py       # De novo overhang/primer/oligo/gBlock design
├── integrations/           # Third-party API connectors
│   ├── addgene_integration.py  # Addgene web scraping, GenBank parsing, API client
│   ├── ncbi_integration.py     # NCBI Entrez gene search + CDS retrieval
│   ├── fpbase_integration.py   # FPbase fluorescent-protein sequence lookup
│   └── literature.py           # Unpaywall open-access full-text lookup
├── data/                   # Static reference data
│   └── codon_tables.py     # Human codon-usage tables (CAI)
├── utils/                  # Shared helpers
│   ├── genbank_utils.py    # GenBank parse/format/export + annotation
│   ├── restriction_utils.py    # Type IIS site checks + silent-mutation design
│   ├── sequence_utils.py   # Generic DNA sequence helpers
│   ├── fasta_utils.py      # FASTA parse/format helpers
│   └── codon_utils.py      # Codon-optimization helpers (uses data/codon_tables)
├── annotation/             # Sequence identification/annotation of user input
│   ├── plasmid_intake.py   # User upload parser (GenBank/FASTA) + plannotate
│   └── custom_annotations.py   # BYOA custom BLAST annotation DB
├── design_tools/           # Construct design advisors + scoring
│   ├── confidence.py       # Design confidence scoring
│   ├── mutations.py        # GoF/LoF mutation design
│   ├── protein_analysis.py # Translation, disorder prediction
│   └── fusion_designer.py  # Fusion protein design advisor
└── library/                # Library package (facade re-exports its public API)
    ├── __init__.py         # Public API: from src.library import load_backbones, ...
    ├── core.py             # Built-in library search/get + Addgene/NCBI/FPbase fallback
    ├── user.py             # BYOL — user GenBank files ($PLASMID_USER_LIBRARY)
    └── vendor.py           # Vendor backbones (Ansa, Twist, ...) -> vendor_backbones.json

app/                        # Web UI + agent
├── app.py                  # Web UI + SSE streaming server + agent loop
├── agent.py                # Claude Agent SDK agent loop
├── streaming.py            # SSE streaming helpers
├── sessions.py             # Session state persistence
├── database.py             # Saved-construct SQLite store (app/constructs.db)
├── batch_worker.py         # Background batch-job worker
├── bulk_planner.py         # Bulk/combinatorial design planning
├── static/                 # Front-end assets (JS/CSS/HTML)
└── system_prompt.md        # Agent system prompt (5-step workflow)

library/                    # JSON DATA (distinct from the src/library/ code package)
├── backbones.json          # Auto-cached Addgene fetches — gitignored, regenerated at runtime
├── inserts.json            # Inserts: fluorescent proteins, reporters, epitope tags, NCBI genes (tracked)
└── vendor_backbones.json   # Vendor-supplied backbones saved via save_vendor_backbone (tracked)

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
# One-time dev setup: editable install with dev extras (pytest, ruff).
# pLannotate is conda-only — see environment.yml for the full env.
pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY etc.

# Web UI (with auto-reload on file changes)
python app/app.py --reload

# Tests
pytest tests/ -v

# Lint / format
ruff check .
ruff format .

# Agent evals (requires ANTHROPIC_API_KEY)
python -m evals.run_agent_evals
```

## Key Design Principle

Every nucleotide in the output comes from a verified source (library JSON, Addgene, NCBI, or user input). The assembly engine is deterministic string splicing. Claude never generates DNA sequences.

## Model

`claude-opus-4-7` — used in both the web UI agent loop and agent evals.

## Dependencies

Declared in `pyproject.toml` (`requirements.txt` mirrors runtime deps for the
plain `pip install -r` workflow). pLannotate is conda-only (`environment.yml`).

- `anthropic` — Claude API client
- `python-dotenv` — environment variable loading
- `requests` — HTTP client for Addgene/NCBI
- `biopython` — NCBI Entrez gene retrieval
- dev extras: `pytest`, `pytest-cov`, `ruff`

## Testing Conventions

- Pipeline tests use rubric scoring (>=90%, no critical failures)
- Agent evals use the Claude Agent SDK with the same tools/system prompt as production
- All tests: `pytest tests/ -v`
- Single pipeline case: `pytest tests/test_pipeline.py -v -k "T1_001"`

## Phase Roadmap

- **Phase 1** (current): Single plasmid design for mammalian cells — assembly engine, validation rubric, Addgene integration, NCBI gene retrieval, protein tagging/fusions, web UI, evals
- **Phase 2**: Multi-plasmid systems, lentiviral packaging, CRISPR guide RNA design
- **Phase 3**: Advanced workflows — codon optimization, restriction enzyme cloning simulation, primer design
