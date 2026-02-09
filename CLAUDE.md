# Claude-Plasmids

Claude-powered plasmid designer built as a collaboration between Anthropic and the Allen Institute.

## Architecture

```
src/                        # Core modules
├── assembler.py            # Deterministic sequence assembly engine
├── library.py              # Backbone/insert library search + Addgene/NCBI fallback
├── ncbi_integration.py     # NCBI Entrez gene search + CDS retrieval
├── server.py               # MCP server (imports from library.py)
├── tools.py                # Standalone tool definitions for Agent SDK
└── addgene_integration.py  # Addgene web scraping, GenBank parsing, API client

app/                        # Web UI + agent
├── app.py                  # Web UI + SSE streaming server + agent loop
├── agent.py                # Claude Agent SDK agent loop
└── system_prompt.md        # Agent system prompt (5-step workflow)

library/                    # JSON data (curated + auto-cached)
├── backbones.json          # Curated backbones + auto-cached Addgene fetches
└── inserts.json            # Inserts: fluorescent proteins, reporters, epitope tags, NCBI genes

evals/                      # Evaluation infrastructure
├── rubric.py               # Allen Institute verification rubric (~32 weighted checks)
├── test_cases.py           # 27 benchmark cases across 3 tiers
├── run_agent_evals.py      # End-to-end agent eval runner (36 cases, Agent SDK)
├── simulated_user.py       # Simulated user for multi-turn disambiguation evals
└── llm_judge.py            # LLM-as-judge grading for transcript quality

tests/                      # Test suite
├── test_assembler.py       # Assembly engine tests
├── test_library.py         # Library function tests
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

`claude-opus-4-5-20251101` — used in both the web UI agent loop and agent evals.

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
