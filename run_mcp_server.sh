#!/usr/bin/env bash
# Launcher for the plasmid-library stdio MCP server.
#
# Use this when an MCP client (Operon, Claude Desktop, etc.) doesn't let you
# set a working directory — it cd's to the repo root and runs `python -m
# src.server` so the relative imports and library paths resolve.
#
# Register in an MCP client with:
#   command: /path/to/claude-plasmids/run_mcp_server.sh

set -euo pipefail
cd "$(dirname "$0")"

# Use the conda env's python if it exists; fall back to whatever's on PATH.
PY="$(command -v python3)"
for candidate in \
    "${CONDA_PREFIX:-}/bin/python" \
    /opt/homebrew/Caskroom/miniforge/base/envs/claude-plasmids/bin/python \
    "$HOME/miniconda3/envs/claude-plasmids/bin/python"
do
    if [[ -x "$candidate" ]]; then
        PY="$candidate"
        break
    fi
done

exec "$PY" -m src.server
