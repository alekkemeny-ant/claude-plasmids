#!/usr/bin/env python3
"""Plasmid Library MCP server (stdio).

Exposes every tool in :data:`src.tools.ALL_TOOLS` over stdio so the plasmid
designer can run as a standalone MCP server in any MCP client (Claude
Desktop, Operon, Cowork, Claude Code, third-party hosts).

The tool set is derived directly from ``ALL_TOOLS`` — adding a tool there
automatically exposes it here, so this file never goes stale.

Run:
    python -m src.server

Or register in an MCP client config:
    {
      "mcpServers": {
        "plasmid-library": {
          "command": "python",
          "args": ["-m", "src.server"],
          "cwd": "/path/to/claude-plasmids"
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import logging
import sys

from mcp.server.stdio import stdio_server

from .tools import ALL_TOOLS, create_plasmid_tools

# stdout is the MCP transport — log to stderr only.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("plasmid-library")


async def main() -> None:
    server = create_plasmid_tools()["instance"]
    logger.info("plasmid-library MCP server: %d tools", len(ALL_TOOLS))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
