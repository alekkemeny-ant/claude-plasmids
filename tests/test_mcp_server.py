"""Tests for the standalone stdio MCP server (src/server.py)."""

import asyncio
import json
import sys

from src.tools import ALL_TOOLS, ALL_TOOL_NAMES, create_plasmid_tools


def test_server_exposes_all_tools():
    """The MCP server must expose exactly the tools in ALL_TOOLS.

    This is the invariant that keeps server.py from drifting out of sync
    with tools.py — both consume create_plasmid_tools() so they share one
    source of truth.
    """
    cfg = create_plasmid_tools()
    assert cfg["name"] == "plasmid-library"
    assert cfg["type"] == "sdk"
    assert len(ALL_TOOLS) == len(ALL_TOOL_NAMES)
    assert len(ALL_TOOL_NAMES) == len(set(ALL_TOOL_NAMES)), "duplicate tool names"


def test_all_tools_have_valid_schemas():
    """Every tool must have a name, description, and JSON Schema input."""
    for t in ALL_TOOLS:
        assert t.name, f"tool missing name: {t}"
        assert t.description, f"{t.name} missing description"
        assert isinstance(t.input_schema, dict), f"{t.name} schema not a dict"
        assert t.input_schema.get("type") == "object", f"{t.name} schema not object"
        assert callable(t.handler), f"{t.name} handler not callable"


def test_stdio_protocol_roundtrip():
    """Launch the server as a real subprocess and verify the MCP handshake."""
    asyncio.run(_run_stdio_roundtrip())


async def _run_stdio_roundtrip():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "src.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=".",
    )

    async def send(msg):
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        await proc.stdin.drain()

    async def recv():
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=20)
        return json.loads(line.decode())

    try:
        await send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "1.0"}},
        })
        init = await recv()
        assert init["result"]["serverInfo"]["name"] == "plasmid-library"

        await send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

        await send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = await recv()
        names = {t["name"] for t in tools["result"]["tools"]}
        assert names == set(ALL_TOOL_NAMES), (
            f"server exposes {len(names)} tools but ALL_TOOLS has {len(ALL_TOOL_NAMES)}; "
            f"missing: {set(ALL_TOOL_NAMES) - names}, extra: {names - set(ALL_TOOL_NAMES)}"
        )

        await send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "list_all_backbones", "arguments": {}}})
        r = await recv()
        assert "Available Backbones" in r["result"]["content"][0]["text"]
    finally:
        proc.terminate()
        await proc.wait()
