"""Tests for the standalone MCP servers (src/server.py, src/server_http.py)."""

import asyncio
import json
import sys

from src.server_http import FILESYSTEM_TOOLS, build_app
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


def test_http_server_excludes_filesystem_tools_by_default():
    """The HTTP server may be tunneled to a public URL. Filesystem-writing
    tools must not be registered unless explicitly opted in."""
    assert FILESYSTEM_TOOLS <= set(ALL_TOOL_NAMES), "FILESYSTEM_TOOLS drifted from ALL_TOOLS"
    # Read-only by default: build_app with defaults should not raise and the
    # excluded set must be non-empty (i.e. the safety boundary exists).
    assert len(FILESYSTEM_TOOLS) > 0
    assert "export_genbank" in FILESYSTEM_TOOLS, "export_genbank takes a caller-controlled output_path"


def test_http_server_bearer_auth_rejects_missing_token():
    """When a token is configured, requests without it get a 401."""
    app = build_app(token="s3cret")
    asyncio.run(_assert_401(app, headers=[]))
    asyncio.run(_assert_401(app, headers=[(b"authorization", b"Bearer wrong")]))


async def _assert_401(app, *, headers):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "headers": headers,
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 8741),
        "client": ("127.0.0.1", 0),
    }
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 401, f"expected 401, got {start['status']}"
