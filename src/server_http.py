#!/usr/bin/env python3
"""Plasmid Library MCP server (HTTP).

Same tool set as :mod:`src.server`, served over HTTP for MCP clients that
can't spawn a local process — e.g. a hosted MCP host like Operon.

Serves both MCP HTTP transports so the client can pick:

* **SSE** (legacy): ``GET /sse`` opens the event stream, ``POST /messages``
  sends requests.
* **Streamable HTTP** (current spec): ``POST /mcp``.

Run::

    python -m src.server_http              # listens on 127.0.0.1:8741
    python -m src.server_http --port 9000

Then register the URL in the MCP client. To expose it to a remote backend,
tunnel it first::

    cloudflared tunnel --url http://127.0.0.1:8741

and register ``https://<tunnel>/sse`` or ``https://<tunnel>/mcp``.

Security
--------
This server binds to ``127.0.0.1`` and runs read-only by default. If you
tunnel it to a public URL, anyone who finds the URL can call the tools.
Defaults and overrides, strongest first:

1. **Filesystem tools are excluded.** ``export_genbank``,
   ``save_vendor_backbone``, and ``import_addgene_to_library`` write to
   disk and are not registered unless ``--allow-writes`` is passed.
   Without it, a remote caller can search and assemble but not persist.
2. **Bearer token** (``--token`` or ``$PLASMID_MCP_TOKEN``) — required
   for every request when set. Set it if the server is reachable beyond
   localhost.
3. **DNS-rebinding protection** is on by default and rejects requests
   with an unexpected ``Host`` header. Tunnels put an arbitrary
   hostname in front of the loopback listener, so you must pass
   ``--allow-any-host`` (and set a token) to use one.
4. **Don't leave it running.** Treat tunnel URLs as ephemeral.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import secrets
import sys

import uvicorn
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

from claude_agent_sdk import create_sdk_mcp_server

from .tools import ALL_TOOLS

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("plasmid-library-http")

_REQUIRED_ACCEPT = b"application/json, text/event-stream"

# Tools that write to the local filesystem. The stdio server (src/server.py)
# is local-only so the trust boundary is the OS user; the HTTP server may be
# tunneled to a public URL, so these are excluded by default.
FILESYSTEM_TOOLS = frozenset(
    {
        "export_genbank",         # writes to caller-controlled output_path
        "save_vendor_backbone",   # writes to PLASMID_USER_LIBRARY
        "import_addgene_to_library",  # writes to library/backbones.json
    }
)


class AcceptHeaderShim:
    """Force the Accept header the Streamable HTTP handler requires.

    ``mcp.server.streamable_http`` does literal ``startswith()`` matching on
    Accept and returns 406 for ``*/*`` or a missing header. Some MCP clients
    don't send an explicit Accept — this shim sets it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = [(k, v) for k, v in scope["headers"] if k != b"accept"]
            headers.append((b"accept", _REQUIRED_ACCEPT))
            scope = {**scope, "headers": headers}
        await self.app(scope, receive, send)


class TrailingSlashShim:
    """Rewrite ``/mcp`` → ``/mcp/`` so the Mount matches without a 307.

    ``Mount("/mcp")`` only matches ``/mcp/...``; Starlette's
    ``redirect_slashes`` adds a 307 for the bare path, but some MCP clients
    don't follow redirects.
    """

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path") in ("/mcp", "/messages"):
            p = scope["path"] + "/"
            scope = {**scope, "path": p, "raw_path": p.encode()}
        await self.inner(scope, receive, send)


class BearerAuth:
    """Reject requests without ``Authorization: Bearer <token>``.

    Constant-time comparison via :func:`secrets.compare_digest`.
    """

    def __init__(self, inner, token: str):
        self.inner = inner
        self._token = token

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope["headers"])
            auth = headers.get(b"authorization", b"").decode("latin-1")
            ok = auth.startswith("Bearer ") and secrets.compare_digest(
                auth[len("Bearer ") :], self._token
            )
            if not ok:
                resp = Response(
                    "unauthorized\n",
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await resp(scope, receive, send)
                return
        await self.inner(scope, receive, send)


def build_app(
    *,
    allow_any_host: bool = False,
    token: str | None = None,
    allow_writes: bool = False,
) -> Starlette:
    tools = (
        ALL_TOOLS
        if allow_writes
        else [t for t in ALL_TOOLS if t.name not in FILESYSTEM_TOOLS]
    )
    server = create_sdk_mcp_server(name="plasmid-library", tools=tools)["instance"]
    init_opts = server.create_initialization_options()

    security = (
        TransportSecuritySettings(enable_dns_rebinding_protection=False)
        if allow_any_host
        else None
    )

    # --- Legacy SSE transport ---
    sse = SseServerTransport("/messages/", security_settings=security)

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read, write):
            await server.run(read, write, init_opts)
        return Response()

    # --- Streamable HTTP transport ---
    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, security_settings=security
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            n_excluded = len(ALL_TOOLS) - len(tools)
            logger.info(
                "plasmid-library HTTP MCP server: %d tools (%d filesystem tools %s)",
                len(tools),
                len(FILESYSTEM_TOOLS),
                "included" if allow_writes else f"excluded; {n_excluded} hidden",
            )
            logger.info("  SSE endpoint:             GET /sse")
            logger.info("  Streamable HTTP endpoint: POST /mcp")
            logger.info("  bearer auth:              %s", "ON" if token else "OFF")
            logger.info("  DNS-rebinding protection: %s", "OFF" if allow_any_host else "ON")
            yield

    streamable = AcceptHeaderShim(manager.handle_request)

    async def health(_request):
        return Response("plasmid-library MCP server\n", media_type="text/plain")

    app = Starlette(
        routes=[
            Route("/", endpoint=health, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages", app=sse.handle_post_message),
            Mount("/mcp", app=streamable),
        ],
        lifespan=lifespan,
    )
    wrapped = TrailingSlashShim(app)
    if token:
        wrapped = BearerAuth(wrapped, token)
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8741)
    parser.add_argument(
        "--allow-any-host",
        action="store_true",
        help="Disable DNS-rebinding protection. Required behind a tunnel "
        "(cloudflared, ngrok). Only do this with --token set.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("PLASMID_MCP_TOKEN"),
        help="Require 'Authorization: Bearer <token>' on every request. "
        "Defaults to $PLASMID_MCP_TOKEN.",
    )
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Expose tools that write to the local filesystem "
        f"({', '.join(sorted(FILESYSTEM_TOOLS))}). Off by default because "
        "the HTTP server may be reachable beyond localhost.",
    )
    args = parser.parse_args()

    if args.allow_any_host and not args.token:
        logger.warning(
            "--allow-any-host without --token: the server will accept "
            "requests from any origin without authentication. Only do this "
            "for short-lived demos."
        )
    if args.host != "127.0.0.1" and not args.token:
        logger.warning(
            "Binding to %s without --token: the server is reachable from "
            "the network without authentication.",
            args.host,
        )

    uvicorn.run(
        build_app(
            allow_any_host=args.allow_any_host,
            token=args.token,
            allow_writes=args.allow_writes,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
