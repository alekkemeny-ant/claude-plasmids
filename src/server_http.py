#!/usr/bin/env python3
"""Plasmid Library MCP server (HTTP).

Same tool set as :mod:`src.server`, served over HTTP for MCP clients that
can't spawn a local process — e.g. the Operon web app's backend.

Serves both MCP HTTP transports so the client can pick:

* **SSE** (legacy, what Operon's connector uses today):
  ``GET /sse`` opens the event stream, ``POST /messages`` sends requests.
* **Streamable HTTP** (current spec): ``POST /mcp``.

Run::

    python -m src.server_http              # listens on 127.0.0.1:8741
    python -m src.server_http --port 9000

Then register the URL in the MCP client. To expose it to a remote backend,
tunnel it first::

    cloudflared tunnel --url http://127.0.0.1:8741

and register ``https://<tunnel>/sse`` (Operon) or ``https://<tunnel>/mcp``.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys

import uvicorn
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

from .tools import ALL_TOOLS, create_plasmid_tools

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("plasmid-library-http")

# Tunnels (cloudflared, ngrok) give an arbitrary public hostname; the SDK's
# default DNS-rebinding protection rejects unknown Host headers. We bind to
# loopback only and the tunnel is the auth boundary, so allow any Host.
_SECURITY = TransportSecuritySettings(enable_dns_rebinding_protection=False)
_REQUIRED_ACCEPT = b"application/json, text/event-stream"


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


def build_app() -> Starlette:
    server = create_plasmid_tools()["instance"]
    init_opts = server.create_initialization_options()

    # --- Legacy SSE transport (what Operon uses) ---
    sse = SseServerTransport("/messages/", security_settings=_SECURITY)

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read, write):
            await server.run(read, write, init_opts)
        return Response()

    # --- Streamable HTTP transport (current spec) ---
    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, security_settings=_SECURITY
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            logger.info("plasmid-library HTTP MCP server: %d tools", len(ALL_TOOLS))
            logger.info("  SSE endpoint:             GET /sse")
            logger.info("  Streamable HTTP endpoint: POST /mcp")
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

    class TrailingSlashShim:
        """Rewrite /mcp -> /mcp/ so the Mount matches without a 307.

        ``Mount("/mcp")`` only matches ``/mcp/...``; Starlette's
        ``redirect_slashes`` adds a 307 for the bare path, but some MCP
        clients don't follow redirects.
        """

        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope.get("type") == "http" and scope.get("path") in ("/mcp", "/messages"):
                p = scope["path"] + "/"
                scope = {**scope, "path": p, "raw_path": p.encode()}
            await self.inner(scope, receive, send)

    return TrailingSlashShim(app)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8741)
    args = parser.parse_args()
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
