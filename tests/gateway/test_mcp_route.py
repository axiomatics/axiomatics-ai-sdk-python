"""
Unit tests for the gateway MCP route: create_proxy(backend) + FastMCPAuthZMiddleware.

Backend is an in-memory FastMCP server (no port). Enforcement is exercised via an in-memory
fastmcp Client against the proxy — the middleware runs at the MCP-message layer regardless
of transport, so this validates the gateway's MCP enforcement wiring without ADS or HTTP.
Identity-over-HTTP (JWT) is covered by the integration tier.
"""
from __future__ import annotations

import asyncio

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server import create_proxy

from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware

from .conftest import FakePdp


def _backend() -> FastMCP:
    backend = FastMCP("backend")

    @backend.tool
    def search_docs(q: str) -> str:
        """Search docs."""
        return "docs result"

    @backend.tool
    def send_email(to: str) -> str:
        """Send email."""
        return "sent"

    return backend


def _proxy_with_authz(fake: FakePdp):
    """Mirror the gateway's MCP route builder, injecting a fake ADS into the middleware."""
    proxy = create_proxy(_backend())
    mw = FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement()))
    mw._ads = fake
    proxy.add_middleware(mw)
    return proxy


def test_mcp_route_visibility_filters_tools():
    proxy = _proxy_with_authz(FakePdp(allowed_tools={"search_docs"}))

    async def go():
        async with Client(proxy) as client:
            return sorted(t.name for t in await client.list_tools())

    names = asyncio.run(go())
    assert names == ["search_docs"]   # send_email hidden by tool-visibility enforcement


def test_mcp_route_invocation_allows_permitted_and_blocks_denied():
    proxy = _proxy_with_authz(FakePdp(allowed_tools={"search_docs"}))

    async def go():
        async with Client(proxy) as client:
            permitted = await client.call_tool("search_docs", {"q": "x"})
            denied_raised = False
            try:
                await client.call_tool("send_email", {"to": "a@b.c"})
            except ToolError:
                denied_raised = True
            return permitted.data, denied_raised

    result, denied_raised = asyncio.run(go())
    assert "docs result" in str(result)    # permitted call reached the backend
    assert denied_raised                    # denied call blocked with ToolError
