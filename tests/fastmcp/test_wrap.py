"""Tests for to_authorized_fastmcp_server — it attaches FastMCPAuthZMiddleware to the server
and returns the same server. A tiny fake stands in for the FastMCP server (only add_middleware
is used)."""
from __future__ import annotations

# Imported from the top-level package to also cover the top-level re-export.
from axiomatics_ai import (
    AuthZConfig,
    FastMCPAuthZMiddleware,
    standard_tool_enforcement,
    to_authorized_fastmcp_server,
)


class _FakeServer:
    def __init__(self):
        self.middleware = []

    def add_middleware(self, mw):
        self.middleware.append(mw)


def test_attaches_our_middleware_and_returns_server():
    server = _FakeServer()
    config = AuthZConfig(enforcement=standard_tool_enforcement())
    returned = to_authorized_fastmcp_server(server, config)
    assert returned is server                                    # returns the same server (chainable)
    assert len(server.middleware) == 1
    assert isinstance(server.middleware[0], FastMCPAuthZMiddleware)
    assert server.middleware[0].config is config                 # our config, not a fresh default


def test_uses_config_context_provider_no_separate_param():
    # The wrapper has no context_provider param — identity comes from config.context_provider.
    from axiomatics_ai.core.config import DefaultServerContextProvider

    provider = DefaultServerContextProvider()
    server = to_authorized_fastmcp_server(_FakeServer(), AuthZConfig(context_provider=provider))
    assert server.middleware[0]._provider is provider
