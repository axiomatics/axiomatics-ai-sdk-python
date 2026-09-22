"""Convenience wrapper for attaching PDP authorization to a FastMCP server.

``to_authorized_fastmcp_server(server, config)`` attaches ``FastMCPAuthZMiddleware(config)`` to
an existing FastMCP server and returns the same server, so it reads like the other
``to_authorized_*`` wrappers::

    mcp = to_authorized_fastmcp_server(FastMCP("my-server"), config)

It is exactly equivalent to the explicit form (also supported), just one call::

    mcp = FastMCP("my-server")
    mcp.add_middleware(FastMCPAuthZMiddleware(config))

Identity extraction is taken from ``config.context_provider`` (default: read the
``Authorization: Bearer`` JWT / HTTP headers). Need a per-call provider override, or to set the
internal ``surface`` label? Use the raw ``FastMCPAuthZMiddleware`` directly.
"""
from __future__ import annotations

from typing import Any

from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.fastmcp.fastmcp_middleware import FastMCPAuthZMiddleware


def to_authorized_fastmcp_server(server: Any, config: AuthZConfig | None = None):
    """Attach ``FastMCPAuthZMiddleware(config)`` to a FastMCP server and return it.

    Equivalent to ``server.add_middleware(FastMCPAuthZMiddleware(config))``. To set the
    ``surface`` label, use the middleware directly.

    Args:
        server (FastMCP): The server; changed in place.
        config (AuthZConfig, optional): What to enforce and where the PDP is.

    Returns:
        FastMCP: ``server``, for chaining.
    """
    server.add_middleware(FastMCPAuthZMiddleware(config))
    return server
