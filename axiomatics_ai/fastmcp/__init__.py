"""Server-side authorization for FastMCP servers.

``FastMCPAuthZMiddleware`` enforces tool visibility, invocation (with human approval by
elicitation) and post-invocation on an MCP server; ``to_authorized_fastmcp_server`` adds it
to a server.
"""
from axiomatics_ai._extras import requires_extra
from axiomatics_ai.core.config import DefaultServerContextProvider
from axiomatics_ai.core.authz_helpers import decode_bearer_jwt_unverified

with requires_extra("fastmcp", "axiomatics_ai.fastmcp"):
    from axiomatics_ai.fastmcp.fastmcp_middleware import FastMCPAuthZMiddleware
    from axiomatics_ai.fastmcp.wrap import to_authorized_fastmcp_server

__all__ = [
    "DefaultServerContextProvider",
    "FastMCPAuthZMiddleware",
    "decode_bearer_jwt_unverified",
    "to_authorized_fastmcp_server",
]
