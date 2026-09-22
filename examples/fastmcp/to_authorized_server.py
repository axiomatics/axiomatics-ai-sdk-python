"""
Example: server-side authorization via the ``to_authorized_fastmcp_server`` wrapper.

Same enforcement as ``authz_server.py`` (tools/list filtering + tools/call authz + record
filtering, identity from the ``Authorization: Bearer`` JWT), but attached with the one-call
convenience wrapper instead of ``mcp.add_middleware(FastMCPAuthZMiddleware(...))``:

    mcp = to_authorized_fastmcp_server(FastMCP("my-server"), config)

Identity extraction comes from ``config.context_provider`` (default: decode the JWT). Use the
raw ``FastMCPAuthZMiddleware`` if you need a per-call provider override.

Prerequisites:
  - ADS running at http://localhost:8080 with a matching domain loaded
  - fastmcp and PyJWT installed: uv sync --extra fastmcp

Run:  uv run python examples/fastmcp/to_authorized_server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastmcp import FastMCP

from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.fastmcp import to_authorized_fastmcp_server
from axiomatics_ai import PDPConnection, standard_tool_enforcement

mcp = FastMCP("axio-fastmcp-wrapper-example")


@mcp.tool()
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Results for '{query}': Getting Started Guide, API Reference, FAQ."


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'."


# One-call integration — attaches FastMCPAuthZMiddleware and returns the server.
mcp = to_authorized_fastmcp_server(
    mcp,
    AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        enforcement=standard_tool_enforcement(),
    ),
)


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
