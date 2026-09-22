"""
Example: server-side authorization enforcement via FastMCP middleware.

This example shows how to add Axiomatics ADS authorization directly to a FastMCP
server. Authorization is enforced on every request — the client never needs to be
aware of it and no changes are required to the LangChain/LangGraph side.

Two enforcement points are active:
  - tools/list  → ADS filters which tools the client is allowed to see
  - tools/call  → ADS checks whether the client is allowed to execute each call

User identity comes from the JWT in the ``Authorization: Bearer`` header. The
full JWT claims dict is forwarded as XACML subject attributes so ADS policies can
match on any claim the IdP includes (e.g. ``username``, ``tenant_id``, ``sub``).

Prerequisites:
  - ADS running at http://localhost:8080 with domain_mcp_tool_visibility.yaml loaded
  - fastmcp and PyJWT installed: uv sync --extra fastmcp

Run the server:
    uv run python examples/fastmcp/authz_server.py

Call it (HTTP/SSE transport) — supply a JWT whose claims match your ADS policy:
    # alice sees all tools; bob sees only search_docs (per domain_mcp_tool_visibility.yaml)
    curl -N http://localhost:8000/sse -H "Authorization: Bearer <alice-jwt>"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastmcp import FastMCP

from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware
from axiomatics_ai import PDPConnection, standard_tool_enforcement

mcp = FastMCP("axio-fastmcp-example")


@mcp.tool()
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Results for '{query}': Getting Started Guide, API Reference, FAQ."


@mcp.tool()
def get_account_balance(account_id: str) -> str:
    """Retrieve the balance for an account."""
    return f"Account {account_id} has a balance of $1,250.00."


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'."


# Attach the Axiomatics middleware.
# - Default context extractor: reads Authorization header, decodes JWT, passes
#   the full claims dict as XACML subject attributes.
# - All three enforcement flags default to True (tool filtering + call authz +
#   record filtering).
# - stdio transport is not supported; use http or sse.
mcp.add_middleware(
    FastMCPAuthZMiddleware(
        AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")), enforcement=standard_tool_enforcement(),
        )
    )
)


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
