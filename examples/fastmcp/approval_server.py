"""
Example: a FastMCP server whose large payments need a human's approval — asked by MCP
elicitation.

The server-side counterpart of the human-approval examples, on the same policy
(domain_human_approval.yaml): payments up to 100 EUR go through; a larger payment or any
cancellation needs a human's approval. With no approver configured, FastMCPAuthZMiddleware
asks the CLIENT'S user with an MCP elicitation (approve / reject) during the tool call, then
checks the answer with the PDP. The client must support elicitation (approval_client.py
does); a client that doesn't gets a deny.

Identity comes from the caller's bearer token, read WITHOUT verification (a demo token from
approval_client.py); see agent-identity.md to verify tokens here.

Prerequisites: ADS at http://localhost:8080 with examples/policies/domain_human_approval.yaml.

Run:
    uv run python examples/fastmcp/approval_server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for approval_console
from approval_console import INVOICES, AuditHook, InvoiceServerContextProvider, list_invoices_text

from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware

load_dotenv()

PORT = int(os.getenv("APPROVAL_SERVER_PORT", "8005"))

mcp = FastMCP("finance")


@mcp.tool
def list_invoices() -> str:
    """List the open invoices."""
    return list_invoices_text()


@mcp.tool
def pay_invoice(invoice_id: str) -> str:
    """Pay an invoice."""
    invoice = INVOICES[invoice_id]
    return f"paid {invoice_id} ({invoice['amount']:.2f} EUR to {invoice['creditor']})"


@mcp.tool
def cancel_invoice(invoice_id: str) -> str:
    """Cancel an invoice."""
    return f"cancelled {invoice_id}"


mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
    connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
    enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
    context_provider=InvoiceServerContextProvider(),
    hooks=[AuditHook()],
)))

if __name__ == "__main__":
    mcp.run(transport="http", port=PORT)
