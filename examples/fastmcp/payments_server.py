"""
Example: a FastMCP payments server that VERIFIES the caller's OAuth access token and enforces
the payment the user approved.

The agent calls this server with a bearer access token (RFC 9068). `AgentIdentityContextProvider`
verifies it — signature against the issuer's published keys (found through OIDC discovery),
issuer, audience, expiry, the `at+jwt` type — and maps its claims to named attributes:
`user_id` (the delegating user), `agent_id` (client_id), `actor_id`, `scopes`, and the one
approved Rich Authorization Request entry (RFC 9396) as `approved_amount` / `approved_creditor`.
The policy (domain_agent_identity.yaml) then allows:

  list_invoices — with the payments.read scope
  pay_invoice   — only for a delegating user, with payments.initiate, inside the approved
                  grant: amount <= approved amount, creditor == approved creditor

The server knows nothing about the tokens' origin except the issuer and audience it trusts. In
this example the issuer is the local test issuer that payments_agent.py starts (see
examples/local_test_issuer.py); in production it is your identity provider's issuer URL.

Prerequisites:
  - ADS running at http://localhost:8080 with domain_agent_identity.yaml loaded
  - uv sync --extra fastmcp

Run the server, then the agent in a second terminal:
    uv run python examples/fastmcp/payments_server.py
    uv run python examples/fastmcp/payments_agent.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastmcp import FastMCP

from axiomatics_ai import (
    AgentIdentityContextProvider,
    AuditLogHook,
    AuthZConfig,
    Claim,
    JwtVerification,
    PDPConnection,
    RarMapping,
    standard_tool_enforcement,
)
from axiomatics_ai.fastmcp import to_authorized_fastmcp_server

ISSUER = os.getenv("ISSUER_URL", "http://127.0.0.1:9400")          # your identity provider
AUDIENCE = os.getenv("PAYMENTS_AUDIENCE", "https://pay-mcp.example.com")  # this server
PORT = int(os.getenv("MCP_PORT", "8003"))

INVOICES = [
    {"invoice_id": "INV-42", "amount": 123.50, "currency": "EUR", "creditor": "Hotel Adlon", "item": "Berlin, 2 nights"},
    {"invoice_id": "INV-43", "amount": 500.00, "currency": "EUR", "creditor": "Hotel Adlon", "item": "Berlin, suite upgrade"},
]

mcp = FastMCP("payments")


@mcp.tool()
def list_invoices() -> list[dict]:
    """List the user's open invoices."""
    return INVOICES


@mcp.tool()
def pay_invoice(invoice_id: str, amount: float, creditor: str) -> str:
    """Pay an invoice: the invoice id, the amount in EUR, and the creditor's name."""
    return f"Paid {amount:.2f} EUR to {creditor} for {invoice_id}."


config = AuthZConfig(
    connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
    # Visibility + invocation. (Post-invocation would filter list_invoices' records, which is
    # not what this example is about.)
    enforcement=standard_tool_enforcement(post_invocation=False),
    context_provider=AgentIdentityContextProvider(
        verification=JwtVerification(issuer=ISSUER, audience=AUDIENCE),   # keys via OIDC discovery
        rar=RarMapping(type="payment_initiation", fields={
            "instructedAmount.amount": Claim("approved_amount", type="double"),  # "123.50" -> 123.5
            "creditorName": "approved_creditor",
        }),
    ),
    hooks=[AuditLogHook()],   # one JSON audit record per decision on the "axiomatics_ai.audit" logger
)
to_authorized_fastmcp_server(mcp, config)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for noisy in ("httpx", "mcp", "uvicorn", "fastmcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"Payments MCP server on :{PORT}/mcp — trusting issuer {ISSUER}, audience {AUDIENCE}")
    mcp.run(transport="http", port=PORT)
