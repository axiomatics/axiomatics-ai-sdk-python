"""
Example: an MCP client that answers the server's approval requests (MCP elicitation).

Calls approval_server.py's tools directly — no model — as alice, and answers each elicitation
at the console (or --approve / --reject):

  1. pay_invoice INV-42 (40 EUR)   → permitted, no approval needed
  2. pay_invoice INV-43 (500 EUR)  → the server asks: approve or reject?
  3. cancel_invoice INV-43         → the server asks: approve or reject?

The elicitation handler is what makes the client able to answer; without one (or with a
client that doesn't support elicitation) the server denies calls that need approval.

Prerequisites:
  - approval_server.py running: uv run python examples/fastmcp/approval_server.py
  - ADS at http://localhost:8080 with examples/policies/domain_human_approval.yaml

Run:
    uv run python examples/fastmcp/approval_client.py [--approve | --reject]
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import jwt
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for approval_console
from approval_console import ask_human

SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://127.0.0.1:{os.getenv('APPROVAL_SERVER_PORT', '8005')}/mcp")
# An unsigned demo identity: the server reads it without verifying (see its docstring).
ALICE = jwt.encode({"username": "alice", "sub": "alice"}, "demo-secret", algorithm="HS256")  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example


async def on_elicitation(message: str, response_type, params, context):
    """The server's question looks like "Approve the call to 'pay_invoice' with {...}? (reason)"."""
    match = re.match(r"Approve the call to '([^']+)' with (\{.*\})\?(?: \((.*)\))?$", message)
    tool, args, reason = match.groups() if match else ("?", message, None)
    answer = ask_human({"tool_name": tool, "args": args, "reason": reason})
    return "approve" if answer["approved"] else "reject"


async def call(client: Client, tool: str, args: dict) -> None:
    print(f"\n[alice] {tool} {args}")
    try:
        result = await client.call_tool(tool, args)
        print(f"[server] {result.content[0].text}")
    except ToolError as exc:
        print(f"[server] {exc}")


async def main() -> None:
    transport = StreamableHttpTransport(SERVER_URL, headers={"Authorization": f"Bearer {ALICE}"})
    async with Client(transport, elicitation_handler=on_elicitation) as client:
        await call(client, "pay_invoice", {"invoice_id": "INV-42"})
        await call(client, "pay_invoice", {"invoice_id": "INV-43"})
        await call(client, "cancel_invoice", {"invoice_id": "INV-43"})


if __name__ == "__main__":
    asyncio.run(main())
