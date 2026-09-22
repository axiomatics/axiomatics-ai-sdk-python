"""
Example: a LangChain agent paying invoices through the verified payments MCP server.

The agent sends an OAuth access token with every MCP call. What it may do depends on what the
token says, as verified and decided on the server (payments_server.py):

  1. alice's token, with her approval of ONE payment (123.50 EUR to Hotel Adlon, as a Rich
     Authorization Request grant) — "Pay invoice INV-42"  → the payment is permitted
  2. the same token — "Pay invoice INV-43" (500 EUR)       → denied: over what she approved
  3. the agent's own token (Client Credentials, no user)   → it can list invoices, not pay

The model is real and decides the tool calls itself, so the exact calls can vary between runs;
the script prints every tool call the agent made and what came back.

The tokens come from the LOCAL TEST ISSUER (examples/local_test_issuer.py), which this script starts on
localhost:9400 as a stand-in for your identity provider: in a real application alice's token
would come from an OAuth login (Authorization Code) and her payment approval from a consent step
(a redirect or CIBA) that puts the grant into the token. Verification and every decision are real.

Prerequisites:
  - payments_server.py running: uv run python examples/fastmcp/payments_server.py
  - ADS running at http://localhost:8080 with domain_agent_identity.yaml loaded
  - .env (or env vars) with LLM_MODEL and the provider's API key

Run:
    uv run python examples/fastmcp/payments_agent.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for the shared local_test_issuer
from local_test_issuer import LocalTestIssuer

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/google/gemini-3.5-flash-lite")
SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8003/mcp")
AUDIENCE = os.getenv("PAYMENTS_AUDIENCE", "https://pay-mcp.example.com")
AGENT_ID = "spiffe://example.com/agents/travel"
SYSTEM = ("You are a travel assistant that pays the user's travel invoices. Use list_invoices to "
          "find an invoice, then pay_invoice with its exact invoice_id, amount and creditor. If a "
          "tool call is denied, say so plainly; do not retry with other values.")


def approved_payment(amount: str, creditor: str) -> dict:
    """The RAR entry the identity provider puts in the token once the user approved a payment."""
    return {"type": "payment_initiation", "actions": ["initiate"], "locations": [AUDIENCE],
            "instructedAmount": {"currency": "EUR", "amount": amount}, "creditorName": creditor}


async def run(label: str, token: str, request: str) -> None:
    print(f"\n── {label}\n   request: {request}")
    client = MultiServerMCPClient({"payments": {
        "transport": "streamable_http", "url": SERVER_URL,
        "headers": {"Authorization": f"Bearer {token}"},
    }})
    tools = await client.get_tools()
    for tool in tools:
        tool.handle_tool_error = True   # a denied call comes back to the model as a tool error
    print(f"   tools the agent can see: {[t.name for t in tools]}")
    agent = create_agent(model=ChatLiteLLM(model=LLM_MODEL), tools=tools, system_prompt=SYSTEM)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": request}]})
    calls = {c["id"]: c for m in result["messages"] for c in getattr(m, "tool_calls", None) or []}
    for message in result["messages"]:
        if message.type == "tool":
            call = calls.get(message.tool_call_id, {})
            args = ", ".join(f"{k}={v}" for k, v in call.get("args", {}).items())
            content = message.content
            if isinstance(content, list):  # MCP results arrive as content blocks
                content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            outcome = " ".join(str(content).split())
            print(f"   {call.get('name')}({args}) -> {outcome[:110]}")
    print(f"   agent: {' '.join(str(result['messages'][-1].content).split())[:220]}")


async def main() -> None:
    issuer = LocalTestIssuer(port=int(os.getenv("LOCAL_ISSUER_PORT", "9400")))
    print(f"Local test issuer (stand-in identity provider) at {issuer.url}")
    alice = issuer.access_token(
        audience=AUDIENCE, sub="alice@example.com", client_id=AGENT_ID,
        scope="payments.read payments.initiate", acr="mfa", txn="txn-8b47",
        authorization_details=[approved_payment("123.50", "Hotel Adlon")],
    )
    agent_alone = issuer.access_token(      # Client Credentials: the agent is its own subject
        audience=AUDIENCE, sub=AGENT_ID, client_id=AGENT_ID, scope="payments.read payments.initiate",
    )
    await run("alice, who approved paying 123.50 EUR to Hotel Adlon", alice, "Pay invoice INV-42.")
    await run("alice, same approval", alice, "Pay invoice INV-43.")
    await run("the agent on its own (Client Credentials, no user)", agent_alone, "Pay invoice INV-42.")
    issuer.close()


if __name__ == "__main__":
    asyncio.run(main())
