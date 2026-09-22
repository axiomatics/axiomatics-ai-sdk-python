"""
Example: a LangChain agent behind your own API, enforcing on the user's verified access token.

A common setup: the agent runs in your backend behind a `/chat` endpoint, and the frontend calls
it with the signed-in user's OAuth access token (`Authorization: Bearer ...`, audience = your
agent API). The agent's backend is the resource server for that token, so it verifies it and
uses its claims for every tool decision. `AgentIdentityContextProvider` does both on the
client path: `token_from` finds the token in the agent's runtime context, the endpoint puts it
there, and the SDK's LangChain middleware checks tool visibility and every tool call against
the policy (domain_agent_identity.yaml):

  list_invoices — with the payments.read scope
  pay_invoice   — only for a delegating user, with payments.initiate, inside the payment the
                  user approved (a Rich Authorization Request grant in the token)

Three calls to /chat:
  1. alice's token, approving 123.50 EUR to Hotel Adlon — "Pay invoice INV-42"  → paid
  2. the same token — "Pay invoice INV-43" (500 EUR)                            → denied
  3. no token at all                → the agent sees no tools; anything it tries is denied

The model is real and decides the tool calls itself, so the exact calls can vary between runs;
the script prints every tool call the agent made and what came back. The tokens come from the
LOCAL TEST ISSUER (examples/local_test_issuer.py), a stand-in for your identity provider; the
HTTP calls to /chat stand in for your frontend. Verification and every decision are real.

If your agent platform already verifies the token (LangGraph Platform auth, AgentCore
Runtime), use `AgentIdentityContextProvider(claims_from=...)` with the verified claims instead.

Prerequisites:
  - ADS running at http://localhost:8080 with domain_agent_identity.yaml loaded
  - .env (or env vars) with LLM_MODEL and the provider's API key

Run:
    uv run python examples/langchain_agents/agent_behind_api.py
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_litellm import ChatLiteLLM
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from axiomatics_ai import (
    AgentIdentityContextProvider,
    AuthZConfig,
    Claim,
    JwtVerification,
    PDPConnection,
    RarMapping,
    create_authorized_agent,
    standard_tool_enforcement,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for the shared local_test_issuer
from local_test_issuer import LocalTestIssuer

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/google/gemini-3.5-flash-lite")
AGENT_API = "https://travel-agent.example.com"      # this API: the audience of the user's token
AGENT_ID = "spiffe://example.com/agents/travel"
PORT = int(os.getenv("AGENT_API_PORT", "8004"))
SYSTEM = ("You are a travel assistant that pays the user's travel invoices. Use list_invoices to "
          "find an invoice, then pay_invoice with its exact invoice_id, amount and creditor. If a "
          "tool call is denied or no tool is available, say so plainly; do not retry with other values.")

INVOICES = [
    {"invoice_id": "INV-42", "amount": 123.50, "currency": "EUR", "creditor": "Hotel Adlon", "item": "Berlin, 2 nights"},
    {"invoice_id": "INV-43", "amount": 500.00, "currency": "EUR", "creditor": "Hotel Adlon", "item": "Berlin, suite upgrade"},
]


@tool
def list_invoices() -> list[dict]:
    """List the user's open invoices."""
    return INVOICES


@tool
def pay_invoice(invoice_id: str, amount: float, creditor: str) -> str:
    """Pay an invoice: the invoice id, the amount in EUR, and the creditor's name."""
    return f"Paid {amount:.2f} EUR to {creditor} for {invoice_id}."


def build_agent(issuer_url: str):
    config = AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        enforcement=standard_tool_enforcement(post_invocation=False),
        context_provider=AgentIdentityContextProvider(
            verification=JwtVerification(issuer=issuer_url, audience=AGENT_API),
            token_from=lambda context: context.get("access_token"),   # put there by /chat
            rar=RarMapping(type="payment_initiation", fields={
                "instructedAmount.amount": Claim("approved_amount", type="double"),
                "creditorName": "approved_creditor",
            }),
        ),
    )
    return create_authorized_agent(
        model=ChatLiteLLM(model=LLM_MODEL), tools=[list_invoices, pay_invoice],
        system_prompt=SYSTEM, config=config,
    )


def build_api(agent) -> Starlette:
    async def chat(request: Request) -> JSONResponse:
        auth = request.headers.get("authorization", "")
        token = auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else None
        body = await request.json()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": body["message"]}]},
            context={"access_token": token},     # the SDK verifies it before every tool decision
        )
        calls = {c["id"]: c for m in result["messages"] for c in getattr(m, "tool_calls", None) or []}
        steps = [
            {"tool": calls.get(m.tool_call_id, {}).get("name"), "args": calls.get(m.tool_call_id, {}).get("args"),
             "result": " ".join(str(m.content).split())[:110]}
            for m in result["messages"] if m.type == "tool"
        ]
        return JSONResponse({"answer": str(result["messages"][-1].content), "steps": steps})

    return Starlette(routes=[Route("/chat", chat, methods=["POST"])])


def serve(app: Starlette) -> None:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", PORT), 1).close()
            return
        except OSError:
            time.sleep(0.1)


def approved_payment(amount: str, creditor: str) -> dict:
    """The RAR entry the identity provider puts in the token once the user approved a payment."""
    return {"type": "payment_initiation", "actions": ["initiate"], "locations": [AGENT_API],
            "instructedAmount": {"currency": "EUR", "amount": amount}, "creditorName": creditor}


def main() -> None:
    issuer = LocalTestIssuer()
    print(f"Local test issuer (stand-in identity provider) at {issuer.url}")
    serve(build_api(build_agent(issuer.url)))
    alice = issuer.access_token(
        audience=AGENT_API, sub="alice@example.com", client_id=AGENT_ID,
        scope="payments.read payments.initiate", txn="txn-8b47",
        authorization_details=[approved_payment("123.50", "Hotel Adlon")],
    )
    calls = [
        ("alice, who approved paying 123.50 EUR to Hotel Adlon", alice, "Pay invoice INV-42."),
        ("alice, same approval", alice, "Pay invoice INV-43."),
        ("no token", None, "Pay invoice INV-42."),
    ]
    with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=180) as frontend:
        for label, token, message in calls:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            reply = frontend.post("/chat", json={"message": message}, headers=headers).json()
            print(f"\n── {label}\n   POST /chat: {message}")
            for step in reply["steps"]:
                args = ", ".join(f"{k}={v}" for k, v in (step["args"] or {}).items())
                print(f"   {step['tool']}({args}) -> {step['result']}")
            print(f"   agent: {' '.join(reply['answer'].split())[:220]}")
    issuer.close()


if __name__ == "__main__":
    main()
