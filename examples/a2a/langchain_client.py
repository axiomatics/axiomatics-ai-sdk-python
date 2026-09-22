"""
A2A CLIENT — a LangChain ``create_agent`` that calls a remote agent, with egress authz.

LangChain has no *official* native A2A client yet (unlike ``langchain-mcp-adapters`` for
MCP; an official ``langchain-a2a-adapters`` is only a proposed feature request as of mid
2026, and the community ``a2a-langchain-adapters`` is third-party). So we follow the
ecosystem convention — surface the remote agent as a tool — and hand-roll a small tool
around the a2a-sdk client. Hand-rolling is deliberate: it lets us attach ``A2AClientAuthZ``
to the httpx client, so every outbound message/send is authorized against ADS *before it
leaves this process*. (When an official adapter ships that exposes an httpx/client hook,
the same ``A2AClientAuthZ`` would attach there instead.)

Identity is the local caller (alice / bob), also sent as a Bearer JWT so a protected
server can enforce too.

  --user alice|bob|both   whose identity to run as (default: both)
  --no-authz              drop A2AClientAuthZ (outbound calls are no longer gated here)

Under domain_a2a.yaml: alice may call billing-agent, bob may not. Acme's billing address
"1200 Market Street" appearing in the output means the remote agent was reached.

Run after langchain_server.py is up (and ADS is running).
Config in .env: PDP_BASE_URL, LLM_MODEL, OPENROUTER_API_KEY
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import jwt
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory, create_text_message_object
from a2a.utils import get_message_text
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_litellm import ChatLiteLLM

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.a2a import A2AClientAuthZ
from axiomatics_ai import PDPConnection

AGENT_URL = "http://127.0.0.1:9103"
CANARY = "1200 Market Street"


def bearer(username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {jwt.encode({'username': username}, 'demo-secret', algorithm='HS256')}"}  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring


def make_billing_tool(user: str, authz: bool):
    auth = None
    if authz:
        auth = A2AClientAuthZ(
            AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080"))),
            context={"username": user},
            agent_name="billing-agent",
        )

    @tool
    async def ask_billing_agent(question: str) -> str:
        """Ask Acme's remote billing agent a question and return its answer."""
        async with httpx.AsyncClient(auth=auth, headers=bearer(user), timeout=60) as http:
            card = await A2ACardResolver(http, base_url=AGENT_URL).get_agent_card()
            client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)
            try:
                replies = []
                async for event in client.send_message(create_text_message_object(content=question)):
                    if not isinstance(event, tuple):  # a Message event (vs a (Task, update) tuple)
                        replies.append(get_message_text(event))
                return " ".join(t for t in replies if t) or "(no reply)"
            except Exception as exc:  # e.g. A2AClientAuthZ denial before send
                return f"ERROR: {exc}"

    return ask_billing_agent


async def run_as(user: str, *, authz: bool) -> None:
    agent = create_agent(
        model=ChatLiteLLM(model=os.getenv("LLM_MODEL")),
        tools=[make_billing_tool(user, authz)],
        system_prompt="Use the ask_billing_agent tool to answer billing questions; relay its answer verbatim.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is Acme's billing address?"}]}
    )
    final = result["messages"][-1].content
    reached = "REACHED billing agent" if CANARY in final else "did NOT reach billing agent"
    print(f"[{user}] authz={'on' if authz else 'OFF'} → {reached}\n    {final.strip()[:300]}")


async def main(users: list[str], *, authz: bool) -> None:
    for user in users:
        await run_as(user, authz=authz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", choices=["alice", "bob", "both"], default="both")
    parser.add_argument("--no-authz", dest="authz", action="store_false",
                        help="drop client-side A2AClientAuthZ")
    args = parser.parse_args()
    users = ["alice", "bob"] if args.user == "both" else [args.user]
    asyncio.run(main(users, authz=args.authz))
