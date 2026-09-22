"""
A2A CLIENT — a Google ADK agent that calls a remote agent, with egress authorization.

An orchestrator ADK agent delegates to a remote ``billing`` agent over A2A. The remote is a
``RemoteA2aAgent`` whose httpx client carries ``A2AClientAuthZ`` — so every outbound
message/send is authorized against ADS *before it leaves this process*. Identity is the
local caller (alice / bob); it is also sent as a Bearer JWT so a protected server can
enforce too.

  --user alice|bob|both   whose identity to run as (default: both)
  --no-authz              drop A2AClientAuthZ (outbound calls are no longer gated here)

Under domain_a2a.yaml: alice may call billing-agent, bob may not. The billing agent knows
one concrete fact — Acme's billing address, "1200 Market Street". If that address appears
in the output, the remote agent was actually reached; if it is missing, the call was
blocked (client-side when authz is on, or by the server if it is protected).

Run after adk_server.py is up (and ADS is running). See adk_server.py for the full order.
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
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.a2a import A2AClientAuthZ
from axiomatics_ai import PDPConnection

AGENT_URL = "http://127.0.0.1:9101"
CARD_URL = f"{AGENT_URL}/.well-known/agent-card.json"
CANARY = "1200 Market Street"


def bearer(username: str) -> dict[str, str]:
    """Identity as a Bearer JWT so a protected server can read the subject from claims."""
    return {"Authorization": f"Bearer {jwt.encode({'username': username}, 'demo-secret', algorithm='HS256')}"}  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring


def build_orchestrator(user: str, *, authz: bool) -> Agent:
    auth = None
    if authz:
        auth = A2AClientAuthZ(
            AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080"))),
            context={"username": user},
            agent_name="billing-agent",  # ADS resource label matching domain_a2a.yaml
        )
    # The client seam: our authz + identity ride on the httpx client ADK uses for A2A.
    httpx_client = httpx.AsyncClient(auth=auth, headers=bearer(user), timeout=60)
    billing = RemoteA2aAgent(name="billing", agent_card=CARD_URL, httpx_client=httpx_client)
    return Agent(
        name="orchestrator",
        model=LiteLlm(os.getenv("LLM_MODEL")),
        instruction="Delegate any billing question to the 'billing' agent, then relay its answer verbatim.",
        sub_agents=[billing],
    )


async def run_as(user: str, *, authz: bool) -> None:
    runner = InMemoryRunner(build_orchestrator(user, authz=authz), app_name="a2a-demo")
    await runner.session_service.create_session(app_name="a2a-demo", user_id=user, session_id=user)
    message = types.Content(role="user", parts=[types.Part(text="What is Acme's billing address?")])
    final = ""
    try:
        async for event in runner.run_async(user_id=user, session_id=user, new_message=message):
            if event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts)
                if text.strip():
                    final = text
        reached = "REACHED billing agent" if CANARY in final else "did NOT reach billing agent"
        print(f"[{user}] authz={'on' if authz else 'OFF'} → {reached}\n    {final.strip()[:300]}")
    except Exception as exc:
        print(f"[{user}] authz={'on' if authz else 'OFF'} → BLOCKED/ERROR: {type(exc).__name__}: {exc}")


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
