"""
A2A client for the gateway demo — an ADK agent that calls the billing agent THROUGH the
gateway. No client-side authorization here: the gateway is the enforcement point.

The client is pointed only at the gateway URL (http://127.0.0.1:9210/billing). It resolves
the agent card from the gateway; the gateway rewrites the card's url to itself, so every
message/send goes back through the gateway and is authorized against ADS. The client never
knows the real agent's address.

(The remote agent is wrapped as an AgentTool the orchestrator calls — see the note in
build_orchestrator on why we use a tool-call here rather than sub_agents/transfer.)

  --user alice|bob|both   whose identity to send as (default: both)

Under domain_gateway.yaml: alice may call billing-agent, bob may not — so alice gets a real
answer and bob's delegation is blocked at the gateway (the orchestrator reports it).

Run after a2a_server.py + gateway.py are up (and ADS with domain_gateway.yaml).
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
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

load_dotenv()

GATEWAY = "http://127.0.0.1:9210"
CARD_URL = f"{GATEWAY}/billing/.well-known/agent-card.json"   # only the gateway URL is known


def bearer(username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {jwt.encode({'username': username}, 'demo-secret', algorithm='HS256')}"}  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring


def build_orchestrator(user: str) -> Agent:
    # Identity travels as a JWT; the gateway reads it and enforces. No A2AClientAuthZ here.
    httpx_client = httpx.AsyncClient(headers=bearer(user), timeout=60)
    billing = RemoteA2aAgent(name="billing", agent_card=CARD_URL, httpx_client=httpx_client)
    # We wrap the remote agent as an AgentTool (call), NOT sub_agents (transfer). Both are
    # valid ADK A2A patterns, but with sub_agents the orchestrator *transfers* control to
    # the remote agent and, if that call fails (e.g. the gateway denies with -32001),
    # control does not return — so the orchestrator produces no final summary. AgentTool
    # instead invokes the remote agent as a tool: the orchestrator gets the result (or the
    # denial) back and can relay it. AgentTool is pure ADK — the tool still speaks A2A.
    return Agent(
        name="orchestrator",
        model=LiteLlm(os.getenv("LLM_MODEL")),
        instruction=(
            "Answer billing questions by calling the 'billing' tool, then relay its answer verbatim. "
            "If the tool is unavailable or the request is denied, tell the user you were "
            "unable to retrieve the billing information."
        ),
        tools=[AgentTool(agent=billing)],
    )


async def run_as(user: str) -> None:
    runner = InMemoryRunner(build_orchestrator(user), app_name="gw-demo")
    await runner.session_service.create_session(app_name="gw-demo", user_id=user, session_id=user)
    message = types.Content(role="user", parts=[types.Part(text="What is Acme's billing address?")])
    final = ""
    try:
        async for event in runner.run_async(user_id=user, session_id=user, new_message=message):
            if event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts)
                if text.strip():
                    final = text
        # We don't hardcode the agent's answer here — just relay it. alice sees the billing
        # address. For bob the gateway deterministically denies the delegated call (-32001,
        # logged above by ADK's A2A client); ADK ends a failed sub-agent transfer without a
        # summary, so `final` is empty — surface that as a blocked call rather than nothing.
        if final.strip():
            print(f"[{user}] → {final.strip()[:400]}")
        else:
            print(f"[{user}] → (no answer — the delegated A2A call was blocked at the gateway)")
    except Exception as exc:
        print(f"[{user}] → BLOCKED/ERROR: {type(exc).__name__}: {exc}")


async def main(users: list[str]) -> None:
    for user in users:
        await run_as(user)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", choices=["alice", "bob", "both"], default="both")
    args = parser.parse_args()
    asyncio.run(main(["alice", "bob"] if args.user == "both" else [args.user]))
