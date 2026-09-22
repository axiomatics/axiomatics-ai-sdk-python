"""
A2A CLIENT — an AWS Strands agent that calls a remote agent, with egress authorization.

A Strands agent is given the A2A client tools from ``A2AClientToolProvider``, whose httpx
client carries ``A2AClientAuthZ`` — so every outbound message/send is authorized against
ADS *before it leaves this process*. Identity is the local caller (alice / bob); it is also
sent as a Bearer JWT so a protected server can enforce too.

  --user alice|bob|both   whose identity to run as (default: both)
  --no-authz              drop A2AClientAuthZ (outbound calls are no longer gated here)

Under domain_a2a.yaml: alice may call billing-agent, bob may not. The billing agent knows
Acme's billing address, "1200 Market Street"; if that appears in the output the remote
agent was reached, otherwise the call was blocked.

Run after strands_server.py is up (and ADS is running).
Config in .env: PDP_BASE_URL, LLM_MODEL, OPENROUTER_API_KEY
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools.a2a_client import A2AClientToolProvider

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.a2a import A2AClientAuthZ
from axiomatics_ai import PDPConnection

AGENT_URL = "http://127.0.0.1:9102"
CANARY = "1200 Market Street"


def bearer(username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {jwt.encode({'username': username}, 'demo-secret', algorithm='HS256')}"}  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring


def run_as(user: str, *, authz: bool) -> None:
    httpx_args = {"headers": bearer(user), "timeout": 60}
    if authz:
        # The client seam: our authz rides on the httpx client Strands uses for A2A.
        httpx_args["auth"] = A2AClientAuthZ(
            AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080"))),
            context={"username": user},
            agent_name="billing-agent",
        )
    provider = A2AClientToolProvider(known_agent_urls=[AGENT_URL], httpx_client_args=httpx_args)
    agent = Agent(
        model=LiteLLMModel(model_id=os.getenv("LLM_MODEL")),
        tools=provider.tools,
        system_prompt="Use the billing agent tools to answer, then relay its answer verbatim.",
    )
    try:
        result = agent("Ask the billing agent for Acme's billing address.")
        text = str(result)
        reached = "REACHED billing agent" if CANARY in text else "did NOT reach billing agent"
        print(f"[{user}] authz={'on' if authz else 'OFF'} → {reached}\n    {text.strip()[:300]}")
    except Exception as exc:
        print(f"[{user}] authz={'on' if authz else 'OFF'} → BLOCKED/ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", choices=["alice", "bob", "both"], default="both")
    parser.add_argument("--no-authz", dest="authz", action="store_false",
                        help="drop client-side A2AClientAuthZ")
    args = parser.parse_args()
    users = ["alice", "bob"] if args.user == "both" else [args.user]
    for u in users:
        run_as(u, authz=args.authz)
