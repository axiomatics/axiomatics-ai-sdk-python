"""
A2A SERVER — an AWS Strands agent, optionally protected by Axiomatics authorization.

The whole server a developer writes: build a Strands agent, expose it over A2A with
``A2AServer(...).to_starlette_app()``, add ONE middleware line, and serve.
``A2AAuthZMiddleware`` authorizes every inbound message/send against ADS before the agent
runs. Identity is read from the caller's ``Authorization: Bearer <JWT>`` header.

  --no-authz   run the same server WITHOUT the middleware (to compare behaviour)

Run order:
  1. Start ADS with the A2A domain (examples/policies/domain_a2a.yaml) on PDP_BASE_URL.
  2. python examples/a2a/strands_server.py           # serves on :9102 (add --no-authz to disable)
  3. python examples/a2a/strands_client.py           # calls this server

Config in .env: PDP_BASE_URL, LLM_MODEL, OPENROUTER_API_KEY
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.multiagent.a2a import A2AServer

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.a2a import A2AAuthZMiddleware
from axiomatics_ai import PDPConnection

PORT = 9102


def build_app(*, authz: bool):
    agent = Agent(
        model=LiteLLMModel(model_id=os.getenv("LLM_MODEL")),
        name="billing",
        description="Billing assistant for Acme Corp.",
        # The billing address is the recognisable fact a caller checks for (see the client).
        system_prompt=(
            "You are the billing assistant for Acme Corp. When asked about billing, state that "
            "the company billing address is '1200 Market Street, Suite 400, San Francisco, CA 94103'. "
            "Answer briefly."
        ),
    )
    app = A2AServer(agent=agent, port=PORT, serve_at_root=True).to_starlette_app()

    if authz:
        # ── The one-line integration: identical to the ADK/LangChain servers. ──────
        app.add_middleware(
            A2AAuthZMiddleware,
            config=AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080"))),
            agent_name="billing-agent",
        )
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-authz", dest="authz", action="store_false",
                        help="serve without the authorization middleware")
    args = parser.parse_args()
    print(f"Strands A2A server on :{PORT}  (authz={'on' if args.authz else 'OFF'})")
    uvicorn.run(build_app(authz=args.authz), host="127.0.0.1", port=PORT)
