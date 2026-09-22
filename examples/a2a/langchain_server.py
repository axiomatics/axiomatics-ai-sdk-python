"""
A2A SERVER — a LangChain ``create_agent`` exposed over A2A, optionally protected.

LangChain has no native A2A server, so you self-wire a2a-sdk: a tiny ``AgentExecutor`` runs
the ``create_agent`` graph and returns its reply. Everything else is identical to the ADK
and Strands servers — including the ONE middleware line. ``A2AAuthZMiddleware`` authorizes
every inbound message/send against ADS before the agent runs; identity is the inbound JWT.

  --no-authz   run the same server WITHOUT the middleware (to compare behaviour)

Run order:
  1. Start ADS with the A2A domain (examples/policies/domain_a2a.yaml) on PDP_BASE_URL.
  2. python examples/a2a/langchain_server.py         # serves on :9103 (add --no-authz to disable)
  3. python examples/a2a/langchain_client.py         # calls this server

Config in .env: PDP_BASE_URL, LLM_MODEL, OPENROUTER_API_KEY
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.a2a import A2AAuthZMiddleware
from axiomatics_ai import PDPConnection

PORT = 9103

# A normal LangChain agent — nothing A2A-specific here.
billing_agent = create_agent(
    model=ChatLiteLLM(model=os.getenv("LLM_MODEL")),
    tools=[],
    system_prompt=(
        "You are the billing assistant for Acme Corp. When asked about billing, state that "
        "the company billing address is '1200 Market Street, Suite 400, San Francisco, CA 94103'. "
        "Answer briefly."
    ),
)


class LangChainAgentExecutor(AgentExecutor):
    """Bridge a2a-sdk ↔ LangChain: run the create_agent graph, return its final message."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        result = await billing_agent.ainvoke(
            {"messages": [{"role": "user", "content": context.get_user_input()}]}
        )
        await event_queue.enqueue_event(new_agent_text_message(result["messages"][-1].content))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def build_app(*, authz: bool):
    card = AgentCard(
        name="billing",
        description="Billing assistant for Acme Corp.",
        url=f"http://127.0.0.1:{PORT}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="billing", name="billing", description="Billing questions", tags=["billing"])],
    )
    handler = DefaultRequestHandler(agent_executor=LangChainAgentExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()

    if authz:
        # ── The one-line integration: identical to the ADK/Strands servers. ────────
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
    print(f"LangChain A2A server on :{PORT}  (authz={'on' if args.authz else 'OFF'})")
    uvicorn.run(build_app(authz=args.authz), host="127.0.0.1", port=PORT)
