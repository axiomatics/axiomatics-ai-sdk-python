"""
A2A authorization on a self-wired a2a-sdk server + client — no LLM required.

Demonstrates both enforcement directions against real ADS (domain_a2a.yaml):

  • Server: A2AAuthZMiddleware wraps the a2a-sdk app. alice may send to billing-agent;
    bob may not (unless skill=refund). A denied send never reaches the agent.
  • Client: A2AClientAuthZ (an httpx.Auth) blocks a disallowed outbound send before it
    leaves the process.

The server app is driven in-process via httpx.ASGITransport so the example runs as a
single script with no open ports. In production you would serve the same app with uvicorn
and point real clients at it — the middleware line is identical.

Run:  python examples/a2a/self_wired_a2a.py   (needs the ADS jar in examples/policies/)
"""
import asyncio
import os

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory, create_text_message_object
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from axiomatics_ai import AuthZConfig
from axiomatics_ai.a2a import A2AAuthorizationError, A2AAuthZMiddleware, A2AClientAuthZ
from axiomatics_ai.core.config import DefaultServerContextProvider
from axiomatics_ai import PDPConnection

AGENT_URL = "http://billing"  # virtual host for the in-process ASGI transport


class EchoExecutor(AgentExecutor):
    """Trivial agent: echoes a fixed reply. No model — keeps the example key-free."""

    async def execute(self, context, event_queue) -> None:
        await event_queue.enqueue_event(new_agent_text_message("echo: request handled"))

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


class HeaderProvider(DefaultServerContextProvider):
    """Read the subject from an ``x-user`` header (avoids JWT setup for the demo)."""

    def get_subject(self, headers):
        user = headers.get("x-user")
        return {"username": user} if user else {}


def build_server_app(ads_url: str):
    card = AgentCard(
        name="billing",
        description="Echo billing agent",
        url=f"{AGENT_URL}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="refund", name="refund", description="Issue refunds", tags=["billing"])],
    )
    handler = DefaultRequestHandler(agent_executor=EchoExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    # THE server-side one-liner — enforce on every inbound message/send.
    app.add_middleware(
        A2AAuthZMiddleware,
        config=AuthZConfig(connection=PDPConnection(base_url=ads_url), context_provider=HeaderProvider()),
        agent_name="billing-agent",
    )
    return app


async def send_as(app, *, user: str, skill: str | None, client_authz: A2AClientAuthZ | None = None):
    """Resolve the agent card and send one message as ``user`` over the in-process transport."""
    transport = httpx.ASGITransport(app=app)
    headers = {"x-user": user}
    async with httpx.AsyncClient(
        transport=transport, base_url=AGENT_URL, headers=headers, auth=client_authz
    ) as http:
        card = await A2ACardResolver(http, base_url=AGENT_URL).get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)
        message = create_text_message_object(content="refund my order")
        if skill is not None:
            message.metadata = {"skillId": skill}
        return [event async for event in client.send_message(message)]


async def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_a2a.yaml loaded (see docs/running-ads.md)
    app = build_server_app(ads_url)

    print("=== SERVER-side enforcement ===")
    print("alice → billing (permitted):")
    print("  ", await send_as(app, user="alice", skill=None))

    print("bob → billing without skill (server denies):")
    try:
        print("  ", await send_as(app, user="bob", skill=None))
    except Exception as exc:  # a2a client surfaces the JSON-RPC deny error
        print("   denied:", type(exc).__name__, exc)

    print("bob → billing with skill=refund (permitted):")
    print("  ", await send_as(app, user="bob", skill="refund"))

    print("\n=== CLIENT-side enforcement (A2AClientAuthZ) ===")
    bob_authz = A2AClientAuthZ(
        AuthZConfig(connection=PDPConnection(base_url=ads_url)), context={"username": "bob"}, agent_name="billing-agent"
    )
    print("bob's own client blocks the disallowed send before it leaves:")
    try:
        await send_as(app, user="bob", skill=None, client_authz=bob_authz)
    except A2AAuthorizationError as exc:
        print("   blocked outbound:", exc)


if __name__ == "__main__":
    asyncio.run(main())
