"""
Gateway integration tests against REAL ADS, gateway served under uvicorn on a real port
(so FastMCP route lifespans run) and clients hitting it over real HTTP.

Two classes, because each needs a different ADS policy domain:
  - TestGatewayA2ACrossProcess → domain_a2a.yaml           (A2A route, cross-process forward)
  - TestGatewayMcpVisibility    → domain_mcp_tool_visibility.yaml (MCP route, tool visibility)

The A2A target is a REAL a2a-sdk echo server; the gateway forwards to it over real HTTP.

NOTE: a2a-sdk / fastmcp are imported *inside* the helpers/tests, not at module top. Those
imports pull in a gRPC/protobuf/opentelemetry stack that, if imported at collection time,
conflicts with chromadb's OTLP exporter and breaks the rag tests. Lazy-importing keeps
collection (which happens even for deselected tests) clean.

Marked ads_integration — skipped when the ADS jar is absent (ads_server fixture).
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import httpx
import jwt
import pytest
import uvicorn

from axiomatics_ai import AuthZConfig
from axiomatics_ai.gateway import GatewayConfig, Route, build_gateway

from .conftest import message_send
from axiomatics_ai import PDPConnection, standard_tool_enforcement

ADS_URL = "http://localhost:8080"
A2A_CANARY = "gateway-reached-echo"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bearer(username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {jwt.encode({'username': username}, 'demo-secret', algorithm='HS256')}"}


def _serve(app, port: int, ready_path: str) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    server.install_signal_handlers = lambda: None  # can't install signal handlers off the main thread
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        try:
            # Any HTTP response means it's listening (the MCP endpoint answers a bare GET with
            # 406 — that still means "up"). httpx, unlike urllib, doesn't raise on 4xx/5xx.
            httpx.get(f"http://127.0.0.1:{port}{ready_path}", timeout=2)
            return server
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"server on :{port} did not start")


def _start_a2a_echo_target(port: int) -> uvicorn.Server:
    """A real a2a-sdk server with an echo agent. Heavy imports kept local (see module note)."""
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill
    from a2a.utils import new_agent_text_message

    class _EchoExecutor(AgentExecutor):
        async def execute(self, context, event_queue) -> None:
            await event_queue.enqueue_event(new_agent_text_message(A2A_CANARY))

        async def cancel(self, context, event_queue) -> None:
            raise NotImplementedError

    card = AgentCard(
        name="billing", description="echo", url=f"http://127.0.0.1:{port}/", version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        skills=[AgentSkill(id="billing", name="billing", description="x", tags=["billing"])],
    )
    handler = DefaultRequestHandler(agent_executor=_EchoExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    return _serve(app, port, "/.well-known/agent-card.json")


@pytest.mark.ads_integration
class TestGatewayA2ACrossProcess:
    domain_yaml = "domain_a2a.yaml"

    def test_forwards_permitted_blocks_denied(self, ads_server):
        target_port, gw_port = _free_port(), _free_port()
        target = _start_a2a_echo_target(target_port)
        cfg = GatewayConfig(authz=AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()), allow_unverified_tokens=True, routes=[
            Route("/billing", f"http://127.0.0.1:{target_port}", protocol="a2a", agent_name="billing-agent"),
        ])
        gateway = _serve(build_gateway(cfg), gw_port, "/billing/.well-known/agent-card.json")
        try:
            base = f"http://127.0.0.1:{gw_port}"
            # the served card's url is rewritten to the gateway (not the target port) so
            # clients route back through the gateway.
            card = httpx.get(f"{base}/billing/.well-known/agent-card.json", timeout=30).json()
            assert card["url"] == f"{base}/billing/"
            assert str(target_port) not in card["url"]

            alice = httpx.post(f"{base}/billing/", content=message_send(), headers=_bearer("alice"), timeout=30).json()
            bob = httpx.post(f"{base}/billing/", content=message_send(), headers=_bearer("bob"), timeout=30).json()
            assert A2A_CANARY in str(alice)             # forwarded cross-process to the echo server
            assert bob["error"]["code"] == -32001       # blocked at the gateway, never forwarded
        finally:
            target.should_exit = True
            gateway.should_exit = True


@pytest.mark.ads_integration
class TestGatewayMcpVisibility:
    domain_yaml = "domain_mcp_tool_visibility.yaml"

    def test_tool_list_filtered_per_subject(self, ads_server):
        from fastmcp import Client, FastMCP
        from fastmcp.client.transports import StreamableHttpTransport

        backend = FastMCP("backend")
        for tool_name in ("search_docs", "get_account_balance", "send_email"):
            @backend.tool(name=tool_name)
            def _tool(q: str = "") -> str:
                return "ok"

        gw_port = _free_port()
        cfg = GatewayConfig(authz=AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()), allow_unverified_tokens=True, routes=[
            Route("/tools", backend, protocol="mcp"),  # in-memory backend (create_proxy accepts it)
        ])
        gateway = _serve(build_gateway(cfg), gw_port, "/tools/")
        try:
            async def list_as(user: str):
                transport = StreamableHttpTransport(f"http://127.0.0.1:{gw_port}/tools/", headers=_bearer(user))
                async with Client(transport) as client:
                    return sorted(t.name for t in await client.list_tools())

            assert asyncio.run(list_as("alice")) == ["get_account_balance", "search_docs", "send_email"]
            assert asyncio.run(list_as("bob")) == ["search_docs"]
        finally:
            gateway.should_exit = True
