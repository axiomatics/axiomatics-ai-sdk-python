"""
A2A integration tests against a REAL ADS instance (domain_a2a.yaml).

Marked ``ads_integration`` — the ``ads_server`` fixture (tests/conftest.py) starts the ADS
jar with domain_a2a.yaml and stops it after the class; the whole class is skipped when the
jar is absent. No network ports are opened: the A2A server app is driven in-process via
``httpx.ASGITransport``, which exercises the full real path end-to-end —

  real async httpx  →  A2AClientAuthZ.auth_flow  →  real ASGI body replay
                    →  A2AAuthZMiddleware  →  downstream agent  →  real ADS decisions

Policy (domain_a2a.yaml): alice → billing-agent (any skill); bob → billing-agent only with
skill "refund"; everything else Deny.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from axiomatics_ai.a2a import A2AAuthZMiddleware, A2AClientAuthZ, A2AAuthorizationError
from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider
from axiomatics_ai import PDPConnection, standard_tool_enforcement

ADS_URL = "http://localhost:8080"


# ── A minimal real A2A-style agent (JSON-RPC echo) ──────────────────────────────

async def _rpc_endpoint(request):
    payload = await request.json()
    return JSONResponse({"jsonrpc": "2.0", "id": payload.get("id"), "result": {"echo": "ok"}})


def _downstream_agent() -> Starlette:
    return Starlette(routes=[Route("/", _rpc_endpoint, methods=["POST"])])


class HeaderProvider(DefaultServerContextProvider):
    """Read the subject from an ``x-user`` header (keeps the test free of JWT plumbing)."""

    def get_subject(self, headers):
        user = headers.get("x-user")
        return {"username": user} if user else {}


def _message_send(text: str = "hi", *, skill: str | None = None) -> dict:
    message = {"role": "user", "parts": [{"kind": "text", "text": text}]}
    if skill is not None:
        message["metadata"] = {"skillId": skill}
    return {"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {"message": message}}


@pytest.mark.ads_integration
class TestA2AAgainstRealAds:
    domain_yaml = "domain_a2a.yaml"

    def _server_app(self):
        return A2AAuthZMiddleware(
            _downstream_agent(),
            AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement(),
                        context_provider=HeaderProvider()),
            agent_name="billing-agent",
        )

    # ── Server-side enforcement (A2AAuthZMiddleware) ────────────────────────────

    def test_server_permits_alice_and_denies_bob(self, ads_server):
        app = self._server_app()

        async def go():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent"
            ) as client:
                alice = await client.post("/", json=_message_send(), headers={"x-user": "alice"})
                bob = await client.post("/", json=_message_send(), headers={"x-user": "bob"})
                return alice.json(), bob.json()

        alice, bob = asyncio.run(go())
        assert alice["result"] == {"echo": "ok"}          # alice permitted → reached agent
        assert bob["error"]["code"] == -32001             # bob denied → JSON-RPC error
        assert "Access denied" in bob["error"]["message"]

    def test_server_permits_bob_with_refund_skill(self, ads_server):
        app = self._server_app()

        async def go():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent"
            ) as client:
                r = await client.post(
                    "/", json=_message_send(skill="refund"), headers={"x-user": "bob"}
                )
                return r.json()

        assert asyncio.run(go())["result"] == {"echo": "ok"}   # skill hint flips the decision

    # ── Client-side enforcement (A2AClientAuthZ), stacked on the same server ─────

    def test_client_blocks_bob_before_send(self, ads_server):
        app = self._server_app()
        authz = A2AClientAuthZ(
            AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()), context={"username": "bob"}, agent_name="billing-agent"
        )

        async def go():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent", auth=authz
            ) as client:
                await client.post("/", json=_message_send(), headers={"x-user": "bob"})

        with pytest.raises(A2AAuthorizationError):
            asyncio.run(go())

    def test_client_permits_bob_with_refund_and_reaches_agent(self, ads_server):
        app = self._server_app()
        authz = A2AClientAuthZ(
            AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()),
            context={"username": "bob"},
            agent_name="billing-agent",
            include_message_text=False,
        )

        async def go():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent", auth=authz
            ) as client:
                # skill=refund → client permits, server (also bob) permits → agent echoes
                r = await client.post(
                    "/", json=_message_send(skill="refund"), headers={"x-user": "bob"}
                )
                return r.json()

        assert asyncio.run(go())["result"] == {"echo": "ok"}
