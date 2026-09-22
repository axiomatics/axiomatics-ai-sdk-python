"""
Unit tests for the gateway: build_gateway wiring, and the A2A route composing
A2AAuthZMiddleware + HttpForwarder (enforce, then forward). Mocked ADS; in-process target
via httpx.ASGITransport. MCP-route request/response is covered by the integration tier.
"""
from __future__ import annotations

import json

import httpx
import pytest

from axiomatics_ai import AuthZConfig
from axiomatics_ai.a2a import A2AAuthZMiddleware
from axiomatics_ai.gateway import GatewayConfig, Route, build_gateway
from axiomatics_ai.gateway.forwarder import HttpForwarder

from .conftest import FakePdp, bearer, drive_asgi, message_send
from axiomatics_ai import PDPConnection, standard_tool_enforcement

CANARY = "REACHED-TARGET"


def _fake_a2a_target():
    """A tiny ASGI 'A2A agent' that records calls and echoes a canary result."""
    calls: list[str] = []

    async def app(scope, receive, send):
        calls.append(scope["path"])
        await receive()
        body = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"note": CANARY}}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    return app, calls


def _a2a_route_app(permit: bool):
    """Compose the A2A route the way build_gateway does, but with an in-process forwarder."""
    target, calls = _fake_a2a_target()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=target))
    mw = A2AAuthZMiddleware(HttpForwarder("http://target", client=client),
                            AuthZConfig(), agent_name="billing-agent")
    mw._ads = FakePdp(permit=permit)
    return mw, calls


# ── A2A route: enforce, then forward ────────────────────────────────────────────

def test_a2a_route_permit_forwards_to_target():
    app, calls = _a2a_route_app(permit=True)
    status, body = drive_asgi(app, body=message_send(), headers=[bearer("alice")])
    assert status == 200
    assert CANARY in body.decode()      # reached the target and relayed its reply
    assert calls == ["/"]               # target actually invoked


def test_a2a_route_deny_blocks_before_target():
    app, calls = _a2a_route_app(permit=False)
    status, body = drive_asgi(app, body=message_send(), headers=[bearer("bob")])
    assert status == 200
    error = json.loads(body)["error"]
    assert error["code"] == -32001      # JSON-RPC deny from the middleware
    assert calls == []                  # target never reached


# ── build_gateway wiring ──────────────────────────────────────────────────────────

def _cfg(*routes: Route) -> GatewayConfig:
    return GatewayConfig(
        routes=list(routes),
        authz=AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement()),
        allow_unverified_tokens=True,   # these wiring tests don't send tokens
    )


def test_build_gateway_mounts_every_route_including_many_of_each():
    cfg = _cfg(
        Route("/billing", "http://billing:9000", protocol="a2a", agent_name="billing-agent"),
        Route("/support", "http://support:9000", protocol="a2a", agent_name="support-agent"),
        Route("/docs", "http://docs:8000/mcp", protocol="mcp"),
        Route("/crm", "http://crm:8000/mcp", protocol="mcp"),
    )
    app = build_gateway(cfg)
    paths = sorted(r.path for r in app.routes)
    assert paths == ["/billing", "/crm", "/docs", "/support"]   # many a2a + many mcp coexist


def test_build_gateway_requires_routes():
    with pytest.raises(ValueError, match="at least one route"):
        build_gateway(_cfg())


# ── Token passthrough and per-route audience ─────────────────────────────────────

def test_mcp_upstream_http_target_never_forwards_incoming_headers():
    from fastmcp.client.transports.http import StreamableHttpTransport

    from axiomatics_ai.gateway.app import _mcp_upstream

    transport = _mcp_upstream(Route("/docs", "http://docs:8000/mcp", protocol="mcp"))
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == "http://docs:8000/mcp"
    assert transport.forward_incoming_headers is False
    transport.forward_incoming_headers = True  # what the proxy client does — pinned
    assert transport.forward_incoming_headers is False


def test_mcp_upstream_opt_in_and_non_url_targets_unchanged():
    from fastmcp import FastMCP

    from axiomatics_ai.gateway.app import _mcp_upstream

    route = Route("/docs", "https://docs/mcp", protocol="mcp", forward_authorization=True)
    assert _mcp_upstream(route) == "https://docs/mcp"
    backend = FastMCP("backend")
    assert _mcp_upstream(Route("/docs", backend, protocol="mcp")) is backend
    assert _mcp_upstream(Route("/docs", "server.py", protocol="mcp")) == "server.py"


def _identity_config():
    from axiomatics_ai import AgentIdentityContextProvider, JwtVerification

    provider = AgentIdentityContextProvider(JwtVerification(issuer="https://idp", audience="https://gateway"))
    return AuthZConfig(connection=PDPConnection(base_url="http://pdp"), context_provider=provider)


def test_route_authz_without_audience_is_the_shared_config():
    from axiomatics_ai.gateway.app import _route_authz

    config = _identity_config()
    assert _route_authz(Route("/docs", "http://docs/mcp", protocol="mcp"), config) is config


def test_route_authz_with_audience_verifies_for_that_audience_and_shares_caches():
    from axiomatics_ai.gateway.app import _route_authz

    config = _identity_config()
    route_config = _route_authz(Route("/docs", "http://docs/mcp", protocol="mcp", audience="https://docs"), config)
    assert route_config is not config
    assert route_config.context_provider.verification.audiences == ["https://docs"]
    assert config.context_provider.verification.audiences == ["https://gateway"]  # parent untouched
    assert route_config.connection is config.connection
    assert route_config._state is config._state

    route_cache = route_config.build_cache()
    route_cache.set({"user_id": "alice"}, {"name": "x"}, {}, {}, True)
    assert config.invalidate_decisions(user_id="alice") == 1   # parent reaches the route's cache


def test_route_audience_needs_a_verifying_provider():
    from axiomatics_ai.gateway.app import _route_authz

    route = Route("/docs", "http://docs/mcp", protocol="mcp", audience="https://docs")
    with pytest.raises(ValueError, match="audience"):
        _route_authz(route, AuthZConfig())
    with pytest.raises(ValueError, match="audience"):
        build_gateway(_cfg(route))


def test_build_gateway_with_route_audiences():
    config = GatewayConfig(
        routes=[
            Route("/docs", "http://docs:8000/mcp", protocol="mcp", audience="https://docs"),
            Route("/billing", "http://billing:9000", protocol="a2a", agent_name="billing-agent", audience="https://billing"),
        ],
        authz=_identity_config(),
    )
    assert sorted(r.path for r in build_gateway(config).routes) == ["/billing", "/docs"]


# ── tokens must be verified ──────────────────────────────────────────────────────

def test_build_gateway_refuses_unverified_tokens_by_default():
    # The gateway is usually the edge: decoding a bearer token without checking its
    # signature would trust a forged one.
    config = GatewayConfig(routes=[Route("/billing", "http://b", protocol="a2a", agent_name="b")])
    with pytest.raises(ValueError, match="forged token"):
        build_gateway(config)
    from axiomatics_ai.core.config import DefaultServerContextProvider

    config.authz = AuthZConfig(context_provider=DefaultServerContextProvider())   # set explicitly: same
    with pytest.raises(ValueError, match="verifying provider"):
        build_gateway(config)


def test_build_gateway_accepts_a_verifying_provider_or_an_explicit_opt_out():
    route = Route("/billing", "http://b", protocol="a2a", agent_name="b")
    assert build_gateway(GatewayConfig(routes=[route], authz=_identity_config()))
    assert build_gateway(GatewayConfig(routes=[route], allow_unverified_tokens=True))


def test_gateway_lifespan_closes_the_forwarders():
    import asyncio

    from axiomatics_ai.gateway.forwarder import HttpForwarder

    closed = []

    async def aclose(self):
        closed.append(self.target_url)

    app = build_gateway(GatewayConfig(
        routes=[Route("/billing", "http://b", protocol="a2a", agent_name="b")], allow_unverified_tokens=True,
    ))

    async def run():
        HttpForwarder.aclose, original = aclose, HttpForwarder.aclose
        try:
            async with app.router.lifespan_context(app):
                pass
        finally:
            HttpForwarder.aclose = original

    asyncio.run(run())
    assert closed == ["http://b"]
