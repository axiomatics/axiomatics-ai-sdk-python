"""Unit tests for A2ACardRewriter — rewrites the card url(s) to the gateway's base."""
from __future__ import annotations

import asyncio
import json

from axiomatics_ai.gateway.rewrite import A2ACardRewriter


def _drive(app, *, path: str, root_path: str, host: str = "gw.example"):
    scope = {
        "type": "http", "method": "GET", "path": path, "root_path": root_path,
        "scheme": "http", "headers": [(b"host", host.encode())], "query_string": b"",
    }

    async def run():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict] = []

        async def send(m):
            sent.append(m)

        await app(scope, receive, send)
        return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")

    return asyncio.run(run())


def _card_target(url: str = "http://target-agent:9999/"):
    async def app(scope, receive, send):
        await receive()
        body = json.dumps({
            "name": "billing", "url": url,
            "additional_interfaces": [{"url": url, "protocol_binding": "JSONRPC"}],
        }).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    return app


def test_rewrites_card_url_and_interfaces_to_gateway_base():
    body = _drive(A2ACardRewriter(_card_target()),
                  path="/billing/.well-known/agent-card.json", root_path="/billing")
    card = json.loads(body)
    assert card["url"] == "http://gw.example/billing/"                       # top-level url rewritten
    assert card["additional_interfaces"][0]["url"] == "http://gw.example/billing/"  # interface url too


def _card_app(card: dict):
    """An ASGI app that serves ``card`` as its agent-card JSON."""
    async def app(scope, receive, send):
        await receive()
        body = json.dumps(card).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    return app


def test_rewrites_0_3_camelcase_additional_interfaces():
    """a2a JSON is camelCase: ``additionalInterfaces``. The snake_case key never matches a
    real 0.3 card, so this is the shape that actually ships."""
    target = "http://127.0.0.1:9314/"
    card = {"name": "billing", "url": target,
            "additionalInterfaces": [{"url": target, "transport": "JSONRPC"}]}
    body = _drive(A2ACardRewriter(_card_app(card)),
                  path="/billing/.well-known/agent-card.json", root_path="/billing")
    out = json.loads(body)
    assert out["url"] == "http://gw.example/billing/"
    assert out["additionalInterfaces"][0]["url"] == "http://gw.example/billing/"


def test_rewrites_1_x_supported_interfaces():
    """a2a-sdk 1.x carries the agent's real addresses in ``supportedInterfaces``; the
    top-level ``url`` is only a 0.3 compatibility field. Rewriting just the compat url masks
    the bypass — assert the ``supportedInterfaces`` url specifically."""
    agent_addr = "http://127.0.0.1:9314/"
    card = {"name": "billing", "url": agent_addr,
            "supportedInterfaces": [{"url": agent_addr, "transport": "JSONRPC"}]}
    body = _drive(A2ACardRewriter(_card_app(card)),
                  path="/billing/.well-known/agent-card.json", root_path="/billing")
    out = json.loads(body)
    assert out["supportedInterfaces"][0]["url"] == "http://gw.example/billing/"


def test_non_card_path_passes_through_untouched():
    async def inner(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"raw-passthrough"})

    body = _drive(A2ACardRewriter(inner), path="/billing/", root_path="/billing")
    assert body == b"raw-passthrough"


def test_public_url_replaces_the_client_controlled_host():
    app = A2ACardRewriter(_card_target(), public_url="https://gateway.example.com/")
    card = json.loads(_drive(app, path="/billing/.well-known/agent-card.json", root_path="/billing", host="evil.example"))
    assert card["url"] == "https://gateway.example.com/billing/"
    assert card["additional_interfaces"][0]["url"] == "https://gateway.example.com/billing/"


def test_legacy_agent_card_path_is_rewritten_too():
    app = A2ACardRewriter(_card_target())
    card = json.loads(_drive(app, path="/billing/.well-known/agent.json", root_path="/billing"))
    assert card["url"] == "http://gw.example/billing/"
