"""``A2ACardRewriter``: rewrite a proxied A2A agent card's URLs to the gateway's address.

Why this exists: an A2A client fetches the agent card, then sends ``message/send`` to
whatever ``url`` the card advertises. If the target's card advertised its *own* direct URL,
a client would fetch the card through the gateway but then talk to the target directly,
bypassing enforcement. So on the A2A route the gateway rewrites the card's ``url`` **and
every per-interface URL** to point back at the gateway, so the client routes every
subsequent call through it. The public base is derived from the request itself
(``GatewayConfig.public_url``, or else the ``Host`` header, + mount prefix).

Interface URLs live under different keys across a2a-sdk lines, and the JSON is camelCase:
a2a-sdk 0.3.x uses ``additionalInterfaces``; a2a-sdk 1.x uses ``supportedInterfaces`` (its
protobuf has no top-level ``url`` — only a JSON compatibility ``url``). We rewrite all of
them: a 1.x-aware client reads ``supportedInterfaces``, so missing it would hand out the
agent's real address and bypass the gateway silently. (Snake_case variants are covered too,
defensively.)

REQUIRED CLIENT SETUP (document this for users): clients must be created pointing at the
**gateway's** per-route URL for every fronted server — e.g. ``http://gateway/billing`` —
NOT the target agent's own address. The client resolves the card from the gateway; this
rewrite then ensures all of its subsequent ``message/send`` calls go back through the
gateway. If a client is pointed directly at a target, the gateway is bypassed entirely.

Pure ASGI; only touches responses to the agent-card path, everything else passes through.
"""
from __future__ import annotations

import json
from typing import Any
from collections.abc import Awaitable, Callable

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

# The current agent-card path, and the legacy one (pre-0.3 a2a) some agents still serve.
_CARD_PATHS = ("/.well-known/agent-card.json", "/.well-known/agent.json")

# Keys under which a card may carry a list of alternate transport endpoints (each a dict with
# its own ``url``). The list is named differently across a2a-sdk lines, and a2a JSON is
# camelCase; the snake_case variants are covered defensively in case a producer emits them.
_INTERFACE_KEYS = (
    "supportedInterfaces", "supported_interfaces",    # a2a-sdk 1.x
    "additionalInterfaces", "additional_interfaces",  # a2a-sdk 0.3.x
)


class A2ACardRewriter:
    """Wrap an ASGI app and rewrite the A2A agent-card URLs in its responses.

    Only responses to the agent-card paths (``/.well-known/agent-card.json`` and the legacy
    ``/.well-known/agent.json``) are touched.

    Args:
        app (Any): The wrapped ASGI app.
        public_url (str, optional): The gateway's address as clients reach it; without it the
            address is built from the request's ``Host`` header and scheme, which a client
            controls (and which reads ``http`` behind a TLS-terminating proxy).
    """

    def __init__(self, app: Any, *, public_url: str | None = None) -> None:
        self.app = app
        self._public_url = public_url.rstrip("/") if public_url else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request on; rewrite the card's URLs if it is an agent-card request.

        Args:
            scope (dict[str, Any]): The ASGI scope.
            receive (Callable): The ASGI receive channel.
            send (Callable): The ASGI send channel.
        """
        if scope.get("type") != "http" or not scope.get("path", "").endswith(_CARD_PATHS):
            await self.app(scope, receive, send)
            return

        start: dict[str, Any] = {}
        chunks: list[bytes] = []

        async def _capture(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                start.update(message)
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = _rewrite_card(b"".join(chunks), _public_base(scope, self._public_url))
                headers = [
                    (k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"
                ]
                headers.append((b"content-length", str(len(body)).encode()))
                await send({"type": "http.response.start", "status": start["status"], "headers": headers})
                await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, _capture)


def _public_base(scope: Scope, public_url: str | None = None) -> str:
    root = scope.get("root_path", "")  # the mount prefix, e.g. /billing
    if public_url:
        return f"{public_url}{root}/"
    headers = {k.lower(): v for k, v in scope.get("headers", [])}
    host = headers.get(b"host", b"localhost").decode("latin-1")
    scheme = scope.get("scheme", "http")
    return f"{scheme}://{host}{root}/"


def _rewrite_card(body: bytes, public_base: str) -> bytes:
    try:
        card = json.loads(body)
    except (ValueError, TypeError):
        return body  # not JSON — leave untouched
    if not isinstance(card, dict):
        return body
    if "url" in card:
        card["url"] = public_base
    for key in _INTERFACE_KEYS:
        for iface in card.get(key) or []:
            if isinstance(iface, dict) and "url" in iface:
                iface["url"] = public_base
    return json.dumps(card).encode()
