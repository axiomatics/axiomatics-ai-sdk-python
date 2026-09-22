"""Server-side A2A authorization — ``A2AAuthZMiddleware`` (the recommended enforcement point).

A **pure ASGI** middleware that authorizes every **inbound** A2A JSON-RPC request before
it reaches the agent's request handler. This is the authoritative place to enforce: it
sits at the resource, protects the agent regardless of which client calls, and cannot be
bypassed by a misconfigured or hostile client (unlike the client-side ``A2AClientAuthZ``,
which is complementary egress control). Being pure ASGI (not a Starlette
``BaseHTTPMiddleware``) it mounts on any ASGI app — Starlette, FastAPI, or the raw app the
a2a-sdk builds — so one implementation covers every framework:

    # Google ADK:
    app = to_a2a(my_agent, port=8000)
    app.add_middleware(A2AAuthZMiddleware, config=AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080")))

    # AWS Strands:
    app = A2AServer(agent=my_agent).to_starlette_app()
    app.add_middleware(A2AAuthZMiddleware, config=AuthZConfig(...))

    # self-wired a2a-sdk:
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    app.add_middleware(A2AAuthZMiddleware, config=AuthZConfig(...))

Because ASGI middleware composes, adding it to a single parent app that mounts many agents
enforces on all of them at once (``agent_name`` falls back to the request path, so PDP
still distinguishes each mounted agent).

Identity comes from the request headers per invocation (``Authorization: Bearer <JWT>`` by
default, via ``DefaultServerContextProvider``) — no per-call wiring needed. On a Deny the
request never reaches the handler; a JSON-RPC error is returned instead.

Every POST must be a single JSON-RPC request, and every method is authorized — not only
``message/send`` / ``SendMessage``, but ``tasks/get``, ``tasks/cancel``, push-notification
config and any method added later. The method is a resource attribute, so a policy tells
them apart. Any other POST (a batch array, a body with no ``method``, invalid JSON, the
HTTP+JSON REST binding) is rejected with a JSON-RPC Invalid Request error. Of non-POST
traffic, only GET / HEAD of the public agent card (and any ``public_paths`` you add, e.g. a
health check) passes through; everything else (REST task reads, SSE subscriptions, the
authenticated extended card) gets HTTP 403.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from collections.abc import Awaitable, Callable

from axiomatics_ai.a2a import _wire
from axiomatics_ai.a2a._enforce import authorize_invocation
from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider

logger = logging.getLogger(__name__)

# JSON-RPC error returned on a Deny. Sent with HTTP 200 (JSON-RPC convention) so a2a-sdk
# clients parse it as a structured error rather than a transport failure. -32001 is in the
# implementation-defined server-error range.
_DENY_ERROR_CODE = -32001
# JSON-RPC Invalid Request: a POST that is not a single JSON-RPC request.
_INVALID_REQUEST_CODE = -32600

# Paths served to anyone with GET / HEAD: the public agent card (current, and the legacy name
# pre-0.3 a2a agents serve). Matched as a suffix, so a mounted agent's card is public too.
PUBLIC_CARD_PATHS: tuple[str, ...] = ("/.well-known/agent-card.json", "/.well-known/agent.json")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class A2AAuthZMiddleware:
    """Pure-ASGI middleware authorizing every inbound A2A JSON-RPC request.

    Args:
        app (Any): The wrapped ASGI app (supplied by ``add_middleware``).
        config (AuthZConfig, optional): The PDP connection, the A2A_INVOCATION action name,
            fail_open and cache settings, and the context provider
            (``DefaultServerContextProvider`` by default: token claims, unverified).
        agent_name (str, optional): The served agent's identity for the PDP; the request path
            by default.
        include_message_text (bool): Add the inbound message text to the resource, so a policy
            can match on content. Off by default (privacy, cache-key churn).
        surface (str, optional): The label of the decision events (the gateway passes
            ``"gateway_a2a_server"``).
        public_paths (tuple[str, ...]): Extra paths served to anyone with GET / HEAD, e.g. a
            health check (``("/health",)``), on top of the public agent-card paths. Matched
            as a suffix of the request path. Every other non-POST request is refused.
    """

    # Default only — the constructor's `surface` argument overrides it into self._surface
    # (the gateway passes "gateway_a2a_server"). Named _DEFAULT_SURFACE rather than _SURFACE
    # so it cannot be confused with that instance attribute.
    _DEFAULT_SURFACE = "a2a_server"

    def __init__(
        self,
        app: Any,
        config: AuthZConfig | None = None,
        *,
        agent_name: str | None = None,
        include_message_text: bool = False,
        surface: str | None = None,
        public_paths: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self._public_paths = PUBLIC_CARD_PATHS + tuple(public_paths)
        self.config = config or AuthZConfig()
        self._provider = self.config.context_provider or DefaultServerContextProvider()
        self._agent_name = agent_name
        self._include_message_text = include_message_text
        self._ads = self.config.build_pdp_client()
        self._cache = self.config.build_cache()
        # Tags emitted decision events; the gateway overrides it to "gateway_a2a_server".
        self._surface = surface or self._DEFAULT_SURFACE

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authorize the request, then pass it on, or answer with an error.

        A GET or HEAD of a public path (the agent card) passes through. Any other non-POST
        request gets HTTP 403: there is no JSON-RPC method to authorize, and passing it would
        expose the REST binding's task reads, the authenticated extended card or anything else
        the app serves. A POST that is not a single JSON-RPC request gets error ``-32600``; a
        denied one gets ``-32001``; none of them reaches the app.

        Args:
            scope (dict[str, Any]): The ASGI scope.
            receive (Callable): The ASGI receive channel.
            send (Callable): The ASGI send channel.
        """
        if scope.get("type") != "http":
            await self.app(scope, receive, send)   # lifespan (and websocket) scopes
            return
        if scope.get("method") != "POST":
            if scope.get("method") in ("GET", "HEAD") and scope.get("path", "").endswith(self._public_paths):
                await self.app(scope, receive, send)
                return
            logger.debug("A2A inbound %s %s: not a public path, rejected", scope.get("method"), scope.get("path"))
            await _send_forbidden(send)
            return

        body = await _read_body(receive)
        rpc = _wire.parse_jsonrpc(body)
        if rpc is None or not isinstance(rpc.get("method"), str):
            # Not a single JSON-RPC request, so there is no method to authorize. Passing it
            # through would let batches, the REST binding or a malformed body skip the PDP.
            logger.debug("A2A inbound POST %s: not a single JSON-RPC request, rejected", scope.get("path"))
            await _send_jsonrpc_error(
                send, None, _INVALID_REQUEST_CODE,
                "Invalid Request: expected a single JSON-RPC request.",
            )
            return

        headers = _headers_dict(scope)
        params = rpc.get("params") if isinstance(rpc.get("params"), dict) else {}
        agent_name = self._agent_name or scope.get("path") or "a2a-agent"
        message_text = _wire.extract_message_text(params) if self._include_message_text else None

        # The PDP call (and any token verification) blocks: run it on a worker thread so a slow
        # PDP doesn't stall every other request on the event loop.
        permitted = await asyncio.to_thread(
            authorize_invocation,
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            provider=self._provider,
            context=headers,
            agent_name=agent_name,
            method=rpc["method"],
            skill_hint=_wire.extract_skill_hint(params),
            message_text=message_text,
            surface=self._surface,
        )
        logger.debug(
            "A2A inbound %s → %s: %s",
            rpc["method"], agent_name, "Permit" if permitted else "Deny",
        )
        if not permitted:
            await _send_jsonrpc_error(
                send, rpc.get("id"), _DENY_ERROR_CODE,
                f"Access denied: not authorized to call '{rpc['method']}' on '{agent_name}'.",
            )
            return

        await self.app(scope, _replay_receive(body, receive), send)


# ── ASGI helpers ────────────────────────────────────────────────────────────────


async def _read_body(receive: Receive) -> bytes:
    """Buffer the full request body from the ASGI receive channel."""
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        more_body = message.get("more_body", False)
    return b"".join(chunks)


def _replay_receive(body: bytes, receive: Receive) -> Receive:
    """Build a receive callable that replays the buffered body, then defers to the real channel.

    After the body, the downstream app's next receive() waits for what the client really does:
    a streaming (SSE) response listens for a disconnect while it streams, so answering
    ``http.disconnect`` at once would end every stream.
    """
    delivered = False

    async def replay() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def _headers_dict(scope: Scope) -> dict[str, str]:
    """Lowercased header dict from the ASGI scope (last value wins on duplicates).

    A request with more than one ``Authorization`` header gets none (no identity, so denied):
    an authentication layer in front may have checked a different one than the dict keeps.
    """
    pairs = [(k.decode("latin-1").lower(), v.decode("latin-1")) for k, v in scope.get("headers", [])]
    headers = dict(pairs)
    if sum(k == "authorization" for k, _ in pairs) > 1:
        logger.warning("A2A request with more than one Authorization header: treated as anonymous")
        headers.pop("authorization")
    return headers


async def _send_forbidden(send: Send) -> None:
    """Refuse a non-POST request outside the public paths with HTTP 403."""
    data = json.dumps({
        "error": "Forbidden: only JSON-RPC POST requests and the public agent card are served.",
    }).encode()
    await send({
        "type": "http.response.start",
        "status": 403,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())],
    })
    await send({"type": "http.response.body", "body": data})


async def _send_jsonrpc_error(send: Send, request_id: Any, code: int, message: str) -> None:
    """Emit a JSON-RPC error response (HTTP 200) for a denied or rejected request."""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
    data = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(data)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": data})
