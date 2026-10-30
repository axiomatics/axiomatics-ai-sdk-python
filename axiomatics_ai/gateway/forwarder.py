"""``HttpForwarder``: a minimal pure-ASGI reverse proxy, with no policy logic.

Enforcement does NOT happen here. On an A2A route the gateway builds
``A2AAuthZMiddleware(HttpForwarder(target_url))``: the middleware runs first and authorizes
the inbound request against PDP. On a Permit it calls this forwarder, which relays the
(already-authorized) request verbatim to the route's target server and returns the response.
On a Deny the middleware short-circuits — it sends a JSON-RPC error (`-32001`, "Access
denied …") back to the caller and never invokes the forwarder. (Same middleware we use
in-process — there it wraps the real agent app; here it wraps a forwarder to a remote agent.)

v1 scope: request/response (POST/JSON-RPC). The response is buffered, not streamed —
SSE/streaming passthrough is a later enhancement.
"""
from __future__ import annotations

import http.cookiejar
from typing import Any
from collections.abc import Awaitable, Callable

import httpx

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

# Hop-by-hop headers (RFC 9110 §7.6.1): they describe one connection, never forwarded either
# way. A ``Connection`` header can name more; those are dropped too.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
    "te", "trailer", "transfer-encoding", "upgrade",
}

# Request headers we must not forward, on top of hop-by-hop: length/host, which httpx sets
# itself; and the client's own ``X-Forwarded-*`` / ``Forwarded``, which a client can forge —
# the forwarder sets them from what it saw instead.
#
# ``authorization`` and ``cookie`` (the caller's credentials for the gateway) are added per
# forwarder unless the route sets forward_authorization.
#
# This is a DENYLIST on purpose: everything else passes through, so application headers reach
# the target unchanged. That is load-bearing for a2a-sdk 1.x, which negotiates the protocol
# version via an ``A2A-Version`` request header (absent → the server defaults to 0.3). Do NOT
# convert this to an allowlist without explicitly permitting ``A2A-Version`` — dropping it
# silently downgrades every 1.x client to 0.3.
_DROP_REQUEST_HEADERS = _HOP_BY_HOP | {
    "host", "content-length",
    "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "x-forwarded-port",
}

# Response headers we must not relay, on top of hop-by-hop: ``content-length`` (set for the
# body we send) and ``content-encoding`` — httpx has already decoded a gzip/br body, so
# relaying the header would make the client decode plain bytes again.
_DROP_RESPONSE_HEADERS = _HOP_BY_HOP | {"content-length", "content-encoding"}


class HttpForwarder:
    """Forward every request to a fixed ``target_url``, preserving path, query and body.

    No policy logic: on an A2A route the middleware in front of it has already authorized the
    request. Hop-by-hop headers and the client's own ``X-Forwarded-*`` / ``Forwarded`` are
    dropped (it sets its own), as is ``Content-Encoding`` on responses httpx already decoded.
    The response is buffered, not streamed.

    Args:
        target_url (str): The target server's base URL.
        client (httpx.AsyncClient, optional): A client to use (e.g. a shared pool, or a mock
            transport in tests); one is created on first use otherwise.
        forward_authorization (bool): Forward the caller's ``Authorization`` and ``Cookie``
            headers. Off by default.
    """

    def __init__(
        self, target_url: str, *, client: httpx.AsyncClient | None = None, forward_authorization: bool = False,
    ) -> None:
        self.target_url = target_url.rstrip("/")
        # The caller's credentials are for the gateway, not the target: drop them unless the
        # route opts in (AIMS §10.8 — don't pass access tokens on to other services).
        credentials = set() if forward_authorization else {"authorization", "cookie"}
        self._drop_headers = _DROP_REQUEST_HEADERS | credentials
        # Default: a real httpx client (cross-process forwarding to a live target). An
        # injected client lets the gateway share one connection pool, and lets unit tests
        # point the forwarder at an in-process app via httpx MockTransport/ASGITransport.
        self._client = _no_cookie_jar(client) if client is not None else None
        self._owns_client = client is None

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = _no_cookie_jar(httpx.AsyncClient(timeout=60))
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client this forwarder created (the gateway's lifespan calls it)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward one request and relay the target's response.

        Args:
            scope (dict[str, Any]): The ASGI scope (HTTP only).
            receive (Callable): The ASGI receive channel.
            send (Callable): The ASGI send channel.
        """
        if scope.get("type") != "http":
            return  # gateway only mounts HTTP routes

        body = await _read_body(receive)
        # Under a Starlette Mount, scope["path"] is the FULL path and scope["root_path"] is
        # the mount prefix (Starlette does not strip it). Forward the path *below* the mount.
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        sub_path = path[len(root):] if root and path.startswith(root) else path
        if _has_dot_segment(sub_path) or "?" in sub_path or "#" in sub_path:
            # scope["path"] is percent-decoded, so "%2e%2e" is ".." and "%3F" is "?" here:
            # forwarded, they would reach paths above (or beside) the target the route was
            # authorized for, or cut off the path the authorization saw (".../x%3F/agent-card.json").
            await _send_bad_request(send)
            return
        url = self.target_url + sub_path
        # scope["query_string"] is the raw bytes after "?" (e.g. b"limit=10"); re-append it
        # so query params survive the hop. (A2A POSTs rarely carry one, but keep it general.)
        query = scope.get("query_string", b"").decode("latin-1")
        if query:
            url = f"{url}?{query}"
        response = await self._client_or_create().request(
            scope["method"], url, content=body, headers=self._request_headers(scope)
        )

        # multi_items keeps repeated headers (several Set-Cookie) apart instead of merging them.
        drop = _DROP_RESPONSE_HEADERS | _connection_tokens(response.headers.get_list("connection"))
        out_headers = [
            (k.encode("latin-1"), v.encode("latin-1"))
            for k, v in response.headers.multi_items()
            if k.lower() not in drop
        ]
        out_headers.append((b"content-length", str(len(response.content)).encode()))
        await send({"type": "http.response.start", "status": response.status_code, "headers": out_headers})
        await send({"type": "http.response.body", "body": response.content})


    def _request_headers(self, scope: Scope) -> list[tuple[str, str]]:
        """Build the headers to send on.

        The caller's headers minus the dropped ones, plus ``X-Forwarded-*`` from what this hop saw
        (the client's address, and the scheme and host it called).
        """
        raw = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in scope.get("headers", [])]
        drop = self._drop_headers | _connection_tokens(v for k, v in raw if k.lower() == "connection")
        headers = [(k, v) for k, v in raw if k.lower() not in drop]
        client = scope.get("client")
        if client:
            headers.append(("x-forwarded-for", str(client[0])))
        headers.append(("x-forwarded-proto", scope.get("scheme", "http")))
        host = next((v for k, v in raw if k.lower() == "host"), None)
        if host:
            headers.append(("x-forwarded-host", host))
        return headers


def _no_cookie_jar(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Make the client keep no cookies.

    One client serves every caller, so a cookie the target sets for one caller would otherwise
    go out on the next caller's requests.
    """
    client.cookies.jar.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    return client


def _connection_tokens(values) -> set[str]:
    """The header names a ``Connection`` header lists — hop-by-hop for this connection."""
    return {token.strip().lower() for value in values for token in value.split(",") if token.strip()}


def _has_dot_segment(path: str) -> bool:
    """Whether a (decoded) path has a ``.`` or ``..`` segment; a backslash counts as a separator."""
    return any(segment in (".", "..") for segment in path.replace("\\", "/").split("/"))


async def _send_bad_request(send: Send) -> None:
    data = b'{"error": "Bad Request: the path has a . or .. segment, or a ? or # in it."}'
    await send({
        "type": "http.response.start",
        "status": 400,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())],
    })
    await send({"type": "http.response.body", "body": data})


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        more_body = message.get("more_body", False)
    return b"".join(chunks)
