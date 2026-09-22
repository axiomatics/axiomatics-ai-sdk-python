"""Unit tests for HttpForwarder — it relays verbatim and contains no policy logic."""
from __future__ import annotations

import httpx
import pytest

from axiomatics_ai.gateway.forwarder import HttpForwarder

from .conftest import drive_asgi


def _forwarder_with_capture(status: int = 200, reply: bytes = b'{"ok":true}', **kw):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = request.content
        captured["auth"] = request.headers.get("authorization")
        captured["headers"] = dict(request.headers)
        return httpx.Response(status, content=reply)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpForwarder("http://target.example", client=client, **kw), captured


def test_forwards_path_body_and_relays_response():
    fwd, captured = _forwarder_with_capture(reply=b'{"echo":"hi"}')
    status, body = drive_asgi(fwd, path="/rpc", body=b'{"m":1}', headers=[(b"authorization", b"Bearer x")])
    assert status == 200
    assert body == b'{"echo":"hi"}'                      # target's response relayed
    assert captured["method"] == "POST"
    assert captured["url"] == "http://target.example/rpc"  # target_url + sub-path
    assert captured["body"] == b'{"m":1}'                  # body forwarded verbatim
    assert captured["auth"] is None                        # caller's token is not passed on


def test_authorization_forwarded_only_when_opted_in():
    fwd, captured = _forwarder_with_capture(forward_authorization=True)
    drive_asgi(fwd, path="/rpc", headers=[(b"authorization", b"Bearer x")])
    assert captured["auth"] == "Bearer x"


def test_other_headers_still_forwarded_when_authorization_dropped():
    fwd, captured = _forwarder_with_capture()
    drive_asgi(fwd, path="/rpc", headers=[
        (b"authorization", b"Bearer x"), (b"a2a-version", b"1.0"), (b"host", b"gateway.example"),
    ])
    assert captured["auth"] is None
    assert captured["headers"]["a2a-version"] == "1.0"
    assert captured["headers"]["host"] == "target.example"  # host still dropped (httpx sets it)


def test_query_string_survives():
    fwd, captured = _forwarder_with_capture()
    drive_asgi(fwd, path="/rpc", query=b"limit=10&x=1")
    assert captured["url"] == "http://target.example/rpc?limit=10&x=1"


def test_relays_non_200_status():
    fwd, _ = _forwarder_with_capture(status=503, reply=b"upstream down")
    status, body = drive_asgi(fwd, path="/")
    assert status == 503 and body == b"upstream down"


# ── header hygiene ────────────────────────────────────────────────────────────────

def _response_headers(fwd, **drive_kw) -> list[tuple[str, str]]:
    """Drive the forwarder once and return the response headers it sent to the caller."""
    import asyncio

    scope = {"type": "http", "method": "POST", "path": "/rpc", "query_string": b"",
             "headers": drive_kw.get("headers", []), "scheme": "https", "client": ("203.0.113.7", 5555)}
    sent: list[dict] = []

    async def run():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await fwd(scope, receive, send)

    asyncio.run(run())
    start = next(m for m in sent if m["type"] == "http.response.start")
    return [(k.decode(), v.decode()) for k, v in start["headers"]]


def test_hop_by_hop_and_connection_listed_headers_are_not_forwarded():
    fwd, captured = _forwarder_with_capture()
    drive_asgi(fwd, path="/rpc", headers=[
        (b"connection", b"keep-alive, x-secret-hop"), (b"keep-alive", b"timeout=5"),
        (b"te", b"trailers"), (b"upgrade", b"websocket"), (b"x-secret-hop", b"1"), (b"x-app", b"kept"),
    ])
    for name in ("keep-alive", "te", "upgrade", "x-secret-hop"):
        assert name not in captured["headers"]
    # httpx sets its own Connection header for the upstream hop; the caller's isn't relayed.
    assert "x-secret-hop" not in captured["headers"].get("connection", "")
    assert captured["headers"]["x-app"] == "kept"


def test_cookie_is_a_credential_and_dropped_unless_opted_in():
    fwd, captured = _forwarder_with_capture()
    drive_asgi(fwd, path="/rpc", headers=[(b"cookie", b"session=gateway")])
    assert "cookie" not in captured["headers"]
    fwd, captured = _forwarder_with_capture(forward_authorization=True)
    drive_asgi(fwd, path="/rpc", headers=[(b"cookie", b"session=gateway")])
    assert captured["headers"]["cookie"] == "session=gateway"


def test_client_forwarded_headers_are_replaced_by_what_the_gateway_saw():
    fwd, captured = _forwarder_with_capture()
    _response_headers(fwd, headers=[
        (b"x-forwarded-for", b"10.0.0.1"), (b"forwarded", b"for=10.0.0.1"),
        (b"x-forwarded-proto", b"http"), (b"host", b"gw.example"),
    ])
    assert captured["headers"]["x-forwarded-for"] == "203.0.113.7"   # the real peer, not the claim
    assert captured["headers"]["x-forwarded-proto"] == "https"
    assert captured["headers"]["x-forwarded-host"] == "gw.example"
    assert "forwarded" not in captured["headers"]


def test_decoded_body_is_not_relayed_as_gzip():
    import gzip

    def handler(request):
        return httpx.Response(200, content=gzip.compress(b'{"ok":true}'), headers={"content-encoding": "gzip"})

    fwd = HttpForwarder("http://t", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    headers = dict(_response_headers(fwd))
    assert "content-encoding" not in headers                  # httpx already decoded the body
    assert headers["content-length"] == str(len(b'{"ok":true}'))


def test_repeated_set_cookie_headers_stay_separate():
    def handler(request):
        return httpx.Response(200, content=b"{}", headers=[("set-cookie", "a=1"), ("set-cookie", "b=2")])

    fwd = HttpForwarder("http://t", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    cookies = [v for k, v in _response_headers(fwd) if k == "set-cookie"]
    assert cookies == ["a=1", "b=2"]


def test_aclose_closes_only_a_client_it_created():
    import asyncio

    injected = httpx.AsyncClient()
    fwd = HttpForwarder("http://t", client=injected)
    asyncio.run(fwd.aclose())
    assert not injected.is_closed                              # the caller owns it
    own = HttpForwarder("http://t")
    client = own._client_or_create()
    asyncio.run(own.aclose())
    assert client.is_closed
    asyncio.run(injected.aclose())


# ── dot segments: never forwarded ─────────────────────────────────────────────────

def _forwarder_counting_calls():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"{}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpForwarder("http://target.example", client=client), calls


@pytest.mark.parametrize("path", [
    "/../admin", "/a/../../admin", "/a/./b", "/.", "/..", "/a/..", "/a\\..\\b", "\\..\\admin", "/a/.\\b",
])
def test_dot_segment_is_rejected_and_not_forwarded(path):
    # scope["path"] is percent-decoded by the server, so "%2e%2e" arrives here as "..".
    fwd, calls = _forwarder_counting_calls()
    status, body = drive_asgi(fwd, path=path)
    assert status == 400
    assert b"Bad Request" in body
    assert calls == []                                    # the target was never called


def test_dot_segment_below_the_mount_root_is_rejected():
    fwd, calls = _forwarder_counting_calls()
    import asyncio

    scope = {"type": "http", "method": "POST", "path": "/billing/../admin", "root_path": "/billing",
             "query_string": b"", "headers": []}
    sent: list[dict] = []

    async def run():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await fwd(scope, receive, send)

    asyncio.run(run())
    assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 400
    assert calls == []


@pytest.mark.parametrize("path", ["/rpc", "/a..b", "/.well-known/x", "/a/.hidden", "/a/b..", "/", ""])
def test_names_with_dots_still_forward(path):
    fwd, calls = _forwarder_counting_calls()
    status, _ = drive_asgi(fwd, path=path)
    assert status == 200
    assert calls == ["http://target.example" + path]


def test_a_cookie_the_target_sets_for_one_caller_is_not_sent_for_the_next():
    """One client serves every caller, so its cookie jar must stay empty: a session cookie the
    target sets for alice would otherwise go upstream on bob's request."""
    seen = []

    def target(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("cookie"))
        return httpx.Response(200, json={"ok": True}, headers={"set-cookie": "session=sess-of-alice; Path=/"})

    fwd = HttpForwarder("http://target.local", client=httpx.AsyncClient(transport=httpx.MockTransport(target)))
    drive_asgi(fwd, path="/rpc", body=b"{}")   # alice
    drive_asgi(fwd, path="/rpc", body=b"{}")   # bob
    assert seen == [None, None]
