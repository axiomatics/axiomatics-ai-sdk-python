"""
Unit tests for A2AAuthZMiddleware (inbound pure-ASGI guard).

Drives the middleware directly with a constructed ASGI scope/receive/send and a fake
downstream app (no web server, no ADS). Asserts: permitted sends reach the downstream and
its body is replayed intact; denied sends never reach it and return a JSON-RPC error;
every JSON-RPC method is authorized and other POSTs are rejected; non-POST traffic bypasses
ADS; identity comes from headers; the PDP call runs off the event loop.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from axiomatics_ai.a2a import A2AAuthZMiddleware
from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider, AttributeNames

from .conftest import FakePdp, jsonrpc, params_v03, params_v1


class Downstream:
    """Fake ASGI app: records that it was reached and echoes the body it received."""

    def __init__(self) -> None:
        self.called = False
        self.received_body = b""

    async def __call__(self, scope, receive, send) -> None:
        self.called = True
        message = await receive()
        self.received_body = message.get("body", b"")
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"jsonrpc":"2.0","id":"1","result":{}}'})


class HeaderProvider(DefaultServerContextProvider):
    """Read the subject from a plain ``x-user`` header (avoids JWT plumbing in unit tests)."""

    def get_subject(self, headers):
        user = headers.get("x-user")
        return {"username": user} if user else {}


def _drive(mw, body: bytes, *, method: str = "POST", path: str = "/", headers=None):
    headers = headers or [(b"x-user", b"alice")]
    scope = {"type": "http", "method": method, "path": path, "headers": headers}

    async def run():
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        await mw(scope, receive, send)
        return sent

    return asyncio.run(run())


def _middleware(fake: FakePdp, **kwargs) -> tuple[A2AAuthZMiddleware, Downstream]:
    downstream = Downstream()
    config = kwargs.pop("config", None) or AuthZConfig(context_provider=HeaderProvider())
    mw = A2AAuthZMiddleware(downstream, config, **kwargs)
    mw._ads = fake
    return mw, downstream


def _response_body(sent: list[dict]) -> dict:
    body = next(m["body"] for m in sent if m["type"] == "http.response.body")
    return json.loads(body)


def test_permit_reaches_downstream_and_replays_body():
    fake = FakePdp(permit=True)
    mw, downstream = _middleware(fake)
    body = jsonrpc("message/send", params_v03())
    _drive(mw, body)
    assert downstream.called is True
    assert downstream.received_body == body     # body replayed intact
    assert len(fake.calls) == 1


def test_deny_blocks_downstream_and_returns_jsonrpc_error():
    fake = FakePdp(permit=False)
    mw, downstream = _middleware(fake)
    sent = _drive(mw, jsonrpc("message/send", params_v03()))
    assert downstream.called is False           # never reached the agent
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200
    error = _response_body(sent)["error"]
    assert error["code"] == -32001
    assert "Access denied" in error["message"]


@pytest.mark.parametrize("params_fn", [params_v03, params_v1])
def test_version_agnostic_deny(params_fn):
    method = "message/send" if params_fn is params_v03 else "SendMessage"
    fake = FakePdp(permit=False)
    mw, downstream = _middleware(fake)
    _drive(mw, jsonrpc(method, params_fn()))
    assert downstream.called is False


@pytest.mark.parametrize("method", [
    "tasks/get", "tasks/cancel", "tasks/resubscribe", "tasks/pushNotificationConfig/set",
    "GetTask", "some/futureMethod",
])
def test_every_jsonrpc_method_is_authorized(method):
    # A caller denied message/send must not read or cancel tasks, or register a webhook.
    fake = FakePdp(permit=False)
    mw, downstream = _middleware(fake)
    sent = _drive(mw, jsonrpc(method, {"id": "t1"}))
    assert downstream.called is False
    assert fake.calls[0]["resource"]["method"] == method
    error = _response_body(sent)["error"]
    assert error["code"] == -32001 and method in error["message"]


def test_permitted_task_method_reaches_downstream():
    fake = FakePdp(permit=True)
    mw, downstream = _middleware(fake)
    _drive(mw, jsonrpc("tasks/get", {"id": "t1"}))
    assert downstream.called is True and len(fake.calls) == 1


@pytest.mark.parametrize("body", [
    pytest.param(b"", id="empty"),
    pytest.param(b"not json", id="invalid-json"),
    pytest.param(b'[{"jsonrpc":"2.0","id":"1","method":"message/send","params":{}}]', id="batch"),
    pytest.param(b'{"message":{"parts":[{"text":"hi"}]}}', id="rest-binding-no-method"),
    pytest.param(b'{"jsonrpc":"2.0","id":"1","method":42}', id="non-string-method"),
])
def test_post_that_is_not_a_single_jsonrpc_request_is_rejected(body):
    fake = FakePdp(permit=True)  # would permit if consulted
    mw, downstream = _middleware(fake)
    sent = _drive(mw, body, path="/v1/message:send")
    assert downstream.called is False
    assert fake.calls == []
    assert _response_body(sent)["error"]["code"] == -32600


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("path", ["/.well-known/agent-card.json", "/.well-known/agent.json", "/billing/.well-known/agent-card.json"])
def test_public_agent_card_passes_without_the_pdp(method, path):
    fake = FakePdp(permit=False)
    mw, downstream = _middleware(fake)
    _drive(mw, b"", method=method, path=path)
    assert downstream.called is True
    assert fake.calls == []


@pytest.mark.parametrize("method,path", [
    ("GET", "/"),                                         # anything that is not the card
    ("GET", "/v1/tasks/t1"),                              # the REST binding's task read
    ("GET", "/agent/authenticatedExtendedCard"),          # the authenticated extended card
    ("PUT", "/.well-known/agent-card.json"),              # a write to the card's path
    ("DELETE", "/"),
])
def test_other_non_post_requests_are_refused(method, path):
    fake = FakePdp(permit=True)
    mw, downstream = _middleware(fake)
    sent = _drive(mw, b"", method=method, path=path)
    assert downstream.called is False
    assert sent[0]["status"] == 403
    assert fake.calls == []


def test_public_paths_add_to_the_card():
    fake = FakePdp(permit=False)
    mw, downstream = _middleware(fake, public_paths=("/health",))
    _drive(mw, b"", method="GET", path="/health")
    assert downstream.called is True
    mw, downstream = _middleware(fake, public_paths=("/health",))
    sent = _drive(mw, b"", method="GET", path="/metrics")
    assert downstream.called is False
    assert sent[0]["status"] == 403


def test_non_http_scopes_pass_through():
    mw, downstream = _middleware(FakePdp(permit=False))

    async def run():
        async def receive():
            return {"type": "lifespan.startup"}

        async def send(message):
            pass

        await mw({"type": "lifespan"}, receive, send)

    asyncio.run(run())
    assert downstream.called is True


def test_identity_and_resource_from_request():
    fake = FakePdp(permit=True)
    mw, downstream = _middleware(fake, agent_name="billing-agent")
    _drive(mw, jsonrpc("message/send", params_v03(skill="refund")),
           headers=[(b"x-user", b"carol")])
    sent = fake.calls[0]
    assert sent["subject"] == {"username": "carol"}
    assert sent["action"] == {"name": "a2a_invocation"}
    assert sent["resource"] == {"agent_name": "billing-agent", "method": "message/send", "skill": "refund"}


def test_agent_name_defaults_to_request_path():
    fake = FakePdp(permit=True)
    mw, downstream = _middleware(fake)
    _drive(mw, jsonrpc("message/send", params_v03()), path="/billing")
    assert fake.calls[0]["resource"]["agent_name"] == "/billing"


def test_agent_name_attribute_is_configurable():
    fake = FakePdp(permit=True)
    config = AuthZConfig(context_provider=HeaderProvider(), attribute_names=AttributeNames(agent="target_agent"))
    mw, downstream = _middleware(fake, config=config, agent_name="billing-agent")
    _drive(mw, jsonrpc("message/send", params_v03()))
    resource = fake.calls[0]["resource"]
    assert resource["target_agent"] == "billing-agent"
    assert "agent_name" not in resource


# ── context_provider resolution ──────────────────────────────────────────────────

class _DistinctiveProvider(DefaultServerContextProvider):
    """Emits a distinctive subject attribute so we can prove which provider ADS saw."""

    def get_subject(self, headers):
        return {"provider_marker": "from-config"}


def test_provider_set_only_on_config_is_used():
    """Regression: a context_provider set on AuthZConfig (not passed as the constructor
    argument) must be the one that builds the ADS request. Before the fix the config field
    was silently dropped and the default provider ran, so the distinctive attribute never
    reached ADS and every request was denied for a mismatched subject dictionary."""
    from axiomatics_ai.core.config import AuthZConfig

    fake = FakePdp(permit=True)
    downstream = Downstream()
    # No context_provider= argument — only on the config.
    mw = A2AAuthZMiddleware(downstream, config=AuthZConfig(context_provider=_DistinctiveProvider()))
    mw._ads = fake

    _drive(mw, jsonrpc("message/send", params_v03()))

    assert fake.calls[0]["subject"] == {"provider_marker": "from-config"}




def test_pdp_call_runs_off_the_event_loop():
    import threading

    fake = FakePdp(permit=True)
    threads = []
    original = fake.check_permission

    def recording(**kw):
        threads.append(threading.current_thread())
        return original(**kw)

    fake.check_permission = recording
    mw, downstream = _middleware(fake)
    _drive(mw, jsonrpc("message/send", params_v03()))
    assert downstream.called and threads[0] is not threading.main_thread()   # the loop runs on main


def test_after_the_body_the_app_receives_from_the_real_channel():
    """A streaming (SSE) response keeps listening for a client disconnect: the middleware must
    pass that on from the real channel, not answer ``http.disconnect`` at once (which ended
    every ``message/stream``)."""
    received = []

    async def app(scope, receive, send):
        received.append(await receive())   # the replayed body
        received.append(await receive())   # what the client really sends next
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = A2AAuthZMiddleware(app, AuthZConfig(context_provider=HeaderProvider()))
    mw._ads = FakePdp(permit=True)
    body = jsonrpc("message/stream", params_v03())
    channel = [{"type": "http.request", "body": body, "more_body": False}, {"type": "client.still.here"}]

    async def receive():
        return channel.pop(0)

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [(b"x-user", b"alice")]}
    asyncio.run(mw(scope, receive, send))
    assert received[0]["body"] == body
    assert received[1] == {"type": "client.still.here"}


def test_a_request_with_two_authorization_headers_is_anonymous():
    """Authentication in front may check the first; the dict would keep the last."""
    from axiomatics_ai.a2a.server import _headers_dict

    scope = {"headers": [(b"authorization", b"Bearer mine"), (b"authorization", b"Bearer forged"), (b"x-a", b"1")]}
    assert _headers_dict(scope) == {"x-a": "1"}
    assert _headers_dict({"headers": [(b"authorization", b"Bearer mine")]}) == {"authorization": "Bearer mine"}
