"""
Unit tests for A2AClientAuthZ (outbound httpx guard).

Drives the auth through a real sync ``httpx.Client`` with a ``MockTransport`` (never hits
the network) and a fake ADS. Asserts: permitted sends reach the transport; denied sends
raise before leaving; non-invocation traffic bypasses ADS; the correct XACML categories
reach ADS; both a2a-sdk wire shapes behave identically.
"""
from __future__ import annotations

import pytest
import httpx

from axiomatics_ai.a2a import A2AClientAuthZ, A2AAuthorizationError
from axiomatics_ai.core.config import AuthZConfig, AttributeNames

from .conftest import FakePdp, jsonrpc, params_v03, params_v1


def _client(authz: A2AClientAuthZ, calls: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {}})

    return httpx.Client(transport=httpx.MockTransport(handler), auth=authz)


def _authz(fake: FakePdp, **kwargs) -> A2AClientAuthZ:
    authz = A2AClientAuthZ(context={"username": "alice", "tenant_id": "acme"}, **kwargs)
    authz._ads = fake  # inject fake ADS (bypass the real build)
    return authz


URL = "http://billing.example.com/"


def test_permit_lets_the_send_through():
    fake = FakePdp(permit=True)
    calls: list[httpx.Request] = []
    client = _client(_authz(fake), calls)
    r = client.post(URL, content=jsonrpc("message/send", params_v03()))
    assert r.status_code == 200
    assert len(calls) == 1          # request reached the transport
    assert len(fake.calls) == 1     # ADS was consulted


def test_deny_blocks_before_the_send():
    fake = FakePdp(permit=False)
    calls: list[httpx.Request] = []
    client = _client(_authz(fake), calls)
    with pytest.raises(A2AAuthorizationError):
        client.post(URL, content=jsonrpc("message/send", params_v03()))
    assert calls == []              # request never left


@pytest.mark.parametrize("params_fn", [params_v03, params_v1])
def test_version_agnostic_deny(params_fn):
    """Both 0.3.x (message/send) and 1.x (SendMessage) are blocked identically."""
    method = "message/send" if params_fn is params_v03 else "SendMessage"
    fake = FakePdp(permit=False)
    calls: list[httpx.Request] = []
    client = _client(_authz(fake), calls)
    with pytest.raises(A2AAuthorizationError):
        client.post(URL, content=jsonrpc(method, params_fn()))
    assert calls == []


def test_non_invocation_traffic_bypasses_ads():
    fake = FakePdp(permit=False)  # would deny if consulted
    calls: list[httpx.Request] = []
    client = _client(_authz(fake), calls)
    r = client.post(URL, content=jsonrpc("tasks/get", {"id": "t1"}))
    assert r.status_code == 200
    assert len(calls) == 1          # passed through
    assert fake.calls == []         # ADS never consulted


def test_xacml_categories_reaching_ads():
    fake = FakePdp(permit=True)
    calls: list[httpx.Request] = []
    client = _client(_authz(fake), calls)
    client.post(URL, content=jsonrpc("message/send", params_v03(skill="refund")))

    sent = fake.calls[0]
    assert sent["subject"] == {"username": "alice", "tenant_id": "acme"}
    assert sent["action"] == {"name": "a2a_invocation"}
    # resource = target agent host + method + skill hint (message text off by default)
    assert sent["resource"] == {
        "agent_name": "billing.example.com", "method": "message/send", "skill": "refund",
    }


def test_include_message_text_adds_content_to_resource():
    fake = FakePdp(permit=True)
    calls: list[httpx.Request] = []
    client = _client(_authz(fake, include_message_text=True), calls)
    client.post(URL, content=jsonrpc("message/send", params_v03("transfer $500")))
    assert fake.calls[0]["resource"]["message"] == "transfer $500"


def test_agent_name_override_is_used_as_resource():
    fake = FakePdp(permit=True)
    calls: list[httpx.Request] = []
    client = _client(_authz(fake, agent_name="billing-agent"), calls)
    client.post(URL, content=jsonrpc("message/send", params_v03()))
    assert fake.calls[0]["resource"]["agent_name"] == "billing-agent"


def test_agent_name_attribute_is_configurable():
    fake = FakePdp(permit=True)
    calls: list[httpx.Request] = []
    client = _client(_authz(fake, config=AuthZConfig(attribute_names=AttributeNames(agent="target_agent"))), calls)
    client.post(URL, content=jsonrpc("message/send", params_v03()))
    resource = fake.calls[0]["resource"]
    assert resource["target_agent"] == "billing.example.com"
    assert "agent_name" not in resource


def test_context_source_is_resolved_per_request():
    """Per-invocation identity: the callable is re-evaluated on each send."""
    fake = FakePdp(permit=True)
    users = iter(["alice", "bob"])
    authz = A2AClientAuthZ(context_source=lambda: {"username": next(users)})
    authz._ads = fake
    calls: list[httpx.Request] = []
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: (calls.append(r), httpx.Response(200, json={}))[1]),
        auth=authz,
    )
    client.post(URL, content=jsonrpc("message/send", params_v03()))
    client.post(URL, content=jsonrpc("message/send", params_v03()))
    assert [c["subject"]["username"] for c in fake.calls] == ["alice", "bob"]


def test_async_client_checks_off_the_event_loop():
    # With an AsyncClient the PDP call (blocking) must not run on the event loop's thread.
    import asyncio
    import threading

    fake = FakePdp(permit=False)
    threads = []
    original = fake.check_permission

    def recording(**kw):
        threads.append(threading.current_thread())
        return original(**kw)

    fake.check_permission = recording

    async def run():
        loop_thread = threading.current_thread()
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), auth=_authz(fake))
        with pytest.raises(A2AAuthorizationError):
            await client.post(URL, content=jsonrpc("message/send", params_v03()))
        await client.aclose()
        return loop_thread

    loop_thread = asyncio.run(run())
    assert threads and threads[0] is not loop_thread
