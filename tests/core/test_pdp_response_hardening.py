"""PDP client hardening against values and answers that used to slip through ``fail_open``.

- A value JSON can't carry (NaN, infinity, a lone surrogate) is left out like any value a PDP
  can't carry; a request that still can't be encoded is refused, never treated as an outage.
- A malformed XACML MultiRequest result gives no decision for its name instead of raising.
- An Indeterminate isn't cached, so a ``fail_open`` fallback is decided again; NotApplicable is.
"""
from __future__ import annotations

import math

import httpx
import pytest

from axiomatics_ai import AuthZConfig, Authorizer, PDPConnection
from axiomatics_ai.core.cache import AuthzDecisionCacheWithTTL
from axiomatics_ai.core.enforcement import authorized_call
from axiomatics_ai.core.pdp import PdpClient, PdpRequestError, xacml

_SUBJECT = {"username": "alice"}
_ACTION = {"name": "tool_invocation"}


def _authorizer(handler, *, protocol: str, fail_open: bool) -> Authorizer:
    transport = httpx.MockTransport(handler)
    return Authorizer(AuthZConfig(connection=PDPConnection(
        base_url="http://pdp", protocol=protocol, fail_open=fail_open, http_client=httpx.Client(transport=transport),
    )))


@pytest.mark.parametrize("protocol", ["xacml", "authzen"])
@pytest.mark.parametrize("fail_open", [False, True])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "\ud800"])
def test_a_value_json_cant_carry_is_dropped_and_the_pdp_decides(protocol, fail_open, value):
    """Like any value a PDP can't carry, it is left out; the PDP still decides (never fail_open)."""
    asked = []

    def deny_all(request):
        asked.append(request)
        return httpx.Response(200, json={"decision": False} if protocol == "authzen" else {"Response": [{"Decision": "Deny"}]})

    authz = _authorizer(deny_all, protocol=protocol, fail_open=fail_open)
    assert authz.is_permitted(action="read", resource="doc", subject={**_SUBJECT, "amount": value}) is False
    assert len(asked) == 1 and b"amount" not in asked[0].content


def test_a_request_json_cant_encode_is_refused_not_resolved_by_fail_open():
    """The encode check stays as a backstop for anything the attribute rules let through."""
    client = PdpClient(base_url="http://pdp", fail_open=True,
                       http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))))
    with pytest.raises(PdpRequestError, match="can't be encoded"):
        client._authorize({"Request": {"x": math.nan}})


def test_an_ordinary_float_still_reaches_the_pdp():
    asked = []

    def deny_all(request):
        asked.append(request)
        return httpx.Response(200, json={"Response": [{"Decision": "Deny"}]})

    authz = _authorizer(deny_all, protocol="xacml", fail_open=True)
    assert authz.is_permitted(action="read", resource="doc", subject={**_SUBJECT, "amount": 1.5}) is False
    assert len(asked) == 1


@pytest.mark.parametrize("result", [
    {"Decision": "Permit", "Category": {"not": "a list"}},
    {"Decision": "Permit", "Category": [{"Attribute": "not a list"}]},
    {"Decision": "Permit", "Category": [{"Attribute": ["not a dict"]}]},
    {"Decision": "Permit", "Category": ["not a dict"]},
    {"Decision": "Permit", "Category": [{"Attribute": [{"AttributeId": "tool_name", "Value": [{"x": 1}]}]}]},
])
def test_a_malformed_multi_result_gives_no_decision_instead_of_raising(result):
    decisions = xacml.parse_multi_response({"Response": [result]}, "tool_name", ["a", "b"])
    assert all(d.decision is None for d in decisions.values())


def test_a_multi_result_may_echo_a_single_value_not_a_list():
    result = {"Decision": "Permit", "Category": [{"Attribute": [{"AttributeId": "tool_name", "Value": "a"}]}]}
    decisions = xacml.parse_multi_response({"Response": [result]}, "tool_name", ["a", "b"])
    assert decisions["a"].decision is not None and decisions["b"].decision is None


@pytest.mark.parametrize("answer", ["Indeterminate"])
def test_an_inconclusive_answer_is_not_cached_under_fail_open(answer):
    """The fallback permit must not outlive the PDP's answer by the cache's TTL."""
    answers = iter([answer, "Deny"])

    def pdp(request):
        return httpx.Response(200, json={"Response": [{"Decision": next(answers)}]})

    client = PdpClient(base_url="http://pdp", fail_open=True, http_client=httpx.Client(transport=httpx.MockTransport(pdp)))
    cache = AuthzDecisionCacheWithTTL(ttl=300, maxsize=16)
    call = dict(pdp=client, cache=cache, config=AuthZConfig(connection=PDPConnection(fail_open=True)),
                surface="test", subject=_SUBJECT, action=_ACTION, resource={"tool_name": "t"}, environment={})
    assert authorized_call(**call) is True    # inconclusive → fail_open permits
    assert authorized_call(**call) is False   # asked again: the PDP now denies


def test_a_not_applicable_answer_is_cached():
    """NotApplicable is the policy's answer (fail_open is applied again on each read)."""
    asked = []

    def pdp(request):
        asked.append(request)
        return httpx.Response(200, json={"Response": [{"Decision": "NotApplicable"}]})

    client = PdpClient(base_url="http://pdp", http_client=httpx.Client(transport=httpx.MockTransport(pdp)))
    cache = AuthzDecisionCacheWithTTL(ttl=300, maxsize=16)
    call = dict(pdp=client, cache=cache, config=AuthZConfig(), surface="test",
                subject=_SUBJECT, action=_ACTION, resource={"tool_name": "t"}, environment={})
    assert authorized_call(**call) is False
    assert authorized_call(**call) is False
    assert len(asked) == 1


def test_numeric_resource_names_are_read_from_a_multi_response():
    result = {"Decision": "Permit", "Category": [{"Attribute": [{"AttributeId": "id", "Value": [101]}]}]}
    assert xacml.parse_multi_response({"Response": [result]}, "id", [101, 102])[101].decision is not None


def test_a_definite_answer_is_still_cached():
    asked = []

    def pdp(request):
        asked.append(request)
        return httpx.Response(200, json={"Response": [{"Decision": "Deny"}]})

    client = PdpClient(base_url="http://pdp", http_client=httpx.Client(transport=httpx.MockTransport(pdp)))
    cache = AuthzDecisionCacheWithTTL(ttl=300, maxsize=16)
    call = dict(pdp=client, cache=cache, config=AuthZConfig(), surface="test",
                subject=_SUBJECT, action=_ACTION, resource={"tool_name": "t"}, environment={})
    assert authorized_call(**call) is False
    assert authorized_call(**call) is False
    assert len(asked) == 1
