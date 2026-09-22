"""
Unit tests for PdpClient.check_permission and check_permission_batch.

ADS HTTP calls are stubbed via monkeypatching _authorize directly, so no
real ADS instance is required.
"""
from __future__ import annotations

from tests.pdp_fakes import patch_pdp_http

import pytest

import httpx

from axiomatics_ai.core.pdp import Decision, PdpClient, PdpRequestError, authzen, xacml

# ── Helpers ────────────────────────────────────────────────────────────────────

_SUBJECT = {"username": "alice"}
_ACTION = {"name": "tool_invocation"}
_RESOURCE = {"tool_name": "search_docs"}
_ENV = {}

_TOOL_NAMES = ["search_docs", "send_email", "get_balance"]


def _client() -> PdpClient:
    return PdpClient(base_url="http://ads-stub")


def _single_response(decision: str) -> dict:
    return {"Response": [{"Decision": decision}]}


def _batch_response(decisions: dict[str, str]) -> dict:
    """Build a MultiRequest response. decisions maps tool_name → Decision string."""
    responses = []
    for name, decision in decisions.items():
        responses.append({
            "Decision": decision,
            "Category": [{"Attribute": [{"AttributeId": "tool_name", "Value": [name]}]}],
        })
    return {"Response": responses}


# ── check_permission — single request ─────────────────────────────────────────

class TestCheckPermission:
    def test_permit_returns_true(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: _single_response("Permit"))
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV) is True

    def test_deny_returns_false(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: _single_response("Deny"))
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV) is False

    def test_not_applicable_fail_closed_returns_false(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: _single_response("NotApplicable"))
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=False) is False

    def test_not_applicable_fail_open_returns_true(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: _single_response("NotApplicable"))
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=True) is True

    def test_indeterminate_fail_closed_returns_false(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: _single_response("Indeterminate"))
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=False) is False

    def test_indeterminate_fail_open_returns_true(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: _single_response("Indeterminate"))
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=True) is True

    def test_ads_unreachable_fail_closed_returns_false(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: None)
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=False) is False

    def test_ads_unreachable_fail_open_returns_true(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: None)
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=True) is True

    def test_deny_always_false_even_with_fail_open(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: _single_response("Deny"))
        assert _client().check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=True) is False


# ── check_permission_batch — multi request ────────────────────────────────────

class TestCheckPermissionBatch:
    def _call(self, client, monkeypatch, response, *, fail_open=False):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req: response)
        return client.check_permission_batch(
            subject=_SUBJECT,
            action={"name": "tool_visibility"},
            resource_names=_TOOL_NAMES,
            environment=_ENV,
            resource_attr_id="tool_name",
            fail_open=fail_open,
        )

    def test_all_permit_returns_all_names(self, monkeypatch):
        resp = _batch_response({n: "Permit" for n in _TOOL_NAMES})
        result = self._call(_client(), monkeypatch, resp)
        assert result == set(_TOOL_NAMES)

    def test_mixed_permit_deny_returns_only_permitted(self, monkeypatch):
        resp = _batch_response({"search_docs": "Permit", "send_email": "Deny", "get_balance": "Permit"})
        result = self._call(_client(), monkeypatch, resp)
        assert result == {"search_docs", "get_balance"}

    def test_not_applicable_fail_closed_excluded(self, monkeypatch):
        resp = _batch_response({"search_docs": "Permit", "send_email": "NotApplicable", "get_balance": "Deny"})
        result = self._call(_client(), monkeypatch, resp, fail_open=False)
        assert result == {"search_docs"}

    def test_not_applicable_fail_open_included(self, monkeypatch):
        resp = _batch_response({"search_docs": "Permit", "send_email": "NotApplicable", "get_balance": "Deny"})
        result = self._call(_client(), monkeypatch, resp, fail_open=True)
        assert result == {"search_docs", "send_email"}

    def test_indeterminate_fail_closed_excluded(self, monkeypatch):
        resp = _batch_response({"search_docs": "Indeterminate", "send_email": "Permit"})
        result = self._call(_client(), monkeypatch, resp, fail_open=False)
        assert result == {"send_email"}

    def test_indeterminate_fail_open_included(self, monkeypatch):
        # get_balance has no result at all: inconclusive too, so fail_open includes it.
        resp = _batch_response({"search_docs": "Indeterminate", "send_email": "Permit"})
        result = self._call(_client(), monkeypatch, resp, fail_open=True)
        assert result == {"search_docs", "send_email", "get_balance"}

    def test_name_without_result_excluded_when_fail_closed(self, monkeypatch):
        resp = _batch_response({"search_docs": "Permit"})
        result = self._call(_client(), monkeypatch, resp, fail_open=False)
        assert result == {"search_docs"}

    def test_deny_always_excluded_even_with_fail_open(self, monkeypatch):
        resp = _batch_response({n: "Deny" for n in _TOOL_NAMES})
        result = self._call(_client(), monkeypatch, resp, fail_open=True)
        assert result == set()

    def test_ads_unreachable_fail_closed_returns_empty(self, monkeypatch):
        result = self._call(_client(), monkeypatch, None, fail_open=False)
        assert result == set()

    def test_ads_unreachable_fail_open_returns_all(self, monkeypatch):
        result = self._call(_client(), monkeypatch, None, fail_open=True)
        assert result == set(_TOOL_NAMES)


# ── Missing subject identity in the batch (subject-centric) check ──────────────

class TestBatchMissingSubjectIdentity:
    """check_permission_batch personalizes per subject, so no subject → nothing to show:
    it short-circuits to fail_open without consulting the PDP. A provider that returns ``None``
    for absent attributes (``{"username": None}``) must behave identically to ``{}`` — nulls
    must not slip past the emptiness guard.

    (The single-request check_permission deliberately has no such guard: a resource-/
    environment-only policy is valid, so an empty subject is sent to the PDP to decide.)
    """

    @pytest.mark.parametrize("subject", [{}, {"username": None}, {"username": None, "tenant_id": None}])
    def test_none_only_subject_matches_empty(self, monkeypatch, subject):
        def _must_not_call(self, req, **kw):
            raise AssertionError("PDP must not be consulted when there is no subject identity")

        monkeypatch.setattr(PdpClient, "_authorize", _must_not_call)
        c = _client()
        assert c.check_permission_batch(
            subject=subject, action=_ACTION, resource_names=_TOOL_NAMES, environment=_ENV,
            resource_attr_id="tool_name", fail_open=False) == set()
        assert c.check_permission_batch(
            subject=subject, action=_ACTION, resource_names=_TOOL_NAMES, environment=_ENV,
            resource_attr_id="tool_name", fail_open=True) == set(_TOOL_NAMES)


# ── AuthZEN protocol ───────────────────────────────────────────────────────────

def _authzen_client() -> PdpClient:
    return PdpClient(base_url="http://ads-stub", protocol="authzen")


def _authzen_single(xacml_decision: str) -> dict:
    """AuthZEN result carrying the ADS context.xacml.decision echo."""
    return {"decision": xacml_decision == "Permit", "context": {"xacml": {"decision": xacml_decision}}}


def _authzen_batch(xacml_decisions: list[str]) -> dict:
    """AuthZEN batch response — decisions are ordered, no echoed attributes."""
    return {"evaluations": [_authzen_single(d) for d in xacml_decisions]}


class TestProtocolValidation:
    def test_unknown_protocol_rejected(self):
        with pytest.raises(ValueError):
            PdpClient(base_url="http://ads-stub", protocol="rego")


class TestAuthZenDecisionNormalization:
    """authzen._decision prefers context.xacml.decision but falls back to the
    plain boolean for PDPs that omit it (the case the API supports optionally)."""

    def test_prefers_xacml_context_notapplicable(self):
        # decision:false but context says NotApplicable → must NOT collapse to Deny
        result = {"decision": False, "context": {"xacml": {"decision": "NotApplicable"}}}
        assert authzen._decision(result) is Decision.NOT_APPLICABLE

    def test_boolean_only_true_is_permit(self):
        assert authzen._decision({"decision": True}) is Decision.PERMIT

    def test_boolean_only_false_is_deny(self):
        assert authzen._decision({"decision": False}) is Decision.DENY

    def test_missing_everything_is_none(self):
        assert authzen._decision({}) is None
        assert authzen._decision(None) is None


class TestAuthZenCheckPermission:
    def _call(self, monkeypatch, response, *, fail_open=False):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req, *, endpoint=None: response)
        return _authzen_client().check_permission(
            subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=fail_open
        )

    def test_permit_true(self, monkeypatch):
        assert self._call(monkeypatch, _authzen_single("Permit")) is True

    def test_deny_false(self, monkeypatch):
        assert self._call(monkeypatch, _authzen_single("Deny")) is False

    def test_not_applicable_fail_closed_false(self, monkeypatch):
        assert self._call(monkeypatch, _authzen_single("NotApplicable"), fail_open=False) is False

    def test_not_applicable_fail_open_true(self, monkeypatch):
        # relies on context.xacml.decision surviving so fail_open can apply to NotApplicable
        assert self._call(monkeypatch, _authzen_single("NotApplicable"), fail_open=True) is True

    def test_deny_stays_false_under_fail_open(self, monkeypatch):
        assert self._call(monkeypatch, _authzen_single("Deny"), fail_open=True) is False

    def test_boolean_only_pdp_true(self, monkeypatch):
        assert self._call(monkeypatch, {"decision": True}) is True

    def test_unreachable_fail_open(self, monkeypatch):
        assert self._call(monkeypatch, None, fail_open=True) is True
        assert self._call(monkeypatch, None, fail_open=False) is False


class TestAuthZenCheckPermissionBatch:
    def _call(self, monkeypatch, response, *, fail_open=False):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req, *, endpoint=None: response)
        return _authzen_client().check_permission_batch(
            subject=_SUBJECT, action={"name": "tool_visibility"}, resource_names=_TOOL_NAMES,
            environment=_ENV, resource_attr_id="tool_name", fail_open=fail_open,
        )

    def test_correlates_by_order(self, monkeypatch):
        # search_docs=Permit, send_email=Deny, get_balance=Permit
        resp = _authzen_batch(["Permit", "Deny", "Permit"])
        assert self._call(monkeypatch, resp) == {"search_docs", "get_balance"}

    def test_not_applicable_fail_open_included(self, monkeypatch):
        resp = _authzen_batch(["Permit", "NotApplicable", "Deny"])
        assert self._call(monkeypatch, resp, fail_open=True) == {"search_docs", "send_email"}

    def test_short_response_is_inconclusive_for_every_name(self, monkeypatch):
        # two results for three resources can't be lined up by position → no decision for any
        resp = _authzen_batch(["Permit", "Permit"])
        assert self._call(monkeypatch, resp, fail_open=False) == set()
        assert self._call(monkeypatch, resp, fail_open=True) == set(_TOOL_NAMES)

    def test_unreachable(self, monkeypatch):
        assert self._call(monkeypatch, None, fail_open=False) == set()
        assert self._call(monkeypatch, None, fail_open=True) == set(_TOOL_NAMES)


class TestAuthorizeHttpHandling:
    """_authorize distinguishes deterministic client errors (4xx → raise, never masked by
    fail_open) from transient failures (5xx / connection errors → None → fail_open)."""

    class _Resp:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.text = "boom" if payload is None else ""
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=None)

    def _client(self, monkeypatch, resp=None, *, raise_conn=False):
        def fake_post(url, **kwargs):
            if raise_conn:
                raise httpx.ConnectError("refused")
            return resp

        patch_pdp_http(monkeypatch, "post", fake_post)
        return PdpClient(base_url="http://ads-stub")

    def test_4xx_raises(self, monkeypatch):
        client = self._client(monkeypatch, self._Resp(400))
        with pytest.raises(PdpRequestError):
            client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=True)

    def test_404_raises(self, monkeypatch):
        client = self._client(monkeypatch, self._Resp(404))
        with pytest.raises(PdpRequestError):
            client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV)

    def test_5xx_fails_open(self, monkeypatch):
        client = self._client(monkeypatch, self._Resp(503))
        assert client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=True) is True
        assert client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=False) is False

    def test_connection_error_fails_open(self, monkeypatch):
        client = self._client(monkeypatch, raise_conn=True)
        assert client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=True) is True

    def test_2xx_returns_decision(self, monkeypatch):
        client = self._client(monkeypatch, self._Resp(200, {"Response": [{"Decision": "Permit"}]}))
        assert client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV) is True


class TestGetSimplifiedConstraints:
    """get_simplified_constraints returns ADS's human-readable rendering of the constraint.

    Display-only: same providedAttributes serialization as get_constraints, but returns a
    string (empty string or None on failure) and must never raise for a transient error.
    """

    _PROVIDED = [
        {"attribute": "username", "values": ["alice"]},
        {"attribute": "department", "values": ["finance"]},
        {"attribute": "name", "values": ["retrieval"]},
    ]

    def test_json_envelope_response_returns_inner_string(self, monkeypatch):
        # ADS 26.1.6 returns {"response": "<expr>"} with content-type application/json.
        captured = {}

        def fake(self, body, *, endpoint=None):
            captured["endpoint"] = endpoint
            captured["body"] = body
            return {"response": 'resource.department in ("finance") AND resource.level <= 3'}

        monkeypatch.setattr(PdpClient, "_authorize", fake)
        result = _client().get_simplified_constraints(provided_attributes=self._PROVIDED)

        assert result == 'resource.department in ("finance") AND resource.level <= 3'
        assert captured["endpoint"] == "/authorization/simplified-constraints"
        # Same providedAttributes serialization as get_constraints.
        assert captured["body"]["providedAttributes"] == [
            {"attribute": "username", "values": ["alice"]},
            {"attribute": "department", "values": ["finance"]},
            {"attribute": "name", "values": ["retrieval"]},
        ]
        assert captured["body"]["pdpDecisionSet"] == ["PERMIT"]

    def test_bare_string_response_passed_through(self, monkeypatch):
        # Cross-version safety net: an ADS that returns a plain string (text/plain).
        monkeypatch.setattr(PdpClient, "_authorize",
                            lambda self, body, *, endpoint=None: "level <= 3")
        assert _client().get_simplified_constraints(provided_attributes=self._PROVIDED) == "level <= 3"

    def test_missing_response_key_returns_empty_string(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize",
                            lambda self, body, *, endpoint=None: {})
        assert _client().get_simplified_constraints(provided_attributes=self._PROVIDED) == ""

    def test_ads_unreachable_returns_none(self, monkeypatch):
        # Transient failure → None, never an exception: display must not break enforcement.
        monkeypatch.setattr(PdpClient, "_authorize",
                            lambda self, body, *, endpoint=None: None)
        assert _client().get_simplified_constraints(provided_attributes=self._PROVIDED) is None

    def test_end_to_end_through_real_authorize(self, monkeypatch):
        """Through the real _authorize: posts to the right endpoint and parses the envelope."""
        captured = {}

        class _Resp:
            status_code = 200
            text = ""
            headers = {"content-type": "application/json"}

            def json(self):
                return {"response": "resource.level <= 3"}

            def raise_for_status(self):
                pass

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return _Resp()

        patch_pdp_http(monkeypatch, "post", fake_post)
        result = _client().get_simplified_constraints(provided_attributes=self._PROVIDED)

        assert result == "resource.level <= 3"
        assert captured["url"] == "http://ads-stub/authorization/simplified-constraints"
        assert captured["json"]["providedAttributes"][0] == {"attribute": "username", "values": ["alice"]}


class TestAuthZenRequestShape:
    """Verify the request bodies the AuthZEN builders emit (captured via _authorize)."""

    def _capture(self, monkeypatch):
        captured = {}

        def fake(self, req, *, endpoint=None):
            captured["req"] = req
            captured["endpoint"] = endpoint
            return _authzen_single("Permit")

        monkeypatch.setattr(PdpClient, "_authorize", fake)
        return captured

    def test_single_shape(self, monkeypatch):
        cap = self._capture(monkeypatch)
        _authzen_client().check_permission(
            subject={"username": "alice"}, action={"name": "tool_invocation"},
            resource={"name": "search_docs"}, environment={"session_id": "s1"},
        )
        assert cap["endpoint"] == "/access/v1/evaluation"
        req = cap["req"]
        # action name is structural (raw AttributeId "name" server-side), not in properties
        assert req["action"]["name"] == "tool_invocation"
        assert req["subject"]["properties"] == {"username": "alice"}
        assert req["resource"]["properties"] == {"name": "search_docs"}
        # environment rides in the request-level context map
        assert req["context"] == {"session_id": "s1"}

    def test_batch_shape(self, monkeypatch):
        cap = self._capture(monkeypatch)
        _authzen_client().check_permission_batch(
            subject={"username": "bob"}, action={"name": "tool_visibility"},
            resource_names=["a", "b"], environment={}, resource_attr_id="tool_name",
        )
        assert cap["endpoint"] == "/access/v1/evaluations"
        req = cap["req"]
        assert req["subject"]["properties"] == {"username": "bob"}
        assert [e["resource"]["properties"] for e in req["evaluations"]] == [
            {"tool_name": "a"}, {"tool_name": "b"},
        ]
        assert "context" not in req  # empty environment omitted


class TestMultiResponseCorrelation:
    """Correlating XACML MultiRequest decisions back to the resources they were about.

    ADS echoes the resource attribute (marked IncludeInResult) into each sub-response; these
    pin that correlation, since getting it wrong permits the wrong resource rather than
    failing loudly. ``_permitted`` resolves each name the way ``check_permission_batch`` does.
    """

    @staticmethod
    def _result(decision: str, name: str, *, attr_id: str = "tool_name") -> dict:
        return {
            "Decision": decision,
            "Category": [{"Attribute": [{"AttributeId": attr_id, "Value": [name]}]}],
        }

    @staticmethod
    def _permitted(response, names, *, fail_open: bool = False) -> set[str]:
        client = PdpClient(base_url="http://ads-stub")
        decisions = xacml.parse_multi_response(response, "tool_name", names)
        return {
            name for name, decision in decisions.items()
            if client.resolve(decision, subject={}, action={}, resource={}, environment={}, fail_open=fail_open).permitted
        }

    def test_permit_included_and_deny_excluded(self):
        response = {"Response": [self._result("Permit", "a"), self._result("Deny", "b")]}
        assert self._permitted(response, ["a", "b"]) == {"a"}

    @pytest.mark.parametrize("decision", ["NotApplicable", "Indeterminate", "Unknown"])
    def test_inconclusive_excluded_by_default_included_under_fail_open(self, decision):
        response = {"Response": [self._result(decision, "a")]}
        assert self._permitted(response, ["a"]) == set()
        assert self._permitted(response, ["a"], fail_open=True) == {"a"}

    def test_deny_stays_excluded_even_under_fail_open(self):
        # fail_open covers inconclusive results; an explicit Deny is never overridden.
        response = {"Response": [self._result("Deny", "a")]}
        assert self._permitted(response, ["a"], fail_open=True) == set()

    def test_other_attribute_ids_are_ignored(self):
        # A result about another attribute decides nothing: "a" has no decision, so it is
        # inconclusive (excluded unless fail_open).
        response = {"Response": [self._result("Permit", "a", attr_id="something_else")]}
        assert self._permitted(response, ["a"]) == set()
        assert self._permitted(response, ["a"], fail_open=True) == {"a"}

    def test_name_echoed_in_several_categories_is_fully_collected(self):
        # A result may echo the attribute in more than one category; every echoed name counts,
        # so a permitted resource is never dropped because it appeared in the second category.
        response = {"Response": [{
            "Decision": "Permit",
            "Category": [
                {"Attribute": [{"AttributeId": "tool_name", "Value": ["a"]}]},
                {"Attribute": [{"AttributeId": "tool_name", "Value": ["b"]}]},
            ],
        }]}
        assert self._permitted(response, ["a", "b"]) == {"a", "b"}

    def test_echoed_name_that_was_not_requested_is_ignored(self):
        response = {"Response": [self._result("Permit", "zzz")]}
        assert self._permitted(response, ["a"]) == set()

    def test_permitted_result_without_the_echoed_attribute_adds_nothing(self):
        response = {"Response": [{"Decision": "Permit", "Category": []}]}
        assert self._permitted(response, ["a"]) == set()

    def test_empty_response_is_empty_set(self):
        assert self._permitted({}, ["a"]) == set()


# ── Endpoints + fail_open defaults ─────────────────────────────────────────────

class TestEndpointsAndFailOpenDefaults:
    """PdpEndpoints routes each call; the client's own fail_open is the default for checks."""

    @staticmethod
    def _capture(monkeypatch, response_json):
        seen: list[str] = []

        def fake_post(url, json, timeout):
            seen.append(url)
            return httpx.Response(200, json=response_json, request=httpx.Request("POST", url))

        patch_pdp_http(monkeypatch, "post", fake_post)
        return seen

    def test_default_endpoints_are_ads_paths(self):
        from axiomatics_ai.core.pdp import PdpEndpoints
        e = PdpEndpoints()
        assert (e.decision, e.health) == ("/authorize", "/health")
        assert (e.authzen_evaluation, e.authzen_evaluations) == ("/access/v1/evaluation", "/access/v1/evaluations")
        assert (e.partial_evaluation, e.partial_evaluation_text) == (
            "/authorization/constraints", "/authorization/simplified-constraints",
        )

    def test_custom_decision_endpoint_is_used_for_xacml(self, monkeypatch):
        from axiomatics_ai.core.pdp import PdpEndpoints
        seen = self._capture(monkeypatch, _single_response("Permit"))
        client = PdpClient(base_url="http://pdp/", endpoints=PdpEndpoints(decision="/domains/d1/pdp"))
        assert client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV)
        assert seen == ["http://pdp/domains/d1/pdp"]

    def test_custom_authzen_endpoints_are_used(self, monkeypatch):
        from axiomatics_ai.core.pdp import PdpEndpoints
        seen = self._capture(monkeypatch, {"decision": True})
        client = PdpClient(
            base_url="http://pdp", protocol="authzen",
            endpoints=PdpEndpoints(authzen_evaluation="/v1/eval", authzen_evaluations="/v1/evals"),
        )
        client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV)
        client.check_permission_batch(
            subject=_SUBJECT, action=_ACTION, resource_names=["a"], environment=_ENV, resource_attr_id="tool_name",
        )
        assert seen == ["http://pdp/v1/eval", "http://pdp/v1/evals"]

    def test_custom_partial_evaluation_endpoints_are_used(self, monkeypatch):
        from axiomatics_ai.core.pdp import PdpEndpoints
        seen = self._capture(monkeypatch, {"response": {}})
        client = PdpClient(
            base_url="http://pdp",
            endpoints=PdpEndpoints(partial_evaluation="/pe", partial_evaluation_text="/pe/text"),
        )
        client.get_constraints(provided_attributes=[])
        client.get_simplified_constraints(provided_attributes=[])
        assert seen == ["http://pdp/pe", "http://pdp/pe/text"]

    def test_custom_health_endpoint_is_used(self, monkeypatch):
        from axiomatics_ai.core.pdp import PdpEndpoints
        seen: list[str] = []
        patch_pdp_http(
            monkeypatch, "get",
            lambda url, timeout: (seen.append(url), httpx.Response(200, request=httpx.Request("GET", url)))[1],
        )
        assert PdpClient(base_url="http://pdp", endpoints=PdpEndpoints(health="/healthz")).healthcheck()
        assert seen == ["http://pdp/healthz"]

    def test_config_endpoints_reach_the_client(self):
        from axiomatics_ai.core.config import AuthZConfig, PDPConnection
        from axiomatics_ai.core.pdp import PdpEndpoints
        client = AuthZConfig(
            connection=PDPConnection(base_url="http://pdp", endpoints=PdpEndpoints(decision="/x")),
        ).build_pdp_client()
        assert client.endpoints.decision == "/x"

    def test_check_permission_defaults_to_client_fail_open(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req, **kw: None)  # unreachable
        assert PdpClient(base_url="http://pdp", fail_open=True).check_permission(
            subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV,
        ) is True
        assert PdpClient(base_url="http://pdp", fail_open=False).check_permission(
            subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV,
        ) is False

    def test_explicit_fail_open_overrides_client_default(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req, **kw: None)
        assert PdpClient(base_url="http://pdp", fail_open=True).check_permission(
            subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment=_ENV, fail_open=False,
        ) is False

    def test_batch_defaults_to_client_fail_open(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req, **kw: None)
        names = ["a", "b"]
        assert PdpClient(base_url="http://pdp", fail_open=True).check_permission_batch(
            subject=_SUBJECT, action=_ACTION, resource_names=names, environment=_ENV, resource_attr_id="tool_name",
        ) == set(names)
        assert PdpClient(base_url="http://pdp").check_permission_batch(
            subject=_SUBJECT, action=_ACTION, resource_names=names, environment=_ENV, resource_attr_id="tool_name",
        ) == set()


class TestAuthzenBatchResourceIds:
    """Each AuthZEN batch entry must carry its own ``resource.id``. PDPs treat entries that share
    an id as one resource: some collapse them to a single decision, others return the
    evaluations out of request order, and the SDK correlates results by position."""

    def test_batch_entries_have_distinct_ids_equal_to_the_resource_name(self):
        req = authzen.build_batch_request(
            subject=_SUBJECT, action=_ACTION, resource_attr_id="tool_name",
            resource_names=_TOOL_NAMES, environment={},
        )
        ids = [e["resource"]["id"] for e in req["evaluations"]]
        assert ids == _TOOL_NAMES
        assert len(set(ids)) == len(ids)
        assert [e["resource"]["properties"]["tool_name"] for e in req["evaluations"]] == _TOOL_NAMES

    def test_single_request_keeps_the_generic_resource_id(self):
        req = authzen.build_request(
            subject=_SUBJECT, action=_ACTION, resource=_RESOURCE, environment={},
        )
        assert req["resource"]["id"] == "resource"


# ── transport: authentication, TLS, pooling, logs ────────────────────────────────

def _capturing_transport(monkeypatch):
    """Capture every request the PDP client really sends (below httpx.Client)."""
    sent: list[httpx.Request] = []

    def handle_request(self, request):
        sent.append(request)
        return httpx.Response(200, json=_single_response("Permit"), request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    return sent


ALICE = {"username": "alice"}


class TestTransport:
    def _check(self, client):
        return client.check_permission(subject=ALICE, action={"name": "a"}, resource={"tool_name": "t"}, environment={})

    def test_headers_and_basic_auth_reach_the_pdp(self, monkeypatch):
        sent = _capturing_transport(monkeypatch)
        client = PdpClient("http://pdp", headers={"X-Api-Key": "k1"}, auth=("svc", "s3cret"))
        assert self._check(client) is True
        assert sent[0].headers["x-api-key"] == "k1"
        assert sent[0].headers["authorization"].startswith("Basic ")
        assert sent[0].headers["content-type"] == "application/json"

    def test_an_httpx_auth_can_set_a_bearer_token(self, monkeypatch):
        class Bearer(httpx.Auth):
            def auth_flow(self, request):
                request.headers["Authorization"] = "Bearer fresh"
                yield request

        sent = _capturing_transport(monkeypatch)
        self._check(PdpClient("http://pdp", auth=Bearer()))
        assert sent[0].headers["authorization"] == "Bearer fresh"

    def test_tls_settings_reach_httpx(self, monkeypatch):
        seen = {}
        real_init = httpx.Client.__init__

        def init(self, **kwargs):
            seen.update(kwargs)
            real_init(self, **{k: v for k, v in kwargs.items() if k not in ("verify", "cert")})

        monkeypatch.setattr(httpx.Client, "__init__", init)
        _capturing_transport(monkeypatch)
        self._check(PdpClient("http://pdp", verify="/etc/ca.pem", cert=("/c.pem", "/k.pem")))
        assert seen["verify"] == "/etc/ca.pem" and seen["cert"] == ("/c.pem", "/k.pem")

    def test_one_pooled_client_serves_every_call(self, monkeypatch):
        _capturing_transport(monkeypatch)
        client = PdpClient("http://pdp")
        self._check(client)
        first = client._http()
        self._check(client)
        assert client._http() is first

    def test_an_injected_client_is_used_and_left_open(self):
        sent = []
        injected = httpx.Client(transport=httpx.MockTransport(
            lambda request: sent.append(request) or httpx.Response(200, json=_single_response("Permit")),
        ))
        client = PdpClient("http://pdp", http_client=injected)
        assert self._check(client) is True and len(sent) == 1
        client.close()
        assert not injected.is_closed
        injected.close()

    def test_close_releases_a_client_it_built(self, monkeypatch):
        _capturing_transport(monkeypatch)
        with PdpClient("http://pdp") as client:
            self._check(client)
            built = client._http()
        assert built.is_closed

    def test_injected_client_and_settings_are_exclusive(self):
        with pytest.raises(ValueError, match="either http_client"):
            PdpClient("http://pdp", http_client=httpx.Client(), headers={"X": "1"})

    def test_connection_settings_flow_from_the_config(self, monkeypatch):
        from axiomatics_ai import AuthZConfig, PDPConnection

        sent = _capturing_transport(monkeypatch)
        config = AuthZConfig(connection=PDPConnection(base_url="http://pdp", headers={"X-Api-Key": "k2"}))
        self._check(config.build_pdp_client())
        assert sent[0].headers["x-api-key"] == "k2"

    def test_credentials_are_not_in_the_repr(self):
        from axiomatics_ai import PDPConnection

        text = repr(PDPConnection(headers={"X-Api-Key": "topsecret"}, auth=("u", "hunter2")))
        assert "topsecret" not in text and "hunter2" not in text

    def test_debug_logs_carry_no_request_or_response_body(self, monkeypatch, caplog):
        import logging

        _capturing_transport(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger="axiomatics_ai.core.pdp.client"):
            PdpClient("http://pdp").check_permission(
                subject={"username": "alice", "email": "alice@corp.example"}, action={"name": "a"},
                resource={"tool_name": "t"}, environment={},
            )
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "http://pdp" in text
        assert "alice@corp.example" not in text and "Permit" not in text


class TestEmptySubject:
    """A subject with no usable attributes (no token, or one that failed verification) is no
    identity to authorize: it never reaches the PDP, and fail_open decides."""

    def _reached(self, monkeypatch):
        calls = []
        patch_pdp_http(monkeypatch, "post", lambda url, **kw: calls.append(url))
        return calls

    @pytest.mark.parametrize("subject", [{}, {"username": None}])
    def test_single_check_denies_without_asking(self, monkeypatch, subject):
        calls = self._reached(monkeypatch)
        assert _client().check_permission(subject=subject, action={}, resource={"tool_name": "t"}, environment={}) is False
        assert calls == []

    def test_fail_open_permits_without_asking(self, monkeypatch):
        calls = self._reached(monkeypatch)
        client = PdpClient("http://pdp", fail_open=True)
        assert client.check_permission(subject={}, action={}, resource={}, environment={}) is True
        assert calls == []


class TestBatchAnswered:
    def test_answered_when_every_name_got_a_decision(self, monkeypatch):
        body = {"Response": [
            {"Decision": "Permit", "Category": [{"CategoryId": "urn:oasis:names:tc:xacml:3.0:attribute-category:resource",
                                                 "Attribute": [{"AttributeId": "tool_name", "Value": "a"}]}]},
            {"Decision": "Deny", "Category": [{"CategoryId": "urn:oasis:names:tc:xacml:3.0:attribute-category:resource",
                                               "Attribute": [{"AttributeId": "tool_name", "Value": "b"}]}]},
        ]}
        patch_pdp_http(monkeypatch, "post", lambda url, **kw: httpx.Response(200, json=body, request=httpx.Request("POST", url)))
        permitted = _client().check_permission_batch(
            subject=ALICE, action={}, resource_names=["a", "b"], environment={}, resource_attr_id="tool_name",
        )
        assert permitted == {"a"} and permitted.answered is True

    def test_not_answered_when_the_pdp_is_unreachable(self, monkeypatch):
        def boom(url, **kw):
            raise httpx.ConnectError("down")

        patch_pdp_http(monkeypatch, "post", boom)
        permitted = PdpClient("http://pdp", fail_open=True).check_permission_batch(
            subject=ALICE, action={}, resource_names=["a"], environment={}, resource_attr_id="tool_name",
        )
        assert permitted == {"a"} and permitted.answered is False


class TestAuthorizeRedirectsAndNonJson:
    """A redirect (3xx) or a 2xx without a JSON decision is a misconfiguration: it raises
    PdpRequestError even with fail_open=True, never a permit."""

    class _Resp:
        def __init__(self, status_code, *, payload=None, text=""):
            self.status_code = status_code
            self.text = text
            self._payload = payload

        def json(self):
            if self._payload is None:
                import json
                return json.loads(self.text)   # ValueError (JSONDecodeError) for HTML/plain text
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=None)

    def _client(self, monkeypatch, resp):
        patch_pdp_http(monkeypatch, "post", lambda url, **kwargs: resp)
        return PdpClient(base_url="http://ads-stub")

    def _check(self, client, *, fail_open):
        return client.check_permission(subject=_SUBJECT, action=_ACTION, resource=_RESOURCE,
                                       environment=_ENV, fail_open=fail_open)

    @pytest.mark.parametrize("status", [301, 302, 307])
    def test_redirect_raises_even_with_fail_open(self, monkeypatch, status):
        client = self._client(monkeypatch, self._Resp(status, text="<html>login</html>"))
        with pytest.raises(PdpRequestError, match=str(status)):
            self._check(client, fail_open=True)

    @pytest.mark.parametrize("text", ["<html><body>Sign in</body></html>", "OK", ""])
    def test_200_without_json_raises_even_with_fail_open(self, monkeypatch, text):
        client = self._client(monkeypatch, self._Resp(200, text=text))
        with pytest.raises(PdpRequestError, match="without a JSON decision"):
            self._check(client, fail_open=True)

    def test_200_json_still_decides(self, monkeypatch):
        client = self._client(monkeypatch, self._Resp(200, payload=_single_response("Deny")))
        assert self._check(client, fail_open=True) is False
        client = self._client(monkeypatch, self._Resp(200, payload=_single_response("Permit")))
        assert self._check(client, fail_open=False) is True

    def test_5xx_still_fails_open(self, monkeypatch):
        client = self._client(monkeypatch, self._Resp(502, text="<html>bad gateway</html>"))
        assert self._check(client, fail_open=True) is True
        assert self._check(client, fail_open=False) is False

    def test_lowercase_deny_from_the_pdp_is_a_deny_under_fail_open(self, monkeypatch):
        client = self._client(monkeypatch, self._Resp(200, payload=_single_response("deny")))
        assert self._check(client, fail_open=True) is False

    def test_a_json_string_body_is_json(self, monkeypatch):
        """A CAQ answer can be a bare JSON string; it is not refused as a non-JSON body."""
        client = self._client(monkeypatch, self._Resp(200, text='"level <= 3"'))
        assert client.get_simplified_constraints(
            provided_attributes=[{"attribute": "username", "values": ["a"]}],
        ) == "level <= 3"


class TestDecisionParse:
    @pytest.mark.parametrize("value, expected", [
        ("Permit", Decision.PERMIT), ("PERMIT", Decision.PERMIT), ("permit", Decision.PERMIT),
        ("deny", Decision.DENY), ("Deny", Decision.DENY), ("DENY", Decision.DENY),
        (" NotApplicable ", Decision.NOT_APPLICABLE), ("notapplicable", Decision.NOT_APPLICABLE),
        ("INDETERMINATE", Decision.INDETERMINATE),
    ])
    def test_case_and_whitespace_insensitive(self, value, expected):
        assert Decision.parse(value) is expected

    @pytest.mark.parametrize("value", [None, True, False, 1, 0, ["Permit"], {"Decision": "Permit"},
                                       "", "  ", "Allow", "Not Applicable", "Permitted"])
    def test_non_strings_and_unknown_values_are_none(self, value):
        assert Decision.parse(value) is None
