"""
Obligations and advice: parsing both wire protocols, what the PEP does with them
(``resolve``), and the approval memory. No network — the PDP's HTTP call is stubbed.
"""
from __future__ import annotations

from tests.pdp_fakes import patch_pdp_http

import logging

import pytest

import axiomatics_ai.core.pdp.obligations as obligations_module
from axiomatics_ai.core.pdp import (
    Advice,
    ApprovalConfig,
    ApprovalDecision,
    ApprovalRequest,
    Decision,
    Obligation,
    ObligationsAndAdviceConfig,
    Outcome,
    PdpClient,
    PdpDecision,
    authzen,
    xacml,
)
from axiomatics_ai.core.pdp.obligations import ApprovalMemory, ObligationContext, args_hash, resolve, ttl_seconds

_CTX = ObligationContext(subject={"user": "alice"}, action={"name": "tool_invocation"}, resource={"tool_name": "pay"}, environment={})
_APPROVAL_ADVICE = Advice("require-approval", {"reason": "Payments need approval", "ttl": "300"})


def _resolve(decision, *, permitted=None, config=None, approval_possible=False):
    if permitted is None:
        permitted = decision.decision is Decision.PERMIT
    return resolve(decision, permitted=permitted, context=_CTX,
                   config=config or ObligationsAndAdviceConfig(), approval_possible=approval_possible)


# ── Parsing ───────────────────────────────────────────────────────────────────────

class TestXacmlParsing:
    def test_obligations_and_advice(self):
        result = {
            "Decision": "Deny",
            "Obligations": [{"Id": "log", "AttributeAssignment": [{"AttributeId": "level", "Value": "audit"}]}],
            "AssociatedAdvice": [{"Id": "require-approval", "AttributeAssignment": [
                {"AttributeId": "reason", "Value": "why"}, {"AttributeId": "ttl", "Value": "300"},
            ]}],
        }
        assert xacml.parse_result(result) == PdpDecision(
            Decision.DENY, (Obligation("log", {"level": "audit"}),),
            (Advice("require-approval", {"reason": "why", "ttl": "300"}),),
        )

    def test_attribute_assigned_twice_becomes_a_list(self):
        result = {"Decision": "Permit", "Obligations": [{"Id": "notify", "AttributeAssignment": [
            {"AttributeId": "to", "Value": "a"}, {"AttributeId": "to", "Value": "b"}, {"AttributeId": "to", "Value": "c"},
        ]}]}
        assert xacml.parse_result(result).obligations == (Obligation("notify", {"to": ["a", "b", "c"]}),)

    def test_malformed_entries_are_skipped(self):
        result = {"Decision": "Permit", "Obligations": [{"AttributeAssignment": []}, "junk", {"Id": "ok"}]}
        assert xacml.parse_result(result).obligations == (Obligation("ok", {}),)

    @pytest.mark.parametrize("response", [None, {}, {"Response": []}, {"Response": "x"}])
    def test_no_result_is_no_decision(self, response):
        assert xacml.parse_response(response) == PdpDecision(None)

    def test_unknown_decision_value_is_none(self):
        assert xacml.parse_result({"Decision": "Maybe"}).decision is None


class TestAuthzenParsing:
    def test_reads_ads_context_xacml(self):
        result = {"decision": False, "context": {"xacml": {
            "decision": "Deny", "advice": [{"id": "require-approval", "attributes": {"ttl": "300"}}],
            "obligations": [{"id": "log", "attributes": {"level": "audit"}}],
        }}}
        assert authzen.parse_result(result) == PdpDecision(
            Decision.DENY, (Obligation("log", {"level": "audit"}),), (Advice("require-approval", {"ttl": "300"}),),
        )

    def test_plain_authzen_pdp_gives_a_bare_decision(self):
        assert authzen.parse_result({"decision": True}) == PdpDecision(Decision.PERMIT)

    def test_batch_by_position(self):
        response = {"evaluations": [
            {"decision": True},
            {"decision": True, "context": {"xacml": {"decision": "Permit", "obligations": [{"id": "log", "attributes": {}}]}}},
            {"decision": False},
        ]}
        decisions = authzen.parse_batch_response(response, ["a", "b", "c"])
        assert decisions["a"] == PdpDecision(Decision.PERMIT)
        assert decisions["b"].obligations == (Obligation("log", {}),)
        assert decisions["c"] == PdpDecision(Decision.DENY)

    def test_a_batch_of_another_length_gives_no_decision(self):
        # Matched by position: with an entry missing, "b" would get c's decision.
        short = {"evaluations": [{"decision": False}, {"decision": True}]}
        long = {"evaluations": [{"decision": True}] * 4}
        for response in (short, long):
            decisions = authzen.parse_batch_response(response, ["a", "b", "c"])
            assert all(d == PdpDecision(None) for d in decisions.values())


# ── resolve ───────────────────────────────────────────────────────────────────────

class TestResolveObligations:
    def test_plain_permit_and_deny(self):
        assert _resolve(PdpDecision(Decision.PERMIT)).outcome is Outcome.PERMIT
        assert _resolve(PdpDecision(Decision.DENY)).outcome is Outcome.DENY

    def test_unknown_obligation_on_permit_denies_by_default(self):
        resolution = _resolve(PdpDecision(Decision.PERMIT, (Obligation("mystery"),)))
        assert resolution.outcome is Outcome.DENY
        assert resolution.detail["unknown_obligations"] == ["mystery"]

    def test_on_unknown_ignore_permits_and_warns_once(self, monkeypatch, caplog):
        monkeypatch.setattr(obligations_module, "_warned_unknown", set())
        config = ObligationsAndAdviceConfig(on_unknown="ignore")
        decision = PdpDecision(Decision.PERMIT, (Obligation("mystery"),))
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.pdp.obligations"):
            assert _resolve(decision, config=config).permitted
            assert _resolve(decision, config=config).permitted
        assert sum("mystery" in r.getMessage() for r in caplog.records) == 1

    def test_unknown_advice_is_ignored(self):
        assert _resolve(PdpDecision(Decision.PERMIT, advice=(Advice("mystery"),))).permitted

    def test_handler_runs_with_the_request(self):
        seen = []
        config = ObligationsAndAdviceConfig(handlers={"log": lambda o, ctx: seen.append((o, ctx))})
        assert _resolve(PdpDecision(Decision.PERMIT, (Obligation("log", {"level": "audit"}),)), config=config).permitted
        assert seen == [(Obligation("log", {"level": "audit"}), _CTX)]

    def test_none_handler_accepts_without_running_anything(self):
        config = ObligationsAndAdviceConfig(handlers={"elsewhere": None})
        assert _resolve(PdpDecision(Decision.PERMIT, (Obligation("elsewhere"),)), config=config).permitted

    @pytest.mark.parametrize("handler", [lambda o, c: False, lambda o, c: 1 / 0])
    def test_failing_handler_denies(self, handler):
        config = ObligationsAndAdviceConfig(handlers={"log": handler})
        resolution = _resolve(PdpDecision(Decision.PERMIT, (Obligation("log"),)), config=config)
        assert resolution.outcome is Outcome.DENY
        assert resolution.detail["failed_obligations"] == ["log"]

    def test_handlers_also_run_on_deny(self):
        seen = []
        config = ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(o.id)})
        assert not _resolve(PdpDecision(Decision.DENY, (Obligation("log"),)), config=config).permitted
        assert seen == ["log"]

    def test_fail_open_permit_carries_no_obligations(self):
        assert _resolve(PdpDecision(None), permitted=True).permitted


class TestResolveApproval:
    def test_deny_with_approval_advice_requires_approval(self):
        resolution = _resolve(PdpDecision(Decision.DENY, advice=(_APPROVAL_ADVICE,)), approval_possible=True)
        assert resolution.outcome is Outcome.APPROVAL_REQUIRED
        assert resolution.approval == (_APPROVAL_ADVICE,)

    def test_permit_with_approval_obligation_requires_approval(self):
        obligation = Obligation("require-approval")
        resolution = _resolve(PdpDecision(Decision.PERMIT, (obligation,)), approval_possible=True)
        assert resolution.outcome is Outcome.APPROVAL_REQUIRED
        assert resolution.approval == (obligation,)

    @pytest.mark.parametrize("decision", [
        PdpDecision(Decision.DENY, advice=(_APPROVAL_ADVICE,)),
        PdpDecision(Decision.PERMIT, (Obligation("require-approval"),)),
    ])
    def test_where_approval_is_not_possible_it_is_a_deny(self, decision):
        resolution = _resolve(decision, approval_possible=False)
        assert resolution.outcome is Outcome.DENY
        assert resolution.detail["approval"] == "can't be given here"

    def test_unknown_obligation_beats_approval(self):
        decision = PdpDecision(Decision.PERMIT, (Obligation("require-approval"), Obligation("mystery")))
        assert _resolve(decision, approval_possible=True).outcome is Outcome.DENY

    def test_handlers_wait_until_after_approval(self):
        seen = []
        config = ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(o.id)})
        decision = PdpDecision(Decision.PERMIT, (Obligation("require-approval"), Obligation("log")))
        assert _resolve(decision, config=config, approval_possible=True).outcome is Outcome.APPROVAL_REQUIRED
        assert seen == []

    def test_custom_approval_id(self):
        config = ObligationsAndAdviceConfig(approval=ApprovalConfig(obligation_or_advice_ids={"four-eyes"}))
        decision = PdpDecision(Decision.DENY, advice=(Advice("four-eyes"),))
        assert _resolve(decision, config=config, approval_possible=True).outcome is Outcome.APPROVAL_REQUIRED
        assert _resolve(PdpDecision(Decision.DENY, advice=(_APPROVAL_ADVICE,)), config=config,
                        approval_possible=True).outcome is Outcome.DENY


class TestConfigValidation:
    def test_on_unknown_must_be_deny_or_ignore(self):
        with pytest.raises(ValueError, match="on_unknown"):
            ObligationsAndAdviceConfig(on_unknown="warn")

    def test_approval_id_cannot_have_a_handler(self):
        with pytest.raises(ValueError, match="approval id"):
            ObligationsAndAdviceConfig(handlers={"require-approval": None})

    def test_handler_must_be_callable(self):
        with pytest.raises(ValueError, match="callable"):
            ObligationsAndAdviceConfig(handlers={"log": "not a function"})

    @pytest.mark.parametrize("kwargs", [
        {"obligation_or_advice_ids": set()}, {"obligation_or_advice_ids": {""}},
        {"obligation_or_advice_ids": "require-approval"}, {"attribute": ""},
    ])
    def test_approval_config_needs_names(self, kwargs):
        with pytest.raises(ValueError):
            ApprovalConfig(**kwargs)


# ── PdpClient applies them on every bool path ─────────────────────────────────────

class _Response:
    def __init__(self, body):
        self.status_code = 200
        self._body = body
        self.text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _stub(monkeypatch, body):
    patch_pdp_http(monkeypatch, "post", lambda *a, **kw: _Response(body))


class TestClientBoolPaths:
    def test_check_permission_denies_on_unknown_obligation(self, monkeypatch):
        _stub(monkeypatch, {"Response": [{"Decision": "Permit", "Obligations": [{"Id": "mystery"}]}]})
        assert PdpClient("http://pdp").check_permission(subject={"user_id": "alice"}, action={}, resource={}, environment={}) is False

    def test_check_permission_runs_handlers(self, monkeypatch):
        _stub(monkeypatch, {"Response": [{"Decision": "Permit", "Obligations": [{"Id": "log"}]}]})
        seen = []
        client = PdpClient("http://pdp", obligations=ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(c.resource)}))
        assert client.check_permission(subject={"user_id": "alice"}, action={}, resource={"tool_name": "t"}, environment={}) is True
        assert seen == [{"tool_name": "t"}]

    def test_check_permission_treats_approval_as_deny(self, monkeypatch):
        _stub(monkeypatch, {"Response": [{"Decision": "Permit", "Obligations": [{"Id": "require-approval"}]}]})
        assert PdpClient("http://pdp").check_permission(subject={"user_id": "alice"}, action={}, resource={}, environment={}) is False

    def test_batch_resolves_each_name_with_its_own_resource(self, monkeypatch):
        def result(name, obligations):
            return {"Decision": "Permit", "Obligations": obligations,
                    "Category": [{"Attribute": [{"AttributeId": "tool_name", "Value": [name]}]}]}
        _stub(monkeypatch, {"Response": [result("a", []), result("b", [{"Id": "mystery"}]), result("c", [{"Id": "log"}])]})
        seen = []
        client = PdpClient("http://pdp", obligations=ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(c.resource)}))
        permitted = client.check_permission_batch(
            subject={"user": "alice"}, action={}, resource_names=["a", "b", "c"], environment={},
            resource_attr_id="tool_name", resource_extra_attrs=[{}, {}, {"doc": 3}],
        )
        assert permitted == {"a", "c"}
        assert seen == [{"tool_name": "c", "doc": 3}]


# ── Helpers and approval memory ───────────────────────────────────────────────────

def _request(args=None, *, ttl=60.0, subject=None) -> ApprovalRequest:
    args = args if args is not None else {"amount": 5}
    return ApprovalRequest(
        request_id="r", tool_name="pay", args=args, args_hash=args_hash({"tool_name": "pay", **args}),
        reasons=(), ttl_seconds=ttl, subject=subject or {"user": "alice"}, asked_by=(_APPROVAL_ADVICE,),
    )


class TestHelpers:
    def test_args_hash_is_order_independent_and_value_sensitive(self):
        assert args_hash({"a": 1, "b": 2}) == args_hash({"b": 2, "a": 1})
        assert args_hash({"a": 1}) != args_hash({"a": 2})

    @pytest.mark.parametrize(("attributes", "expected"), [
        ({"ttl": "300"}, 300.0), ({"ttl": 1.5}, 1.5), ({"ttl": ["60"]}, 60.0),
        ({}, None), ({"ttl": "soon"}, None), ({"ttl": "0"}, None), ({"ttl": "-5"}, None),
    ])
    def test_ttl_seconds(self, attributes, expected):
        assert ttl_seconds((Advice("require-approval", attributes),)) == expected


class TestApprovalMemory:
    def test_remembers_for_the_same_caller_and_arguments(self):
        memory, answer = ApprovalMemory(), ApprovalDecision(True, approver="boss")
        memory.remember(_request(), answer)
        assert memory.recall(_request()) == answer
        assert memory.recall(_request({"amount": 6})) is None
        assert memory.recall(_request(subject={"user": "bob"})) is None

    def test_single_use_and_rejections_are_not_stored(self):
        memory = ApprovalMemory()
        memory.remember(_request(ttl=None), ApprovalDecision(True))
        memory.remember(_request(), ApprovalDecision(False))
        assert memory.recall(_request()) is None

    def test_expires(self, monkeypatch):
        memory, now = ApprovalMemory(), [1000.0]
        monkeypatch.setattr(obligations_module.time, "monotonic", lambda: now[0])
        memory.remember(_request(ttl=10), ApprovalDecision(True))
        now[0] += 11
        assert memory.recall(_request(ttl=10)) is None

    def test_invalidate_by_subject(self):
        memory = ApprovalMemory()
        memory.remember(_request(subject={"user": "alice"}), ApprovalDecision(True))
        memory.remember(_request(subject={"user": "bob"}), ApprovalDecision(True))
        assert memory.invalidate(user="alice") == 1
        assert memory.recall(_request(subject={"user": "alice"})) is None
        assert memory.recall(_request(subject={"user": "bob"})) is not None
        assert memory.invalidate() == 1


class TestSeveralApprovals:
    def test_several_ids_are_all_approval(self):
        config = ObligationsAndAdviceConfig(approval=ApprovalConfig(obligation_or_advice_ids={"require-approval", "four-eyes"}))
        decision = PdpDecision(Decision.PERMIT, (Obligation("require-approval"), Obligation("four-eyes")))
        resolution = _resolve(decision, config=config, approval_possible=True)
        assert resolution.outcome is Outcome.APPROVAL_REQUIRED
        assert [a.id for a in resolution.approval] == ["require-approval", "four-eyes"]

    def test_an_id_outside_the_set_is_an_unknown_obligation(self):
        decision = PdpDecision(Decision.PERMIT, (Obligation("require-approval"), Obligation("four-eyes")))
        assert _resolve(decision, approval_possible=True).outcome is Outcome.DENY

    def test_deny_collects_advice_and_obligations(self):
        decision = PdpDecision(Decision.DENY, (Obligation("require-approval"),), (Advice("require-approval"),))
        assert len(_resolve(decision, approval_possible=True).approval) == 2

    def test_handler_for_an_approval_id_is_rejected(self):
        with pytest.raises(ValueError, match="approval ids"):
            ObligationsAndAdviceConfig(
                handlers={"four-eyes": None},
                approval=ApprovalConfig(obligation_or_advice_ids={"require-approval", "four-eyes"}),
            )

    @pytest.mark.parametrize(("ttls", "expected"), [
        (["300", "60"], 60.0), (["300", None], None), ([None], None), (["30"], 30.0),
    ])
    def test_smallest_ttl_wins_and_any_single_use_makes_it_single_use(self, ttls, expected):
        items = tuple(Advice("require-approval", {} if t is None else {"ttl": t}) for t in ttls)
        assert ttl_seconds(items) == expected

    def test_reasons_are_collected_once_each(self):
        from axiomatics_ai.core.pdp.obligations import reasons
        items = (Advice("a", {"reason": "Over 10k"}), Obligation("b", {"reason": ["Finance", "Over 10k"]}), Advice("c"))
        assert reasons(items) == ("Over 10k", "Finance")

    def test_run_handlers_false_only_looks(self):
        seen = []
        config = ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(1) or False})
        decision = PdpDecision(Decision.PERMIT, (Obligation("log"),))
        assert resolve(decision, permitted=True, context=_CTX, config=config, approval_possible=False, run_handlers=False).permitted
        assert seen == []
