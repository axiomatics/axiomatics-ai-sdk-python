"""
Human approval of tool calls in the enforcement core (``authorized_tool_invocation``): the
approver is asked when the policy requires it, the call is evaluated again with the approval
attribute, approvals with a ttl are reused, and everything else fails closed. The PDP is
stubbed at ``enforcement.evaluate_tool_call``.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, AuthzHook, standard_tool_enforcement
from axiomatics_ai.core.enforcement import authorized_call, authorized_tool_invocation
from axiomatics_ai.core.pdp import (
    Advice,
    ApprovalConfig,
    ApprovalDecision,
    Decision,
    Obligation,
    ObligationsAndAdviceConfig,
    PdpDecision,
)

SUBJECT = {"user_id": "alice"}
ACTION = {"name": "tool_invocation"}
RESOURCE = {"tool_name": "pay_invoice", "invoice_id": "INV-43", "amount": 500}
ASK = PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"reason": "Over 100 EUR"}),))
PERMIT = PdpDecision(Decision.PERMIT)
APPROVED = ApprovalDecision(True, approver="manager")


class _Pdp:
    """The PDP: asks for approval until the approval attribute is set, then answers
    ``after`` (Permit by default). Records each request's environment."""

    def __init__(self, first=ASK, after=PERMIT, attribute="approval_granted"):
        self.first, self.after, self.attribute = first, after, attribute
        self.environments: list[dict] = []

    def __call__(self, pdp, *, subject, action, resource, environment):
        self.environments.append(dict(environment))
        return self.after if environment.get(self.attribute) is True else self.first


class _Approver:
    def __init__(self, answer=APPROVED):
        self.answer = answer
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class _Recorder(AuthzHook):
    def __init__(self):
        self.decisions = []

    def on_decision(self, event):
        self.decisions.append(event)


@pytest.fixture
def pdp(monkeypatch):
    fake = _Pdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
    return fake


def _config(approver=None, *, hooks=None, **approval) -> AuthZConfig:
    return AuthZConfig(
        enforcement=standard_tool_enforcement(),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver, **approval)),
        hooks=hooks or [],
    )


def _decide(config, resource=RESOURCE, *, cache=None, subject=SUBJECT, **kwargs):
    return authorized_tool_invocation(
        pdp=config.build_pdp_client(), cache=cache, config=config, surface="test",
        subject=subject, action=ACTION, resource=resource, environment={"channel": "chat"}, user_messages=[],
        **kwargs,
    )


def _call(config, resource=RESOURCE, *, cache=None, subject=SUBJECT) -> bool:
    return _decide(config, resource, cache=cache, subject=subject).permitted


class TestApprovalFlow:
    def test_approved_call_is_evaluated_again_with_the_attribute(self, pdp):
        approver = _Approver()
        assert _call(_config(approver)) is True
        assert pdp.environments == [{"channel": "chat"}, {"channel": "chat", "approval_granted": True}]

    def test_the_approver_sees_the_call(self, pdp):
        approver = _Approver()
        _call(_config(approver))
        (request,) = approver.requests
        assert request.tool_name == "pay_invoice"
        assert request.args == {"invoice_id": "INV-43", "amount": 500}
        assert request.reason == "Over 100 EUR"
        assert request.subject == SUBJECT
        assert request.ttl_seconds is None
        assert request.asked_by == ASK.advice
        assert request.reasons == ("Over 100 EUR",)

    def test_permit_with_approval_obligation_works_the_same(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.PERMIT, (Obligation("require-approval"),)))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        assert _call(_config(_Approver())) is True

    def test_rejection_denies_without_a_second_evaluation(self, pdp):
        assert _call(_config(_Approver(ApprovalDecision(False, approver="manager")))) is False
        assert len(pdp.environments) == 1

    def test_second_decision_is_final(self, monkeypatch):
        fake = _Pdp(after=PdpDecision(Decision.DENY))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        assert _call(_config(_Approver())) is False

    def test_asking_again_after_approval_is_a_deny(self, monkeypatch):
        fake = _Pdp(after=ASK)   # a policy that never stops asking
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        approver = _Approver()
        assert _call(_config(approver)) is False
        assert len(approver.requests) == 1

    def test_no_approver_denies_with_a_warning(self, pdp, caplog):
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            assert _call(_config()) is False
        assert any("no approver is configured" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("answer", [RuntimeError("down"), "yes", None])
    def test_approver_failure_or_garbage_is_a_rejection(self, pdp, answer):
        assert _call(_config(_Approver(answer))) is False
        assert len(pdp.environments) == 1

    def test_custom_attribute_and_id(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.DENY, advice=(Advice("four-eyes"),)), attribute="human_ok")
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        assert _call(_config(_Approver(), obligation_or_advice_ids={"four-eyes"}, attribute="human_ok")) is True
        assert fake.environments[-1]["human_ok"] is True

    def test_authorized_call_never_waits_for_approval(self, pdp):
        config = _config(_Approver())
        assert authorized_call(
            pdp=config.build_pdp_client(), cache=None, config=config, surface="test",
            subject=SUBJECT, action=ACTION, resource=RESOURCE, environment={},
        ) is False


class TestAsyncApprover:
    @staticmethod
    def _async_approver(requests):
        async def approver(request):
            await asyncio.sleep(0)
            requests.append(request)
            return ApprovalDecision(True, approver="async")
        return approver

    def test_from_a_thread_without_a_loop(self, pdp):
        requests = []
        config = _config(self._async_approver(requests))

        async def main():
            return await asyncio.to_thread(_call, config)
        assert asyncio.run(main()) is True
        assert len(requests) == 1

    def test_from_sync_code_inside_a_running_loop(self, pdp):
        requests = []
        config = _config(self._async_approver(requests))

        async def main():
            return _call(config)   # a sync framework hook called on the loop thread
        assert asyncio.run(main()) is True
        assert len(requests) == 1


class TestReuse:
    @pytest.fixture
    def ttl_pdp(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"ttl": "300"}),)))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        return fake

    def test_approval_with_ttl_is_reused_for_the_same_call(self, ttl_pdp):
        approver = _Approver()
        config = _config(approver)
        assert _call(config) and _call(config)
        assert len(approver.requests) == 1

    def test_different_arguments_need_a_new_approval(self, ttl_pdp):
        approver = _Approver()
        config = _config(approver)
        _call(config)
        _call(config, {**RESOURCE, "amount": 900})
        assert len(approver.requests) == 2

    def test_other_caller_needs_a_new_approval(self, ttl_pdp):
        approver = _Approver()
        config = _config(approver)
        _call(config)
        _call(config, subject={"user_id": "bob"})
        assert len(approver.requests) == 2

    def test_single_use_by_default(self, pdp):
        approver = _Approver()
        config = _config(approver)
        _call(config)
        _call(config)
        assert len(approver.requests) == 2

    def test_invalidate_decisions_drops_approvals(self, ttl_pdp):
        approver = _Approver()
        config = _config(approver)
        _call(config)
        config.invalidate_decisions(user_id="alice")
        _call(config)
        assert len(approver.requests) == 2

    def test_rejection_is_not_remembered(self, ttl_pdp):
        approver = _Approver(ApprovalDecision(False))
        config = _config(approver)
        _call(config)
        _call(config)
        assert len(approver.requests) == 2


class TestEventsAndCache:
    def test_both_decisions_and_the_answer_are_logged_under_one_correlation_id(self, pdp):
        recorder = _Recorder()
        _call(_config(_Approver(), hooks=[recorder]))
        assert [e.decision for e in recorder.decisions] == ["approval_required", "approved", "permit"]
        assert len({e.correlation_id for e in recorder.decisions}) == 1
        answer = recorder.decisions[1].detail
        assert answer["approver"] == "manager" and answer["source"] == "approver" and answer["request_id"]
        assert recorder.decisions[0].detail["advice"] == [
            {"id": "require-approval", "attributes": {"reason": "Over 100 EUR"}},
        ]

    def test_rejection_event(self, pdp):
        recorder = _Recorder()
        _call(_config(_Approver(ApprovalDecision(False, note="too much")), hooks=[recorder]))
        assert [e.decision for e in recorder.decisions] == ["approval_required", "rejected", "deny"]
        assert recorder.decisions[1].detail["note"] == "too much"
        assert recorder.decisions[2].detail == {
            "not_approved": True, "request_id": recorder.decisions[1].detail["request_id"],
        }

    def test_cached_decisions_still_run_handlers(self, monkeypatch):
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call",
                            _Pdp(first=PdpDecision(Decision.PERMIT, (Obligation("log"),))))
        seen = []
        config = AuthZConfig(
            enforcement=standard_tool_enforcement(),
            obligations_and_advice=ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(o.id)}),
        )
        cache = config.build_cache()
        assert _call(config, cache=cache) and _call(config, cache=cache)
        assert seen == ["log", "log"]

    def test_unreachable_pdp_is_not_cached(self, monkeypatch):
        calls = []

        def unreachable(pdp, **request):
            calls.append(1)
            return PdpDecision(None)
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", unreachable)
        config = AuthZConfig(enforcement=standard_tool_enforcement())
        cache = config.build_cache()
        assert _call(config, cache=cache) is False
        assert _call(config, cache=cache) is False
        assert len(calls) == 2


class TestTwoPolicyForms:
    def test_permit_with_obligation_needs_one_pdp_call(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.PERMIT, (Obligation("require-approval"),)))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        recorder = _Recorder()
        assert _call(_config(_Approver(), hooks=[recorder])) is True
        assert len(fake.environments) == 1
        assert [e.decision for e in recorder.decisions] == ["approval_required", "approved", "permit"]
        assert recorder.decisions[-1].detail["evaluated_again"] is False

    def test_deny_with_advice_asks_the_pdp_again(self, pdp):
        recorder = _Recorder()
        assert _call(_config(_Approver(), hooks=[recorder])) is True
        assert len(pdp.environments) == 2
        assert recorder.decisions[-1].detail["evaluated_again"] is True

    def test_other_obligations_are_handled_after_approval(self, monkeypatch):
        seen = []
        fake = _Pdp(first=PdpDecision(Decision.PERMIT, (Obligation("require-approval"), Obligation("log"))))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)

        def approver(request):
            seen.append("asked")
            return APPROVED
        config = AuthZConfig(
            enforcement=standard_tool_enforcement(),
            obligations_and_advice=ObligationsAndAdviceConfig(
                handlers={"log": lambda o, c: seen.append("log")}, approval=ApprovalConfig(approver=approver),
            ),
        )
        assert _call(config) is True
        assert seen == ["asked", "log"]

    def test_failing_obligation_after_approval_denies(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.PERMIT, (Obligation("require-approval"), Obligation("log"))))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        config = AuthZConfig(
            enforcement=standard_tool_enforcement(),
            obligations_and_advice=ObligationsAndAdviceConfig(
                handlers={"log": lambda o, c: False}, approval=ApprovalConfig(approver=_Approver()),
            ),
        )
        assert _call(config) is False


class TestFrameworkInterrupts:
    def test_pending_when_the_engine_can_interrupt_and_nothing_answered(self, pdp):
        recorder = _Recorder()
        decision = _decide(_config(hooks=[recorder]), can_interrupt=True)
        assert decision.permitted is False
        assert decision.pending.tool_name == "pay_invoice"
        assert [e.decision for e in recorder.decisions] == ["approval_required"]   # nothing answered yet

    def test_the_approver_answers_before_any_interrupt(self, pdp):
        decision = _decide(_config(_Approver()), can_interrupt=True)
        assert decision.permitted is True and decision.pending is None

    @pytest.mark.parametrize(("answer", "permitted", "label"), [
        (ApprovalDecision(True, approver="alice"), True, "approved"),
        (ApprovalDecision(False), False, "rejected"),
    ])
    def test_resume_with_the_answer(self, pdp, answer, permitted, label):
        recorder = _Recorder()
        decision = _decide(_config(hooks=[recorder]), approval=answer, can_interrupt=True)
        assert decision.permitted is permitted and decision.pending is None
        # the "approval required" outcome was logged when the engine paused; not again here
        assert recorder.decisions[0].decision == label
        assert recorder.decisions[0].detail["source"] == "interrupt"

    def test_without_interrupt_or_approver_the_rejection_is_logged(self, pdp):
        recorder = _Recorder()
        decision = _decide(_config(hooks=[recorder]))
        assert decision.permitted is False and decision.pending is None
        assert [e.decision for e in recorder.decisions] == ["approval_required", "rejected", "deny"]
        assert recorder.decisions[1].detail["source"] == "none"

    def test_remembered_approval_answers_before_an_interrupt(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"ttl": "300"}),)))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        config = _config()
        assert _decide(config, approval=APPROVED, can_interrupt=True).permitted
        assert _decide(config, can_interrupt=True).permitted   # reused, no new pause


class TestPayloadAndAnswers:
    def test_payload_is_json_safe_and_leaves_out_the_subject(self, pdp):
        import json
        request = _decide(_config(), {**RESOURCE, "when": object()}, can_interrupt=True).pending
        payload = request.to_payload()
        json.dumps(payload)
        assert payload["type"] == "axiomatics.approval_request"
        assert payload["tool_name"] == "pay_invoice" and payload["reason"] == "Over 100 EUR"
        assert "subject" not in payload and payload["request_id"] == request.request_id

    @pytest.mark.parametrize(("answer", "approved"), [
        (True, True), (False, False), ({"approved": True, "approver": "bob"}, True),
        ({"approved": "yes"}, False), ({}, False), ("y", False), (None, False), (1, False),
        (ApprovalDecision(True), True),
    ])
    def test_from_answer_only_an_explicit_true_approves(self, answer, approved):
        assert ApprovalDecision.from_answer(answer).approved is approved

    def test_from_answer_keeps_approver_and_note(self):
        decision = ApprovalDecision.from_answer({"approved": True, "approver": "bob", "note": "ok"})
        assert (decision.approver, decision.note) == ("bob", "ok")


class TestDenialMessage:
    def test_policy_denial_and_rejection_read_differently(self):
        from axiomatics_ai.core.enforcement import ToolCallDecision, denial_message
        assert denial_message("pay", ToolCallDecision(permitted=False)) == "Access denied: not authorized to call 'pay'."
        assert denial_message("pay", ToolCallDecision(permitted=False, not_approved=True)) == "'pay' was not approved, so it did not run."

    def test_rejection_sets_not_approved(self, pdp):
        assert _decide(_config(_Approver(ApprovalDecision(False)))).not_approved is True
        assert _decide(_config()).not_approved is True                     # no approver at all
        assert _decide(_config(_Approver()), {**RESOURCE, "tool_name": "x"}).not_approved is False

    def test_policy_change_between_pause_and_answer_is_logged(self, monkeypatch):
        # Paused for approval; by the time the answer comes back the policy simply denies.
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", _Pdp(first=PdpDecision(Decision.DENY)))
        recorder = _Recorder()
        decision = _decide(_config(hooks=[recorder]), approval=APPROVED, can_interrupt=True)
        assert decision.permitted is False
        assert [e.decision for e in recorder.decisions] == ["deny"]


class TestCombinedApprovals:
    def test_one_request_one_answer_for_several_approval_obligations(self, monkeypatch):
        first = PdpDecision(Decision.PERMIT, (
            Obligation("require-approval", {"reason": "Over 10k", "ttl": "300"}),
            Obligation("four-eyes", {"reason": "Finance must agree", "ttl": "60", "approver_role": "finance"}),
        ))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", _Pdp(first=first))
        approver = _Approver()
        config = _config(approver, obligation_or_advice_ids={"require-approval", "four-eyes"})
        assert _call(config) is True
        (request,) = approver.requests
        assert request.reasons == ("Over 10k", "Finance must agree")
        assert request.reason == "Over 10k; Finance must agree"
        assert request.ttl_seconds == 60.0
        assert request.asked_by[1].attributes["approver_role"] == "finance"
        assert request.to_payload()["reasons"] == ["Over 10k", "Finance must agree"]


class TestPreCheck:
    """``needs_human_approval``: the pre-check behind OpenAI's needs_approval."""

    def _check(self, config, resource=RESOURCE):
        from axiomatics_ai.core.enforcement import needs_human_approval
        return needs_human_approval(
            pdp=config.build_pdp_client(), cache=None, config=config, surface="test",
            subject=SUBJECT, action=ACTION, resource=resource, environment={}, user_messages=[],
        )

    def test_returns_the_request_and_logs_only_approval_required(self, pdp):
        recorder = _Recorder()
        request = self._check(_config(hooks=[recorder]))
        assert request.tool_name == "pay_invoice"
        assert [e.decision for e in recorder.decisions] == ["approval_required"]

    def test_a_plain_permit_is_not_logged_and_runs_no_handler(self, monkeypatch):
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", _Pdp(first=PdpDecision(Decision.PERMIT, (Obligation("log"),))))
        seen, recorder = [], _Recorder()
        config = AuthZConfig(
            enforcement=standard_tool_enforcement(), hooks=[recorder],
            obligations_and_advice=ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(1)}),
        )
        assert self._check(config) is None
        assert seen == [] and recorder.decisions == []

    def test_an_approver_means_no_pause(self, pdp):
        assert self._check(_config(_Approver())) is None

    def test_a_remembered_approval_means_no_pause(self, monkeypatch):
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call",
                            _Pdp(first=PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"ttl": "300"}),))))
        config = _config()
        assert _decide(config, approval=APPROVED, can_interrupt=True).permitted
        assert self._check(config) is None


class TestApprovalRequestAndAuditTrail:
    """The demo's bug reports: what the approver is shown, and whether the audit trail tells
    one call's approval story end to end."""

    def test_the_approver_sees_the_calls_own_arguments(self, pdp):
        # A context provider that forwards only some arguments must not hide the rest from the
        # human approving the call.
        approver = _Approver()
        _call_args = {"invoice_id": "INV-43", "note": "urgent, pay today"}
        _decide(_config(approver), args=_call_args)
        assert approver.requests[0].args == _call_args

    def test_an_unforwarded_argument_still_binds_the_approval(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"ttl": "300"}),)))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        approver = _Approver()
        config = _config(approver)
        _decide(config, args={"invoice_id": "INV-43", "iban": "A"})
        _decide(config, args={"invoice_id": "INV-43", "iban": "A"})     # remembered
        _decide(config, args={"invoice_id": "INV-43", "iban": "B"})     # same resource, new args
        assert len(approver.requests) == 2

    def test_one_call_id_is_one_correlation_id_and_one_request_id(self, pdp):
        recorder = _Recorder()
        config = _config(hooks=[recorder])
        paused = _decide(config, can_interrupt=True, call_id="call-1")
        _decide(config, approval=APPROVED, can_interrupt=True, call_id="call-1")   # the resume
        assert [e.decision for e in recorder.decisions] == ["approval_required", "approved", "permit"]
        assert len({e.correlation_id for e in recorder.decisions}) == 1
        request_ids = {recorder.decisions[0].detail["request_id"], recorder.decisions[1].detail["request_id"]}
        assert request_ids == {paused.pending.request_id}

    def test_different_calls_get_different_ids(self, pdp):
        recorder = _Recorder()
        config = _config(hooks=[recorder])
        first = _decide(config, can_interrupt=True, call_id="call-1")
        second = _decide(config, can_interrupt=True, call_id="call-2")
        assert first.pending.request_id != second.pending.request_id
        assert recorder.decisions[0].correlation_id != recorder.decisions[1].correlation_id

    def test_a_rejection_ends_with_a_final_deny(self, pdp):
        recorder = _Recorder()
        _decide(_config(hooks=[recorder]), approval=ApprovalDecision(False), can_interrupt=True, call_id="c")
        assert [e.decision for e in recorder.decisions] == ["rejected", "deny"]
        assert recorder.decisions[1].detail["not_approved"] is True

    def test_a_refusal_where_nobody_was_asked_is_not_a_humans_rejection(self, pdp):
        recorder = _Recorder()
        nobody = ApprovalDecision(False, note="the client could not be asked", asked=False)
        _decide(_config(hooks=[recorder]), approval=nobody, can_interrupt=True)
        assert recorder.decisions[0].detail["source"] == "none"
        human = _Recorder()
        _decide(_config(hooks=[human]), approval=ApprovalDecision(False), can_interrupt=True)
        assert human.decisions[0].detail["source"] == "interrupt"


def test_pre_check_and_the_calls_own_check_share_ids(pdp):
    # OpenAI: needs_approval pauses the run, the call's own check runs after the answer.
    from axiomatics_ai.core.enforcement import needs_human_approval

    recorder = _Recorder()
    config = _config(hooks=[recorder])
    request = needs_human_approval(
        pdp=config.build_pdp_client(), cache=None, config=config, surface="test", subject=SUBJECT,
        action=ACTION, resource=RESOURCE, environment={"channel": "chat"}, user_messages=[],
        args={"invoice_id": "INV-43"}, call_id="call-1",
    )
    _decide(config, approval=APPROVED, args={"invoice_id": "INV-43"}, call_id="call-1")
    assert request.args == {"invoice_id": "INV-43"}
    assert [e.decision for e in recorder.decisions] == ["approval_required", "approved", "permit"]
    assert len({e.correlation_id for e in recorder.decisions}) == 1
    assert recorder.decisions[1].detail["request_id"] == request.request_id


class TestResumeWithoutReasking:
    """A paused call is remembered with the decision that asked for approval, so the PDP is
    not asked that again: not by a replay of the paused step before the answer (LangGraph,
    Strands run it again), and not when the answer arrives."""

    def test_advice_form_asks_the_pdp_twice_in_all(self, pdp):
        recorder = _Recorder()
        config = _config(hooks=[recorder])
        _decide(config, can_interrupt=True, call_id="c1")                       # pauses
        replay = _decide(config, can_interrupt=True, call_id="c1")              # replay, no answer yet
        final = _decide(config, approval=APPROVED, can_interrupt=True, call_id="c1")
        assert replay.pending is not None and final.permitted is True
        # the first ask, then the evaluation with the approval attribute — nothing in between
        assert pdp.environments == [{"channel": "chat"}, {"channel": "chat", "approval_granted": True}]
        assert [e.decision for e in recorder.decisions] == ["approval_required", "approved", "permit"]

    def test_obligation_form_asks_the_pdp_once(self, monkeypatch):
        fake = _Pdp(first=PdpDecision(Decision.PERMIT, (Obligation("require-approval"),)))
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        config = _config()
        _decide(config, can_interrupt=True, call_id="c1")
        assert _decide(config, approval=APPROVED, can_interrupt=True, call_id="c1").permitted is True
        assert len(fake.environments) == 1      # the approval fulfils the obligation: no new ask

    def test_the_replay_returns_the_same_request(self, pdp):
        config = _config()
        first = _decide(config, can_interrupt=True, call_id="c1")
        assert _decide(config, can_interrupt=True, call_id="c1").pending is first.pending

    def test_an_answered_call_is_forgotten(self, pdp):
        config = _config()
        _decide(config, can_interrupt=True, call_id="c1")
        _decide(config, approval=ApprovalDecision(False), can_interrupt=True, call_id="c1")
        _decide(config, can_interrupt=True, call_id="c1")     # the same id again is a new check
        assert len(pdp.environments) == 2

    def test_changed_arguments_are_a_new_check(self, pdp):
        config = _config()
        _decide(config, can_interrupt=True, call_id="c1", args={"invoice_id": "INV-43"})
        _decide(config, can_interrupt=True, call_id="c1", args={"invoice_id": "INV-44"})
        assert len(pdp.environments) == 2

    def test_without_a_call_id_nothing_is_remembered(self, pdp):
        config = _config()
        _decide(config, can_interrupt=True)
        _decide(config, approval=APPROVED, can_interrupt=True)
        assert len(pdp.environments) == 3      # ask, ask again, evaluate with the approval

    def test_the_pre_check_is_reused_by_the_calls_own_check(self, pdp):
        from axiomatics_ai.core.enforcement import needs_human_approval

        config = _config()
        needs_human_approval(
            pdp=config.build_pdp_client(), cache=None, config=config, surface="test", subject=SUBJECT,
            action=ACTION, resource=RESOURCE, environment={"channel": "chat"}, user_messages=[], call_id="c1",
        )
        assert _decide(config, approval=APPROVED, call_id="c1").permitted is True
        assert len(pdp.environments) == 2


def test_decision_events_carry_obligation_and_advice_attributes(pdp):
    recorder = _Recorder()
    _decide(_config(hooks=[recorder]), can_interrupt=True)
    assert recorder.decisions[0].detail["advice"] == [
        {"id": "require-approval", "attributes": {"reason": "Over 100 EUR"}},
    ]
