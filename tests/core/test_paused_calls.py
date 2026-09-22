"""
Calls paused for a human's approval (``PausedCalls``) and what the enforcement core does with
them: a pause expires, is bounded, is keyed by the caller, and is dropped by
``invalidate_decisions``. Also: a remembered approval is reused only while the asking rule
allows reuse, and a caller can't set the approval attribute itself. The PDP is stubbed at
``enforcement.evaluate_tool_call``.
"""
from __future__ import annotations

import logging

import pytest

import axiomatics_ai.core.enforcement as enforcement_module
import axiomatics_ai.core.pdp.obligations as obligations_module
from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.core.enforcement import authorized_tool_invocation, needs_human_approval
from axiomatics_ai.core.pdp import (
    Advice,
    ApprovalConfig,
    ApprovalDecision,
    ApprovalRequest,
    Decision,
    ObligationsAndAdviceConfig,
    PdpDecision,
)
from axiomatics_ai.core.pdp.obligations import PausedCalls

ALICE = {"user_id": "alice"}
BOB = {"user_id": "bob"}
ACTION = {"name": "tool_invocation"}
RESOURCE = {"tool_name": "pay_invoice", "invoice_id": "INV-43"}
ASK = PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"reason": "Over 100 EUR"}),))
ASK_TTL = PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"ttl": "300"}),))
PERMIT = PdpDecision(Decision.PERMIT)
APPROVED = ApprovalDecision(True, approver="manager")


def _request(subject=ALICE, request_id="r1") -> ApprovalRequest:
    return ApprovalRequest(
        request_id=request_id, tool_name="pay_invoice", args={}, args_hash="h", reasons=(),
        ttl_seconds=None, subject=dict(subject), asked_by=(),
    )


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(obligations_module.time, "monotonic", fake)
    return fake


# ── PausedCalls on its own ──────────────────────────────────────────────────────

class TestPausedCalls:
    def test_get_returns_request_and_decision(self):
        paused = PausedCalls()
        request = _request()
        paused.put(("k",), request, ASK)
        assert paused.get(("k",)) == (request, ASK)
        assert paused.get(("other",)) is None

    def test_an_entry_expires_after_the_ttl(self, clock):
        paused = PausedCalls(ttl_seconds=10)
        paused.put(("k",), _request(), ASK)
        clock.now += 9.9
        assert paused.get(("k",)) is not None
        clock.now += 0.1
        assert paused.get(("k",)) is None
        clock.now -= 5      # an expired entry is gone, not merely hidden
        assert paused.get(("k",)) is None

    def test_putting_again_restarts_the_ttl(self, clock):
        paused = PausedCalls(ttl_seconds=10)
        paused.put(("k",), _request(), ASK)
        clock.now += 8
        paused.put(("k",), _request(), ASK)
        clock.now += 8
        assert paused.get(("k",)) is not None

    def test_maxsize_drops_the_least_recently_put(self):
        paused = PausedCalls(maxsize=2)
        for key in ("a", "b", "c"):
            paused.put((key,), _request(), ASK)
        assert paused.get(("a",)) is None
        assert paused.get(("b",)) is not None and paused.get(("c",)) is not None

    def test_putting_an_existing_key_makes_it_most_recent(self):
        paused = PausedCalls(maxsize=2)
        paused.put(("a",), _request(), ASK)
        paused.put(("b",), _request(), ASK)
        paused.put(("a",), _request(), ASK)
        paused.put(("c",), _request(), ASK)
        assert paused.get(("b",)) is None and paused.get(("a",)) is not None

    def test_invalidate_drops_the_matching_subjects_entries_and_counts_them(self):
        paused = PausedCalls()
        paused.put(("a1",), _request(ALICE), ASK)
        paused.put(("a2",), _request({**ALICE, "tenant_id": "t1"}), ASK)
        paused.put(("b1",), _request(BOB), ASK)
        assert paused.invalidate(user_id="alice") == 2
        assert paused.get(("a1",)) is None and paused.get(("a2",)) is None
        assert paused.get(("b1",)) is not None

    def test_invalidate_needs_every_given_attribute_to_match(self):
        paused = PausedCalls()
        paused.put(("a1",), _request({**ALICE, "tenant_id": "t1"}), ASK)
        paused.put(("a2",), _request({**ALICE, "tenant_id": "t2"}), ASK)
        assert paused.invalidate(user_id="alice", tenant_id="t1") == 1
        assert paused.get(("a2",)) is not None

    def test_invalidate_without_attributes_drops_all(self):
        paused = PausedCalls()
        paused.put(("a",), _request(ALICE), ASK)
        paused.put(("b",), _request(BOB), ASK)
        assert paused.invalidate() == 2
        assert paused.invalidate() == 0

    def test_invalidate_with_no_match_drops_nothing(self):
        paused = PausedCalls()
        paused.put(("a",), _request(ALICE), ASK)
        assert paused.invalidate(user_id="carol") == 0
        assert paused.get(("a",)) is not None


# ── through the enforcement core ─────────────────────────────────────────────────

class _Pdp:
    """Asks with ``first`` until the approval attribute is set, then permits."""

    def __init__(self, first=ASK):
        self.first = first
        self.environments: list[dict] = []
        self.subjects: list[dict] = []

    def __call__(self, pdp, *, subject, action, resource, environment):
        self.environments.append(dict(environment))
        self.subjects.append(dict(subject))
        return PERMIT if environment.get("approval_granted") is True else self.first


class _Approver:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return APPROVED


@pytest.fixture
def pdp(monkeypatch):
    fake = _Pdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
    return fake


def _config(approver=None) -> AuthZConfig:
    return AuthZConfig(
        enforcement=standard_tool_enforcement(),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
    )


def _decide(config, *, subject=ALICE, environment=None, **kwargs):
    return authorized_tool_invocation(
        pdp=config.build_pdp_client(), cache=None, config=config, surface="test",
        subject=subject, action=ACTION, resource=RESOURCE,
        environment={"channel": "chat"} if environment is None else environment, user_messages=[],
        **kwargs,
    )


def _pre_check(config, *, subject=ALICE, environment=None, call_id="c1"):
    return needs_human_approval(
        pdp=config.build_pdp_client(), cache=None, config=config, surface="test",
        subject=subject, action=ACTION, resource=RESOURCE,
        environment={"channel": "chat"} if environment is None else environment, user_messages=[],
        call_id=call_id,
    )


class TestPausedKeyedByCaller:
    def test_another_caller_reusing_a_call_id_is_checked_anew(self, pdp):
        config = _config()
        alice = _decide(config, can_interrupt=True, call_id="X")
        bob = _decide(config, subject=BOB, can_interrupt=True, call_id="X")
        assert len(pdp.environments) == 2                 # bob's call went to the PDP
        assert pdp.subjects[-1] == BOB
        assert bob.pending is not alice.pending
        assert bob.pending.subject == BOB and alice.pending.subject == ALICE

    def test_the_same_caller_still_replays_without_the_pdp(self, pdp):
        config = _config()
        first = _decide(config, can_interrupt=True, call_id="X")
        _decide(config, subject=BOB, can_interrupt=True, call_id="X")
        replay = _decide(config, can_interrupt=True, call_id="X")
        assert replay.pending is first.pending
        assert len(pdp.environments) == 2

    def test_a_pre_check_pause_is_not_reused_for_another_caller(self, pdp):
        config = _config()
        alice_request = _pre_check(config, call_id="X")
        assert alice_request is not None
        # bob answers under the same call id (providers may number ids per run): his call is
        # evaluated from scratch, not with alice's pause; which caller may resume a paused run
        # is the application's to enforce
        assert _decide(config, subject=BOB, approval=APPROVED, call_id="X").permitted is True
        assert pdp.subjects == [ALICE, BOB, BOB]

    def test_an_expired_pause_is_checked_anew(self, pdp, clock):
        config = _config()
        _decide(config, can_interrupt=True, call_id="X")
        clock.now += 3601
        _decide(config, can_interrupt=True, call_id="X")
        assert len(pdp.environments) == 2


class TestInvalidateDropsPausedCalls:
    def test_invalidate_decisions_drops_the_subjects_pauses(self, pdp):
        config = _config()
        _decide(config, can_interrupt=True, call_id="X")
        _decide(config, subject=BOB, can_interrupt=True, call_id="Y")
        config.invalidate_decisions(user_id="alice")
        _decide(config, can_interrupt=True, call_id="X")                   # checked anew
        _decide(config, subject=BOB, can_interrupt=True, call_id="Y")      # still paused
        assert pdp.subjects == [ALICE, BOB, ALICE]

    def test_invalidate_decisions_reaches_the_paused_store(self, pdp):
        config = _config()
        _decide(config, can_interrupt=True, call_id="X")
        _decide(config, subject=BOB, can_interrupt=True, call_id="Y")
        config.invalidate_decisions(user_id="alice")
        assert config._state.paused.invalidate() == 1                      # only bob's remained


class TestRecallOnlyWhenTheRuleAllowsReuse:
    def test_a_remembered_approval_is_not_reused_once_the_rule_is_single_use(self, pdp):
        approver = _Approver()
        config = _config(approver)
        pdp.first = ASK_TTL
        assert _decide(config).permitted
        assert _decide(config).permitted
        assert len(approver.requests) == 1                 # reused under the ttl rule
        pdp.first = ASK                                    # the rule now asks without a ttl
        assert _decide(config).permitted
        assert len(approver.requests) == 2                 # the human is asked again

    def test_the_pre_check_pauses_again_once_the_rule_is_single_use(self, pdp):
        config = _config()
        pdp.first = ASK_TTL
        assert _decide(config, approval=APPROVED, can_interrupt=True).permitted
        assert _pre_check(config, call_id="c2") is None    # remembered under the ttl rule
        pdp.first = ASK
        assert _pre_check(config, call_id="c3") is not None

    def test_an_interrupt_is_raised_again_once_the_rule_is_single_use(self, pdp):
        config = _config()
        pdp.first = ASK_TTL
        assert _decide(config, approval=APPROVED, can_interrupt=True).permitted
        pdp.first = ASK
        assert _decide(config, can_interrupt=True).pending is not None


class TestCallerCannotSelfApprove:
    @pytest.fixture(autouse=True)
    def _reset_warning(self, monkeypatch):
        monkeypatch.setattr(enforcement_module, "_warned_approval_attribute", False)

    def test_the_approval_attribute_is_dropped_before_the_pdp(self, pdp):
        decision = _decide(_config(), environment={"channel": "chat", "approval_granted": True})
        assert decision.permitted is False
        assert pdp.environments == [{"channel": "chat"}]

    def test_dropped_in_the_pre_check_too(self, pdp):
        request = _pre_check(_config(), environment={"channel": "chat", "approval_granted": True})
        assert request is not None
        assert pdp.environments == [{"channel": "chat"}]

    def test_a_real_approval_still_sets_it(self, pdp):
        decision = _decide(_config(_Approver()), environment={"approval_granted": True})
        assert decision.permitted is True
        assert pdp.environments == [{}, {"approval_granted": True}]

    def test_a_custom_attribute_is_dropped(self, monkeypatch):
        seen: list[dict] = []

        def fake(pdp, *, subject, action, resource, environment):
            seen.append(dict(environment))
            return ASK
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        config = AuthZConfig(
            enforcement=standard_tool_enforcement(),
            obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(attribute="human_ok")),
        )
        _decide(config, environment={"human_ok": True, "approval_granted": True})
        assert seen == [{"approval_granted": True}]       # only the configured attribute is reserved

    def test_the_warning_is_logged_once(self, pdp, caplog):
        config = _config()
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            _decide(config, environment={"approval_granted": True})
            _decide(config, environment={"approval_granted": True})
        warnings = [r for r in caplog.records if "reserved for the SDK" in r.getMessage()]
        assert len(warnings) == 1

    def test_no_warning_without_the_attribute(self, pdp, caplog):
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            _decide(_config(), environment={"channel": "chat"})
        assert not any("reserved for the SDK" in r.getMessage() for r in caplog.records)


class TestAnswerBoundToThePausedCall:
    def test_an_answer_after_the_arguments_changed_is_refused(self, pdp):
        """The human saw INV-43; the call resumed with INV-999 (an edited checkpoint)."""
        config = _config()
        shown = _decide(config, can_interrupt=True, call_id="X")
        assert shown.pending is not None and shown.pending.args == {"invoice_id": "INV-43"}
        edited = authorized_tool_invocation(
            pdp=config.build_pdp_client(), cache=None, config=config, surface="test",
            subject=ALICE, action=ACTION, resource={**RESOURCE, "invoice_id": "INV-999"},
            environment={"channel": "chat"}, user_messages=[], approval=APPROVED, call_id="X",
        )
        assert edited.permitted is False and edited.not_approved is True

    def test_the_answer_for_the_call_as_paused_still_counts(self, pdp):
        config = _config()
        _decide(config, can_interrupt=True, call_id="X")
        assert _decide(config, approval=APPROVED, call_id="X").permitted is True

    def test_another_callers_pause_under_the_same_call_id_doesnt_block_an_answer(self, pdp):
        """Providers may number call ids per run: alice and bob both paused under "X"."""
        config = _config()
        _decide(config, can_interrupt=True, call_id="X")
        _decide(config, subject=BOB, can_interrupt=True, call_id="X")
        assert _decide(config, approval=APPROVED, call_id="X").permitted is True

    def test_a_pause_on_another_of_the_apps_agents_doesnt_block_the_answer(self, pdp):
        """A tool shared by two agents (an OpenAI handoff): paused under one, run under the other."""
        config = _config()
        _decide(config, subject={**ALICE, "agent_name": "triage"}, can_interrupt=True, call_id="X")
        decision = _decide(config, subject={**ALICE, "agent_name": "payments"}, approval=APPROVED, call_id="X")
        assert decision.permitted is True
