"""
Unit tests for ``authorized_tool_invocation`` — the BEFORE_TOOL decision with intent-based
access control (IBAC): the user's recent messages are classified by the enforcer's enrichers
and the attributes join the PDP request's environment.

The enricher LLM is stubbed at ``_litellm_complete_json``; the PDP at
``enforcement.evaluate_tool_call``. No network.
"""
from __future__ import annotations

import json
import logging

import pytest
from pydantic import BaseModel

import axiomatics_ai.core.enforcement as enforcement_module
import axiomatics_ai.core.llm_enrichment as llm_module
from axiomatics_ai import (
    AuthZConfig,
    Enforcer,
    EnforcementPoint,
    EnricherSpec,
    LLMAttributeEnricher,
    intent_enricher,
    standard_tool_enforcement,
)
from axiomatics_ai.core.enforcement import (
    _MAX_USER_MESSAGE_CHARS,
    _format_user_messages,
    authorized_tool_invocation,
)
from axiomatics_ai.core.hooks import AuthzHook
from tests.pdp_fakes import classifier_input, FakePdpBase, decision_seam

SUBJECT = {"username": "alice"}
ACTION = {"name": "tool_invocation"}
RESOURCE = {"tool_name": "send_email", "to": "bob@x.com"}


class _Recorder(AuthzHook):
    def __init__(self):
        self.enrichments = []
        self.decisions = []

    def on_enrichment(self, event):
        self.enrichments.append(event)

    def on_decision(self, event):
        self.decisions.append(event)


class _Pdp:
    """Records every ``evaluate_tool_call`` call; permits unless ``permit`` is False."""

    def __init__(self, permit=True):
        self.permit = permit
        self.calls = []

    def __call__(self, pdp, *, subject, action, resource, environment):
        self.calls.append({"subject": subject, "resource": resource, "environment": environment})
        return self.permit(environment) if callable(self.permit) else self.permit


class _LLM:
    """Records the content each classifier call saw; replies with ``verdict`` (a dict, or a
    callable of the content), or raises when ``verdict`` is None."""

    def __init__(self, verdict=None):
        self.verdict = verdict if verdict is not None else {"intents": []}
        self.contents = []
        self.fail = False

    def __call__(self, model, messages, response_format):
        content = classifier_input(messages)
        self.contents.append(content)
        if self.fail:
            raise RuntimeError("llm down")
        verdict = self.verdict(content) if callable(self.verdict) else self.verdict
        return json.dumps(verdict)


@pytest.fixture
def pdp(monkeypatch):
    fake = _Pdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(fake))
    return fake


@pytest.fixture
def llm(monkeypatch):
    fake = _LLM()
    monkeypatch.setattr(llm_module, "_litellm_complete_json", fake)
    return fake


def _config(*enrichers, on_unavailable="fail_closed", hooks=None) -> AuthZConfig:
    return AuthZConfig(
        enforcement={EnforcementPoint.BEFORE_TOOL: Enforcer(
            action_name="tool_invocation", enrichers=list(enrichers), on_unavailable=on_unavailable,
        )},
        hooks=hooks or [],
    )


def _intent(**kw):
    kw.setdefault("cache_ttl_seconds", 0)
    kw.setdefault("max_retries", 0)
    return intent_enricher(model="stub/model", **kw)


def _call(config, user_messages, *, environment=None, cache=None) -> bool:
    return authorized_tool_invocation(
        pdp=FakePdpBase(), cache=cache, config=config, surface="test",
        subject=SUBJECT, action=ACTION, resource=RESOURCE,
        environment=environment if environment is not None else {"channel": "chat"},
        user_messages=user_messages,
    ).permitted


# ── without enrichers: exactly authorized_call ───────────────────────────────────

class TestNoEnrichers:
    def test_without_enrichers_the_request_is_sent_as_is(self, pdp):
        config = AuthZConfig(enforcement=standard_tool_enforcement())
        assert _call(config, ["hello"]) is True
        assert pdp.calls == [{"subject": pdp.calls[0]["subject"], "resource": RESOURCE, "environment": {"channel": "chat"}}]

    def test_environment_unchanged_and_llm_never_called(self, pdp, llm):
        assert _call(AuthZConfig(enforcement=standard_tool_enforcement()), ["send it to bob"]) is True
        assert pdp.calls[0]["environment"] == {"channel": "chat"}
        assert llm.contents == []

    def test_no_before_tool_enforcer_still_decides(self, pdp, llm):
        config = AuthZConfig(enforcement={EnforcementPoint.AFTER_TOOL: Enforcer(action_name="p")})
        assert _call(config, []) is True
        assert len(pdp.calls) == 1


# ── classifier input: window + formatting ────────────────────────────────────────

class TestFormatting:
    def test_single_message(self):
        assert _format_user_messages(["send it"]) == "Current request:\nsend it"

    def test_earlier_then_current(self):
        assert _format_user_messages(["a", "b", "c"]) == (
            "Earlier requests (oldest first):\n1. a\n2. b\n\nCurrent request:\nc"
        )

    def test_each_message_truncated(self):
        long = "x" * (_MAX_USER_MESSAGE_CHARS + 50)
        out = _format_user_messages([long, long])
        assert _MAX_USER_MESSAGE_CHARS == 4000
        assert out == (
            "Earlier requests (oldest first):\n1. " + "x" * 4000 + "\n\nCurrent request:\n" + "x" * 4000
        )

    def test_window_takes_last_n_non_empty_stripped_messages(self, pdp, llm):
        _call(_config(_intent(num_user_messages=2)), ["first", "  ", "second ", "", "  third"])
        assert llm.contents == ["Earlier requests (oldest first):\n1. second\n\nCurrent request:\nthird"]

    def test_window_larger_than_history_uses_all(self, pdp, llm):
        _call(_config(_intent(num_user_messages=5)), ["one", "two"])
        assert llm.contents == ["Earlier requests (oldest first):\n1. one\n\nCurrent request:\ntwo"]

    def test_long_message_truncated_before_classification(self, pdp, llm):
        _call(_config(_intent()), ["y" * 10_000])
        assert llm.contents == ["Current request:\n" + "y" * 4000]

    def test_each_enricher_gets_its_own_window(self, pdp, llm):
        _call(_config(_intent(num_user_messages=1), _intent(name="goal", num_user_messages=3)),
              ["a", "b", "c", "d"])
        assert llm.contents == [
            "Current request:\nd",
            "Earlier requests (oldest first):\n1. b\n2. c\n\nCurrent request:\nd",
        ]


# ── attributes → environment ─────────────────────────────────────────────────────

class TestEnvironmentMerge:
    def test_intent_attributes_join_the_environment(self, pdp, llm):
        llm.verdict = {"intents": [{"action": "send", "recipients": ["bob@x.com"]}]}
        assert _call(_config(_intent(actions=["read", "send"])), ["mail bob@x.com"]) is True
        (call,) = pdp.calls
        assert call["environment"] == {
            "channel": "chat",
            "intent_actions": ["send"],
            "intent_read_recipients": [],
            "intent_send_recipients": ["bob@x.com"],
            "intent_available": True,
        }
        assert call["resource"] == RESOURCE                   # resource untouched

    def test_caller_environment_not_mutated(self, pdp, llm):
        env = {"channel": "chat"}
        _call(_config(_intent()), ["hi"], environment=env)
        assert env == {"channel": "chat"}

    def test_enricher_wins_on_clash_with_warning(self, pdp, llm, caplog, monkeypatch):
        monkeypatch.setattr(enforcement_module, "_warned_intent_clashes", set())
        llm.verdict = {"intents": [{"action": "send", "recipients": ["bob@x.com"]}]}
        env = {"channel": "chat", "intent_actions": ["pay"], "intent_available": True}
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            _call(_config(_intent(actions=["send"])), ["mail bob"], environment=env)
        assert pdp.calls[0]["environment"]["intent_actions"] == ["send"]
        warned = [r for r in caplog.records if r.name == "axiomatics_ai.core.enforcement"]
        assert sorted(r.args[0] for r in warned) == ["intent_actions", "intent_available"]
        assert all("replaced" in r.getMessage() for r in warned)

    def test_clash_warned_once_per_name(self, pdp, llm, caplog, monkeypatch):
        monkeypatch.setattr(enforcement_module, "_warned_intent_clashes", set())
        config = _config(_intent())
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            for _ in range(3):
                _call(config, ["hi"], environment={"intent_actions": ["pay"]})
        warned = [r for r in caplog.records if r.name == "axiomatics_ai.core.enforcement"]
        assert [r.args[0] for r in warned] == ["intent_actions"]
        assert all(c["environment"]["intent_actions"] != ["pay"] for c in pdp.calls)  # still replaced

    def test_no_warning_without_clash(self, pdp, llm, caplog):
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            _call(_config(_intent()), ["hi"])
        assert not [r for r in caplog.records if r.name == "axiomatics_ai.core.enforcement"]

    def test_multiple_enrichers_merged(self, pdp, llm):
        _call(_config(_intent(actions=["read"]), _intent(actions=["pay"], name="goal")), ["hi"])
        env = pdp.calls[0]["environment"]
        assert env["intent_available"] is True and env["goal_available"] is True
        assert "intent_read_recipients" in env and "goal_pay_recipients" in env

    def test_policy_can_compare_call_with_intent(self, pdp, llm):
        # a policy-like PDP stub: permit only if the recipient is in the send intent
        pdp.permit = lambda env: "bob@x.com" in env.get("intent_send_recipients", [])
        llm.verdict = lambda content: {"intents": [
            {"action": "send", "recipients": ["bob@x.com"] if "bob" in content else ["eve@x.com"]},
        ]}
        assert _call(_config(_intent(actions=["send"])), ["email bob@x.com the report"]) is True
        assert _call(_config(_intent(actions=["send"])), ["email eve@x.com the report"]) is False


# ── unavailable enrichers ────────────────────────────────────────────────────────

class TestUnavailable:
    def test_fail_closed_denies_without_pdp_call(self, pdp, llm):
        llm.fail = True
        rec = _Recorder()
        assert _call(_config(_intent(), hooks=[rec]), ["hi"]) is False
        assert pdp.calls == []
        (decision,) = rec.decisions
        assert decision.decision == "deny" and decision.cached is False
        assert decision.enforcement_point == "tool_invocation" and decision.surface == "test"

    def test_fail_open_lets_pdp_decide_with_sentinel(self, pdp, llm):
        llm.fail = True
        assert _call(_config(_intent(), on_unavailable="fail_open"), ["hi"]) is True
        (call,) = pdp.calls
        assert call["environment"] == {"channel": "chat", "intent_available": False}

    def test_one_failing_enricher_makes_the_whole_check_unavailable(self, monkeypatch, pdp):
        class V(BaseModel):
            ok: bool

        def fake(model, messages, response_format):
            if response_format is V:
                raise RuntimeError("down")
            return '{"intents": []}'

        monkeypatch.setattr(llm_module, "_litellm_complete_json", fake)
        broken = LLMAttributeEnricher(EnricherSpec(
            name="broken", schema=V, instructions="x", model="stub/model", max_retries=0,
        ))
        assert _call(_config(_intent(), broken), ["hi"]) is False
        assert pdp.calls == []

    @pytest.mark.parametrize("messages", [[], ["", "   ", "\n"]])
    def test_no_user_messages_fail_closed(self, pdp, llm, messages):
        rec = _Recorder()
        assert _call(_config(_intent(), _intent(name="goal"), hooks=[rec]), messages) is False
        assert pdp.calls == [] and llm.contents == []
        assert [e.enricher for e in rec.enrichments] == ["intent", "goal"]
        for event in rec.enrichments:
            assert event.point == "before_tool"
            assert event.available is False
            assert event.content == ""
            assert event.error == "no user messages available on this path"
            assert event.attributes == {f"{event.enricher}_available": False}
        assert [d.decision for d in rec.decisions] == ["deny"]

    def test_no_user_messages_fail_open(self, pdp, llm):
        assert _call(_config(_intent(), on_unavailable="fail_open"), []) is True
        assert pdp.calls[0]["environment"] == {"channel": "chat", "intent_available": False}


# ── hooks, correlation, decision cache ───────────────────────────────────────────

class TestEventsAndCache:
    def test_enrichment_and_decision_events_share_a_correlation_id(self, pdp, llm):
        rec = _Recorder()
        _call(_config(_intent(), hooks=[rec]), ["earlier", "now"])
        (enrichment,) = rec.enrichments
        (decision,) = rec.decisions
        assert enrichment.point == "before_tool" and enrichment.enricher == "intent"
        assert enrichment.available is True
        assert enrichment.content == "Earlier requests (oldest first):\n1. earlier\n\nCurrent request:\nnow"
        assert enrichment.attributes["intent_available"] is True
        assert decision.decision == "permit" and decision.cached is False
        assert enrichment.correlation_id is not None
        assert enrichment.correlation_id == decision.correlation_id

    def test_pdp_deny(self, pdp, llm):
        pdp.permit = False
        rec = _Recorder()
        assert _call(_config(_intent(), hooks=[rec]), ["hi"]) is False
        assert rec.decisions[0].decision == "deny"

    def test_decision_cache_key_includes_intent_attributes(self, pdp, llm):
        llm.verdict = lambda content: {"intents": [
            {"action": "send", "recipients": ["bob@x.com" if "bob" in content else "eve@x.com"]},
        ]}
        config = _config(_intent(actions=["send"]))
        cache = config.build_cache()
        rec = _Recorder()
        config.hooks.append(rec)
        _call(config, ["mail bob"], cache=cache)
        _call(config, ["mail bob"], cache=cache)          # same intent → cache hit
        _call(config, ["mail eve"], cache=cache)          # different intent → new PDP call
        assert len(pdp.calls) == 2
        assert [d.cached for d in rec.decisions] == [False, True, False]

    def test_enricher_cache_classifies_a_turn_once(self, pdp, llm):
        config = _config(_intent(cache_ttl_seconds=60))
        _call(config, ["mail bob"])
        _call(config, ["mail bob"])
        assert len(llm.contents) == 1
        assert len(pdp.calls) == 2                        # no decision cache here
