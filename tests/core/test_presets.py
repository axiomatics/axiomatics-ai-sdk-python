"""
Tests for the preset building blocks in ``core.presets`` (also re-exported from the top-level
package): the enricher factories (schema/attributes + model resolution) and that the strategy
presets remain importable after the move out of ``core.post_invocation``.
"""
from __future__ import annotations

import json

import pytest

import axiomatics_ai.core.llm_enrichment as llm_module
from axiomatics_ai import (
    ColumnMaskStrategy,
    EnforcementPoint,
    RecordFilterStrategy,
    intent_enricher,
    pii_enricher,
    prompt_injection_enricher,
    toxicity_enricher,
)


def test_strategies_are_importable_and_construct():
    assert callable(RecordFilterStrategy().apply)
    assert callable(ColumnMaskStrategy("ssn").apply)


def test_enricher_attribute_vocabularies():
    inj = prompt_injection_enricher(model="stub/model")
    assert inj.spec.name == "prompt_injection"
    assert set(inj.spec.schema.model_fields) == {"injection_detected", "attack_type"}

    pii = pii_enricher(model="stub/model")
    assert set(pii.spec.schema.model_fields) == {"contains_ssn", "contains_credit_card"}

    tox = toxicity_enricher(model="stub/model")
    assert set(tox.spec.schema.model_fields) == {"toxic", "severity"}


def test_field_descriptions_present_for_the_classifier():
    # descriptions flow into the JSON schema the model fills — they must be set
    fields = prompt_injection_enricher(model="stub/model").spec.schema.model_fields
    assert all(f.description for f in fields.values())


def test_custom_name_is_used():
    e = prompt_injection_enricher(model="stub/model", name="jailbreak")
    assert e.spec.name == "jailbreak"
    assert e.spec.status_attribute == "jailbreak_available"


# ── model resolution: explicit → env → error ────────────────────────────────────

def test_explicit_model_wins(monkeypatch):
    monkeypatch.setenv("AXIOMATICS_ENRICHER_MODEL", "env/model")
    assert prompt_injection_enricher(model="explicit/model").spec.model == "explicit/model"


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("AXIOMATICS_ENRICHER_MODEL", "env/model")
    assert pii_enricher().spec.model == "env/model"


def test_raises_when_neither_model_nor_env(monkeypatch):
    monkeypatch.delenv("AXIOMATICS_ENRICHER_MODEL", raising=False)
    with pytest.raises(ValueError, match="AXIOMATICS_ENRICHER_MODEL"):
        toxicity_enricher()


# ── allowed_points: each preset declares where it is meant to run ───────────────

def test_preset_allowed_points():
    assert prompt_injection_enricher(model="stub/model").spec.allowed_points == {EnforcementPoint.BEFORE_MODEL}
    assert pii_enricher(model="stub/model").spec.allowed_points == {EnforcementPoint.AFTER_MODEL}
    assert toxicity_enricher(model="stub/model").spec.allowed_points == {
        EnforcementPoint.BEFORE_MODEL, EnforcementPoint.AFTER_MODEL,
    }
    assert intent_enricher(model="stub/model").spec.allowed_points == {EnforcementPoint.BEFORE_TOOL}


def test_presets_other_than_intent_do_not_cache():
    for factory in (prompt_injection_enricher, pii_enricher, toxicity_enricher):
        assert factory(model="stub/model").spec.cache_ttl_seconds == 0


# ── intent_enricher ────────────────────────────────────────────────────────────

DEFAULT_ACTIONS = ("read", "search", "write", "send", "delete", "pay")


def _enrich(monkeypatch, enricher, verdict: dict):
    monkeypatch.setattr(llm_module, "_litellm_complete_json", lambda *a, **k: json.dumps(verdict))
    return enricher.enrich("Current request:\nsend the report to bob@x.com")


def test_intent_defaults():
    e = intent_enricher(model="stub/model")
    assert e.spec.name == "intent"
    assert e.spec.status_attribute == "intent_available"
    assert e.spec.num_user_messages == 5
    assert e.spec.cache_ttl_seconds == 3600
    assert e.spec.max_retries == 1
    assert e.actions == DEFAULT_ACTIONS


def test_intent_is_exported_from_core_and_top_level():
    import axiomatics_ai
    import axiomatics_ai.core as core
    assert "intent_enricher" in axiomatics_ai.__all__ and "intent_enricher" in core.__all__
    assert core.intent_enricher is intent_enricher


def test_intent_flattens_verdict_into_attributes(monkeypatch):
    outcome = _enrich(monkeypatch, intent_enricher(model="stub/model"), {"intents": [
        {"action": "send", "recipients": ["bob@x.com", "carol@x.com"]},
        {"action": "read", "recipients": []},
        {"action": "send", "recipients": ["bob@x.com", "dave@x.com"]},   # dup action + dup recipient
    ]})
    assert outcome.available is True
    assert outcome.attributes == {
        "intent_actions": ["send", "read"],                          # ordered, unique
        "intent_read_recipients": [],
        "intent_search_recipients": [],
        "intent_write_recipients": [],
        "intent_send_recipients": ["bob@x.com", "carol@x.com", "dave@x.com"],
        "intent_delete_recipients": [],
        "intent_pay_recipients": [],
        "intent_available": True,
    }


def test_intent_empty_verdict_still_emits_every_recipients_attribute(monkeypatch):
    outcome = _enrich(monkeypatch, intent_enricher(model="stub/model"), {"intents": []})
    assert outcome.attributes["intent_actions"] == []
    for action in DEFAULT_ACTIONS:
        assert outcome.attributes[f"intent_{action}_recipients"] == []
    assert outcome.attributes["intent_available"] is True


def test_intent_custom_actions_and_name(monkeypatch):
    e = intent_enricher(model="stub/model", actions=["read", "refund", "refund", "approve"], name="goal")
    assert e.actions == ("read", "refund", "approve")                 # de-duplicated, order kept
    outcome = _enrich(monkeypatch, e, {"intents": [{"action": "refund", "recipients": ["ACC-1"]}]})
    assert outcome.attributes == {
        "goal_actions": ["refund"],
        "goal_read_recipients": [],
        "goal_refund_recipients": ["ACC-1"],
        "goal_approve_recipients": [],
        "goal_available": True,
    }


def test_intent_action_outside_vocabulary_is_unavailable(monkeypatch):
    e = intent_enricher(model="stub/model", actions=["read"], max_retries=0)
    outcome = _enrich(monkeypatch, e, {"intents": [{"action": "send", "recipients": []}]})
    assert outcome.available is False
    assert outcome.attributes == {"intent_available": False}


def test_intent_schema_restricts_action_to_the_vocabulary():
    schema = intent_enricher(model="stub/model", actions=["read", "send"]).spec.schema.model_json_schema()
    intent = schema["$defs"]["Intent"]
    assert intent["properties"]["action"]["enum"] == ["read", "send"]
    assert intent["properties"]["recipients"]["type"] == "array"
    assert set(intent["required"]) == {"action", "recipients"}
    assert "intents" in schema["properties"]
    assert all(p.get("description") for p in intent["properties"].values())


@pytest.mark.parametrize("actions", [(), []])
def test_intent_rejects_empty_actions(actions):
    with pytest.raises(ValueError, match="at least one action"):
        intent_enricher(model="stub/model", actions=actions)


@pytest.mark.parametrize("bad", ["Send", "send-email", "1pay", "send email", "", "_x"])
def test_intent_rejects_non_identifier_action_names(bad):
    with pytest.raises(ValueError, match="lowercase identifiers"):
        intent_enricher(model="stub/model", actions=["read", bad])


def test_intent_accepts_identifier_action_names():
    assert intent_enricher(model="stub/model", actions=["a", "send_email2"]).actions == ("a", "send_email2")


def test_intent_rejects_num_user_messages_below_one():
    with pytest.raises(ValueError):
        intent_enricher(model="stub/model", num_user_messages=0)


def test_intent_custom_window_and_cache():
    e = intent_enricher(model="stub/model", num_user_messages=2, cache_ttl_seconds=0)
    assert e.spec.num_user_messages == 2
    assert e.spec.cache_ttl_seconds == 0


def test_intent_env_model_fallback(monkeypatch):
    monkeypatch.setenv("AXIOMATICS_ENRICHER_MODEL", "env/model")
    assert intent_enricher().spec.model == "env/model"
