"""Tests for config diagnostics and connectivity: PostInvocationContext.action_name,
AuthZConfig.validate()/summary()/check_connection(), PdpClient.healthcheck()/__repr__,
and the decision cache's clear()/__len__/__repr__."""
from __future__ import annotations

from tests.pdp_fakes import patch_pdp_http

import httpx
import pytest

from axiomatics_ai import (
    AuthZConfig,
    ColumnMaskStrategy,
    Enforcer,
    EnforcementPoint,
    PDPConnection,
    RecordFilterStrategy,
    AttributeNames,
    EnricherSpec,
    LLMAttributeEnricher,
    RetrievalConfig,
    intent_enricher,
    pii_enricher,
    prompt_injection_enricher,
    standard_tool_enforcement,
    toxicity_enricher,
)
from axiomatics_ai.core.pdp import PdpClient
from axiomatics_ai.core.post_invocation import PostInvocationContext


# ── PostInvocationContext.action_name (the tool_decorator custom-strategy fix) ──────

def test_action_name_property_returns_action_name():
    ctx = PostInvocationContext(
        pdp_client=None, subject={}, environment={}, tool_name="get_records",
        args={"id": "1"}, resource={"tool_name": "get_records", "id": "1"},
        action={"name": "post_invocation"}, fail_open=False,
    )
    assert ctx.action_name == "post_invocation"
    assert ctx.action == {"name": "post_invocation"}  # the dict is still available
    assert ctx.resource == {"tool_name": "get_records", "id": "1"}  # stored as given


def test_action_name_empty_when_missing():
    ctx = PostInvocationContext(
        pdp_client=None, subject={}, environment={}, tool_name="t",
        args={}, resource={}, action={}, fail_open=False,
    )
    assert ctx.action_name == ""


# ── AuthZConfig.validate() ──────────────────────────────────────────────────────────

def test_validate_clean_config_has_no_problems():
    cfg = AuthZConfig(connection=PDPConnection(), enforcement=standard_tool_enforcement())
    assert cfg.validate() == []


def test_validate_flags_empty_tool_name_attribute():
    cfg = AuthZConfig(attribute_names=AttributeNames(tool=""), enforcement=standard_tool_enforcement())
    assert any("attribute_names.tool" in p for p in cfg.validate())


def test_validate_flags_empty_agent_name_attribute():
    cfg = AuthZConfig(attribute_names=AttributeNames(agent=""))
    assert any("attribute_names.agent" in p for p in cfg.validate())


def test_validate_visibility_with_invocation_is_ok():
    # The batch attribute id now comes from AuthZConfig.attribute_names.tool, not the Enforcer.
    cfg = AuthZConfig(enforcement={
        EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="v"),
        EnforcementPoint.BEFORE_TOOL: Enforcer(action_name="i"),
    })
    assert cfg.validate() == []


def test_validate_flags_visibility_without_invocation():
    # Visibility only hides tools; a hidden tool can still be called by name.
    cfg = AuthZConfig(enforcement={EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="v")})
    assert any("TOOL_VISIBILITY is enforced without BEFORE_TOOL" in p for p in cfg.validate())


def test_validate_flags_empty_mcp_server_attribute():
    cfg = AuthZConfig(attribute_names=AttributeNames(mcp_server=""))
    assert any("attribute_names.mcp_server" in p for p in cfg.validate())


def test_validate_flags_on_unrecognized_output_outside_after_tool():
    cfg = AuthZConfig(enforcement={
        EnforcementPoint.BEFORE_TOOL: Enforcer(action_name="i", on_unrecognized_output="fail_closed"),
    })
    assert any("on_unrecognized_output" in p and "AFTER_TOOL-only" in p for p in cfg.validate())


def test_validate_flags_strategy_on_non_after_tool_point():
    cfg = AuthZConfig(enforcement={
        EnforcementPoint.BEFORE_TOOL: Enforcer(action_name="i", strategy=ColumnMaskStrategy("x")),
    })
    assert any("before_tool" in p and "AFTER_TOOL-only" in p for p in cfg.validate())


def test_validate_record_filter_is_ok():
    # RecordFilterStrategy carries its own key_attribute (default "record_hash").
    cfg = AuthZConfig(enforcement={
        EnforcementPoint.AFTER_TOOL: Enforcer(action_name="p", strategy=RecordFilterStrategy()),
    })
    assert cfg.validate() == []


def test_record_filter_rejects_empty_key_attribute():
    with pytest.raises(ValueError, match="key_attribute"):
        RecordFilterStrategy(key_attribute="")


def test_validate_column_mask_is_ok():
    cfg = AuthZConfig(enforcement={
        EnforcementPoint.AFTER_TOOL: Enforcer(action_name="p", strategy=ColumnMaskStrategy("x")),
    })
    assert cfg.validate() == []


def test_validate_flags_retrieval_with_authzen():
    cfg = AuthZConfig(
        connection=PDPConnection(protocol="authzen"),
        retrieval=RetrievalConfig(),
    )
    assert any("authzen" in p.lower() and "xacml" in p.lower() for p in cfg.validate())


# ── validate(): enricher placement ──────────────────────────────────────────────────

def _at(point: EnforcementPoint, *enrichers, action_name="x") -> AuthZConfig:
    return AuthZConfig(enforcement={point: Enforcer(action_name=action_name, enrichers=list(enrichers))})


def test_validate_presets_at_their_points_are_ok():
    m = "stub/model"
    assert _at(EnforcementPoint.BEFORE_TOOL, intent_enricher(model=m)).validate() == []
    assert _at(EnforcementPoint.BEFORE_MODEL, prompt_injection_enricher(model=m), toxicity_enricher(model=m)).validate() == []
    assert _at(EnforcementPoint.AFTER_MODEL, pii_enricher(model=m), toxicity_enricher(model=m)).validate() == []


@pytest.mark.parametrize("point", [
    EnforcementPoint.TOOL_VISIBILITY, EnforcementPoint.AFTER_TOOL, EnforcementPoint.A2A_INVOCATION,
])
def test_validate_flags_enrichers_at_a_point_that_never_runs_them(point):
    problems = _at(point, intent_enricher(model="stub/model")).validate()
    assert f"Enforcer at {point.value} sets enrichers, which never run at this point." in problems


def test_validate_flags_enricher_outside_its_allowed_points():
    problems = _at(EnforcementPoint.BEFORE_MODEL, intent_enricher(model="stub/model")).validate()
    assert problems == ["Enricher 'intent' is meant for before_tool, not before_model."]


def test_validate_lists_allowed_points_sorted():
    problems = _at(EnforcementPoint.BEFORE_TOOL, toxicity_enricher(model="stub/model")).validate()
    assert problems == ["Enricher 'toxicity' is meant for after_model, before_model, not before_tool."]


def test_validate_flags_each_misplaced_enricher():
    problems = _at(
        EnforcementPoint.AFTER_MODEL,
        prompt_injection_enricher(model="stub/model"), pii_enricher(model="stub/model"),
        intent_enricher(model="stub/model", name="goal"),
    ).validate()
    assert problems == [
        "Enricher 'prompt_injection' is meant for before_model, not after_model.",
        "Enricher 'goal' is meant for before_tool, not after_model.",
    ]


def test_validate_custom_enricher_without_allowed_points_is_ok_anywhere_that_runs_enrichers():
    from pydantic import BaseModel

    class Verdict(BaseModel):
        flag: bool

    enr = LLMAttributeEnricher(EnricherSpec(name="c", schema=Verdict, instructions="x", model="stub/model"))
    for point in (EnforcementPoint.BEFORE_MODEL, EnforcementPoint.AFTER_MODEL, EnforcementPoint.BEFORE_TOOL):
        assert _at(point, enr).validate() == []


def test_validate_flags_zero_user_message_window_at_before_tool():
    # intent_enricher rejects 0 itself; a hand-built EnricherSpec is caught by validate().
    from pydantic import BaseModel

    class Verdict(BaseModel):
        flag: bool

    enr = LLMAttributeEnricher(EnricherSpec(
        name="c", schema=Verdict, instructions="x", model="stub/model", num_user_messages=0,
    ))
    problems = _at(EnforcementPoint.BEFORE_TOOL, enr).validate()
    assert problems == ["Enricher 'c' at before_tool needs num_user_messages >= 1."]


def test_validate_flags_intent_enrichers_with_authzen():
    cfg = AuthZConfig(
        connection=PDPConnection(protocol="authzen"),
        enforcement={EnforcementPoint.BEFORE_TOOL: Enforcer(
            action_name="tool_invocation", enrichers=[intent_enricher(model="stub/model")],
        )},
    )
    problems = cfg.validate()
    assert len(problems) == 1 and "protocol='xacml'" in problems[0] and "lists" in problems[0]


def test_validate_intent_enrichers_with_xacml_are_ok():
    assert _at(EnforcementPoint.BEFORE_TOOL, intent_enricher(model="stub/model")).validate() == []


def test_misplaced_enricher_is_logged_at_construction(caplog):
    with caplog.at_level("WARNING"):
        _at(EnforcementPoint.AFTER_TOOL, intent_enricher(model="stub/model"))
    assert any("never run at this point" in r.message for r in caplog.records)


def test_post_init_logs_problems(caplog):
    with caplog.at_level("WARNING"):
        AuthZConfig(attribute_names=AttributeNames(tool=""))
    assert any("AuthZConfig" in r.message for r in caplog.records)


# ── AuthZConfig.summary() ───────────────────────────────────────────────────────────

def test_summary_lists_enforced_points_and_connection():
    cfg = AuthZConfig(
        connection=PDPConnection(base_url="http://ads:9999", protocol="authzen"),
        enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
    )
    s = cfg.summary()
    assert "http://ads:9999" in s and "authzen" in s and "tool_visibility" in s


def test_summary_none_when_nothing_enforced():
    assert "none" in AuthZConfig().summary()


# ── PdpClient.healthcheck() + HealthCheckResult ──────────────────────────────────

def test_healthcheck_ok(monkeypatch):
    def fake_get(url, timeout):
        assert url.endswith("/health")
        return httpx.Response(200, request=httpx.Request("GET", url))
    patch_pdp_http(monkeypatch, "get", fake_get)
    r = PdpClient("http://localhost:8080").healthcheck()
    assert r.ok and bool(r) is True
    assert r.status_code == 200 and r.error is None and r.latency_ms is not None


def test_healthcheck_bad_status(monkeypatch):
    patch_pdp_http(
        monkeypatch, "get",
        lambda url, timeout: httpx.Response(503, request=httpx.Request("GET", url)),
    )
    r = PdpClient("http://localhost:8080").healthcheck()
    assert not r and r.status_code == 503 and "503" in r.error


def test_healthcheck_unreachable_does_not_raise(monkeypatch):
    def boom(url, timeout):
        raise httpx.ConnectError("nope")
    patch_pdp_http(monkeypatch, "get", boom)
    r = PdpClient("http://localhost:8080").healthcheck()
    assert not r and r.status_code is None and "could not reach PDP" in r.error


def test_check_connection_delegates(monkeypatch):
    patch_pdp_http(
        monkeypatch, "get",
        lambda url, timeout: httpx.Response(200, request=httpx.Request("GET", url)),
    )
    assert AuthZConfig(connection=PDPConnection(base_url="http://x:1")).check_connection().ok


def test_client_repr_shows_key_fields():
    r = repr(PdpClient("http://localhost:8080/", protocol="xacml", fail_open=True))
    assert "PdpClient(" in r and "fail_open=True" in r and "xacml" in r


# ── decision cache clear() / __len__ / __repr__ ─────────────────────────────────────

def test_cache_len_clear_and_repr():
    cache = AuthZConfig(connection=PDPConnection(cache_ttl_seconds=300, cache_maxsize=8)).build_cache()
    assert len(cache) == 0
    cache.set({"u": "a"}, {"name": "x"}, {"name": "t"}, {}, True)
    assert len(cache) == 1
    assert "size=1" in repr(cache)
    cache.clear()
    assert len(cache) == 0


def test_disabled_cache_len_and_repr():
    cache = AuthZConfig(connection=PDPConnection(cache_ttl_seconds=0)).build_cache()
    assert len(cache) == 0
    cache.clear()  # no-op, must not raise
    assert "disabled" in repr(cache)
