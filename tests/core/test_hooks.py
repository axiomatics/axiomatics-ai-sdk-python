"""
Unit tests for the observe-only hooks: transport events (from PdpClient), decision
events (from the shared enforcement helpers), correlation grouping, and exception isolation.

ADS HTTP is stubbed by monkeypatching ``httpx.post`` so the real ``_authorize`` runs and
emits, or by monkeypatching ``PdpClient._authorize`` where only the decision matters.
No real ADS instance is required.
"""
from __future__ import annotations

import hashlib
import json
import logging

import httpx
import pytest

from axiomatics_ai.core.pdp import PdpClient, PdpRequestError
from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.core.enforcement import (
    apply_post_invocation,
    authorized_call,
    authorized_tools,
    emit_post_invocation_skip,
    enrich_and_authorize,
)
from axiomatics_ai.core.hooks import (
    AuditLogHook,
    AuthzHook,
    DecisionEvent,
    HookDispatcher,
    InvalidationEvent,
    LoggingHook,
    verify_audit_trail,
)
from axiomatics_ai.core.post_invocation import PostInvocationContext
from axiomatics_ai import PDPConnection, standard_tool_enforcement
from tests.pdp_fakes import patch_pdp_http, FakePdpBase


class RecordingHook(AuthzHook):
    def __init__(self) -> None:
        self.requests = []
        self.responses = []
        self.decisions = []
        self.invalidations = []

    def on_pdp_request(self, event):
        self.requests.append(event)

    def on_pdp_response(self, event):
        self.responses.append(event)

    def on_decision(self, event):
        self.decisions.append(event)

    def on_invalidation(self, event):
        self.invalidations.append(event)


def _ok_response(decision: str = "Permit") -> httpx.Response:
    # A request must be attached or Response.raise_for_status() (called in _authorize on 2xx)
    # raises "request instance has not been set".
    return httpx.Response(
        200,
        json={"Response": [{"Decision": decision}]},
        request=httpx.Request("POST", "http://ads/authorize"),
    )


# ── Transport events ────────────────────────────────────────────────────────────

class TestTransportEvents:
    def test_request_and_response_emitted(self, monkeypatch):
        patch_pdp_http(monkeypatch, "post", lambda *a, **k: _ok_response("Permit"))
        hook = RecordingHook()
        client = PdpClient(base_url="http://ads", dispatcher=HookDispatcher([hook]))

        assert client.check_permission(subject={"u": "a"}, action={"name": "x"}, resource={"name": "r"}, environment={}) is True
        assert len(hook.requests) == 1
        assert len(hook.responses) == 1
        assert hook.requests[0].endpoint == "/authorize"
        assert hook.responses[0].status_code == 200
        assert hook.responses[0].error is None
        assert hook.responses[0].latency_ms >= 0

    def test_transient_failure_emits_error(self, monkeypatch):
        def boom(*a, **k):
            raise httpx.ConnectError("no route")
        patch_pdp_http(monkeypatch, "post", boom)
        hook = RecordingHook()
        client = PdpClient(base_url="http://ads", dispatcher=HookDispatcher([hook]), fail_open=True)

        assert client.check_permission(subject={"u": "a"}, action={"name": "x"}, resource={"name": "r"}, environment={}, fail_open=True) is True
        assert hook.responses[0].error is not None
        assert hook.responses[0].status_code is None

    def test_client_error_emits_response_then_raises(self, monkeypatch):
        patch_pdp_http(monkeypatch, "post", lambda *a, **k: httpx.Response(400, text="undeclared attribute"))
        hook = RecordingHook()
        client = PdpClient(base_url="http://ads", dispatcher=HookDispatcher([hook]))

        with pytest.raises(PdpRequestError):
            client.check_permission(subject={"u": "a"}, action={"name": "x"}, resource={"name": "r"}, environment={})
        assert hook.responses[0].status_code == 400
        assert hook.responses[0].error is not None

    def test_no_hooks_is_quiet(self, monkeypatch):
        patch_pdp_http(monkeypatch, "post", lambda *a, **k: _ok_response("Deny"))
        # No dispatcher → default no-op; nothing to assert beyond "does not raise".
        client = PdpClient(base_url="http://ads")
        assert client.check_permission(subject={"u": "a"}, action={"name": "x"}, resource={"name": "r"}, environment={}) is False


# ── Exception isolation ─────────────────────────────────────────────────────────

class TestExceptionIsolation:
    def test_raising_hook_does_not_break_enforcement(self, monkeypatch):
        patch_pdp_http(monkeypatch, "post", lambda *a, **k: _ok_response("Permit"))

        class BoomHook(AuthzHook):
            def on_pdp_request(self, event):
                raise RuntimeError("boom")
            def on_pdp_response(self, event):
                raise RuntimeError("boom")

        rec = RecordingHook()
        client = PdpClient(base_url="http://ads", dispatcher=HookDispatcher([BoomHook(), rec]))
        # The bad hook must not prevent the call from succeeding, nor stop later hooks.
        assert client.check_permission(subject={"u": "a"}, action={"name": "x"}, resource={"name": "r"}, environment={}) is True
        assert len(rec.responses) == 1


# ── Correlation ─────────────────────────────────────────────────────────────────

class TestCorrelation:
    def test_helper_groups_its_own_events(self, monkeypatch):
        patch_pdp_http(monkeypatch, "post", lambda *a, **k: _ok_response("Permit"))
        hook = RecordingHook()
        config = AuthZConfig(connection=PDPConnection(base_url="http://ads", cache_ttl_seconds=0), hooks=[hook])
        ads = config.build_pdp_client()

        authorized_call(
            pdp=ads, cache=None, config=config, surface="test",
            subject={"u": "a"}, action={"name": "x"}, resource={"name": "r"}, environment={},
        )
        # Transport (request+response) and the decision share one non-None correlation id.
        cids = {e.correlation_id for e in (hook.requests + hook.responses + hook.decisions)}
        assert len(cids) == 1 and None not in cids

    def test_outer_scope_groups_multiple_helpers(self, monkeypatch):
        patch_pdp_http(monkeypatch, "post", lambda *a, **k: _ok_response("Permit"))
        hook = RecordingHook()
        config = AuthZConfig(connection=PDPConnection(base_url="http://ads", cache_ttl_seconds=0), hooks=[hook])
        ads = config.build_pdp_client()
        dispatcher = config.build_hook_dispatcher()

        with dispatcher.correlation():
            authorized_call(pdp=ads, cache=None, config=config, surface="test",
                            subject={"u": "a"}, action={"name": "x"}, resource={"name": "r1"}, environment={})
            authorized_call(pdp=ads, cache=None, config=config, surface="test",
                            subject={"u": "a"}, action={"name": "x"}, resource={"name": "r2"}, environment={})
        cids = {e.correlation_id for e in hook.decisions}
        assert len(hook.decisions) == 2
        assert len(cids) == 1 and None not in cids  # reentrant: both reuse the outer id


# ── Decision events + cache flag ─────────────────────────────────────────────────

class TestDecisionEvents:
    def test_invocation_decision_and_cache_flag(self, monkeypatch):
        # Count wire calls to prove the second decision is a cache hit (no ADS call).
        calls = {"n": 0}
        def counting_authorize(self, request, *, endpoint="/authorize"):
            calls["n"] += 1
            return {"Response": [{"Decision": "Permit"}]}
        monkeypatch.setattr(PdpClient, "_authorize", counting_authorize)

        hook = RecordingHook()
        config = AuthZConfig(connection=PDPConnection(base_url="http://ads"), hooks=[hook])  # caching on (default TTL)
        ads = config.build_pdp_client()
        cache = config.build_cache()

        kw = dict(pdp=ads, cache=cache, config=config, surface="test",
                  subject={"u": "a"}, action={"name": "inv"}, resource={"name": "tool_x"}, environment={})
        assert authorized_call(**kw) is True
        assert authorized_call(**kw) is True

        assert calls["n"] == 1  # second call served from cache
        assert [d.decision for d in hook.decisions] == ["permit", "permit"]
        assert [d.cached for d in hook.decisions] == [False, True]
        assert all(d.enforcement_point == "tool_invocation" and d.surface == "test" for d in hook.decisions)

    def test_visibility_decision_detail(self, monkeypatch):
        def authorize(self, request, *, endpoint="/authorize"):
            # Permit only "a" via the MultiRequest echo shape.
            return {"Response": [{"Decision": "Permit", "Category": [{"Attribute": [{"AttributeId": "tool_name", "Value": ["a"]}]}]}]}
        monkeypatch.setattr(PdpClient, "_authorize", authorize)

        hook = RecordingHook()
        config = AuthZConfig(
            connection=PDPConnection(base_url="http://ads"),
            enforcement=standard_tool_enforcement(),
            hooks=[hook],
        )
        ads = config.build_pdp_client()

        permitted = authorized_tools(
            pdp=ads, cache=None, config=config, surface="test",
            subject={"u": "a"}, action={"name": "vis"}, tool_names=["a", "b"], environment={},
        )
        assert permitted == {"a"}
        (event,) = hook.decisions
        assert event.enforcement_point == "tool_visibility"
        assert event.decision == "partial"
        assert event.detail == {"permitted": 1, "total": 2}


# ── Environment on decision events ──────────────────────────────────────────────

_ENV = {"transaction_id": "tx-1", "risk": "low"}


class _FakePdp(FakePdpBase):
    def check_permission(self, **kw):
        return True

    def check_permission_batch(self, *, resource_names, **kw):
        return set(resource_names)


def _config(hook, **kw) -> AuthZConfig:
    return AuthZConfig(connection=PDPConnection(base_url="http://ads"), hooks=[hook], **kw)


class TestDecisionEnvironment:
    def test_event_defaults(self):
        event = DecisionEvent(surface="s", enforcement_point="p", subject={}, action={}, resource={}, decision="permit")
        assert event.environment == {}
        assert event.timestamp > 0

    def test_dispatcher_passes_environment(self):
        hook = RecordingHook()
        HookDispatcher([hook]).decision(
            surface="s", enforcement_point="p", subject={}, action={}, resource={}, decision="permit",
            environment=_ENV,
        )
        HookDispatcher([hook]).decision(
            surface="s", enforcement_point="p", subject={}, action={}, resource={}, decision="permit",
        )
        assert [e.environment for e in hook.decisions] == [_ENV, {}]

    def test_visibility_fresh_and_cached(self):
        hook = RecordingHook()
        config = _config(hook, enforcement=standard_tool_enforcement())
        cache = config.build_cache()
        for _ in range(2):
            authorized_tools(pdp=_FakePdp(), cache=cache, config=config, surface="t",
                             subject={"u": "a"}, action={"name": "vis"}, tool_names=["a"], environment=_ENV)
        assert [(e.cached, e.environment) for e in hook.decisions] == [(False, _ENV), (True, _ENV)]

    def test_invocation_fresh_and_cached(self):
        hook = RecordingHook()
        config = _config(hook)
        cache = config.build_cache()
        for _ in range(2):
            authorized_call(pdp=_FakePdp(), cache=cache, config=config, surface="t",
                            subject={"u": "a"}, action={"name": "inv"}, resource={"tool_name": "x"},
                            environment=_ENV)
        assert [(e.cached, e.environment) for e in hook.decisions] == [(False, _ENV), (True, _ENV)]

    def _post_ctx(self) -> PostInvocationContext:
        return PostInvocationContext(
            pdp_client=_FakePdp(), subject={"u": "a"}, environment=_ENV, tool_name="t", args={},
            resource={"tool_name": "t"}, action={"name": "post_invocation"}, fail_open=False,
        )

    def test_post_invocation(self):
        class Passthrough:
            def apply(self, records, ctx):
                return records

        hook = RecordingHook()
        config = _config(hook)
        apply_post_invocation(strategy=Passthrough(), records=[1], post_ctx=self._post_ctx(), config=config, surface="t")
        emit_post_invocation_skip(post_ctx=self._post_ctx(), config=config, surface="t", reason="no records")
        assert [(e.decision, e.environment) for e in hook.decisions] == [("passthrough", _ENV), ("skipped", _ENV)]

    def test_guard(self):
        hook = RecordingHook()
        config = _config(hook)
        enrich_and_authorize(
            pdp=_FakePdp(), cache=None, config=config, surface="t", enforcement_point="before_model",
            subject={"u": "a"}, action={"name": "before_model"}, resource={}, environment=_ENV,
            enrichers=[], content="hello", block_reason="blocked",
        )
        (event,) = hook.decisions
        assert event.enforcement_point == "before_model" and event.environment == _ENV

    def test_authorizer_single_and_batch(self):
        from axiomatics_ai import Authorizer

        hook = RecordingHook()
        authz = Authorizer(_config(hook))
        authz.pdp = _FakePdp()
        authz.is_permitted(action="read", resource="r", subject={"u": "a"}, environment=_ENV)
        for _ in range(2):
            authz.is_permitted_batch(action="read", resources=["r1", "r2"], subject={"u": "a"}, environment=_ENV)
        assert [e.environment for e in hook.decisions] == [_ENV, _ENV, _ENV]
        assert [e.cached for e in hook.decisions[1:]] == [False, True]

    def test_retrieval(self):
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore

        from axiomatics_ai import RetrievalConfig
        from axiomatics_ai.rag.filtering import compute_filter
        from tests.rag.conftest import BET_PERMIT_ALL

        class Provider:
            def get_subject(self, ctx):
                return {"user_id": "u1"}

            def get_action(self, name, ctx):
                return {"name": name}

            def get_environment(self, ctx):
                return {"session_id": "s1", "dropped": None}

        class Pdp:
            def get_constraints(self, *, provided_attributes):
                return BET_PERMIT_ALL

        hook = RecordingHook()
        compute_filter(
            vectorstore=InMemoryVectorStore(FakeEmbeddings(size=8)), provider=Provider(), context={},
            config=_config(hook, retrieval=RetrievalConfig()), pdp=Pdp(),
        )
        (event,) = hook.decisions
        assert event.enforcement_point == "retrieval"
        assert event.environment == {"session_id": "s1"}


# ── Invalidation events ─────────────────────────────────────────────────────────

class TestInvalidationEvents:
    def test_dispatcher_emits_with_correlation(self):
        hook = RecordingHook()
        dispatcher = HookDispatcher([hook])
        with dispatcher.correlation():
            dispatcher.invalidation(subject={"user_id": "alice"}, removed=3, reason="revoked")
        (event,) = hook.invalidations
        assert isinstance(event, InvalidationEvent)
        assert (event.subject, event.removed, event.reason) == ({"user_id": "alice"}, 3, "revoked")
        assert event.correlation_id is not None and event.timestamp > 0

    def test_base_hook_ignores_invalidation(self):
        HookDispatcher([AuthzHook()]).invalidation(subject={}, removed=0)  # no-op, must not raise

    def test_logging_hook(self, caplog):
        with caplog.at_level(logging.INFO, logger="axiomatics_ai.hooks"):
            HookDispatcher([LoggingHook()]).invalidation(subject={}, removed=2, reason="risk")
        text = caplog.text
        assert "Cache invalidation" in text and "ALL" in text and "removed=2" in text and "'risk'" in text


# ── AuditLogHook ─────────────────────────────────────────────────────────────────

def _decision_event(**kw) -> DecisionEvent:
    base = dict(
        surface="fastmcp", enforcement_point="tool_invocation",
        subject={"user_id": "alice", "agent_id": "agent-1", "actor_id": "orch", "agent_name": "researcher", "scopes": ["read"]},
        action={"name": "tool_invocation"}, resource={"tool_name": "search"}, decision="permit",
        cached=True, detail={"k": 1}, correlation_id="cid-1", environment=_ENV, timestamp=0.0,
    )
    base.update(kw)
    return DecisionEvent(**base)


def _recompute(record: dict) -> str:
    """The documented hash: SHA-256 over the record minus hash/signature, sorted keys, no spaces."""
    body = {k: v for k, v in record.items() if k not in ("hash", "signature")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _chained_trail(**hook_kwargs) -> list[dict]:
    records: list[dict] = []
    hook = AuditLogHook(records.append, hash_chain=True, **hook_kwargs)
    hook.on_decision(_decision_event())
    hook.on_invalidation(InvalidationEvent(subject={}, removed=1, timestamp=1.0))
    hook.on_decision(_decision_event(decision="deny", timestamp=2.0))
    return [json.loads(json.dumps(r)) for r in records]   # as stored and read back


class TestAuditLogHook:
    def test_decision_record(self):
        records = []
        AuditLogHook(records.append).on_decision(_decision_event())
        (record,) = records
        assert record == {
            "event": "decision",
            "timestamp": "1970-01-01T00:00:00+00:00",
            "correlation_id": "cid-1",
            "agent_id": "agent-1", "user_id": "alice", "actor_id": "orch", "agent_name": "researcher",
            "surface": "fastmcp", "enforcement_point": "tool_invocation", "action": "tool_invocation",
            "resource": {"tool_name": "search"}, "decision": "permit", "cached": True,
            "subject": {
                "user_id": "alice", "agent_id": "agent-1", "actor_id": "orch", "agent_name": "researcher",
                "scopes": ["read"],
            },
            "environment": _ENV, "detail": {"k": 1},
        }
        assert "hash" not in record and "previous_hash" not in record

    def test_custom_attribute_names_and_missing_values(self):
        records = []
        hook = AuditLogHook(records.append, agent_attribute="client", user_attribute="username")
        hook.on_decision(_decision_event(subject={"client": "c1", "username": "bob"}, action={}))
        record = records[0]
        assert (record["agent_id"], record["user_id"], record["actor_id"], record["action"]) == ("c1", "bob", None, None)

    def test_remediation_record(self):
        records = []
        AuditLogHook(records.append).on_invalidation(
            InvalidationEvent(subject={"user_id": "alice"}, removed=2, reason="revoked", correlation_id="c", timestamp=0.0),
        )
        assert records == [{
            "event": "remediation", "timestamp": "1970-01-01T00:00:00+00:00", "correlation_id": "c",
            "remediation": "cache_invalidation", "subject": {"user_id": "alice"}, "removed": 2, "reason": "revoked",
        }]

    def test_logger_sink_emits_json_lines(self, caplog):
        logger = logging.getLogger("test.audit")
        with caplog.at_level(logging.INFO, logger="test.audit"):
            AuditLogHook(logger).on_decision(_decision_event())
        (line,) = [r.getMessage() for r in caplog.records if r.name == "test.audit"]
        assert json.loads(line)["decision"] == "permit"

    def test_default_sink_is_audit_logger(self, caplog):
        with caplog.at_level(logging.INFO, logger="axiomatics_ai.audit"):
            AuditLogHook().on_decision(_decision_event())
        events = [json.loads(r.getMessage())["event"] for r in caplog.records if r.name == "axiomatics_ai.audit"]
        assert events == ["decision"]

    def test_hash_chain_is_recomputable(self):
        records = _chained_trail()
        assert [r["seq"] for r in records] == [1, 2, 3]
        assert len({r["chain_id"] for r in records}) == 1
        previous = ""
        for record in records:
            assert record["previous_hash"] == previous
            assert record["hash"] == _recompute(record)
            previous = record["hash"]
        assert verify_audit_trail(records) == []

    def test_verify_detects_edit_removal_and_reorder(self):
        records = _chained_trail()
        edited = [records[0], {**records[1], "removed": 99}, records[2]]
        assert any("content does not match" in p for p in verify_audit_trail(edited))
        removed = verify_audit_trail([records[0], records[2]])
        assert any("expected seq 2" in p for p in removed)
        assert any("previous_hash" in p for p in removed)
        assert verify_audit_trail(records[1:])                     # the start is missing
        assert verify_audit_trail([records[1], records[0], records[2]])
        assert verify_audit_trail([{"event": "decision"}]) == ["record 0 (chain None, seq None): not a chained record"]

    def test_verify_accepts_interleaved_chains(self):
        first, second = _chained_trail(), _chained_trail()        # e.g. two workers
        assert first[0]["chain_id"] != second[0]["chain_id"]
        interleaved = [first[0], second[0], first[1], second[1], second[2], first[2]]
        assert verify_audit_trail(interleaved) == []

    @pytest.mark.parametrize("algorithm", ["ES256", "EdDSA"])
    def test_signed_records_verify_with_the_public_key(self, algorithm):
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519

        private = ec.generate_private_key(ec.SECP256R1()) if algorithm == "ES256" else ed25519.Ed25519PrivateKey.generate()
        records = _chained_trail(signing_key=private, signing_algorithm=algorithm, signing_key_id="audit-1")
        assert all(r["signature"] for r in records)
        assert verify_audit_trail(records, public_key=private.public_key()) == []

        # Rewriting a record and recomputing the chain is caught by the signatures.
        forged = [dict(r) for r in records]
        forged[1]["removed"] = 99
        forged[1]["hash"] = _recompute(forged[1])
        problems = verify_audit_trail(forged, public_key=private.public_key())
        assert any("signature is for another record" in p for p in problems)

        other = ec.generate_private_key(ec.SECP256R1()) if algorithm == "ES256" else ed25519.Ed25519PrivateKey.generate()
        assert all("signature does not verify" in p for p in verify_audit_trail(records, public_key=other.public_key()))

    def test_signing_needs_chain_and_an_asymmetric_algorithm(self):
        from cryptography.hazmat.primitives.asymmetric import ec

        key = ec.generate_private_key(ec.SECP256R1())
        with pytest.raises(ValueError, match="hash_chain=True"):
            AuditLogHook(lambda r: None, signing_key=key)
        with pytest.raises(ValueError, match="signing_algorithm"):
            AuditLogHook(lambda r: None, hash_chain=True, signing_key=b"secret", signing_algorithm="HS256")

    def test_lone_surrogate_is_recorded_not_lost(self):
        records = []
        hook = AuditLogHook(records.append, hash_chain=True)
        for tool in ("a", "\ud800", "b"):
            hook.on_decision(_decision_event(resource={"tool_name": tool}))
        assert [r["seq"] for r in records] == [1, 2, 3] and verify_audit_trail(records) == []

    def test_forked_worker_starts_its_own_chain(self, monkeypatch):
        records = []
        hook = AuditLogHook(records.append, hash_chain=True)
        hook.on_decision(_decision_event())
        monkeypatch.setattr("axiomatics_ai.core.hooks.os.getpid", lambda: -42)   # as after os.fork()
        hook.on_decision(_decision_event())
        assert records[1]["chain_id"] != records[0]["chain_id"] and records[1]["seq"] == 1
        assert verify_audit_trail(records) == []

    def test_verify_takes_json_lines_and_refuses_duplicate_keys(self):
        lines = [json.dumps(r) for r in _chained_trail()]
        assert verify_audit_trail(lines) == []
        forged = '{"decision": "Permit", ' + lines[0][1:]
        assert verify_audit_trail([forged, *lines[1:]])[0].startswith("record 0: not a valid JSON record")
        assert verify_audit_trail(["null", "[]"]) == ["record 0: not a JSON object", "record 1: not a JSON object"]

    def test_chained_record_values_are_json(self):
        records = []
        AuditLogHook(records.append, hash_chain=True).on_decision(_decision_event(detail={"when": object()}))
        assert isinstance(records[0]["detail"]["when"], str)
        assert verify_audit_trail(records) == []

    def test_wired_through_config_and_enforcement(self):
        records = []
        config = AuthZConfig(connection=PDPConnection(base_url="http://ads"), hooks=[AuditLogHook(records.append)])
        authorized_call(pdp=_FakePdp(), cache=None, config=config, surface="t",
                        subject={"user_id": "alice"}, action={"name": "inv"}, resource={"tool_name": "x"},
                        environment=_ENV)
        (record,) = records
        assert record["user_id"] == "alice" and record["environment"] == _ENV and record["correlation_id"]


def test_tool_visibility_does_not_cache_a_fail_open_result():
    from axiomatics_ai import AuthZConfig, PDPConnection
    from axiomatics_ai.core.enforcement import authorized_tools
    from axiomatics_ai.core.pdp import PermittedNames

    calls = []

    class Pdp:
        def check_permission_batch(self, **kw):
            calls.append(1)
            # first: PDP down, fail_open showed everything; then: the PDP is back
            return PermittedNames.of(kw["resource_names"], answered=False) if len(calls) == 1 \
                else PermittedNames.of([], answered=True)

    config = AuthZConfig(connection=PDPConnection(fail_open=True))
    cache = config.build_cache()
    call = dict(pdp=Pdp(), cache=cache, config=config, surface="t", subject={"u": "a"},
                action={"name": "v"}, tool_names=["send_email"], environment={})
    assert authorized_tools(**call) == {"send_email"}       # the outage fallback
    assert authorized_tools(**call) == set()                # the PDP's real answer, not the cache
    assert authorized_tools(**call) == set()                # a real answer is cached
    assert len(calls) == 2
