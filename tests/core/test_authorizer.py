"""
Unit tests for the direct path: ``Authorizer`` answers through the config's context
provider, decision cache, hooks, and fail_open — with no framework in between.
"""
from __future__ import annotations

from axiomatics_ai import Authorizer, AuthZConfig, AuthzHook, PDPConnection
from axiomatics_ai.core.config import DefaultClientContextProvider
from axiomatics_ai.core.pdp import PdpClient
from tests.pdp_fakes import patch_pdp_http, FakePdpBase


class FakePdp(FakePdpBase):
    """Records every call; answers from ``single`` (bool), ``batch`` (set), ``tree`` / ``text``."""

    def __init__(self, *, single: bool = True, batch: set[str] | None = None, tree=None, text=None):
        self.single = single
        self.batch = batch if batch is not None else set()
        self.tree = tree
        self.text = text
        self.calls: list[dict] = []

    def check_permission(self, **kw):
        self.calls.append({"kind": "single", **kw})
        return self.single

    def check_permission_batch(self, **kw):
        self.calls.append({"kind": "batch", **kw})
        return set(self.batch)

    def get_constraints(self, **kw):
        self.calls.append({"kind": "tree", **kw})
        return self.tree

    def get_simplified_constraints(self, **kw):
        self.calls.append({"kind": "text", **kw})
        return self.text


class Recorder(AuthzHook):
    def __init__(self):
        self.decisions = []

    def on_decision(self, event):
        self.decisions.append(event)


def _authorizer(config: AuthZConfig | None = None, **fake_kw) -> tuple[Authorizer, FakePdp]:
    authz = Authorizer(config or AuthZConfig(connection=PDPConnection(base_url="http://pdp")))
    fake = FakePdp(**fake_kw)
    authz.pdp = fake
    return authz, fake


# ── is_permitted ───────────────────────────────────────────────────────────────

class TestIsPermitted:
    def test_default_provider_maps_context_and_merges_args(self):
        authz, fake = _authorizer(single=True)
        assert authz.is_permitted(
            action="read_document", resource="doc-42", args={"classification": "internal"},
            context={"username": "alice", "session_id": "s1", "ignored": "x"},
        ) is True
        call = fake.calls[0]
        assert call["subject"] == {"username": "alice"}
        assert call["action"] == {"name": "read_document"}
        assert call["resource"] == {"resource_name": "doc-42", "classification": "internal"}
        assert call["environment"] == {"session_id": "s1"}

    def test_custom_resource_name_attribute(self):
        authz, fake = _authorizer()
        authz.is_permitted(
            action="read_document", resource="doc-42", resource_name_attribute="document_id",
            context={"username": "alice"},
        )
        assert fake.calls[0]["resource"] == {"document_id": "doc-42"}

    def test_explicit_subject_and_environment_bypass_provider(self):
        authz, fake = _authorizer(single=False)
        assert authz.is_permitted(
            action="a", resource="r", subject={"employee_id": "E1", "dept": None}, environment={"region": "eu"},
        ) is False
        assert fake.calls[0]["subject"] == {"employee_id": "E1"}   # None values stripped
        assert fake.calls[0]["environment"] == {"region": "eu"}

    def test_custom_context_provider_from_config(self):
        class Provider(DefaultClientContextProvider):
            def get_subject(self, context):
                return {"employee_id": context["emp"]}

            def get_action(self, action_name, context):
                return {"id": action_name, "channel": "api"}

        authz, fake = _authorizer(AuthZConfig(context_provider=Provider()))
        authz.is_permitted(action="a", resource="r", context={"emp": "E7"})
        assert fake.calls[0]["subject"] == {"employee_id": "E7"}
        assert fake.calls[0]["action"] == {"id": "a", "channel": "api"}

    def test_fail_open_comes_from_connection(self, monkeypatch):
        monkeypatch.setattr(PdpClient, "_authorize", lambda self, req, **kw: None)  # unreachable
        assert Authorizer(AuthZConfig(connection=PDPConnection(fail_open=True))).is_permitted(
            action="a", resource="r", context={"username": "alice"},
        ) is True
        assert Authorizer(AuthZConfig(connection=PDPConnection(fail_open=False))).is_permitted(
            action="a", resource="r", context={"username": "alice"},
        ) is False

    def test_cache_hit_skips_pdp_and_marks_event_cached(self):
        rec = Recorder()
        authz, fake = _authorizer(AuthZConfig(hooks=[rec]), single=True)
        for _ in range(2):
            authz.is_permitted(action="a", resource="r", context={"username": "alice"})
        assert len(fake.calls) == 1
        assert [e.cached for e in rec.decisions] == [False, True]
        assert rec.decisions[0].surface == "direct"
        assert rec.decisions[0].enforcement_point == "a"
        assert rec.decisions[0].decision == "permit"

    def test_cache_disabled_calls_every_time(self):
        authz, fake = _authorizer(AuthZConfig(connection=PDPConnection(cache_ttl_seconds=0)))
        for _ in range(3):
            authz.is_permitted(action="a", resource="r", context={"username": "alice"})
        assert len(fake.calls) == 3

    def test_clear_cache_forces_a_fresh_decision(self):
        authz, fake = _authorizer()
        authz.is_permitted(action="a", resource="r", context={"username": "alice"})
        authz.clear_cache()
        authz.is_permitted(action="a", resource="r", context={"username": "alice"})
        assert len(fake.calls) == 2

    def test_custom_surface_label(self):
        rec = Recorder()
        authz = Authorizer(AuthZConfig(hooks=[rec]), surface="billing-api")
        authz.pdp = FakePdp()
        authz.is_permitted(action="a", resource="r", context={"username": "alice"})
        assert rec.decisions[0].surface == "billing-api"


# ── is_permitted_batch ─────────────────────────────────────────────────────────

class TestIsPermittedBatch:
    def test_batch_request_shape_and_result(self):
        rec = Recorder()
        authz, fake = _authorizer(AuthZConfig(hooks=[rec]), batch={"search_docs"})
        out = authz.is_permitted_batch(
            action="tool_visibility", resource_name_attribute="tool_name",
            resources=["search_docs", "send_email"], context={"username": "bob"},
        )
        assert out == {"search_docs"}
        call = fake.calls[0]
        assert call["kind"] == "batch"
        assert call["resource_names"] == ["search_docs", "send_email"]
        assert call["resource_attr_id"] == "tool_name"
        assert call["subject"] == {"username": "bob"}
        assert call["fail_open"] is False
        ev = rec.decisions[0]
        assert (ev.decision, ev.detail) == ("partial", {"permitted": 1, "total": 2})

    def test_batch_default_resource_name_attribute(self):
        authz, fake = _authorizer()
        authz.is_permitted_batch(action="v", resources=["a"], context={"username": "alice"})
        assert fake.calls[0]["resource_attr_id"] == "resource_name"

    def test_batch_is_cached_per_name_set(self):
        authz, fake = _authorizer(batch={"a"})
        kw = dict(action="v", resources=["a", "b"], context={"username": "alice"})
        authz.is_permitted_batch(**kw)
        authz.is_permitted_batch(**kw)
        authz.is_permitted_batch(action="v", resources=["a", "b", "c"], context={"username": "alice"})
        assert len(fake.calls) == 2

    def test_fail_open_forwarded_to_batch(self):
        authz, fake = _authorizer(AuthZConfig(connection=PDPConnection(fail_open=True)))
        authz.is_permitted_batch(action="v", resources=["a"], context={"username": "alice"})
        assert fake.calls[0]["fail_open"] is True


# ── partial evaluation ─────────────────────────────────────────────────────────

class TestPartialEvaluation:
    def test_tree_request_flattens_subject_action_environment(self):
        tree = {"operator": "AND", "children": []}
        authz, fake = _authorizer(tree=tree)
        out = authz.partial_evaluation(
            action="retrieval", context={"username": "alice", "session_id": "s1"},
            decision_set=["PERMIT", "NOT_APPLICABLE"], exclude_indeterminate=True,
        )
        assert out is tree
        call = fake.calls[0]
        assert call["kind"] == "tree"
        assert call["provided_attributes"] == [
            {"attribute": "username", "values": ["alice"]},
            {"attribute": "name", "values": ["retrieval"]},
            {"attribute": "session_id", "values": ["s1"]},
        ]
        assert call["pdp_decision_set"] == ["PERMIT", "NOT_APPLICABLE"]
        assert call["exclude_indeterminate"] is True
        assert call["mocked_attributes"] is None

    def test_explicit_subject_with_list_values_and_mocked_attributes(self):
        authz, fake = _authorizer(tree={})
        mocked = [{"attribute": "resource.dept", "values": ["eng"]}]
        authz.partial_evaluation(
            action="read", subject={"roles": ["a", "b"], "gone": None}, environment={},
            mocked_attributes=mocked,
        )
        call = fake.calls[0]
        assert call["provided_attributes"] == [
            {"attribute": "roles", "values": ["a", "b"]},
            {"attribute": "name", "values": ["read"]},
        ]
        assert call["mocked_attributes"] is mocked

    def test_text_variant_uses_simplified_endpoint(self):
        authz, fake = _authorizer(text="department = finance")
        assert authz.partial_evaluation_text(action="read", context={"username": "bob"}) == "department = finance"
        assert fake.calls[0]["kind"] == "text"
        assert fake.calls[0]["provided_attributes"][0] == {"attribute": "username", "values": ["bob"]}

    def test_transient_failure_returns_none(self):
        authz, _ = _authorizer(tree=None, text=None)
        assert authz.partial_evaluation(action="read", context={"username": "bob"}) is None
        assert authz.partial_evaluation_text(action="read", context={"username": "bob"}) is None


# ── Diagnostics ────────────────────────────────────────────────────────────────

def test_check_connection_delegates_to_pdp(monkeypatch):
    import httpx
    patch_pdp_http(
        monkeypatch, "get",
        lambda url, timeout: httpx.Response(200, request=httpx.Request("GET", url)),
    )
    assert Authorizer(AuthZConfig()).check_connection().ok


def test_repr_mentions_surface():
    assert "surface='direct'" in repr(Authorizer(AuthZConfig()))


# ── Decision cache: credential expiry and invalidation ────────────────────────

def _in_context(fn):
    """Run ``fn`` in a copy of the current context, so credential_expires_at doesn't leak."""
    import contextvars

    return contextvars.copy_context().run(fn)


class TestDecisionCacheLifetime:
    def _cache(self, ttl=60):
        from axiomatics_ai.core.cache import AuthzDecisionCacheWithTTL

        return AuthzDecisionCacheWithTTL(ttl=ttl, maxsize=100)

    def _lifetime(self, cache):
        (entry,) = cache._cache.values()
        return entry.lifetime

    def test_ttl_without_credential(self):
        cache = self._cache()
        cache.set({"u": "a"}, {"name": "x"}, {}, {}, True)
        assert self._lifetime(cache) == 60
        assert cache.get({"u": "a"}, {"name": "x"}, {}, {}) is True

    def test_capped_at_credential_expiry(self):
        import time

        from axiomatics_ai.core.cache import credential_expires_at

        cache = self._cache()

        def run():
            credential_expires_at.set(time.time() + 10)
            cache.set({"u": "a"}, {"name": "x"}, {}, {}, True)

        _in_context(run)
        assert 9 < self._lifetime(cache) <= 10

    def test_ttl_wins_when_credential_outlives_it(self):
        import time

        from axiomatics_ai.core.cache import credential_expires_at

        cache = self._cache(ttl=5)

        def run():
            credential_expires_at.set(time.time() + 3600)
            cache.set({"u": "a"}, {"name": "x"}, {}, {}, True)

        _in_context(run)
        assert self._lifetime(cache) == 5

    def test_expired_credential_not_cached(self):
        import time

        from axiomatics_ai.core.cache import credential_expires_at

        cache = self._cache()

        def run():
            credential_expires_at.set(time.time() - 1)
            cache.set({"u": "a"}, {"name": "x"}, {}, {}, True)

        _in_context(run)
        assert len(cache) == 0

    def test_entry_expires_with_credential(self):
        import time

        from axiomatics_ai.core.cache import credential_expires_at

        cache = self._cache()

        def run():
            credential_expires_at.set(time.time() + 0.05)
            cache.set({"u": "a"}, {"name": "x"}, {}, {}, True)

        _in_context(run)
        time.sleep(0.1)
        assert cache.get({"u": "a"}, {"name": "x"}, {}, {}) is None

    def test_invalidate_by_subject_and_all(self):
        cache = self._cache()
        for user in ("alice", "bob"):
            for tool in ("t1", "t2"):
                cache.set({"user_id": user, "agent_id": "a1"}, {"name": "x"}, {"tool_name": tool}, {}, True)
        assert cache.invalidate(user_id="alice") == 2
        assert cache.invalidate(user_id="alice") == 0
        assert cache.invalidate(user_id="bob", agent_id="other") == 0   # all attributes must match
        assert cache.invalidate(agent_id="a1", user_id="bob") == 2
        cache.set({"user_id": "c"}, {"name": "x"}, {}, {}, True)
        cache.set({"user_id": "d"}, {"name": "x"}, {}, {}, True)
        assert cache.invalidate() == 2
        assert len(cache) == 0

    def test_disabled_cache_invalidates_nothing(self):
        cache = self._cache(ttl=0)
        cache.set({"u": "a"}, {"name": "x"}, {}, {}, True)
        assert cache.invalidate() == 0
        assert "disabled" in repr(cache)

    def test_repr_shows_ttl(self):
        assert "ttl=60s" in repr(self._cache())


class TestInvalidateDecisions:
    def test_reaches_every_cache_and_emits_hook(self):
        class Rec(AuthzHook):
            def __init__(self):
                self.events = []

            def on_invalidation(self, event):
                self.events.append(event)

        rec = Rec()
        config = AuthZConfig(connection=PDPConnection(base_url="http://pdp"), hooks=[rec])
        a, b = config.build_cache(), config.build_cache()
        a.set({"user_id": "alice"}, {"name": "x"}, {}, {}, True)
        b.set({"user_id": "alice"}, {"name": "y"}, {}, {}, True)
        b.set({"user_id": "bob"}, {"name": "y"}, {}, {}, True)

        assert config.invalidate_decisions(reason="revoked", user_id="alice") == 2
        assert (len(a), len(b)) == (0, 1)
        assert config.invalidate_decisions() == 1
        assert [(e.subject, e.removed, e.reason) for e in rec.events] == [
            ({"user_id": "alice"}, 2, "revoked"), ({}, 1, ""),
        ]

    def test_caches_weakly_held(self):
        import gc

        config = AuthZConfig(connection=PDPConnection(base_url="http://pdp"))
        config.build_cache()
        gc.collect()
        assert len(config._state.caches) == 0

    def test_authorizer_decides_afresh_after_invalidation(self):
        authz, fake = _authorizer()
        authz.is_permitted(action="a", resource="r", context={"user_id": "alice"})
        authz.is_permitted(action="a", resource="r", context={"user_id": "alice"})
        assert len(fake.calls) == 1
        assert authz.config.invalidate_decisions(user_id="alice") == 1
        authz.is_permitted(action="a", resource="r", context={"user_id": "alice"})
        assert len(fake.calls) == 2


def test_a_fail_open_batch_result_is_not_cached():
    # During a PDP outage the fallback (every name, or none) must not outlive it by the TTL.
    from axiomatics_ai.core.pdp import PermittedNames

    authz, fake = _authorizer()
    calls = []
    fake.check_permission_batch = lambda **kw: calls.append(1) or PermittedNames.of([], answered=False)
    kw = dict(action="v", resources=["a"], context={"username": "alice"})
    authz.is_permitted_batch(**kw)
    authz.is_permitted_batch(**kw)
    assert len(calls) == 2          # asked again: nothing was cached
