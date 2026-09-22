"""End-to-end tests for to_authorized_vectorstore against real, locally-runnable stores.

Each test seeds a real store (InMemory, Chroma, Qdrant), wraps it, fakes the ADS reverse
query and ambient identity, then runs an actual search and asserts the store returned only
the permitted documents — exercising the full pipeline including the real per-store filter
dialect. (OpenSearch/Pinecone require live services; their dialect output is asserted in
test_ir_to_vectorstore_filter.py.)
"""
from __future__ import annotations

import asyncio

import pytest

from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.rag import to_authorized_vectorstore
from axiomatics_ai import PDPConnection, RetrievalConfig
from tests.rag.conftest import (
    BET_DENY_ALL,
    BET_DEPT_OR_PUBLIC,
    BET_PERMIT_ALL,
    BET_UNBOUND_SUBJECT,
    STORE_FACTORIES,
    make_inmemory,
)


def _departments(results):
    return sorted(doc.metadata["department"] for doc in results)


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_search_returns_only_permitted_documents(factory, fake_authz):
    store = to_authorized_vectorstore(factory(), AuthZConfig(retrieval=RetrievalConfig()))
    fake_authz(BET_DEPT_OR_PUBLIC)

    results = store.similarity_search("anything", k=10)

    assert _departments(results) == ["engineering", "engineering", "public"]
    assert "finance" not in _departments(results)


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_permit_all_returns_everything(factory, fake_authz):
    store = to_authorized_vectorstore(factory(), AuthZConfig(retrieval=RetrievalConfig()))
    fake_authz(BET_PERMIT_ALL)

    results = store.similarity_search("anything", k=10)

    assert len(results) == 4  # no constraint → whole corpus


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_deny_all_returns_nothing(factory, fake_authz):
    store = to_authorized_vectorstore(factory(), AuthZConfig(retrieval=RetrievalConfig()))
    fake_authz(BET_DENY_ALL)

    assert store.similarity_search("anything", k=10) == []


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_ads_unreachable_fails_closed(factory, fake_authz):
    store = to_authorized_vectorstore(factory(), AuthZConfig(connection=PDPConnection(fail_open=False), retrieval=RetrievalConfig()))
    fake_authz(None)  # ADS unreachable

    assert store.similarity_search("anything", k=10) == []


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_ads_unreachable_fail_open_returns_unfiltered(factory, fake_authz):
    store = to_authorized_vectorstore(factory(), AuthZConfig(connection=PDPConnection(fail_open=True), retrieval=RetrievalConfig()))
    fake_authz(None)

    assert len(store.similarity_search("anything", k=10)) == 4


def test_no_identity_fails_closed_by_default(monkeypatch):
    """No identity resolved → fail closed (no documents), NOT an unfiltered passthrough.
    Leaving it unfiltered would silently return the whole corpus to an unauthenticated caller."""
    from axiomatics_ai.rag import wrap as wrap_module

    monkeypatch.setattr(wrap_module, "_resolve_context", lambda: None)
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))

    assert store.similarity_search("anything", k=10) == []


def test_no_identity_passes_through_only_under_fail_open(monkeypatch):
    """The unfiltered passthrough on missing identity is opt-in via fail_open."""
    from axiomatics_ai.rag import wrap as wrap_module

    monkeypatch.setattr(wrap_module, "_resolve_context", lambda: None)
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(connection=PDPConnection(fail_open=True), retrieval=RetrievalConfig()))

    assert len(store.similarity_search("anything", k=10)) == 4


def test_authz_context_binds_identity_for_static_search(monkeypatch):
    """Static (non-agent) path: `with authz_context(...)` supplies the identity."""
    from axiomatics_ai.core.pdp import PdpClient
    from axiomatics_ai.rag import authz_context

    monkeypatch.setattr(PdpClient, "get_constraints", lambda self, **kw: BET_DEPT_OR_PUBLIC)
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))

    # No context bound → fail closed (no agent runtime, no authz_context, default fail_open=False).
    assert store.similarity_search("x", k=10) == []

    # Inside the block, identity is bound and the search is filtered.
    with authz_context({"user_id": "u1"}):
        results = store.similarity_search("x", k=10)
    assert _departments(results) == ["engineering", "engineering", "public"]

    # Binding is scoped to the block — back to fail-closed afterwards.
    assert store.similarity_search("x", k=10) == []


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_untranslatable_bet_fails_closed(factory, fake_authz):
    """A partial subject leaks unbound subject attributes into the residual → untranslatable.
    Fail closed rather than silently returning the whole corpus."""
    store = to_authorized_vectorstore(factory(), AuthZConfig(connection=PDPConnection(fail_open=False), retrieval=RetrievalConfig()))
    fake_authz(BET_UNBOUND_SUBJECT)

    assert store.similarity_search("anything", k=10) == []


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_untranslatable_bet_fail_open_returns_unfiltered(factory, fake_authz):
    """Under fail_open, an untranslatable residual passes through (opt-in)."""
    store = to_authorized_vectorstore(factory(), AuthZConfig(connection=PDPConnection(fail_open=True), retrieval=RetrievalConfig()))
    fake_authz(BET_UNBOUND_SUBJECT)

    assert len(store.similarity_search("anything", k=10)) == 4


@pytest.mark.parametrize("fail_open,expected_len", [(False, 0), (True, 4)])
def test_all_none_subject_is_treated_as_no_identity(monkeypatch, fail_open, expected_len):
    """A provider returning only None-valued attributes is no identity — identical to ``{}``:
    fail closed by default, pass through under fail_open. Nulls must not slip past the guard,
    and ADS must never be consulted."""
    from axiomatics_ai.core.pdp import PdpClient
    from axiomatics_ai.core.config import DefaultClientContextProvider
    from axiomatics_ai.rag import authz_context

    class NoneProvider(DefaultClientContextProvider):
        def get_subject(self, ctx):
            return {"username": None, "user_id": None}

    def _fail(self, **kw):
        raise AssertionError("ADS must not be queried when there is no subject identity")

    monkeypatch.setattr(PdpClient, "get_constraints", _fail)

    store = to_authorized_vectorstore(
        make_inmemory(), AuthZConfig(context_provider=NoneProvider(), connection=PDPConnection(fail_open=fail_open), retrieval=RetrievalConfig())
    )
    # Context resolves (non-None), but the provider's subject strips to {} → no identity.
    with authz_context({"anything": 1}):
        assert len(store.similarity_search("x", k=10)) == expected_len


@pytest.mark.parametrize("config", [None, AuthZConfig()])
def test_wrapping_without_retrieval_config_raises(config):
    # A store that looks authorized but searches unfiltered is worse than an error.
    with pytest.raises(ValueError, match="retrieval"):
        to_authorized_vectorstore(make_inmemory(), config)


def test_ads_queried_once_per_search(fake_authz, monkeypatch):
    """The re-entrancy guard prevents a delegating search method from re-querying ADS."""
    from axiomatics_ai.core.pdp import PdpClient

    calls = {"n": 0}
    fake_authz(BET_DEPT_OR_PUBLIC)
    original = PdpClient.get_constraints

    def counting(self, **kw):
        calls["n"] += 1
        return original(self, **kw)

    monkeypatch.setattr(PdpClient, "get_constraints", counting)
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))
    store.similarity_search("anything", k=10)

    assert calls["n"] == 1


# ── sync / async parity ────────────────────────────────────────────────────────────
# The sync and async wrappers are separate closures around one shared decision
# (`_plan_search`). If they ever drift, one call path silently skips filtering while the
# other enforces — so the async surface is pinned against the sync one here.
# (asyncio.run rather than pytest-asyncio: this repo has no async test plugin configured.)

@pytest.mark.parametrize("bet,expected", [
    pytest.param(BET_DEPT_OR_PUBLIC, ["engineering", "engineering", "public"], id="filtered"),
    pytest.param(BET_PERMIT_ALL, ["engineering", "engineering", "finance", "public"], id="permit-all"),
    pytest.param(BET_DENY_ALL, [], id="deny-all"),
    pytest.param(BET_UNBOUND_SUBJECT, [], id="untranslatable-fails-closed"),
])
def test_async_search_matches_sync_search(bet, expected, fake_authz):
    fake_authz(bet)
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))

    sync_results = store.similarity_search("anything", k=10)
    async_results = asyncio.run(store.asimilarity_search("anything", k=10))

    assert _departments(sync_results) == expected
    assert _departments(async_results) == expected


def test_async_wrapper_does_not_double_filter_a_nested_search(fake_authz, monkeypatch):
    # The bypass flag exists so a store whose async search delegates to its own sync search
    # is not filtered twice (which would merge the filter into already-filtered kwargs).
    from axiomatics_ai.core.pdp import PdpClient

    calls = {"n": 0}
    fake_authz(BET_DEPT_OR_PUBLIC)
    original = PdpClient.get_constraints   # the fake: captured after fake_authz, never the network

    def counting(self, **kw):
        calls["n"] += 1
        return original(self, **kw)

    monkeypatch.setattr(PdpClient, "get_constraints", counting)
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))

    asyncio.run(store.asimilarity_search("anything", k=10))

    assert calls["n"] == 1


# ── fail closed on constraints the translator cannot render ────────────────────────────

def _not(node):
    return {"apply": {"function": "not", "arguments": [node]}}


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_not_filters_the_negated_documents(factory, fake_authz):
    from tests.rag.conftest import bag_eq

    store = to_authorized_vectorstore(factory(), AuthZConfig(retrieval=RetrievalConfig()))
    fake_authz(_not(bag_eq("resource.department", "finance")))

    results = _departments(store.similarity_search("anything", k=10))
    assert "finance" not in results
    if factory.__name__ == "make_qdrant":
        # Qdrant's LangChain translator has no != comparator: the filter can't be rendered, so
        # the search fails closed instead of returning unfiltered documents.
        assert results == []


def test_unknown_function_returns_nothing(fake_authz):
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))
    fake_authz({"apply": {"function": "stringStartsWith", "arguments": [
        {"constant": {"value": "fin", "type": "string"}},
        {"attributeDesignator": {"attributeName": "resource.department"}},
    ]}})

    assert store.similarity_search("anything", k=10) == []


# ── every search method is filtered ───────────────────────────────────────────────────

def test_by_vector_searches_are_filtered(fake_authz):
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))
    fake_authz(BET_DENY_ALL)
    vector = store.embeddings.embed_query("anything")

    assert store.similarity_search_by_vector(vector, k=10) == []
    assert store.max_marginal_relevance_search_by_vector(vector, k=4) == []
    assert store.similarity_search_with_score_by_vector(vector, k=10) == []
    assert asyncio.run(store.asimilarity_search_by_vector(vector, k=10)) == []
    assert asyncio.run(store.amax_marginal_relevance_search_by_vector(vector, k=4)) == []


def test_generic_search_is_filtered(fake_authz):
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))
    fake_authz(BET_DEPT_OR_PUBLIC)

    results = store.search("anything", search_type="similarity", k=10)
    assert _departments(results) == ["engineering", "engineering", "public"]


def test_async_search_reverse_query_runs_off_the_event_loop(monkeypatch):
    import threading

    from axiomatics_ai.core.pdp import PdpClient

    threads = []
    monkeypatch.setattr(wrap_module_for_tests(), "_resolve_context", lambda: {"user_id": "u1"})
    monkeypatch.setattr(PdpClient, "get_constraints",
                        lambda self, **kw: threads.append(threading.current_thread()) or BET_DEPT_OR_PUBLIC)
    store = to_authorized_vectorstore(make_inmemory(), AuthZConfig(retrieval=RetrievalConfig()))

    async def run():
        return threading.current_thread(), await store.asimilarity_search("anything", k=10)

    loop_thread, results = asyncio.run(run())
    assert _departments(results) == ["engineering", "engineering", "public"]
    assert threads and threads[0] is not loop_thread


def wrap_module_for_tests():
    from axiomatics_ai.rag import wrap as wrap_module

    return wrap_module
