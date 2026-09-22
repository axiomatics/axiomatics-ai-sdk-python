"""Unit tests for compute_filter orchestration (fake provider + fake ADS)."""
from __future__ import annotations

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.rag.filtering import compute_filter
from tests.rag.conftest import BET_DENY_ALL, BET_DEPT_OR_PUBLIC, BET_PERMIT_ALL
from axiomatics_ai import PDPConnection, RetrievalConfig


class _Provider:
    def get_subject(self, ctx):
        return {"user_id": ctx["user_id"], "department": ctx["department"]}

    def get_action(self, name, ctx):
        return {"name": name}

    def get_environment(self, ctx):
        return {"session_id": ctx.get("session_id")} if ctx.get("session_id") else {}


class _Pdp:
    def __init__(self, bet):
        self.bet = bet
        self.provided = None

    def get_constraints(self, *, provided_attributes):
        self.provided = provided_attributes
        return self.bet


def _store():
    return InMemoryVectorStore(FakeEmbeddings(size=8))


def _run(bet, config=None, ctx=None):
    ads = _Pdp(bet)
    outcome = compute_filter(
        vectorstore=_store(),
        provider=_Provider(),
        context=ctx or {"user_id": "u1", "department": "engineering"},
        config=config or AuthZConfig(retrieval=RetrievalConfig()),
        pdp=ads,
    )
    return outcome, ads


def test_permit_all_no_filter():
    outcome, _ = _run(BET_PERMIT_ALL)
    assert outcome.deny_all is False
    assert outcome.filter_kwargs is None


def test_deny_all():
    outcome, _ = _run(BET_DENY_ALL)
    assert outcome.deny_all is True


def test_constraint_produces_filter_kwargs():
    outcome, _ = _run(BET_DEPT_OR_PUBLIC)
    assert outcome.deny_all is False
    assert callable(outcome.filter_kwargs["filter"])  # in-memory predicate


def test_provided_attributes_include_subject_action_and_environment():
    _outcome, ads = _run(BET_PERMIT_ALL, ctx={"user_id": "u1", "department": "eng", "session_id": "s9"})
    assert {"attribute": "user_id", "values": ["u1"]} in ads.provided
    assert {"attribute": "department", "values": ["eng"]} in ads.provided
    assert {"attribute": "name", "values": ["retrieval"]} in ads.provided  # default retrieval action
    assert {"attribute": "session_id", "values": ["s9"]} in ads.provided


def test_custom_retrieval_action_name():
    _outcome, ads = _run(BET_PERMIT_ALL, config=AuthZConfig(retrieval=RetrievalConfig(action_name="doc_read")))
    assert {"attribute": "name", "values": ["doc_read"]} in ads.provided


def test_ads_unreachable_fails_closed():
    outcome, _ = _run(None, config=AuthZConfig(connection=PDPConnection(fail_open=False), retrieval=RetrievalConfig()))
    assert outcome.deny_all is True


def test_ads_unreachable_fail_open():
    outcome, _ = _run(None, config=AuthZConfig(connection=PDPConnection(fail_open=True), retrieval=RetrievalConfig()))
    assert outcome.deny_all is False
    assert outcome.filter_kwargs is None
