"""Integration test: RAG retrieval filtering against the real ADS/CAQ service.

Starts ADS with examples/policies/domain_rag_retrieval.yaml, issues a CAQ reverse query for a
subject, and verifies the returned BET translates (via the SDK) into a filter that keeps
exactly the documents the subject may read. Auto-skipped when the ADS jar is absent.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from axiomatics_ai.core.pdp import PdpClient
from axiomatics_ai.rag import caq_to_ir, ir_to_filter_kwargs
from tests.conftest import ADS_URL

pytestmark = pytest.mark.ads_integration

# department ∈ {engineering, public, finance}; level ∈ {1, 2, 3}
_CORPUS = [
    Document("eng L1", metadata={"department": "engineering", "level": 1}),
    Document("eng L3", metadata={"department": "engineering", "level": 3}),
    Document("public L1", metadata={"department": "public", "level": 1}),
    Document("finance L1", metadata={"department": "finance", "level": 1}),
]


def _constraints(department: str, level: int) -> dict | None:
    ads = PdpClient(base_url=ADS_URL, timeout=10.0)
    return ads.get_constraints(provided_attributes=[
        {"attribute": "username", "values": ["bob"]},
        {"attribute": "department", "values": [department]},
        {"attribute": "level", "values": [level]},
        {"attribute": "name", "values": ["retrieval"]},
    ])


def _kept(bet: dict | None):
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore

    result = caq_to_ir(bet)
    assert result.deny_all is False and result.tree is not None
    predicate = ir_to_filter_kwargs(result.tree, InMemoryVectorStore(FakeEmbeddings(size=8)))["filter"]
    return sorted(d.page_content for d in _CORPUS if predicate(d))


@pytest.mark.usefixtures("ads_server")
class TestRagRetrievalCaqPolicy:
    domain_yaml = "domain_rag_retrieval.yaml"

    def test_engineering_level2_keeps_own_dept_and_public_within_clearance(self):
        # (dept ∈ {engineering, public}) AND level <= 2
        kept = _kept(_constraints("engineering", 2))
        assert kept == ["eng L1", "public L1"]  # eng L3 excluded by level, finance by dept

    def test_finance_level3_keeps_finance_and_public(self):
        kept = _kept(_constraints("finance", 3))
        assert kept == ["finance L1", "public L1"]
