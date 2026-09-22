"""Shared fixtures/helpers for RAG retrieval-filtering tests.

Sets a protobuf compatibility flag before any store package (Chroma) imports it, then
provides real, locally-runnable vector stores (InMemory, Chroma, Qdrant) seeded with the
same documents, plus a helper to fake the ADS reverse query and the ambient identity.
"""
from __future__ import annotations

import os
import uuid

# chromadb pulls in protobuf-generated modules that fail under the C++ descriptor impl on
# newer Python; force the pure-Python impl before anything imports protobuf.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from axiomatics_ai.core.pdp import PdpClient
from axiomatics_ai.rag import wrap as wrap_module

_VECTOR_SIZE = 16

# Corpus: department ∈ {engineering, public, finance}, level ∈ {1, 2, 3}.
DOCS = [
    Document(page_content="engineering onboarding", metadata={"department": "engineering", "level": 1}),
    Document(page_content="engineering roadmap", metadata={"department": "engineering", "level": 2}),
    Document(page_content="public holiday policy", metadata={"department": "public", "level": 1}),
    Document(page_content="finance q3 revenue", metadata={"department": "finance", "level": 3}),
]


def _embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=_VECTOR_SIZE)


# ── store factories (each returns a fresh store seeded with DOCS) ────────────────

def make_inmemory():
    from langchain_core.vectorstores import InMemoryVectorStore

    store = InMemoryVectorStore(_embeddings())
    store.add_documents(DOCS)
    return store


def make_chroma():
    pytest.importorskip("langchain_chroma")
    pytest.importorskip("chromadb")
    from langchain_chroma import Chroma

    # Unique collection per instance — Chroma persists collections by name within a process.
    store = Chroma(embedding_function=_embeddings(), collection_name=f"rag_docs_{uuid.uuid4().hex}")
    store.add_documents(DOCS)
    return store


def make_qdrant():
    pytest.importorskip("langchain_qdrant")
    pytest.importorskip("qdrant_client")
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    client = QdrantClient(location=":memory:")
    client.create_collection(
        "rag_docs", vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE)
    )
    store = QdrantVectorStore(client=client, collection_name="rag_docs", embedding=_embeddings())
    store.add_documents(DOCS)
    return store


# Parametrization for end-to-end tests: real stores that run locally.
STORE_FACTORIES = [
    pytest.param(make_inmemory, id="inmemory"),
    pytest.param(make_chroma, id="chroma"),
    pytest.param(make_qdrant, id="qdrant"),
]


# ── ADS / identity fakes ──────────────────────────────────────────────────────

@pytest.fixture
def fake_authz(monkeypatch):
    """Return an ``install(bet, context=...)`` that fakes the ADS BET and ambient identity.

    ``bet`` is the raw ADS ``/authorization/constraints`` response body (or None to
    simulate ADS being unreachable). ``context`` is the identity the wrapper reads from
    the (faked) agent runtime.
    """
    def install(bet, context=None):
        monkeypatch.setattr(wrap_module, "_resolve_context", lambda: context or {"user_id": "u1"})
        monkeypatch.setattr(PdpClient, "get_constraints", lambda self, **kw: bet)

    return install


# ── BET builders (mirroring ADS /authorization/constraints output) ──────────────

def bag_eq(attr: str, value: str) -> dict:
    return {"apply": {"function": "anyOfAny", "arguments": [
        {"functionElement": {"name": "equal"}},
        {"attributeDesignator": {"attributeName": attr}},
        {"apply": {"function": "list", "arguments": [{"constant": {"value": value, "type": "string"}}]}},
    ]}}


def bet_or(*args) -> dict:
    return {"apply": {"function": "or", "arguments": list(args)}}


# Permit if department ∈ {engineering, public}.
BET_DEPT_OR_PUBLIC = bet_or(bag_eq("resource.department", "engineering"),
                            bag_eq("resource.department", "public"))
BET_PERMIT_ALL = {"constant": {"value": "true"}}
BET_DENY_ALL = {"constant": {"value": "false"}}


def bag_eq_unbound(attr: str, other_attr: str) -> dict:
    """``attr == other_attr`` with both operands designators (no constant) — the shape ADS
    returns when a subject attribute is left unbound (a partial/under-specified identity)."""
    return {"apply": {"function": "anyOfAny", "arguments": [
        {"functionElement": {"name": "equal"}},
        {"attributeDesignator": {"attributeName": attr}},
        {"attributeDesignator": {"attributeName": other_attr}},
    ]}}


# Residual from a partial subject: resource.department compared to the unbound subject
# attribute ``department`` — untranslatable, must fail closed rather than leak everything.
BET_UNBOUND_SUBJECT = bag_eq_unbound("resource.department", "department")
