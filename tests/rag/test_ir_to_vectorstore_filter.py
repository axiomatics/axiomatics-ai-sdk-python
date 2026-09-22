"""Unit tests for rendering the neutral IR into native store dialects.

Chroma and Qdrant are additionally covered end-to-end (real filtered search) in
test_wrap.py. OpenSearch and Pinecone need live services, so here we assert their exact
native filter output — proving our IR compiles to their query language precisely — plus a
render-level breadth matrix across many other dialects.

Requires langchain-community (the 'rag' extra) for the translators.
"""
from __future__ import annotations

import importlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.structured_query import StructuredQuery
from langchain_core.vectorstores import InMemoryVectorStore

from axiomatics_ai.rag.ir_to_vectorstore_filter import (
    UnsupportedVectorStoreError,
    _to_langchain_ir,
    ir_to_filter_kwargs,
)

# (department == engineering OR department == public) AND level <= 1
_TREE = {"and": [
    {"or": [{"cmp": "eq", "key": "department", "value": "engineering"},
            {"cmp": "eq", "key": "department", "value": "public"}]},
    {"cmp": "le", "key": "level", "value": 1},
]}


def _render(module: str, translator_cls: str):
    mod = importlib.import_module(f"langchain_community.query_constructors.{module}")
    translator = getattr(mod, translator_cls)()
    _query, kwargs = translator.visit_structured_query(StructuredQuery(query="", filter=_to_langchain_ir(_TREE)))
    return kwargs


# ── the four the integration targets: chroma, qdrant (e2e in test_wrap.py) +
#    opensearch, pinecone (exact dialect here, since they need live services) ──────

def test_pinecone_exact_dialect():
    assert _render("pinecone", "PineconeTranslator")["filter"] == {"$and": [
        {"$or": [{"department": {"$eq": "engineering"}}, {"department": {"$eq": "public"}}]},
        {"level": {"$lte": 1}},
    ]}


def test_opensearch_exact_dialect():
    assert _render("opensearch", "OpenSearchTranslator")["filter"] == {"bool": {"must": [
        {"bool": {"should": [
            {"term": {"metadata.department.keyword": "engineering"}},
            {"term": {"metadata.department.keyword": "public"}},
        ]}},
        {"range": {"metadata.level": {"lte": 1}}},
    ]}}


def test_chroma_exact_dialect():
    assert _render("chroma", "ChromaTranslator")["filter"] == {"$and": [
        {"$or": [{"department": {"$eq": "engineering"}}, {"department": {"$eq": "public"}}]},
        {"level": {"$lte": 1}},
    ]}


# ── render-level breadth across many dialects ───────────────────────────────────
# Each store expects the filter under a different kwarg key; assert our IR renders to
# that dialect under the right key. (Qdrant excluded — its translator needs qdrant-client
# to build native objects; it is covered end-to-end in test_wrap.py.)

_DIALECTS = [
    ("chroma", "filter"), ("pinecone", "filter"), ("opensearch", "filter"),
    ("elasticsearch", "filter"), ("pgvector", "filter"), ("astradb", "filter"),
    ("dashvector", "filter"), ("vectara", "filter"), ("milvus", "expr"),
    ("weaviate", "where_filter"), ("mongodb_atlas", "pre_filter"),
    ("deeplake", "tql"), ("supabase", "postgrest_filter"), ("dingo", "search_params"),
]


@pytest.mark.parametrize("module,expected_key", _DIALECTS)
def test_dialect_renders_under_expected_kwarg(module, expected_key):
    mod = importlib.import_module(f"langchain_community.query_constructors.{module}")
    translator_cls = next(c for c in dir(mod) if c.endswith("Translator"))
    ir = _to_langchain_ir({"cmp": "eq", "key": "department", "value": "engineering"})
    _query, kwargs = getattr(mod, translator_cls)().visit_structured_query(
        StructuredQuery(query="", filter=ir)
    )
    assert expected_key in kwargs, f"{module}: expected {expected_key!r}, got {list(kwargs)}"
    assert "engineering" in str(kwargs[expected_key])


def test_membership_or_of_equals_renders_on_chroma():
    # caq_to_ir lowers membership to OR-of-equals so it works even on stores (Chroma)
    # whose translator rejects the IN comparator.
    from langchain_community.query_constructors.chroma import ChromaTranslator

    tree = {"or": [{"cmp": "eq", "key": "region", "value": "eu"},
                   {"cmp": "eq", "key": "region", "value": "us"}]}
    _query, kwargs = ChromaTranslator().visit_structured_query(
        StructuredQuery(query="", filter=_to_langchain_ir(tree))
    )
    assert kwargs["filter"] == {"$or": [{"region": {"$eq": "eu"}}, {"region": {"$eq": "us"}}]}


# ── in-memory predicate ────────────────────────────────────────────────────────

def test_inmemory_returns_filter_predicate_that_matches_metadata():
    store = InMemoryVectorStore(FakeEmbeddings(size=8))
    predicate = ir_to_filter_kwargs(_TREE, store)["filter"]
    assert callable(predicate)
    assert predicate(Document("x", metadata={"department": "engineering", "level": 1})) is True
    assert predicate(Document("x", metadata={"department": "finance", "level": 3})) is False
    assert predicate(Document("x", metadata={})) is False


# ── resolver robustness ────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_inmemory_comparison_on_a_missing_key_is_false(op):
    # XACML: a comparison against an empty attribute bag does not match, so a document
    # without `classification` is not let through by `classification != "secret"`.
    store = InMemoryVectorStore(FakeEmbeddings(size=4))
    predicate = ir_to_filter_kwargs({"cmp": op, "key": "classification", "value": "secret"}, store)["filter"]
    assert predicate(Document("x", metadata={})) is False
    assert predicate(Document("x", metadata={"classification": None})) is False


def test_inmemory_ne_matches_a_present_different_value():
    store = InMemoryVectorStore(FakeEmbeddings(size=4))
    predicate = ir_to_filter_kwargs({"cmp": "ne", "key": "classification", "value": "secret"}, store)["filter"]
    assert predicate(Document("x", metadata={"classification": "public"})) is True
    assert predicate(Document("x", metadata={"classification": "secret"})) is False


def test_fallback_resolver_used_when_langchain_resolver_broken(monkeypatch):
    """If LangChain's resolver raises ImportError (version skew), fall back to our map."""
    import axiomatics_ai.rag.ir_to_vectorstore_filter as mod

    class Chroma:  # name matches our fallback map entry
        pass

    import langchain_classic.retrievers.self_query.base as base

    def _boom(_vs):
        raise ImportError("cannot import name 'DatabricksVectorSearch'")

    monkeypatch.setattr(base, "_get_builtin_translator", _boom)
    assert type(mod._get_translator(Chroma())).__name__ == "ChromaTranslator"


def test_unsupported_store_raises():
    class WeirdStore:
        pass

    with pytest.raises(UnsupportedVectorStoreError):
        ir_to_filter_kwargs(_TREE, WeirdStore())
