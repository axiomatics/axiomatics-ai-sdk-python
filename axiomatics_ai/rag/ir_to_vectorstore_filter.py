"""Render a store-neutral filter tree into the search arguments a vector store expects.

The tree comes from ``caq_to_ir``.

We return the translator's **entire keyword-argument dict**, not just a ``filter`` value,
because the metadata-filter kwarg name varies by store: Chroma/Pinecone/OpenSearch use
``filter``, Milvus uses ``expr``, Weaviate ``where_filter``, MongoDB Atlas ``pre_filter``,
Deeplake ``tql``, Supabase ``postgrest_filter``, and so on. The caller merges the whole
dict into the search call — exactly how LangChain's own ``SelfQueryRetriever`` applies it.

Two rendering paths:

* **Dict-filter stores** — we build LangChain's structured-query IR
  (:mod:`langchain_core.structured_query`) and hand it to the translator LangChain itself
  resolves via ``_get_builtin_translator``. We deliberately reuse LangChain's resolver
  rather than keep our own store→translator table: every dialect LangChain supports (and
  any it adds later, plus per-store quirks like Qdrant's ``metadata_key``) works without a
  code change here.
* **In-memory stores** (``InMemoryVectorStore``) — whose ``filter`` is a
  ``Callable[[Document], bool]`` rather than a dict, and which LangChain's resolver does
  not support — get a predicate compiled directly from the neutral tree.

The resolver lives in ``langchain-community`` (the SDK's ``rag`` extra); it is imported
lazily so importing this module never requires it.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

from axiomatics_ai._extras import requires_extra

logger = logging.getLogger(__name__)

# Neutral op → LangChain Comparator enum member name.
_CMP_TO_COMPARATOR = {
    "eq": "EQ", "ne": "NE", "gt": "GT", "gte": "GTE", "ge": "GTE",
    "lt": "LT", "lte": "LTE", "le": "LTE", "in": "IN",
}

# Store classes whose ``filter`` is a predicate callable, not a dict. These live in
# langchain-core (always available) and are not handled by LangChain's dict-filter
# resolver, so we render a predicate for them ourselves.
_PREDICATE_STORES = {"InMemoryVectorStore"}


class UnsupportedVectorStoreError(Exception):
    """Raised when the authorization filter can't be rendered for a vector store.

    The store is unknown to LangChain's resolver, the ``rag`` extra is missing, or its
    translator rejects part of the filter (e.g. Qdrant has no ``!=``).
    """


# ── neutral tree → LangChain structured-query IR ────────────────────────────────

def _to_langchain_ir(node: dict):
    """Our neutral IR → ``langchain_core.structured_query`` Comparison/Operation."""
    with requires_extra("rag", "Retrieval filtering"):
        from langchain_core.structured_query import (
            Comparator,
            Comparison,
            Operation,
            Operator,
        )

    if "and" in node:
        return Operation(operator=Operator.AND, arguments=[_to_langchain_ir(k) for k in node["and"]])
    if "or" in node:
        return Operation(operator=Operator.OR, arguments=[_to_langchain_ir(k) for k in node["or"]])
    comparator = getattr(Comparator, _CMP_TO_COMPARATOR[node["cmp"]])
    return Comparison(comparator=comparator, attribute=node["key"], value=node["value"])


# Robustness backstop for stores we guarantee. Keyed on the store's class name → the
# translator's ``query_constructors`` module + class, plus how to build it from the store
# instance (some translators need a store-derived arg). This is only consulted when
# LangChain's own resolver is unavailable/broken (see _get_translator); delegation stays
# primary so newly added dialects work without touching this map.
#   value: (module, translator_class, build) where build is:
#     None          → Translator()
#     "from_vectorstore" → Translator.from_vectorstore(vs)
#     callable(vs)  → Translator(**callable(vs))
_FALLBACK_TRANSLATORS: dict[str, tuple[str, str, Any]] = {
    "Chroma": ("chroma", "ChromaTranslator", None),
    "Pinecone": ("pinecone", "PineconeTranslator", None),
    "PineconeVectorStore": ("pinecone", "PineconeTranslator", None),
    "QdrantVectorStore": ("qdrant", "QdrantTranslator",
                          lambda vs: {"metadata_key": getattr(vs, "metadata_payload_key", "metadata")}),
    "Qdrant": ("qdrant", "QdrantTranslator",
               lambda vs: {"metadata_key": getattr(vs, "metadata_payload_key", "metadata")}),
    "OpenSearchVectorSearch": ("opensearch", "OpenSearchTranslator", None),
    "Milvus": ("milvus", "MilvusTranslator", None),
    "Zilliz": ("milvus", "MilvusTranslator", None),
    "PGVector": ("pgvector", "PGVectorTranslator", None),
    "ElasticsearchStore": ("elasticsearch", "ElasticsearchTranslator", None),
    "Weaviate": ("weaviate", "WeaviateTranslator", None),
    "WeaviateVectorStore": ("weaviate", "WeaviateTranslator", None),
    "MongoDBAtlasVectorSearch": ("mongodb_atlas", "MongoDBAtlasTranslator", None),
    "DatabricksVectorSearch": ("databricks_vector_search", "DatabricksVectorSearchTranslator", None),
    "Redis": ("redis", "RedisTranslator", "from_vectorstore"),
    "AstraDB": ("astradb", "AstraDBTranslator", None),
    "AstraDBVectorStore": ("astradb", "AstraDBTranslator", None),
    "SupabaseVectorStore": ("supabase", "SupabaseVectorTranslator", None),
}


def _get_translator(vectorstore: Any):
    """Resolve the query translator for ``vectorstore``.

    Delegates to LangChain's own ``_get_builtin_translator`` first, so we track its store
    coverage and per-store quirks automatically. That function eager-imports a block of
    optional store classes, so on some langchain-community versions it raises before
    inspecting the store; in that case we fall back to :data:`_FALLBACK_TRANSLATORS`,
    importing only the one translator module the store needs. Raises
    :class:`UnsupportedVectorStoreError` if neither path resolves.
    """
    try:
        from langchain_classic.retrievers.self_query.base import _get_builtin_translator

        return _get_builtin_translator(vectorstore)
    except ValueError as exc:  # resolver ran but doesn't support this store
        raise UnsupportedVectorStoreError(str(exc)) from exc
    except ImportError:
        pass  # resolver import block broke (version skew) — fall back below

    entry = _FALLBACK_TRANSLATORS.get(type(vectorstore).__name__)
    if entry is None:
        raise UnsupportedVectorStoreError(
            f"No query translator available for vector store '{type(vectorstore).__name__}'. "
            f"Install the SDK's 'rag' extra, or file an issue to add this store."
        )
    module_name, class_name, build = entry
    try:
        module = __import__(
            f"langchain_community.query_constructors.{module_name}", fromlist=[class_name]
        )
    except ImportError as exc:
        raise UnsupportedVectorStoreError(
            "Rendering vector-store metadata filters needs langchain-community. "
            "Install the SDK's 'rag' extra (e.g. `uv add 'axiomatics-ai-sdk[rag]'`)."
        ) from exc
    translator_cls = getattr(module, class_name)
    if build == "from_vectorstore":
        return translator_cls.from_vectorstore(vectorstore)
    if callable(build):
        return translator_cls(**build(vectorstore))
    return translator_cls()


def _render_dict_filter_kwargs(tree: dict, vectorstore: Any) -> dict:
    with requires_extra("rag", "Retrieval filtering"):
        from langchain_core.structured_query import StructuredQuery

    translator = _get_translator(vectorstore)
    structured = StructuredQuery(query="", filter=_to_langchain_ir(tree))
    try:
        _query, kwargs = translator.visit_structured_query(structured)
    except ValueError as exc:
        # The translator rejects part of the filter (e.g. Qdrant has no NE comparator).
        raise UnsupportedVectorStoreError(
            f"Translator for '{type(vectorstore).__name__}' cannot render the authorization "
            f"filter: {exc}"
        ) from exc
    if not kwargs:
        # A translator that produced no kwarg would mean an unfiltered search — refuse
        # rather than silently return everything for this subject.
        raise UnsupportedVectorStoreError(
            f"Translator for '{type(vectorstore).__name__}' produced no filter kwargs; "
            f"cannot safely apply the authorization filter."
        )
    return dict(kwargs)


# ── neutral tree → in-memory predicate ──────────────────────────────────────────

_PREDICATE_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
    "in": lambda a, b: a in b,
}


def _render_predicate(tree: dict) -> Callable[[Any], bool]:
    """Compile the neutral tree into a ``Callable[[Document], bool]`` over ``doc.metadata``.

    Every comparison on a missing (or ``None``) metadata key is false, ``ne`` included: in
    XACML a comparison against an empty attribute bag does not match, so a document without
    the attribute is never let through by ``classification != "secret"``.
    """

    def evaluate(node: dict, metadata: dict) -> bool:
        if "and" in node:
            return all(evaluate(k, metadata) for k in node["and"])
        if "or" in node:
            return any(evaluate(k, metadata) for k in node["or"])
        value = metadata.get(node["key"])
        return value is not None and _PREDICATE_OPS[node["cmp"]](value, node["value"])

    def predicate(doc: Any) -> bool:
        return evaluate(tree, getattr(doc, "metadata", {}) or {})

    return predicate


# ── public API ──────────────────────────────────────────────────────────────────

def ir_to_filter_kwargs(tree: dict, vectorstore: Any) -> dict:
    """Render the neutral filter tree into the search keyword arguments for a vector store.

    The keyword is whatever the store expects (``filter``, ``expr``, ``where_filter``, ...);
    in-memory stores get ``{"filter": <predicate>}``. The caller merges the dict into the
    search call.

    Args:
        tree (dict): The neutral filter tree.
        vectorstore (VectorStore): The store.

    Returns:
        dict: The search keyword arguments.

    Raises:
        UnsupportedVectorStoreError: No filter can be rendered for the store.
    """
    if type(vectorstore).__name__ in _PREDICATE_STORES:
        return {"filter": _render_predicate(tree)}
    return _render_dict_filter_kwargs(tree, vectorstore)
