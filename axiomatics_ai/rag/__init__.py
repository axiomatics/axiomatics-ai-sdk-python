"""Authorization-aware retrieval for RAG pipelines.

Wrap a vector store so every search is filtered by an ADS reverse query (CAQ):

    from axiomatics_ai.rag import to_authorized_vectorstore

    vector_store = to_authorized_vectorstore(vector_store, config)

The filter is pushed down into the store at query time, translated to the store's own
dialect via LangChain's query translators. Works unchanged inside deepagents and
``create_agent`` retrieval tools and with ``as_retriever()``.

Pipeline: CAQ BET ──caq_to_ir──▶ neutral IR ──ir_to_vectorstore_filter──▶ native filter.
"""
from axiomatics_ai._extras import requires_extra

with requires_extra("rag", "axiomatics_ai.rag"):
    from axiomatics_ai.rag.caq_to_ir import FilterResult, caq_to_ir
    from axiomatics_ai.rag.filtering import FilterOutcome, compute_filter
    from axiomatics_ai.rag.ir_to_vectorstore_filter import (
        UnsupportedVectorStoreError,
        ir_to_filter_kwargs,
    )
    from axiomatics_ai.rag.wrap import authz_context, to_authorized_vectorstore

__all__ = [
    "FilterOutcome",
    "FilterResult",
    "UnsupportedVectorStoreError",
    "authz_context",
    "caq_to_ir",
    "compute_filter",
    "ir_to_filter_kwargs",
    "to_authorized_vectorstore",
]
