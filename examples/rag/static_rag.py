"""
Example: authorization-filtered retrieval in a static (non-agent) RAG pipeline.

Retrieval here is a plain ``as_retriever()`` used directly — no agent, so there is no agent
runtime to read identity from. Instead the caller identity is bound per request with
``with authz_context(user): ...``. That is backed by a contextvar, so one shared wrapped
store can safely serve many concurrent users (each request only sees its own identity).

This example uses a **real Qdrant store** in in-memory mode (``:memory:``) — so it exercises
Qdrant's native filter dialect while creating no files on disk. No LLM is required; the
script just runs the retriever and prints what each user gets, so it is fully runnable
against ADS alone.

The policy (domain_rag_retrieval.yaml) permits a document iff
    (doc.department == user.department OR doc.department == "public")
    AND doc.level <= user.level

Prerequisites:
  - SDK installed with the RAG + examples extras: uv sync --extra rag (in this repo; the examples group comes with uv sync)
  - ADS running at PDP_BASE_URL with domain_rag_retrieval.yaml loaded

Run:
    uv run python static_rag.py

Domain: examples/policies/domain_rag_retrieval.yaml
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from axiomatics_ai import AuthZConfig, authz_context, to_authorized_vectorstore
from axiomatics_ai import PDPConnection, RetrievalConfig

_PDP_BASE_URL = os.getenv("PDP_BASE_URL", "http://localhost:8080")
_VECTOR_SIZE = 32


@dataclass
class UserContext:
    username: str
    department: str
    level: int


class UserContextProvider:
    def get_subject(self, ctx: UserContext) -> dict[str, Any]:
        return {"username": ctx.username, "department": ctx.department, "level": ctx.level}

    def get_action(self, action_name: str, ctx: UserContext) -> dict[str, Any]:
        return {"name": action_name}

    def get_resource(
        self, resource_name: str, ctx: UserContext, *, args=None, resource_name_attribute: str | None = None
    ) -> dict[str, Any]:
        return {resource_name_attribute: resource_name} if resource_name_attribute else {}

    def get_environment(self, ctx: UserContext) -> dict[str, Any]:
        return {}


_DOCS = [
    Document("Engineering onboarding guide.", metadata={"department": "engineering", "level": 1}),
    Document("Engineering architecture roadmap.", metadata={"department": "engineering", "level": 2}),
    Document("Confidential engineering incident review.", metadata={"department": "engineering", "level": 3}),
    Document("Company holiday policy.", metadata={"department": "public", "level": 1}),
    Document("Finance quarterly revenue report.", metadata={"department": "finance", "level": 3}),
    Document("Finance expense guidelines.", metadata={"department": "finance", "level": 1}),
]


def _build_store() -> QdrantVectorStore:
    """A real Qdrant store, entirely in memory — no files written to disk."""
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "docs", vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE)
    )
    store = QdrantVectorStore(
        client=client, collection_name="docs", embedding=DeterministicFakeEmbedding(size=_VECTOR_SIZE)
    )
    store.add_documents(_DOCS)
    return store


def main() -> None:
    config = AuthZConfig(connection=PDPConnection(base_url=_PDP_BASE_URL), retrieval=RetrievalConfig(), context_provider=UserContextProvider())

    # Wrap once; the same shared store/retriever is reused for every user.
    store = to_authorized_vectorstore(_build_store(), config)
    retriever = store.as_retriever(search_kwargs={"k": 10})

    alice = UserContext(username="alice", department="engineering", level=2)
    bob = UserContext(username="bob", department="finance", level=3)

    for user in (alice, bob):
        print(f"\n── {user.username} (dept={user.department}, level={user.level}) ──")
        # Per-request identity — task-local, safe under concurrency.
        with authz_context(user):
            docs = retriever.invoke("company documentation")
        for d in docs:
            print(f"  - {d.page_content} [{d.metadata['department']}/L{d.metadata['level']}]")


if __name__ == "__main__":
    main()
