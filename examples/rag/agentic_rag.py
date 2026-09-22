"""
Example: authorization-aware agentic RAG.

Retrieval is an ordinary agent tool that calls ``vector_store.similarity_search`` — the
default pattern from the LangChain docs. Wrapping the store once with
``to_authorized_vectorstore`` makes every search reverse-query ADS and push a
metadata filter down into the store, so each user only retrieves documents they are
allowed to see. The tool code is unchanged.

The policy (domain_rag_retrieval.yaml) permits a document iff
    (doc.department == user.department OR doc.department == "public")
    AND doc.level <= user.level

So with the corpus below:
  alice (engineering, level 2) → engineering L1/L2 + public L1   (not engineering L3, not finance)
  bob   (finance, level 3)     → finance L1/L3 + public L1

Prerequisites:
  - SDK installed with RAG + agents extras: uv sync --extra rag --extra langchain-agents (in this repo; the examples group comes with uv sync)
  - ADS running at PDP_BASE_URL with domain_rag_retrieval.yaml loaded
  - .env with LLM_MODEL and the appropriate API key

Run:
    uv run python agentic_rag.py

Domain: examples/policies/domain_rag_retrieval.yaml
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_litellm import ChatLiteLLM

from axiomatics_ai import AuthZConfig, to_authorized_vectorstore
from axiomatics_ai import PDPConnection, RetrievalConfig

load_dotenv()

_PDP_BASE_URL = os.getenv("PDP_BASE_URL", "http://localhost:8080")
_LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")
_LLM_API_BASE = os.getenv("LLM_API_BASE") or None


# ── Application context + provider (supplies department/level as subject attrs) ──

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


# ── Corpus + authorized vector store ─────────────────────────────────────────────

_DOCS = [
    Document("Engineering onboarding guide.", metadata={"department": "engineering", "level": 1}),
    Document("Engineering architecture roadmap.", metadata={"department": "engineering", "level": 2}),
    Document("Confidential engineering incident review.", metadata={"department": "engineering", "level": 3}),
    Document("Company holiday policy.", metadata={"department": "public", "level": 1}),
    Document("Finance quarterly revenue report.", metadata={"department": "finance", "level": 3}),
    Document("Finance expense guidelines.", metadata={"department": "finance", "level": 1}),
]

_config = AuthZConfig(connection=PDPConnection(base_url=_PDP_BASE_URL), retrieval=RetrievalConfig(), context_provider=UserContextProvider())

_vector_store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
_vector_store.add_documents(_DOCS)
# One line: every search on this store is now authorization-filtered.
_vector_store = to_authorized_vectorstore(_vector_store, _config)


@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search internal documentation.

    Args:
        query: What to search for.
    """
    docs = _vector_store.similarity_search(query, k=10)
    if not docs:
        return "No accessible documents matched."
    return "\n".join(f"- {d.page_content} [{d.metadata['department']}/L{d.metadata['level']}]" for d in docs)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_for_user(agent, ctx: UserContext, question: str) -> None:
    print(f"\n── {ctx.username} (dept={ctx.department}, level={ctx.level}) ──")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, context=ctx)
    print(result["messages"][-1].content)


def main() -> None:
    kwargs: dict = {"model": _LLM_MODEL}
    if _LLM_API_BASE:
        kwargs["api_base"] = _LLM_API_BASE

    agent = create_agent(model=ChatLiteLLM(**kwargs), tools=[search_documentation])

    alice = UserContext(username="alice", department="engineering", level=2)
    bob = UserContext(username="bob", department="finance", level=3)

    question = "Search the documentation for anything you can find and list the titles you retrieved."
    run_for_user(agent, alice, question)
    run_for_user(agent, bob, question)


if __name__ == "__main__":
    main()
