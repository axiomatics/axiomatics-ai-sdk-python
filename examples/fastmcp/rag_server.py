"""
Example: authorization-filtered RAG inside a FastMCP tool.

A RAG tool searches a vector store wrapped with ``to_authorized_vectorstore``. That wrapper
resolves the caller from the *ambient* authorization context — which nothing populates on a
FastMCP server. ``FastMCPAuthZMiddleware`` closes the gap: its ``on_call_tool`` binds the
request's ``Authorization`` header into the ambient context for the duration of each call
(regardless of which enforcement points are configured), so the wrapped store filters per-user with **no
per-tool wiring**. The tool body is the ordinary ``vector_store.similarity_search(query, k=...)``
— unchanged.

Here all three enforcement flags are **off**, so the middleware only re-establishes identity
for RAG filtering — the "backend RAG server behind the gateway" topology, where the
gateway already did visibility / invocation / post-invocation. The gateway does not pass the
caller's token on by default (no token passthrough), so that route needs
``Route(..., forward_authorization=True)`` for this server to see the identity. In a single-process server you
would leave those flags on; RAG filtering works the same either way.

The policy (domain_rag_retrieval.yaml) permits a document iff
    (doc.department == user.department OR doc.department == "public")
    AND doc.level <= user.level

so with the corpus below:
  alice (engineering, level 2) → engineering L1/L2 + public L1   (not engineering L3, not finance)
  bob   (finance, level 3)     → finance L1/L3 + public L1

Identity travels in the JWT: its ``username``/``department``/``level`` claims are forwarded as
XACML subject attributes by the default server-side provider. Claims are read without
signature verification — in production the token is authenticated upstream of this server;
see the trust-boundary note in the fastmcp README.

Prerequisites:
  - SDK installed with RAG + fastmcp extras: uv sync --extra rag --extra fastmcp (the examples dependency group comes with uv sync)
  - ADS running at PDP_BASE_URL with domain_rag_retrieval.yaml loaded

Run the server:
    uv run python examples/fastmcp/rag_server.py

    # Custom ADS URL and server port
    PDP_BASE_URL=http://myads:8080 MCP_PORT=8002 uv run python examples/fastmcp/rag_server.py

Then query it with rag_client.py.

Domain: examples/policies/domain_rag_retrieval.yaml
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from axiomatics_ai import AuthZConfig, to_authorized_vectorstore
from axiomatics_ai.fastmcp import DefaultServerContextProvider, FastMCPAuthZMiddleware
from axiomatics_ai import PDPConnection, RetrievalConfig

load_dotenv()

_PDP_BASE_URL = os.getenv("PDP_BASE_URL", "http://localhost:8080")
_MCP_PORT = int(os.getenv("MCP_PORT", "8002"))

# One config, shared by the store wrapper and the middleware. A server-side provider reads
# identity from the HTTP headers the middleware binds. Enforcement flags are off: the
# middleware only re-establishes the caller's identity so the wrapped store filters (the
# RAG-only backend topology). Leave the flags on in a single-process server to also enforce
# tool visibility / invocation / post-invocation — RAG filtering works the same either way.
_config = AuthZConfig(
    connection=PDPConnection(base_url=_PDP_BASE_URL),
    context_provider=DefaultServerContextProvider(),
    retrieval=RetrievalConfig(),  # RAG-only backend: retrieval filtering, no tool enforcement
)


# ── Corpus + authorized vector store ─────────────────────────────────────────────

_DOCS = [
    Document("Engineering onboarding guide.", metadata={"department": "engineering", "level": 1}),
    Document("Engineering architecture roadmap.", metadata={"department": "engineering", "level": 2}),
    Document("Confidential engineering incident review.", metadata={"department": "engineering", "level": 3}),
    Document("Company holiday policy.", metadata={"department": "public", "level": 1}),
    Document("Finance quarterly revenue report.", metadata={"department": "finance", "level": 3}),
    Document("Finance expense guidelines.", metadata={"department": "finance", "level": 1}),
]

_vector_store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
_vector_store.add_documents(_DOCS)
# One line: every search on this store is now authorization-filtered.
_vector_store = to_authorized_vectorstore(_vector_store, _config)


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("axio-rag-example")


@mcp.tool()
def search_documentation(query: str) -> str:
    """Search internal documentation. Returns a JSON list of the matching titles."""
    docs = _vector_store.similarity_search(query, k=10)  # unchanged — filtered by ADS
    return json.dumps([d.page_content for d in docs])


# stdio is not supported; use http or sse.
mcp.add_middleware(FastMCPAuthZMiddleware(_config))


if __name__ == "__main__":
    mcp.run(transport="http", port=_MCP_PORT)
