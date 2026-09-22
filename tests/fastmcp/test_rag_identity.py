"""
Tests for the ambient-identity binding FastMCPAuthZMiddleware performs around a tool call,
which is what makes a wrapped RAG store filter inside a FastMCP tool.

``on_call_tool`` binds the request's HTTP headers into the ambient authz context (the
``authz_context`` contextvar that ``to_authorized_vectorstore`` reads) for the duration of
the call. Nothing else populates that contextvar on a FastMCP server, so without this a
wrapped store searched inside a tool has no subject and returns unfiltered (or fails closed).

Asserted here:
  - inside a real FastMCP server, the ADS reverse query receives the JWT subject and the
    resulting native filter reaches the wrapped store's search kwargs — with the enforcement
    flags off (RAG-only backend) and with tool invocation on (RAG coexisting with enforcement);
  - no Authorization header → no subject → the store is left unfiltered;
  - the bound identity is task-local: two concurrent requests from different identities never
    leak into each other.

ADS is stubbed at ``PdpClient.get_constraints`` and get_http_headers at the dependencies
level, so no real ADS is required.
"""
from __future__ import annotations

import asyncio
import contextvars
import json

import jwt
import pytest
from fastmcp import Client, FastMCP

import axiomatics_ai.core.enforcement as enforcement_module
import axiomatics_ai.fastmcp.fastmcp_middleware as fmw_module
from axiomatics_ai.core.pdp import PdpClient
from axiomatics_ai.core.authz_helpers import decode_bearer_jwt_unverified
from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider, PDPConnection, RetrievalConfig, standard_tool_enforcement
from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware
from axiomatics_ai.fastmcp.fastmcp_middleware import bind_ambient_identity
from axiomatics_ai.rag import to_authorized_vectorstore, wrap as wrap_module
from tests.rag.conftest import BET_DEPT_OR_PUBLIC
from tests.pdp_fakes import decision_seam


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def _make_jwt(claims: dict) -> str:
    return jwt.encode(claims, "secret", algorithm="HS256")


def _auth_header(claims: dict) -> dict[str, str]:
    return {"authorization": f"Bearer {_make_jwt(claims)}"}


def _make_store():
    """A real InMemoryVectorStore (a supported predicate store) that records the ``filter``
    kwarg reaching its search, so we can assert the authorization filter is pushed down."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.vectorstores import InMemoryVectorStore

    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    store.add_documents([
        Document(page_content="eng doc", metadata={"department": "engineering"}),
        Document(page_content="public doc", metadata={"department": "public"}),
        Document(page_content="finance doc", metadata={"department": "finance"}),
    ])

    captured: dict = {}
    original = store.similarity_search

    def recording(query, **kwargs):
        captured["kwargs"] = kwargs
        return original(query, **kwargs)

    store.similarity_search = recording  # instance attr; wrapped by to_authorized_vectorstore
    return store, captured


# ── RAG filtering inside a FastMCP tool ──────────────────────────────────────────

@pytest.mark.parametrize("config,extra_setup", [
    pytest.param(
        AuthZConfig(
            context_provider=DefaultServerContextProvider(),
            retrieval=RetrievalConfig(),
        ),
        None,
        id="rag-only-backend",  # all enforcement off — the gateway-backend topology
    ),
    pytest.param(
        AuthZConfig(
            context_provider=DefaultServerContextProvider(),
            enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
            retrieval=RetrievalConfig(),
        ),
        "permit_invocation",  # enforcement on — RAG filtering coexists with it
        id="with-invocation-enforcement",
    ),
])
def test_wrapped_store_is_filtered_inside_tool(monkeypatch, config, extra_setup):
    """The tool searches a wrapped store; the middleware binds identity so it is filtered,
    whether or not the other enforcement points are enabled."""
    captured_provided: dict = {}

    def fake_constraints(self, *, provided_attributes, **kw):
        captured_provided["provided"] = provided_attributes
        return BET_DEPT_OR_PUBLIC  # permit department ∈ {engineering, public}

    monkeypatch.setattr(PdpClient, "get_constraints", fake_constraints)
    monkeypatch.setattr(fmw_module, "get_http_headers",
                        lambda **kw: _auth_header({"username": "alice", "tenant_id": "acme"}))
    if extra_setup == "permit_invocation":
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))

    store, captured_kwargs = _make_store()
    # A store configured for server-side identity: the ambient context is the headers dict.
    to_authorized_vectorstore(store, config)

    mcp = FastMCP("rag-server")

    @mcp.tool()
    async def search_docs(query: str) -> str:
        """Search documents; returns the departments of the results."""
        docs = store.similarity_search(query, k=10)
        return json.dumps(sorted(d.metadata["department"] for d in docs))

    mcp.add_middleware(FastMCPAuthZMiddleware(config))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "anything"})

    result = _run(run())

    # The store returned only the permitted departments — the filter was pushed down.
    assert json.loads(result.data) == ["engineering", "public"]
    # The filter kwarg actually reached the store's search call.
    assert "filter" in captured_kwargs["kwargs"]
    # ADS received the JWT subject on the reverse query.
    provided = {p["attribute"]: p["values"] for p in captured_provided["provided"]}
    assert provided["username"] == ["alice"]
    assert provided["tenant_id"] == ["acme"]


@pytest.mark.parametrize("fail_open,expected", [
    pytest.param(False, [], id="fail-closed"),
    pytest.param(True, ["engineering", "finance", "public"], id="fail-open"),
])
def test_no_auth_header_fails_closed_without_consulting_ads(monkeypatch, fail_open, expected):
    """No Authorization header → empty subject → no identity to authorize against. The SDK
    fails closed (no documents) by default, or passes through only under fail_open — and does
    NOT consult ADS, since an unscoped reverse query would come back unfiltered and leak the
    corpus. This is the fix for the fail-open-on-missing-identity vulnerability."""
    def _fail(self, **kw):
        raise AssertionError("ADS must not be queried when there is no subject identity")

    monkeypatch.setattr(PdpClient, "get_constraints", _fail)
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: {})

    config = AuthZConfig(
        context_provider=DefaultServerContextProvider(),
        connection=PDPConnection(fail_open=fail_open),
        retrieval=RetrievalConfig(),
    )
    store, _ = _make_store()
    to_authorized_vectorstore(store, config)

    mcp = FastMCP("rag-server")

    @mcp.tool()
    async def search_docs(query: str) -> str:
        """Search documents."""
        docs = store.similarity_search(query, k=10)
        return json.dumps(sorted(d.metadata["department"] for d in docs))

    mcp.add_middleware(FastMCPAuthZMiddleware(config))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "anything"})

    result = _run(run())
    assert json.loads(result.data) == expected


# ── Task-locality of the bound identity ──────────────────────────────────────────

class _FakeMessage:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeContext:
    """Minimal stand-in for a FastMCP MiddlewareContext for on_call_tool."""
    def __init__(self, name="search_docs", arguments=None):
        self.message = _FakeMessage(name, arguments or {})


def test_bound_identity_stays_task_local(monkeypatch):
    """Two concurrent tool calls from different identities do not leak into each other.

    Each request runs in its own asyncio task; the contextvar the middleware sets is
    copy-on-task, so binding alice's headers in one task never affects bob's even when the
    two interleave across an await.
    """
    # get_http_headers reports whatever the running task set — mirrors a per-request header.
    current_headers: contextvars.ContextVar[dict] = contextvars.ContextVar("current_headers")
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: current_headers.get())

    mw = FastMCPAuthZMiddleware(AuthZConfig(
        context_provider=DefaultServerContextProvider(),
        enforcement=None,
    ))
    seen: dict[str, str] = {}

    async def one_request(user: str):
        current_headers.set(_auth_header({"username": user}))

        async def call_next(_ctx):
            # Yield so the two requests interleave while both have identity bound.
            await asyncio.sleep(0.02)
            bound = wrap_module._resolve_context()
            seen[user] = decode_bearer_jwt_unverified(bound)["username"]
            return "ok"

        await mw.on_call_tool(_FakeContext(), call_next)

    async def run():
        await asyncio.gather(one_request("alice"), one_request("bob"))

    _run(run())
    assert seen == {"alice": "alice", "bob": "bob"}


def test_no_double_binding_when_identity_already_bound(monkeypatch):
    """bind_ambient_identity is a no-op when an outer scope already bound an identity,
    so it never clobbers a tool's own explicit authz_context."""
    from axiomatics_ai.rag import authz_context

    monkeypatch.setattr(fmw_module, "get_http_headers",
                        lambda **kw: _auth_header({"username": "inner"}))

    outer = _auth_header({"username": "outer"})
    with authz_context(outer), bind_ambient_identity():
        # The outer binding is preserved — the request headers are not applied.
        assert decode_bearer_jwt_unverified(wrap_module._resolve_context())["username"] == "outer"
    # Cleanly restored afterwards.
    assert wrap_module._resolve_context() is None
