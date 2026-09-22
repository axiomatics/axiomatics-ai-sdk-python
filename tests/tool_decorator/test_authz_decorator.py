"""
Unit tests for @protected_tool decorator.

Authorization model under test:
  - Tool always executes.
  - No post_processor → result returned as-is, ADS never called.
  - post_processor provided → always called; the strategy/callable is responsible
    for ADS interaction and transformation.

Each test builds a real FastMCP server with decorated tools, stubs out ADS at
the authz_helpers level and _get_http_headers at the decorator module level,
then uses fastmcp.Client to trigger tool calls and asserts on observable outcomes.

No real ADS instance is required.
"""
from __future__ import annotations

import asyncio

import axiomatics_ai.core.authz_helpers as helpers_module
import jwt
import pytest
from fastmcp import Client, FastMCP

import axiomatics_ai.tool_decorator.decorator as dec_module
from axiomatics_ai import ColumnMaskStrategy
from axiomatics_ai.core.pdp import PdpClient
from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.core.post_invocation import PostInvocationContext
from axiomatics_ai import PDPConnection, standard_tool_enforcement
from axiomatics_ai.tool_decorator import (
    configure_authz_decorator,
    protected_tool,
)
from tests.pdp_fakes import decision_seam


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def _make_jwt(claims: dict) -> str:
    return jwt.encode(claims, "secret", algorithm="HS256")


def _auth_headers(claims: dict) -> dict[str, str]:
    return {"authorization": f"Bearer {_make_jwt(claims)}"}


ALICE_CLAIMS = {"username": "alice", "tenant_id": "acme"}

_PATIENT_RECORDS = [
    {"id": "p1", "name": "Alice Smith", "dob": "1985-03-12", "email": "alice@example.com", "diagnosis": "hypertension"},
    {"id": "p2", "name": "Bob Jones",   "dob": "1972-07-08", "email": "bob@example.com",   "diagnosis": "diabetes"},
]


@pytest.fixture(autouse=True)
def reset_config():
    dec_module._decorator_config = None
    yield
    dec_module._decorator_config = None


def _setup(monkeypatch, *, fail_open: bool = False):
    configure_authz_decorator(config=AuthZConfig(connection=PDPConnection(fail_open=fail_open), enforcement=standard_tool_enforcement()))
    monkeypatch.setattr(dec_module, "_get_http_headers", lambda **kw: _auth_headers(ALICE_CLAIMS))


# ── Tool always executes ───────────────────────────────────────────────────────

def test_tool_executes_with_no_post_processor_and_ads_not_called(monkeypatch):
    """Without a post_processor the tool runs and ADS is never called."""
    ads_calls: list = []
    configure_authz_decorator(config=AuthZConfig(enforcement=standard_tool_enforcement()))
    monkeypatch.setattr(
        helpers_module,
        "evaluate_tool_call",
        decision_seam(lambda *a, **kw: ads_calls.append(1) or False),
    )

    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool()
    def ping(msg: str) -> str:
        return f"pong: {msg}"

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("ping", {"msg": "hello"})

    result = _run(run())
    assert "pong: hello" in result.data
    assert not ads_calls, "ADS should not be called when no post_processor is set"


def test_tool_executes_with_post_processor_permit_returns_full_result(monkeypatch):
    """ADS Permit → ColumnMaskStrategy returns full result (no masking)."""
    _setup(monkeypatch)
    monkeypatch.setattr(PdpClient, "check_permission", lambda self, **kw: True)

    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool(post_processor=ColumnMaskStrategy("dob", "email"))
    def get_patients() -> list:
        return _PATIENT_RECORDS

    result = _run(_run_tool(mcp, "get_patients", {}))
    records = result.data
    assert records[0]["dob"] == "1985-03-12"
    assert records[0]["email"] == "alice@example.com"


def test_tool_executes_with_post_processor_deny_returns_restricted_result(monkeypatch):
    """ADS Deny → ColumnMaskStrategy masks the named fields."""
    _setup(monkeypatch)
    monkeypatch.setattr(PdpClient, "check_permission", lambda self, **kw: False)

    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool(post_processor=ColumnMaskStrategy("dob", "email"))
    def get_patients() -> list:
        return _PATIENT_RECORDS

    result = _run(_run_tool(mcp, "get_patients", {}))
    records = result.data
    for r in records:
        assert r["dob"] == "***"
        assert r["email"] == "***"
        assert r["name"] != "***"
        assert r["diagnosis"] != "***"


# ── fail_open behaviour ────────────────────────────────────────────────────────

def test_fail_open_returns_full_result_when_ads_unreachable(monkeypatch):
    """fail_open=True, ADS unreachable → strategy permits, full result returned."""
    _setup(monkeypatch, fail_open=True)
    monkeypatch.setattr(PdpClient, "check_permission", lambda self, *, fail_open=False, **kw: fail_open)

    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool(post_processor=ColumnMaskStrategy("dob"))
    def get_patients() -> list:
        return _PATIENT_RECORDS

    result = _run(_run_tool(mcp, "get_patients", {}))
    assert result.data[0]["dob"] == "1985-03-12"


def test_fail_closed_applies_post_processor_when_ads_unreachable(monkeypatch):
    """fail_open=False (default), ADS unreachable → strategy denies, fields masked."""
    _setup(monkeypatch, fail_open=False)
    monkeypatch.setattr(PdpClient, "check_permission", lambda self, *, fail_open=False, **kw: fail_open)

    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool(post_processor=ColumnMaskStrategy("dob"))
    def get_patients() -> list:
        return _PATIENT_RECORDS

    result = _run(_run_tool(mcp, "get_patients", {}))
    assert result.data[0]["dob"] == "***"


# ── PostInvocationContext ──────────────────────────────────────────────────────

def test_post_processor_receives_correct_context(monkeypatch):
    """post_processor always receives a populated PostInvocationContext."""
    _setup(monkeypatch)

    received: list = []

    async def capture(result, ctx: PostInvocationContext):
        received.append(ctx)
        return result

    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool(post_processor=capture)
    def get_info(item_id: str) -> dict:
        return {"id": item_id, "value": "data"}

    _run(_run_tool(mcp, "get_info", {"item_id": "X1"}))

    assert received, "post_processor was never called"
    ctx = received[0]
    assert ctx.tool_name == "get_info"
    assert ctx.args == {"item_id": "X1"}
    assert ctx.subject.get("username") == "alice"


# ── Tool arguments forwarded via context ───────────────────────────────────────

def test_tool_arguments_forwarded_to_ads(monkeypatch):
    """Tool call arguments appear in the resource dict sent to ADS via ColumnMaskStrategy."""
    captured: list[dict] = []

    def capturing_permit(self, *, resource, **kw):
        captured.append(resource)
        return True

    _setup(monkeypatch)
    monkeypatch.setattr(PdpClient, "check_permission", capturing_permit)

    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool(post_processor=ColumnMaskStrategy("field"))
    def get_record(record_id: str) -> dict:
        return {"record_id": record_id, "field": "sensitive"}

    _run(_run_tool(mcp, "get_record", {"record_id": "REC-42"}))

    assert captured, "evaluate_tool_call was never called"
    # ColumnMaskStrategy's single post-invocation request: tool name + arguments.
    assert captured == [{"tool_name": "get_record", "record_id": "REC-42"}]


# ── configure_authz_decorator guard ───────────────────────────────────────────

def test_missing_configure_raises_runtime_error():
    """Calling a @protected_tool without configure_authz_decorator raises RuntimeError."""
    mcp = FastMCP("test")

    @mcp.tool()
    @protected_tool(post_processor=ColumnMaskStrategy("field"))
    def some_tool() -> dict:
        return {"field": "value"}

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("some_tool", {})

    with pytest.raises(Exception, match="configure_authz_decorator"):
        _run(run())


# ── Shared helper ──────────────────────────────────────────────────────────────

async def _run_tool(mcp, name, args):
    async with Client(mcp) as client:
        return await client.call_tool(name, args)


# ── called directly from Python ────────────────────────────────────────────────

def test_positional_arguments_reach_the_policy(monkeypatch):
    # FastMCP always passes keyword arguments; a direct Python call may not.
    _setup(monkeypatch)
    received: list = []

    async def capture(result, ctx: PostInvocationContext):
        received.append(ctx.args)
        return result

    @protected_tool(post_processor=capture)
    def get_record(record_id: str, verbose: bool = False) -> dict:
        return {"record_id": record_id}

    _run(get_record("REC-42", verbose=True))
    assert received == [{"record_id": "REC-42", "verbose": True}]


def test_a_sync_tool_sees_the_callers_context(monkeypatch):
    # The sync tool runs on a worker thread; the caller's contextvars (the ambient identity
    # a wrapped RAG store reads) must reach it.
    from axiomatics_ai.rag.wrap import _explicit_context, authz_context

    _setup(monkeypatch)
    seen: list = []

    @protected_tool()
    def search() -> str:
        seen.append(_explicit_context.get())
        return "ok"

    async def call():
        with authz_context({"username": "alice"}):
            return await search()

    _run(call())
    assert seen == [{"username": "alice"}]
