"""
Tests for the explicit `authorized_mcp_tools(tools, config, *, context)` wrapper — filters a
loaded MCP tool list through ADS with no monkeypatching. ADS is stubbed at
`enforcement.get_permitted_tools` tools are MagicMocks with a
`.name`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, PDPConnection, authorized_mcp_tools, standard_tool_enforcement

ALICE = {"username": "alice"}
VIS_ONLY = standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False)


def _tool(name: str):
    t = MagicMock()
    t.name = name
    return t


TOOLS = [_tool("search_docs"), _tool("get_balance"), _tool("send_email")]


def test_filters_to_permitted(monkeypatch):
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, **kw: {"search_docs", "get_balance"},
    )
    result = authorized_mcp_tools(TOOLS, AuthZConfig(enforcement=VIS_ONLY), context=ALICE)
    assert {t.name for t in result} == {"search_docs", "get_balance"}  # send_email filtered


def test_no_context_returns_no_tools_and_skips_ads(monkeypatch):
    # No caller identity: showing every tool would fail open.
    calls = {"n": 0}
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda *a, **k: calls.update(n=calls["n"] + 1) or set())
    result = authorized_mcp_tools(TOOLS, AuthZConfig(enforcement=VIS_ONLY), context=None)
    assert result == [] and calls["n"] == 0


def test_no_context_under_fail_open_returns_all(monkeypatch):
    monkeypatch.setattr(enforcement_module, "get_permitted_tools", lambda *a, **k: set())
    config = AuthZConfig(enforcement=VIS_ONLY, connection=PDPConnection(fail_open=True))
    assert authorized_mcp_tools(TOOLS, config, context=None) == TOOLS


def test_no_visibility_enforcer_returns_all_and_skips_ads(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda *a, **k: calls.update(n=calls["n"] + 1) or set())
    result = authorized_mcp_tools(TOOLS, AuthZConfig(enforcement={}), context=ALICE)  # no TOOL_VISIBILITY
    assert result == TOOLS and calls["n"] == 0


def test_empty_tool_list():
    assert authorized_mcp_tools([], AuthZConfig(enforcement=VIS_ONLY), context=ALICE) == []
