"""
The gateway MCP route serves tools only by default: behind ``create_proxy(backend)`` +
``_tools_only_middleware()``, the backend's resources and prompts are hidden and refused, its
tools still listed. ``Route.mcp_resources_and_prompts=True`` leaves them out of the picture.
In-memory FastMCP backend and client (no port).
"""
from __future__ import annotations

import asyncio

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import McpError, PromptError, ResourceError
from fastmcp.server import create_proxy

import axiomatics_ai.gateway.app as app_module
from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.gateway import Route


def _backend() -> FastMCP:
    backend = FastMCP("backend")

    @backend.tool
    def search_docs(q: str) -> str:
        """Search docs."""
        return "docs result"

    @backend.resource("data://secret")
    def secret() -> str:
        """A secret resource."""
        return "TOP-SECRET"

    @backend.resource("data://users/{user_id}")
    def user(user_id: str) -> str:
        """A user record."""
        return f"user {user_id}"

    @backend.prompt
    def greet(name: str) -> str:
        """A prompt."""
        return f"Hello {name}"

    return backend


def _proxy(tools_only: bool = True):
    proxy = create_proxy(_backend())
    if tools_only:
        proxy.add_middleware(app_module._tools_only_middleware())
    return proxy


def _run(proxy, fn):
    async def go():
        async with Client(proxy) as client:
            return await fn(client)
    return asyncio.run(go())


def test_tools_are_still_listed_and_callable():
    proxy = _proxy()
    assert [t.name for t in _run(proxy, lambda c: c.list_tools())] == ["search_docs"]
    result = _run(proxy, lambda c: c.call_tool("search_docs", {"q": "x"}))
    assert "docs result" in str(result.data)


def test_resources_templates_and_prompts_are_hidden():
    proxy = _proxy()
    assert _run(proxy, lambda c: c.list_resources()) == []
    assert _run(proxy, lambda c: c.list_resource_templates()) == []
    assert _run(proxy, lambda c: c.list_prompts()) == []


@pytest.mark.parametrize("uri", ["data://secret", "data://users/42"])
def test_reading_a_resource_is_refused(uri):
    with pytest.raises((ResourceError, McpError)) as exc_info:
        _run(_proxy(), lambda c: c.read_resource(uri))
    assert "TOP-SECRET" not in str(exc_info.value)


def test_getting_a_prompt_is_refused():
    with pytest.raises((PromptError, McpError)):
        _run(_proxy(), lambda c: c.get_prompt("greet", {"name": "bob"}))


def test_without_the_middleware_the_backend_serves_them():
    # Control: the proxy itself would pass them through.
    proxy = _proxy(tools_only=False)
    assert [str(r.uri) for r in _run(proxy, lambda c: c.list_resources())] == ["data://secret"]
    assert [p.name for p in _run(proxy, lambda c: c.list_prompts())] == ["greet"]
    contents = _run(proxy, lambda c: c.read_resource("data://secret"))
    assert "TOP-SECRET" in str(contents)


# ── _build_mcp_subapp wiring ─────────────────────────────────────────────────────

@pytest.mark.parametrize("opt_in, expected", [(False, 1), (True, 0)])
def test_build_mcp_subapp_adds_tools_only_unless_opted_in(monkeypatch, opt_in, expected):
    made = []
    real = app_module._tools_only_middleware
    monkeypatch.setattr(app_module, "_tools_only_middleware", lambda: made.append(1) or real())
    authz = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"),
                        enforcement=standard_tool_enforcement())
    route = Route("/docs", _backend(), protocol="mcp", mcp_resources_and_prompts=opt_in)
    app_module._build_mcp_subapp(route, authz)
    assert len(made) == expected


def test_route_defaults_to_tools_only():
    assert Route("/docs", "http://docs/mcp", protocol="mcp").mcp_resources_and_prompts is False
