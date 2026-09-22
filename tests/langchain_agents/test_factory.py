"""
Tests for the explicit agent factories (create_authorized_agent / create_authorized_deep_agent),
focused on the middleware chaining: our middleware is injected outermost, the caller's middleware
is preserved after it, and an already-present instance is not duplicated. Plus one end-to-end
test that the injected middleware actually enforces (visibility filtering) through the factory.
"""
from __future__ import annotations

import pytest

import langchain.agents as lc_agents
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain.tools import tool

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import (
    AuthZConfig,
    LangChainAgentAuthZMiddleware,
    create_authorized_agent,
    create_authorized_deep_agent,
    standard_tool_enforcement,
)
from axiomatics_ai.langchain_agents.factory import _authz_middleware_chain

CONFIG = AuthZConfig(enforcement=standard_tool_enforcement())


class _Other(AgentMiddleware):
    """A non-Axiomatics middleware, to check the caller's middleware is preserved."""


# ── the injection helper (the chaining contract) ─────────────────────────────────

def test_chain_prepends_our_middleware_when_none_given():
    chain = _authz_middleware_chain(CONFIG, None)
    assert len(chain) == 1 and isinstance(chain[0], LangChainAgentAuthZMiddleware)


def test_chain_ours_is_outermost_then_caller_middleware():
    other = _Other()
    chain = _authz_middleware_chain(CONFIG, [other])
    assert isinstance(chain[0], LangChainAgentAuthZMiddleware)   # ours runs outermost (first)
    assert chain[1] is other                                     # caller's preserved, after ours


def test_chain_does_not_duplicate_existing_instance():
    existing = LangChainAgentAuthZMiddleware(CONFIG)
    other = _Other()
    chain = _authz_middleware_chain(CONFIG, [existing, other])
    assert chain == [existing, other]                            # left as-is, no second injected
    assert sum(isinstance(m, LangChainAgentAuthZMiddleware) for m in chain) == 1


# ── the factories forward the right middleware= to the underlying creator ─────────

def _capture_create_agent(monkeypatch, module, attr="create_agent"):
    seen = {}
    monkeypatch.setattr(module, attr, lambda *a, **k: seen.update(args=a, kwargs=k) or "AGENT")
    return seen


def test_create_authorized_agent_injects_and_forwards(monkeypatch):
    seen = _capture_create_agent(monkeypatch, lc_agents)
    other = _Other()
    agent = create_authorized_agent(model="M", tools=["T"], config=CONFIG, middleware=[other], system_prompt="hi")
    assert agent == "AGENT"
    mw = seen["kwargs"]["middleware"]
    assert isinstance(mw[0], LangChainAgentAuthZMiddleware) and mw[1] is other
    assert seen["kwargs"]["model"] == "M" and seen["kwargs"]["tools"] == ["T"]  # forwarded verbatim
    assert seen["kwargs"]["system_prompt"] == "hi"                              # arbitrary kwargs pass through


def test_create_authorized_agent_dedupes(monkeypatch):
    seen = _capture_create_agent(monkeypatch, lc_agents)
    existing = LangChainAgentAuthZMiddleware(CONFIG)
    create_authorized_agent(config=CONFIG, middleware=[existing])
    assert seen["kwargs"]["middleware"] == [existing]


def test_create_authorized_deep_agent_injects(monkeypatch):
    # The factory also reads deepagents' general-purpose sub-agent spec, so patch the real
    # module's creator rather than replacing the package.
    deepagents = pytest.importorskip("deepagents")
    seen = {}
    monkeypatch.setattr(deepagents, "create_deep_agent", lambda *a, **k: seen.update(kwargs=k) or "DEEP")
    agent = create_authorized_deep_agent(config=CONFIG)
    assert agent == "DEEP"
    assert isinstance(seen["kwargs"]["middleware"][0], LangChainAgentAuthZMiddleware)


# ── end-to-end: the injected middleware actually enforces through the factory ─────

class _StubChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _ToolCapturing(AgentMiddleware):
    """Placed after ours; captures the tool names that reach the model (post-filter)."""
    def __init__(self):
        self.seen: set[str] = set()

    def wrap_model_call(self, request, handler):
        self.seen = {t.name for t in request.tools}
        return handler(request)


@tool
def search_docs(query: str) -> str:
    """Search docs."""
    return "ok"


@tool
def send_email(to: str, body: str) -> str:
    """Send email."""
    return "sent"


def test_factory_agent_enforces_visibility(monkeypatch):
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, **kw: {"search_docs"},
    )
    spy = _ToolCapturing()
    config = AuthZConfig(enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False))
    agent = create_authorized_agent(
        model=_StubChatModel(responses=[AIMessage(content="ok")]),
        tools=[search_docs, send_email],
        config=config,
        middleware=[spy],
    )
    agent.invoke({"messages": [{"role": "user", "content": "help"}]}, context={"username": "alice"})
    assert spy.seen == {"search_docs"}   # send_email filtered out by our injected middleware
