"""
Tests for sub-agent authorization in create_authorized_deep_agent (and the agent name the
factories pass to the middleware): every declarative sub-agent spec gets the middleware, a
general-purpose spec is added when missing, compiled sub-agents must come from our factories,
remote ones only warn, and the main agent leaves the ``task`` output unprocessed. Plus an
end-to-end run (scripted fake model, recorded PDP) through ``task`` into a sub-agent's tool.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

import deepagents
import langchain.agents as lc_agents
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.types import Command

import axiomatics_ai.langchain_agents.middleware as mw_module
from axiomatics_ai import (
    AuthZConfig,
    Enforcer,
    EnforcementPoint,
    LangChainAgentAuthZMiddleware,
    RecordFilterStrategy,
    create_authorized_agent,
    create_authorized_deep_agent,
    standard_tool_enforcement,
)
from axiomatics_ai.core.enforcement import ToolCallDecision
from tests.google_adk.conftest import FakePdp
from tests.pdp_fakes import patch_pdp_http

CONFIG = AuthZConfig(enforcement=standard_tool_enforcement())


class _FakeModel(GenericFakeChatModel):
    """Scripted answers; tools are ignored (the script names them)."""

    def bind_tools(self, tools, **kwargs):
        return self


class _Other(AgentMiddleware):
    """A non-Axiomatics middleware, to check a spec's own middleware is kept."""


@tool
def lookup(q: str) -> str:
    """Look up."""
    return "found"


def _capture_deep(monkeypatch):
    seen = {}
    monkeypatch.setattr(deepagents, "create_deep_agent", lambda *a, **k: seen.update(kwargs=k) or SimpleNamespace())
    return seen


def _authz_of(spec):
    found = [m for m in spec.get("middleware", []) if isinstance(m, LangChainAgentAuthZMiddleware)]
    assert len(found) == 1
    return found[0]


# ── declarative sub-agent specs ──────────────────────────────────────────────────

def test_every_spec_gets_the_middleware_first_named_after_it(monkeypatch):
    seen = _capture_deep(monkeypatch)
    other = _Other()
    create_authorized_deep_agent(config=CONFIG, subagents=[
        {"name": "researcher", "description": "d", "system_prompt": "p"},
        {"name": "writer", "description": "d", "system_prompt": "p", "middleware": [other]},
    ])
    specs = {s["name"]: s for s in seen["kwargs"]["subagents"]}
    assert set(specs) == {"general-purpose", "researcher", "writer"}
    for name, spec in specs.items():
        first = spec["middleware"][0]
        assert isinstance(first, LangChainAgentAuthZMiddleware)
        assert first.agent_name == name and first.config is CONFIG
    assert specs["writer"]["middleware"][1] is other          # caller's middleware kept, after ours


def test_spec_with_its_own_middleware_keeps_it(monkeypatch):
    seen = _capture_deep(monkeypatch)
    own = LangChainAgentAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(visibility=False)))
    create_authorized_deep_agent(config=CONFIG, subagents=[
        {"name": "special", "description": "d", "system_prompt": "p", "middleware": [own]},
    ])
    spec = next(s for s in seen["kwargs"]["subagents"] if s["name"] == "special")
    assert spec["middleware"] == [own]                        # not replaced, not duplicated
    assert own.config is not CONFIG


def test_general_purpose_added_when_missing_with_skills(monkeypatch):
    seen = _capture_deep(monkeypatch)
    create_authorized_deep_agent(config=CONFIG, skills=["/skills/"])
    (gp,) = seen["kwargs"]["subagents"]
    assert gp["name"] == "general-purpose"
    assert gp["skills"] == ["/skills/"]
    assert _authz_of(gp).agent_name == "general-purpose"


def test_general_purpose_added_without_skills_key_when_none(monkeypatch):
    seen = _capture_deep(monkeypatch)
    create_authorized_deep_agent(config=CONFIG)
    (gp,) = seen["kwargs"]["subagents"]
    assert "skills" not in gp


def test_existing_general_purpose_spec_is_not_duplicated(monkeypatch):
    seen = _capture_deep(monkeypatch)
    create_authorized_deep_agent(config=CONFIG, subagents=[
        {"name": "general-purpose", "description": "mine", "system_prompt": "p"},
    ])
    (gp,) = seen["kwargs"]["subagents"]
    assert gp["description"] == "mine"
    assert _authz_of(gp).agent_name == "general-purpose"


def test_main_agent_middleware_skips_task_output(monkeypatch):
    seen = _capture_deep(monkeypatch)
    create_authorized_deep_agent(config=CONFIG, name="main")
    main = seen["kwargs"]["middleware"][0]
    assert isinstance(main, LangChainAgentAuthZMiddleware)
    assert main._unchecked_output_tools == {"task"} and main.agent_name == "main"
    # sub-agents' own middleware still post-processes everything
    assert _authz_of(seen["kwargs"]["subagents"][0])._unchecked_output_tools == frozenset()


# ── compiled and remote sub-agents ───────────────────────────────────────────────

def test_unauthorized_compiled_subagent_is_rejected(monkeypatch):
    _capture_deep(monkeypatch)
    plain = create_agent(model=_FakeModel(messages=iter([])), tools=[lookup])
    with pytest.raises(ValueError, match=r"Compiled sub-agent .raw."):
        create_authorized_deep_agent(config=CONFIG, subagents=[
            {"name": "raw", "description": "d", "runnable": plain},
        ])


def test_compiled_subagent_from_our_factories_is_accepted(monkeypatch):
    built = create_authorized_agent(model=_FakeModel(messages=iter([])), tools=[lookup], config=CONFIG)
    deep = create_authorized_deep_agent(_FakeModel(messages=iter([])), tools=[lookup], config=CONFIG)
    assert built._axiomatics_authorized is True and deep._axiomatics_authorized is True

    seen = _capture_deep(monkeypatch)
    create_authorized_deep_agent(config=CONFIG, subagents=[
        {"name": "a", "description": "d", "runnable": built},
        {"name": "b", "description": "d", "runnable": deep},
    ])
    specs = {s["name"]: s for s in seen["kwargs"]["subagents"]}
    assert specs["a"]["runnable"] is built and specs["b"]["runnable"] is deep
    assert "middleware" not in specs["a"]                     # passed through untouched


def test_remote_subagent_warns_and_passes_through(monkeypatch, caplog):
    seen = _capture_deep(monkeypatch)
    remote = {"name": "remote", "description": "d", "graph_id": "g"}
    with caplog.at_level(logging.WARNING, logger="axiomatics_ai.langchain_agents.factory"):
        create_authorized_deep_agent(config=CONFIG, subagents=[remote])
    assert remote in seen["kwargs"]["subagents"]
    assert any("remote" in r.getMessage() for r in caplog.records)


def test_create_authorized_agent_passes_name_to_middleware(monkeypatch):
    seen = {}
    monkeypatch.setattr(lc_agents, "create_agent", lambda *a, **k: seen.update(kwargs=k) or SimpleNamespace())
    create_authorized_agent(config=CONFIG, name="x")
    assert seen["kwargs"]["middleware"][0].agent_name == "x"
    assert seen["kwargs"]["name"] == "x"                      # still forwarded


# ── post-invocation on a Command result ──────────────────────────────────────────

def _post_config() -> AuthZConfig:
    return AuthZConfig(enforcement={
        EnforcementPoint.AFTER_TOOL: Enforcer(action_name="post", strategy=RecordFilterStrategy()),
    })


def _request(name: str):
    return SimpleNamespace(
        tool_call={"name": name, "args": {}, "id": "c1"},
        runtime=SimpleNamespace(context={"user_id": "alice"}),
    )


def test_post_invocation_filters_tool_messages_inside_a_command():
    middleware = LangChainAgentAuthZMiddleware(_post_config())
    middleware._ads = FakePdp(permit=False)                   # deny-all → records dropped
    records = json.dumps([{"id": 1}, {"id": 2}])
    command = Command(update={"messages": [ToolMessage(content=records, tool_call_id="c1")], "other": 1})
    out = middleware._apply_post_invocation(command, _request("list_docs"))
    assert isinstance(out, Command) and out.update["other"] == 1
    assert json.loads(out.update["messages"][0].content) == []


def test_post_invocation_leaves_unchecked_tool_output():
    middleware = LangChainAgentAuthZMiddleware(_post_config())
    middleware._ads = FakePdp(permit=False)
    middleware._unchecked_output_tools = frozenset({"task"})
    command = Command(update={"messages": [ToolMessage(content="[{\"id\": 1}]", tool_call_id="c1")]})
    assert middleware._apply_post_invocation(command, _request("task")) is command


# ── end-to-end: main → task → sub-agent tool ─────────────────────────────────────

def test_subagent_tool_call_is_authorized_with_callers_identity(monkeypatch):
    seen = []

    def record(**kw):
        seen.append((kw["resource"].get("tool_name"), kw["subject"], kw["action"]))
        return ToolCallDecision(permitted=True)

    monkeypatch.setattr(mw_module, "authorized_tool_invocation", record)

    def no_network(url, **kwargs):
        raise AssertionError(f"unexpected PDP call to {url}")

    patch_pdp_http(monkeypatch, "post", no_network)

    model = _FakeModel(messages=iter([
        AIMessage("", tool_calls=[{"name": "task", "id": "c1",
                                   "args": {"subagent_type": "general-purpose", "description": "look up"}}]),
        AIMessage("", tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "c2"}]),
        AIMessage("sub done"),
        AIMessage("final"),
    ]))
    config = AuthZConfig(enforcement={
        EnforcementPoint.BEFORE_TOOL: Enforcer(action_name="invoke"),
        EnforcementPoint.AFTER_TOOL: Enforcer(action_name="post", strategy=RecordFilterStrategy()),
    })
    agent = create_authorized_deep_agent(model, tools=[lookup], config=config, name="main")
    out = agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, context={"user_id": "alice"})

    assert out["messages"][-1].content == "final"             # AFTER_TOOL on task's Command didn't crash
    by_tool = {name: subject for name, subject, _ in seen}
    assert by_tool["task"] == {"agent_name": "main", "user_id": "alice"}
    assert by_tool["lookup"] == {"agent_name": "general-purpose", "user_id": "alice"}
    # the sub-agent's tool result came back through task
    assert any(isinstance(m, ToolMessage) and "sub done" in str(m.content) for m in out["messages"])
