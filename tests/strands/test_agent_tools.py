"""
Tests for agents used as tools in authorized_strands_agent: the handler reaches every agent
under ``agent.as_tool()`` once each, an agent authorized first with another config keeps it,
the subject names the agent making the call, a permitted call to an agent tool hands the
caller's context to the inner agent until the call ends, and an agent tool's result is not
post-processed. Real Strands agents for the tree, Strands-shaped fakes for the events, and
one end-to-end run with scripted models (PDP stubbed).
"""
from __future__ import annotations

import contextvars

import pytest

pytest.importorskip("strands")

from strands import Agent, tool
from strands.interventions import Proceed
from strands.models import Model

import axiomatics_ai.core.enforcement as enforcement_module
import axiomatics_ai.strands.authz as strands_authz
from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.strands.authz import (
    _AuthzInterventionHandler,
    _caller_context,
    _StrandsAuthZ,
    authorized_strands_agent,
)
from tests.approval_fakes import PERMIT

from .conftest import FakeAgent, FakePdp, FakeState, FakeToolEvent

CONFIG = AuthZConfig(enforcement=standard_tool_enforcement())
ALICE = {"username": "alice"}


def _callback_count(agent) -> int:
    return sum(len(v) for v in agent.hooks._registered_callbacks.values())


def _record_attach(monkeypatch) -> list:
    attached = []
    original = strands_authz._attach
    monkeypatch.setattr(strands_authz, "_attach",
                        lambda agent, config: attached.append((agent, config)) or original(agent, config))
    return attached


def _tree():
    leaf = Agent(name="leaf", tools=[])
    middle = Agent(name="middle", tools=[leaf.as_tool()])
    root = Agent(name="root", tools=[middle.as_tool(), leaf.as_tool(name="leaf_again")])
    return root, middle, leaf


# ── walking agents used as tools ─────────────────────────────────────────────────

def test_handler_reaches_every_agent_tool_once(monkeypatch):
    attached = _record_attach(monkeypatch)
    root, middle, leaf = _tree()
    authorized_strands_agent(root, CONFIG)
    assert sorted(a.name for a, _ in attached) == ["leaf", "middle", "root"]   # leaf twice in tree, once here
    assert all(config is CONFIG for _, config in attached)
    assert all(a._axiomatics_authorized for a in (root, middle, leaf))


def test_second_call_registers_nothing_more(monkeypatch):
    root, middle, leaf = _tree()
    authorized_strands_agent(root, CONFIG)
    counts = [_callback_count(a) for a in (root, middle, leaf)]
    attached = _record_attach(monkeypatch)
    authorized_strands_agent(root, CONFIG)
    assert attached == []
    assert [_callback_count(a) for a in (root, middle, leaf)] == counts


def test_agent_authorized_first_with_another_config_keeps_it(monkeypatch):
    root, middle, leaf = _tree()
    own = AuthZConfig(enforcement=standard_tool_enforcement(visibility=False))
    authorized_strands_agent(middle, own)              # middle and leaf get `own`
    attached = _record_attach(monkeypatch)
    authorized_strands_agent(root, CONFIG)
    assert [(a.name, c) for a, c in attached] == [("root", CONFIG)]


# ── the acting agent in the subject ──────────────────────────────────────────────

def _engine(fake: FakePdp, config: AuthZConfig = CONFIG) -> _StrandsAuthZ:
    az = _StrandsAuthZ(config)
    az._ads = fake
    return az


def _named(name: str, **kwargs) -> FakeAgent:
    agent = FakeAgent(**kwargs)
    agent.name = name
    return agent


def test_before_tool_subject_names_the_agent():
    fake = FakePdp(permit=True)
    _engine(fake).before_tool(FakeToolEvent("search_docs", agent=_named("researcher"), invocation_state=ALICE))
    assert fake.calls[-1]["subject"] == {"username": "alice", "agent_name": "researcher"}


def test_before_tool_subject_without_agent_name():
    fake = FakePdp(permit=True)
    _engine(fake).before_tool(FakeToolEvent("search_docs", invocation_state=ALICE))
    assert fake.calls[-1]["subject"] == {"username": "alice"}


# ── the caller's context while an agent tool runs ────────────────────────────────

def _agent_tool_event(name="inner", *, tool_use_id="t1", **kwargs) -> FakeToolEvent:
    event = FakeToolEvent(name, **kwargs)
    event.tool_use["toolUseId"] = tool_use_id
    event.selected_tool = Agent(name=name, tools=[]).as_tool()
    return event


def _isolated(fn):
    """Run in a copy of the current context, so a leaked ContextVar can't reach other tests."""
    return contextvars.copy_context().run(fn)


def test_permitted_agent_tool_call_hands_the_callers_context_to_the_inner_agent():
    def run():
        engine = _engine(FakePdp(permit=True))
        handler = _AuthzInterventionHandler(engine, CONFIG)
        event = _agent_tool_event(agent=_named("root"), invocation_state=ALICE)
        assert isinstance(handler.before_tool_call(event), Proceed)
        assert _caller_context.get()["username"] == "alice"

        inner = _engine(FakePdp(permit=True))
        # the inner agent's own event carries no identity: it reads the caller's
        assert inner._context(FakeToolEvent("search_docs", agent=_named("inner")))["username"] == "alice"
        # its own state and invocation state still take precedence
        own_state = _named("inner")
        own_state.state = FakeState({"username": "carol"})
        assert inner._context(FakeToolEvent("search_docs", agent=own_state))["username"] == "carol"
        assert inner._context(FakeToolEvent("search_docs", invocation_state={"username": "dave"}))["username"] == "dave"

        handler.after_tool_call(event)
        assert _caller_context.get() is None
        assert engine._caller_tokens == {}

    _isolated(run)


def test_inner_agent_tool_call_is_authorized_with_the_callers_identity():
    def run():
        outer = _AuthzInterventionHandler(_engine(FakePdp(permit=True)), CONFIG)
        event = _agent_tool_event(agent=_named("root"), invocation_state=ALICE)
        outer.before_tool_call(event)
        fake = FakePdp(permit=True)
        _engine(fake).before_tool(FakeToolEvent("search_docs", agent=_named("inner")))
        assert fake.calls[-1]["subject"] == {"username": "alice", "agent_name": "inner"}
        outer.after_tool_call(event)

    _isolated(run)


def test_denied_agent_tool_call_sets_no_caller_context():
    def run():
        engine = _engine(FakePdp(permit=False))
        assert isinstance(engine.before_tool(_agent_tool_event(invocation_state=ALICE)), str)
        assert _caller_context.get() is None and engine._caller_tokens == {}

    _isolated(run)


def test_plain_tool_call_sets_no_caller_context():
    def run():
        engine = _engine(FakePdp(permit=True))
        event = FakeToolEvent("search_docs", invocation_state=ALICE)
        event.selected_tool = object()
        assert engine.before_tool(event) is None
        assert _caller_context.get() is None

    _isolated(run)


def test_release_without_a_token_is_a_no_op():
    def run():
        engine = _engine(FakePdp(permit=True))
        engine.release_caller_context(FakeToolEvent("search_docs"))
        assert _caller_context.get() is None

    _isolated(run)


# ── agent tool results ───────────────────────────────────────────────────────────

def test_after_tool_skips_an_agent_tool():
    fake = FakePdp(permit=False)                       # deny-all would drop every record
    event = _agent_tool_event(invocation_state=ALICE, result={"items": [{"id": 1}]})
    assert _engine(fake).after_tool(event) is None
    assert fake.calls == []


# ── end-to-end: a real agent calling an agent tool ───────────────────────────────

class _ScriptedModel(Model):
    """Calls ``tool`` (with ``tool_input``) on the first turn, then answers ``answer``."""

    def __init__(self, tool: str, tool_input: str, answer: str):
        self.tool, self.tool_input, self.answer = tool, tool_input, answer
        self.turns = 0

    def update_config(self, **model_config):
        pass

    def get_config(self):
        return {}

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.turns += 1
        yield {"messageStart": {"role": "assistant"}}
        if self.turns == 1:
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"id-{self.tool}", "name": self.tool}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": self.tool_input}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": self.answer}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def test_inner_agent_runs_its_tool_with_the_callers_identity(monkeypatch):
    seen = []

    def decide(pdp, *, subject, action, resource, environment):
        seen.append((resource.get("tool_name"), dict(subject)))
        return PERMIT

    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decide)

    @tool
    def lookup(q: str) -> str:
        """Look up."""
        return "found"

    inner = Agent(name="inner", model=_ScriptedModel("lookup", '{"q": "x"}', "inner done"),
                  tools=[lookup], callback_handler=None)
    outer = Agent(name="outer", model=_ScriptedModel("inner", '{"input": "go"}', "outer done"),
                  tools=[inner.as_tool()], state={"username": "alice"}, callback_handler=None)
    config = AuthZConfig(enforcement=standard_tool_enforcement(visibility=False))

    def run():
        result = authorized_strands_agent(outer, config)("hi")
        assert "outer done" in str(result)
        assert _caller_context.get() is None

    _isolated(run)
    assert seen == [
        ("inner", {"username": "alice", "agent_name": "outer"}),
        ("lookup", {"username": "alice", "agent_name": "inner"}),
    ]
