"""
Tests for sub-agent authorization in authorized_adk_agent: the callbacks reach every agent
below the root (sub_agents, also through workflow agents, and the agents AgentTools wrap)
once each, an agent authorized first with another config keeps it, the subject names the
agent making the call, and an AgentTool's result is not post-processed. Real ADK agent
objects for the tree (no model is called); ADK-shaped fakes for the callback contexts.
"""
from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools.agent_tool import AgentTool

from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.google_adk import authorized_adk_agent
from axiomatics_ai.google_adk.authz import _AdkAuthZ

from .conftest import FakeContext, FakeLlmRequest, FakePdp, FakeTool

pytestmark = pytest.mark.filterwarnings("ignore:.*deprecated in favor of Workflow:DeprecationWarning")

CONFIG = AuthZConfig(enforcement=standard_tool_enforcement())
ALICE = {"username": "alice"}
MODEL = "gemini-2.0-flash"   # a name only: nothing here calls the model


def _ours(agent, field="before_tool_callback") -> list[_AdkAuthZ]:
    existing = getattr(agent, field)
    callbacks = existing if isinstance(existing, list) else [existing]
    return [cb.__self__ for cb in callbacks if isinstance(getattr(cb, "__self__", None), _AdkAuthZ)]


def _tree():
    leaf_a = LlmAgent(name="leaf_a", model=MODEL)
    leaf_b = LlmAgent(name="leaf_b", model=MODEL)
    seq = SequentialAgent(name="seq", sub_agents=[leaf_a])
    par = ParallelAgent(name="par", sub_agents=[leaf_b])
    tooled = LlmAgent(name="tooled", model=MODEL)
    root = LlmAgent(name="root", model=MODEL, sub_agents=[seq, par], tools=[AgentTool(agent=tooled)])
    return root, seq, par, leaf_a, leaf_b, tooled


class _AgentContext(FakeContext):
    """A CallbackContext / ToolContext that also names the agent it runs for."""
    def __init__(self, state, agent_name):
        super().__init__(state)
        self.agent_name = agent_name


# ── walking the agent tree ───────────────────────────────────────────────────────

def test_callbacks_reach_sub_agents_workflow_children_and_agent_tools():
    root, seq, par, leaf_a, leaf_b, tooled = _tree()
    authorized_adk_agent(root, CONFIG)
    for agent in (root, leaf_a, leaf_b, tooled):
        for field in ("before_model_callback", "before_tool_callback", "after_tool_callback"):
            assert len(_ours(agent, field)) == 1, (agent.name, field)
    for workflow in (seq, par):                                 # workflow agents call no model/tool
        assert not hasattr(workflow, "before_tool_callback") or not _ours(workflow)


def test_one_engine_shared_across_the_tree():
    root, *_rest, tooled = _tree()
    authorized_adk_agent(root, CONFIG)
    assert _ours(root)[0] is _ours(tooled)[0]


def test_second_call_adds_no_second_callback():
    root, _seq, _par, leaf_a, _leaf_b, tooled = _tree()
    authorized_adk_agent(root, CONFIG)
    authorized_adk_agent(root, CONFIG)
    for agent in (root, leaf_a, tooled):
        for field in ("before_model_callback", "before_tool_callback", "after_tool_callback"):
            assert len(_ours(agent, field)) == 1


def test_agent_shared_twice_in_the_tree_is_attached_once():
    shared = LlmAgent(name="shared", model=MODEL)
    root = LlmAgent(name="root", model=MODEL, sub_agents=[shared], tools=[AgentTool(agent=shared)])
    authorized_adk_agent(root, CONFIG)
    assert len(_ours(shared)) == 1


def test_agent_authorized_first_with_another_config_keeps_it():
    root, _seq, _par, _leaf_a, _leaf_b, tooled = _tree()
    own = AuthZConfig(enforcement=standard_tool_enforcement(visibility=False, post_invocation=False))
    authorized_adk_agent(tooled, own)
    authorized_adk_agent(root, CONFIG)
    (engine,) = _ours(tooled)
    assert engine.config is own
    assert tooled.before_model_callback is None                 # the root's config added nothing
    assert _ours(root)[0].config is CONFIG


# ── the acting agent in the subject ──────────────────────────────────────────────

def _engine(fake: FakePdp) -> _AdkAuthZ:
    az = _AdkAuthZ(CONFIG)
    az._ads = fake
    return az


def test_before_tool_subject_names_the_agent():
    fake = FakePdp(permit=True)
    _engine(fake).before_tool(FakeTool("search_docs"), {}, _AgentContext(ALICE, "researcher"))
    assert fake.calls[-1]["subject"] == {"username": "alice", "agent_name": "researcher"}


def test_before_tool_subject_without_agent_name():
    fake = FakePdp(permit=True)
    _engine(fake).before_tool(FakeTool("search_docs"), {}, FakeContext(ALICE))
    assert fake.calls[-1]["subject"] == {"username": "alice"}


def test_visibility_subject_names_the_agent():
    seen = []
    fake = FakePdp(permit=True)
    original = fake.check_permission_batch
    fake.check_permission_batch = lambda **kw: seen.append(kw["subject"]) or original(**kw)
    _engine(fake).before_model(_AgentContext(ALICE, "researcher"), FakeLlmRequest(["search_docs"]))
    assert seen == [{"username": "alice", "agent_name": "researcher"}]


def test_after_tool_subject_names_the_agent():
    seen = []
    fake = FakePdp(permit=True)
    for method in ("check_permission", "check_permission_batch"):
        original = getattr(fake, method)
        setattr(fake, method, lambda _o=original, **kw: seen.append(kw["subject"]) or _o(**kw))
    out = _engine(fake).after_tool(FakeTool("list_docs"), {}, _AgentContext(ALICE, "researcher"),
                                   {"items": [{"id": 1}]})
    assert out == {"items": [{"id": 1}]}
    assert seen and all(s == {"username": "alice", "agent_name": "researcher"} for s in seen)


# ── AgentTool results ────────────────────────────────────────────────────────────

def test_after_tool_skips_an_agent_tool():
    fake = FakePdp(permit=False)                                # deny-all would drop every record
    tool = AgentTool(agent=LlmAgent(name="inner", model=MODEL))
    out = _engine(fake).after_tool(tool, {}, _AgentContext(ALICE, "root"), {"items": [{"id": 1}]})
    assert out is None and fake.calls == []
