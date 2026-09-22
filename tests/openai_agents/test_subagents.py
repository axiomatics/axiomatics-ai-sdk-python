"""
The OpenAI adapter beyond one agent: ``authorized_openai_agent`` walks into handoff targets and
agents used as tools (once each, cycles fine, an already-authorized agent kept as it is), each
request names the acting agent, and an agent tool's answer is not post-processed. Also the SDK
approval answers the adapter reads (``_sdk_approval``) and its bookkeeping of paused calls.
Uses real ``agents`` objects (no model is run); the PDP is faked.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from agents import Agent, function_tool, handoff
from agents.items import ToolApprovalItem
from agents.run_context import RunContextWrapper
from openai.types.responses import ResponseFunctionToolCall

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AttributeNames, AuthZConfig, standard_tool_enforcement
from axiomatics_ai.openai_agents.authz import _OpenAIAuthZ, _sdk_approval, authorized_openai_agent
from tests.approval_fakes import ApprovalPdp, DecisionRecorder

from .conftest import FakeAgent, FakePdp, FakeRunContext, FakeTool, FakeToolContext

ALICE = {"username": "alice"}


def _engine_of(agent) -> _OpenAIAuthZ:
    """The engine in the closure of an authorized agent's get_all_tools."""
    for cell in agent.get_all_tools.__closure__ or ():
        if isinstance(cell.cell_contents, _OpenAIAuthZ):
            return cell.cell_contents
    raise AssertionError("not authorized: no engine in get_all_tools")


def _wrap_depth(agent) -> int:
    """How many times get_all_tools was wrapped: follow the wrapped originals in the closures."""
    depth, fn = 0, agent.get_all_tools
    while True:
        cells = [c.cell_contents for c in (getattr(fn, "__closure__", None) or ())]
        if not any(isinstance(c, _OpenAIAuthZ) for c in cells):
            return depth
        depth += 1
        fn = next(c for c in cells if callable(c) and getattr(c, "__name__", "") == "get_all_tools")


def _config(**kwargs) -> AuthZConfig:
    return AuthZConfig(enforcement=standard_tool_enforcement(), **kwargs)


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return q


# ── walking the agent tree ───────────────────────────────────────────────────────

class TestAgentTree:
    def test_handoffs_and_agent_tools_are_authorized(self):
        direct = Agent(name="direct")
        via_handoff = Agent(name="via_handoff")
        as_tool = Agent(name="as_tool")
        root = Agent(
            name="root", tools=[lookup, as_tool.as_tool(tool_name="ask_as_tool", tool_description="d")],
            handoffs=[direct, handoff(via_handoff)],
        )
        config = _config()
        assert authorized_openai_agent(root, config) is root
        for agent in (root, direct, via_handoff, as_tool):
            engine = _engine_of(agent)
            assert engine.config is config
            assert engine.agent_name == agent.name

    def test_recursively(self):
        grandchild = Agent(name="grandchild")
        child_tool_agent = Agent(name="tool_child", handoffs=[grandchild])
        child = Agent(name="child", tools=[child_tool_agent.as_tool(tool_name="t", tool_description="d")])
        root = Agent(name="root", handoffs=[child])
        authorized_openai_agent(root, _config())
        assert _engine_of(grandchild).agent_name == "grandchild"
        assert _engine_of(child_tool_agent).agent_name == "tool_child"

    def test_twice_does_not_wrap_twice(self):
        child = Agent(name="child")
        root = Agent(name="root", handoffs=[child])
        config = _config()
        authorized_openai_agent(root, config)
        root_tools, child_tools = root.get_all_tools, child.get_all_tools
        authorized_openai_agent(root, config)
        assert root.get_all_tools is root_tools and child.get_all_tools is child_tools
        assert _wrap_depth(root) == 1 and _wrap_depth(child) == 1

    def test_cycles_are_fine_and_each_agent_is_wrapped_once(self):
        a = Agent(name="a")
        b = Agent(name="b", handoffs=[a])
        a.handoffs = [b, a]
        authorized_openai_agent(a, _config())
        assert _wrap_depth(a) == 1 and _wrap_depth(b) == 1

    def test_an_agent_reached_twice_is_wrapped_once(self):
        shared = Agent(name="shared")
        root = Agent(
            name="root", handoffs=[shared, handoff(shared)],
            tools=[shared.as_tool(tool_name="ask_shared", tool_description="d")],
        )
        authorized_openai_agent(root, _config())
        assert _wrap_depth(shared) == 1

    def test_an_already_authorized_agent_keeps_its_config(self):
        child = Agent(name="child")
        root = Agent(name="root", handoffs=[child])
        child_config, root_config = _config(), _config()
        authorized_openai_agent(child, child_config)
        authorized_openai_agent(root, root_config)
        assert _engine_of(child).config is child_config
        assert _engine_of(root).config is root_config

    def test_a_handoff_naming_no_agent_is_warned_about(self, caplog):
        class _HandBuiltHandoff:
            agent_name = "elsewhere"   # no _agent_ref: its on_invoke_handoff returns the agent

        root = Agent(name="root")
        root.handoffs = [_HandBuiltHandoff()]
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.openai_agents.authz"):
            authorized_openai_agent(root, _config())
        assert any("'elsewhere' names no agent" in r.getMessage() for r in caplog.records)
        assert _engine_of(root).agent_name == "root"


# ── the acting agent in the subject ──────────────────────────────────────────────

class TestActingAgentAttribute:
    def test_each_agent_names_itself_in_the_subject(self):
        child = Agent(name="billing")
        root = Agent(name="triage", handoffs=[child])
        authorized_openai_agent(root, _config())
        for agent, name in ((root, "triage"), (child, "billing")):
            engine = _engine_of(agent)
            fake = FakePdp(permit=True)
            engine._ads = fake
            assert engine.invocation_denial(ALICE, "lookup", {}, []) is None
            assert fake.calls[-1]["subject"] == {"agent_name": name, "username": "alice"}

    def test_a_custom_attribute(self):
        agent = Agent(name="billing")
        authorized_openai_agent(agent, _config(attribute_names=AttributeNames(acting_agent="acting")))
        engine = _engine_of(agent)
        engine._ads = fake = FakePdp(permit=True)
        engine.invocation_denial(ALICE, "lookup", {}, [])
        assert fake.calls[-1]["subject"] == {"acting": "billing", "username": "alice"}

    def test_turned_off(self):
        agent = Agent(name="billing")
        authorized_openai_agent(agent, _config(attribute_names=AttributeNames(acting_agent=None)))
        engine = _engine_of(agent)
        engine._ads = fake = FakePdp(permit=True)
        engine.invocation_denial(ALICE, "lookup", {}, [])
        assert fake.calls[-1]["subject"] == {"username": "alice"}

    def test_not_sent_without_a_caller_identity(self):
        agent = Agent(name="billing")
        authorized_openai_agent(agent, _config())
        engine = _engine_of(agent)
        engine._ads = fake = FakePdp(permit=True)
        engine.invocation_denial({}, "lookup", {}, [])
        assert fake.calls[-1]["subject"] == {}           # no identity stays no identity


# ── an agent used as a tool ──────────────────────────────────────────────────────

def _agent_tool(result):
    tool = FakeTool("ask_sub", result=result)
    tool._is_agent_tool = True
    tool._agent_instance = None
    return tool


def test_an_agent_tools_output_is_not_post_processed():
    records = {"items": [{"id": 1}, {"id": 2}]}
    agent = FakeAgent([_agent_tool(records)])
    authorized_openai_agent(agent, _config())
    _engine_of(agent)._ads = FakePdp(allowed_tools={"ask_sub"}, permit=False)   # would drop every record
    tools = asyncio.run(agent.get_all_tools(FakeRunContext(ALICE)))
    assert asyncio.run(tools[0].on_invoke_tool(FakeToolContext(ALICE), "{}")) == records


def test_an_agent_tool_is_still_checked_before_it_runs():
    agent = FakeAgent([_agent_tool("answer")])
    authorized_openai_agent(agent, AuthZConfig(enforcement=standard_tool_enforcement(visibility=False)))
    _engine_of(agent)._ads = FakePdp(allowed_tools=set())
    tools = asyncio.run(agent.get_all_tools(FakeRunContext(ALICE)))
    out = asyncio.run(tools[0].on_invoke_tool(FakeToolContext(ALICE), "{}"))
    assert isinstance(out, str) and "Access denied" in out


# ── SDK approval answers ─────────────────────────────────────────────────────────

_OWNER = Agent(name="owner")


def _item(call_id: str, tool_name: str = "pay") -> ToolApprovalItem:
    raw = ResponseFunctionToolCall(type="function_call", id=f"fc-{call_id}", call_id=call_id, name=tool_name, arguments="{}")
    return ToolApprovalItem(agent=_OWNER, raw_item=raw, tool_name=tool_name)


class TestSdkApproval:
    def test_unanswered_is_none(self):
        assert _sdk_approval(RunContextWrapper(context={}), "pay", "c1") is None

    def test_no_call_id_or_no_approval_api_is_none(self):
        wrapper = RunContextWrapper(context={})
        wrapper.approve_tool(_item("c1"))
        assert _sdk_approval(wrapper, "pay", None) is None
        assert _sdk_approval(object(), "pay", "c1") is None

    @pytest.mark.parametrize("paused_here", [True, False])
    def test_an_approval_of_the_exact_call_counts(self, paused_here):
        wrapper = RunContextWrapper(context={})
        wrapper.approve_tool(_item("c1"))
        decision = _sdk_approval(wrapper, "pay", "c1", paused_here=paused_here)
        assert decision is not None and decision.approved is True

    def test_a_blanket_approval_counts_for_the_call_paused_here(self):
        wrapper = RunContextWrapper(context={})
        wrapper.approve_tool(_item("c1"), always_approve=True)
        assert wrapper._approvals["pay"].approved is True
        assert _sdk_approval(wrapper, "pay", "c2", paused_here=True).approved is True

    def test_a_blanket_approval_does_not_count_for_a_call_not_paused_here(self):
        wrapper = RunContextWrapper(context={})
        wrapper.approve_tool(_item("c1"), always_approve=True)
        assert _sdk_approval(wrapper, "pay", "c2", paused_here=False) is None

    def test_paused_here_is_the_default(self):
        wrapper = RunContextWrapper(context={})
        wrapper.approve_tool(_item("c1"), always_approve=True)
        assert _sdk_approval(wrapper, "pay", "c2").approved is True

    @pytest.mark.parametrize("paused_here", [True, False])
    @pytest.mark.parametrize("always", [True, False])
    def test_a_rejection_always_counts(self, paused_here, always):
        wrapper = RunContextWrapper(context={})
        wrapper.reject_tool(_item("c1"), always_reject=always)
        decision = _sdk_approval(wrapper, "pay", "c1", paused_here=paused_here)
        assert decision is not None and decision.approved is False

    def test_another_tools_approval_does_not_count(self):
        wrapper = RunContextWrapper(context={})
        wrapper.approve_tool(_item("c1", tool_name="other"))
        assert _sdk_approval(wrapper, "pay", "c1", paused_here=False) is None


class TestPendingBookkeeping:
    @pytest.fixture
    def pdp(self, monkeypatch):
        fake = ApprovalPdp()
        monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
        return fake

    def test_forget_pending_says_whether_the_call_was_pending(self):
        engine = _OpenAIAuthZ(_config())
        engine.remember_pending("c1", "pay", ALICE, {})
        assert engine.forget_pending("c1") is True
        assert engine.forget_pending("c1") is False
        assert engine.forget_pending("never") is False
        assert engine.forget_pending(None) is False

    def test_log_rejections_forgets_and_logs_only_rejected_calls(self, pdp):
        recorder = DecisionRecorder()
        engine = _OpenAIAuthZ(_config(hooks=[recorder]))
        engine.remember_pending("ok", "pay", ALICE, {"invoice_id": "1"})
        engine.remember_pending("no", "pay", ALICE, {"invoice_id": "2"})
        engine.remember_pending("open", "pay", ALICE, {"invoice_id": "3"})
        wrapper = RunContextWrapper(context=ALICE)
        wrapper.approve_tool(_item("ok"))
        wrapper.reject_tool(_item("no"))
        engine.log_rejections(wrapper)
        assert recorder.decisions == ["rejected", "deny"]
        assert recorder.events[0].detail["note"] == "rejected with RunState.reject"
        assert engine.forget_pending("ok") is True       # approved: left for on_invoke_tool
        assert engine.forget_pending("open") is True     # not answered yet
        assert engine.forget_pending("no") is False      # rejected: forgotten

    def test_a_blanket_reject_in_one_callers_run_leaves_another_callers_call_alone(self, pdp):
        recorder = DecisionRecorder()
        engine = _OpenAIAuthZ(_config(hooks=[recorder]))
        bob = {"username": "bob"}
        engine.remember_pending("alice-call", "pay", ALICE, {"invoice_id": "1"})
        engine.remember_pending("bob-call", "pay", bob, {"invoice_id": "2"})
        bobs_run = RunContextWrapper(context=bob)
        bobs_run.reject_tool(_item("bob-call"), always_reject=True)   # rejects every "pay" call
        engine.log_rejections(bobs_run)
        assert {e.subject.get("username") for e in recorder.events} == {"bob"}
        assert engine.forget_pending("bob-call") is False     # bob's: logged and forgotten
        assert engine.forget_pending("alice-call") is True    # alice's: still paused

    def test_an_approval_runs_its_call_once(self):
        """A saved RunState resumed again must not run the tool again on the same approval."""
        engine = _OpenAIAuthZ(_config())
        assert engine.use_approval("c1", ALICE, {"x": 1}) is True
        assert engine.use_approval("c1", ALICE, {"x": 1}) is False     # the same call again
        assert engine.use_approval("c2", ALICE, {"x": 1}) is True
        # a provider that numbers call ids per run: the same id for another caller or arguments
        assert engine.use_approval("c1", {"username": "bob"}, {"x": 1}) is True
        assert engine.use_approval("c1", ALICE, {"x": 2}) is True

    def test_an_approved_call_keeps_its_blanket_approval_on_invoke(self, pdp):
        # log_rejections ran (next turn) before on_invoke_tool: the call is still "paused here",
        # so a blanket always_approve still counts for it.
        engine = _OpenAIAuthZ(_config())
        engine.remember_pending("c1", "pay", ALICE, {})
        wrapper = RunContextWrapper(context=ALICE)
        wrapper.approve_tool(_item("c1"), always_approve=True)
        engine.log_rejections(wrapper)
        paused_here = engine.forget_pending("c1")
        assert paused_here is True
        assert _sdk_approval(wrapper, "pay", "c1", paused_here=paused_here).approved is True
