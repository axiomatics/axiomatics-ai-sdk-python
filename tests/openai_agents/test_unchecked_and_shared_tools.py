"""
OpenAI adapter: tools the call-by-call check can't reach (hosted tools without
``on_invoke_tool``) are decided by name at BEFORE_TOOL in one batch; the before-model guardrail
runs before the model (not in parallel); and a FunctionTool shared by two authorized agents is
checked by the engine of the agent making the call. Real ``agents`` objects, fake PDP, no model.
"""
from __future__ import annotations

import json

import asyncio

from agents import Agent, FileSearchTool, WebSearchTool, function_tool
from agents.run_context import RunContextWrapper
from agents.tool_context import ToolContext

from axiomatics_ai import AuthZConfig, Enforcer, EnforcementPoint, standard_tool_enforcement
from axiomatics_ai.openai_agents.authz import (
    _OpenAIAuthZ,
    _without_denied_unchecked_tools,
    authorized_openai_agent,
)
from tests.approval_fakes import DecisionRecorder

from .conftest import FakePdp, FakeTool

ALICE = {"username": "alice"}


class BatchRecordingPdp(FakePdp):
    """FakePdp that also records each batched request (action + names)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batches: list[dict] = []

    def check_permission_batch(self, *, subject, action, resource_names, environment,
                               resource_attr_id, resource_extra_attrs=None, fail_open=False) -> set[str]:
        self.batches.append({"subject": subject, "action": action, "names": sorted(resource_names)})
        return super().check_permission_batch(
            subject=subject, action=action, resource_names=resource_names, environment=environment,
            resource_attr_id=resource_attr_id, resource_extra_attrs=resource_extra_attrs, fail_open=fail_open,
        )


def _engine_of(agent) -> _OpenAIAuthZ:
    return agent._axiomatics_engine


def _config(**kwargs) -> AuthZConfig:
    enforcement = kwargs.pop("enforcement", None) or standard_tool_enforcement(
        invocation_action="tool_call", post_invocation=False,
    )
    return AuthZConfig(enforcement=enforcement, **kwargs)


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"found {q}"


# ── hosted tools: decided by name at BEFORE_TOOL ─────────────────────────────────

class TestWithoutDeniedUncheckedTools:
    def _engine(self, fake) -> _OpenAIAuthZ:
        engine = _OpenAIAuthZ(_config())
        engine._ads = fake
        return engine

    def test_denied_hosted_tools_removed_permitted_kept_function_tools_untouched(self):
        fake = BatchRecordingPdp(allowed_tools={"web_search"})
        web, files, fn = WebSearchTool(), FileSearchTool(vector_store_ids=["v"]), FakeTool("lookup")
        out = _without_denied_unchecked_tools(ALICE, [web, files, fn], self._engine(fake))
        assert out == [web, fn]                         # file_search denied; function tool untouched
        assert len(fake.batches) == 1                   # one batch for all the hosted tools
        assert fake.batches[0]["names"] == ["file_search", "web_search"]
        assert fake.batches[0]["action"] == {"name": "tool_call"}   # the BEFORE_TOOL action

    def test_no_pdp_call_without_unchecked_tools(self):
        fake = BatchRecordingPdp(allowed_tools=set())
        tools = [FakeTool("a"), FakeTool("b")]
        assert _without_denied_unchecked_tools(ALICE, tools, self._engine(fake)) == tools
        assert fake.batches == [] and fake.calls == []

    def test_all_permitted_keeps_everything(self):
        fake = BatchRecordingPdp(permit=True)
        tools = [WebSearchTool(), FileSearchTool(vector_store_ids=["v"])]
        assert _without_denied_unchecked_tools(ALICE, tools, self._engine(fake)) == tools


class TestHostedToolsThroughTheAgent:
    def _agent(self, config, fake) -> Agent:
        agent = Agent(name="assistant", tools=[lookup, WebSearchTool(), FileSearchTool(vector_store_ids=["v"])])
        authorized_openai_agent(agent, config)
        _engine_of(agent)._ads = fake
        return agent

    def test_hosted_tools_decided_at_tool_invocation(self):
        recorder = DecisionRecorder()
        fake = BatchRecordingPdp(permit=True)
        # Visibility shows every tool; at BEFORE_TOOL the policy permits only web_search.
        original = fake.check_permission_batch

        def batch(**kw):
            permitted = original(**kw)
            return permitted if kw["action"] == {"name": "tool_visibility"} else permitted & {"web_search"}

        fake.check_permission_batch = batch
        agent = self._agent(_config(hooks=[recorder]), fake)
        tools = asyncio.run(agent.get_all_tools(RunContextWrapper(context=dict(ALICE))))
        assert sorted(t.name for t in tools) == ["lookup", "web_search"]   # file_search removed
        hosted = [e for e in recorder.events if e.enforcement_point == "tool_invocation"]
        assert len(hosted) == 1
        assert hosted[0].action == {"name": "tool_call"}
        assert hosted[0].resource == {"tool_names": ["file_search", "web_search"]}
        visibility = [e for e in recorder.events if e.enforcement_point == "tool_visibility"]
        assert len(visibility) == 1 and visibility[0].action == {"name": "tool_visibility"}

    def test_only_visibility_enforced_asks_nothing_extra(self):
        fake = BatchRecordingPdp(allowed_tools={"lookup", "web_search"})
        config = AuthZConfig(enforcement={EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="tool_visibility")})
        agent = self._agent(config, fake)
        tools = asyncio.run(agent.get_all_tools(RunContextWrapper(context=dict(ALICE))))
        assert sorted(t.name for t in tools) == ["lookup", "web_search"]   # visibility only
        assert [b["action"] for b in fake.batches] == [{"name": "tool_visibility"}]
        assert fake.calls == []


# ── before-model guardrail is blocking ───────────────────────────────────────────

def test_before_model_guardrail_is_not_run_in_parallel():
    agent = Agent(name="assistant")
    authorized_openai_agent(agent, AuthZConfig(enforcement={
        EnforcementPoint.BEFORE_MODEL: Enforcer(action_name="before_model"),
    }))
    (guard,) = agent.input_guardrails
    assert guard.name == "axiomatics-before-model"
    assert guard.run_in_parallel is False


# ── a FunctionTool shared by two authorized agents ───────────────────────────────

class TestSharedFunctionTool:
    def _setup(self):
        @function_tool
        def shared_lookup(q: str) -> str:
            """Look something up."""
            return f"found {q}"

        a = Agent(name="a", tools=[shared_lookup])
        b = Agent(name="b", tools=[shared_lookup])
        authorized_openai_agent(a, _config(enforcement=standard_tool_enforcement(
            visibility=False, post_invocation=False, invocation_action="a_call")))
        authorized_openai_agent(b, _config(enforcement=standard_tool_enforcement(
            visibility=False, post_invocation=False, invocation_action="b_call")))
        fake_a, fake_b = FakePdp(permit=True), FakePdp(permit=True)
        _engine_of(a)._ads, _engine_of(b)._ads = fake_a, fake_b
        # Wrapping happens in get_all_tools; a wraps it first, b's call finds it already wrapped.
        for agent in (a, b):
            asyncio.run(agent.get_all_tools(RunContextWrapper(context=dict(ALICE))))
        return shared_lookup, a, b, fake_a, fake_b

    def _call(self, tool, agent, call_id="c1"):
        # The query is the call id, so the decision cache never answers a later call.
        args = json.dumps({"q": call_id})
        ctx = ToolContext(
            context=dict(ALICE), tool_name=tool.name, tool_call_id=call_id,
            tool_arguments=args, agent=agent,
        )
        return asyncio.run(tool.on_invoke_tool(ctx, args))

    def test_checked_by_the_calling_agents_engine(self):
        tool, a, b, fake_a, fake_b = self._setup()
        assert self._call(tool, b) == "found c1"
        assert fake_a.calls == []
        assert fake_b.calls[-1]["subject"]["agent_name"] == "b"
        assert fake_b.calls[-1]["action"] == {"name": "b_call"}

        assert self._call(tool, a, "c2") == "found c2"
        assert fake_a.calls[-1]["subject"]["agent_name"] == "a"
        assert fake_a.calls[-1]["action"] == {"name": "a_call"}

    def test_unknown_or_missing_agent_falls_back_to_the_first_engine(self):
        tool, _a, _b, fake_a, fake_b = self._setup()
        for i, agent in enumerate((Agent(name="stranger"), None)):
            assert self._call(tool, agent, f"c{i}") == f"found c{i}"
        assert len(fake_a.calls) == 2 and fake_b.calls == []
        assert all(c["subject"]["agent_name"] == "a" for c in fake_a.calls)

    def test_denied_by_the_calling_agents_policy(self):
        tool, a, b, fake_a, fake_b = self._setup()
        fake_b.permit = False
        out = self._call(tool, b)
        assert isinstance(out, str) and "Access denied" in out
        assert self._call(tool, a, "c2") == "found c2"
