"""
Human approval on the OpenAI Agents path: with no approver configured, a tool call the policy
wants approved goes through the SDK's own approval flow (``needs_approval`` →
``result.interruptions`` → ``RunState.approve`` / ``reject``). Runs a real Runner with a
scripted test model; the PDP is stubbed.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from agents import Agent, Runner, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.core.pdp import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig
from axiomatics_ai.openai_agents import authorized_openai_agent
from tests.approval_fakes import PERMIT, ApprovalPdp, DecisionRecorder


class _ScriptedModel(Model):
    """Calls pay_invoice on the first turn, then answers "done"."""

    def __init__(self):
        self.turns = 0

    async def get_response(self, *args, **kwargs):
        self.turns += 1
        if self.turns == 1:
            output = [ResponseFunctionToolCall(
                type="function_call", id="fc1", call_id="c1", name="pay_invoice",
                arguments=json.dumps({"invoice_id": "INV-43"}),
            )]
        else:
            output = [ResponseOutputMessage(
                id="m1", type="message", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
            )]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def pdp(monkeypatch):
    fake = ApprovalPdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
    return fake


def _agent(approver=None, recorder=None, *, needs_approval=False):
    ran: list[str] = []

    @function_tool(needs_approval=needs_approval)
    def pay_invoice(invoice_id: str) -> str:
        """Pay an invoice."""
        ran.append(invoice_id)
        return "paid"

    config = AuthZConfig(
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
        hooks=[recorder] if recorder else [],
    )
    return authorized_openai_agent(Agent(name="payments", model=_ScriptedModel(), tools=[pay_invoice]), config), ran


def _run_and_answer(agent, verdict):
    async def main():
        paused = await Runner.run(agent, "pay INV-43", context={"username": "alice"})
        interruptions = paused.interruptions
        if verdict is None:
            return interruptions, None
        state = paused.to_state()
        getattr(state, verdict)(interruptions[0])
        return interruptions, await Runner.run(agent, state)
    return asyncio.run(main())


def test_pauses_with_an_sdk_approval_item(pdp):
    agent, ran = _agent()
    interruptions, _ = _run_and_answer(agent, None)
    assert [i.tool_name for i in interruptions] == ["pay_invoice"]
    assert ran == []


def test_approved_runs_the_tool_after_a_second_pdp_check(pdp):
    recorder = DecisionRecorder()
    agent, ran = _agent(recorder=recorder)
    _, result = _run_and_answer(agent, "approve")
    assert ran == ["INV-43"] and result.final_output == "done"
    assert pdp.environments[-1] == {"approval_granted": True}
    assert recorder.decisions[-2:] == ["approved", "permit"]


def test_rejected_is_logged_and_the_tool_never_runs(pdp):
    recorder = DecisionRecorder()
    agent, ran = _agent(recorder=recorder)
    _run_and_answer(agent, "reject")
    assert ran == []
    assert recorder.decisions[-2:] == ["rejected", "deny"]
    assert recorder.events[-2].detail["note"] == "rejected with RunState.reject"


def test_an_approver_answers_without_pausing(pdp):
    agent, ran = _agent(approver=lambda r: ApprovalDecision(True))
    interruptions, _ = _run_and_answer(agent, None)
    assert interruptions == [] and ran == ["INV-43"]


def test_the_tools_own_needs_approval_is_kept(monkeypatch):
    # The PDP permits, but the tool itself asks for approval: the SDK still pauses.
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", ApprovalPdp(first=PERMIT))
    agent, ran = _agent(needs_approval=True)
    interruptions, _ = _run_and_answer(agent, None)
    assert [i.tool_name for i in interruptions] == ["pay_invoice"] and ran == []


def test_an_ordinary_call_is_logged_once_and_runs_handlers_once(monkeypatch):
    # needs_approval's pre-check logs nothing and runs no handler; on_invoke_tool does both, once.
    from axiomatics_ai.core.pdp import Decision, Obligation, PdpDecision
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", ApprovalPdp(first=PdpDecision(Decision.PERMIT, (Obligation("log"),))))
    recorder, seen = DecisionRecorder(), []
    ran: list[str] = []

    @function_tool
    def pay_invoice(invoice_id: str) -> str:
        """Pay an invoice."""
        ran.append(invoice_id)
        return "paid"

    config = AuthZConfig(
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False), hooks=[recorder],
        obligations_and_advice=ObligationsAndAdviceConfig(handlers={"log": lambda o, c: seen.append(1)}),
    )
    agent = authorized_openai_agent(Agent(name="payments", model=_ScriptedModel(), tools=[pay_invoice]), config)
    _run_and_answer(agent, None)
    assert ran == ["INV-43"]
    assert recorder.decisions == ["permit"]
    assert seen == [1]


def test_approval_flow_logs_each_step_once(pdp):
    recorder = DecisionRecorder()
    agent, _ = _agent(recorder=recorder)
    _run_and_answer(agent, "approve")
    assert recorder.decisions == ["approval_required", "approved", "permit"]


def test_the_calls_own_check_reuses_the_pre_check(pdp, monkeypatch):
    # needs_approval asked the PDP; after approve, on_invoke_tool must not ask that again.
    monkeypatch.setattr(AuthZConfig, "build_cache", lambda self: None)   # every check is a PDP call
    recorder = DecisionRecorder()
    agent, ran = _agent(recorder=recorder)
    _run_and_answer(agent, "approve")
    assert ran == ["INV-43"]
    assert pdp.environments == [{}, {"approval_granted": True}]
    assert len({e.correlation_id for e in recorder.events}) == 1
