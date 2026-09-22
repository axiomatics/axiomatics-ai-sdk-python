"""
Human approval on the Strands path: with no approver configured, a tool call the policy wants
approved pauses the agent with a Strands interrupt (``axiomatics-approval``); the app resumes
with an ``interruptResponse``. Runs a real Strands agent with a scripted test model; the PDP
is stubbed.
"""
from __future__ import annotations

import pytest
from strands import Agent, tool
from strands.models import Model

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.core.pdp import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig
from axiomatics_ai.strands import authorized_strands_agent
from axiomatics_ai.strands.authz import _StrandsAuthZ
from tests.approval_fakes import ApprovalPdp, DecisionRecorder

from .conftest import FakeToolEvent


class _ScriptedModel(Model):
    """Calls pay_invoice on the first turn, then answers "done"."""

    def __init__(self):
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
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "pay_invoice"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"invoice_id": "INV-43"}'}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


@pytest.fixture
def pdp(monkeypatch):
    fake = ApprovalPdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
    return fake


def _config(approver=None, recorder=None):
    return AuthZConfig(
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
        hooks=[recorder] if recorder else [],
    )


def _agent(config):
    ran: list[str] = []

    @tool
    def pay_invoice(invoice_id: str) -> str:
        """Pay an invoice."""
        ran.append(invoice_id)
        return "paid"

    agent = Agent(model=_ScriptedModel(), tools=[pay_invoice], state={"username": "alice"}, callback_handler=None)
    return authorized_strands_agent(agent, config), ran


def test_pauses_with_the_approval_request(pdp):
    agent, ran = _agent(_config())
    result = agent("pay INV-43")
    assert result.stop_reason == "interrupt"
    (pause,) = result.interrupts
    assert pause.name == "axiomatics-approval"
    assert pause.reason["type"] == "axiomatics.approval_request" and pause.reason["tool_name"] == "pay_invoice"
    assert ran == []


@pytest.mark.parametrize(("response", "expected"), [
    ({"approved": True, "approver": "manager"}, ["INV-43"]),
    ({"approved": False}, []),
    ("y", []),   # only an explicit True approves
])
def test_resume(pdp, response, expected):
    recorder = DecisionRecorder()
    agent, ran = _agent(_config(recorder=recorder))
    paused = agent("pay INV-43")
    result = agent([{"interruptResponse": {"interruptId": paused.interrupts[0].id, "response": response}}])
    assert result.stop_reason == "end_turn"
    assert ran == expected
    assert ("approved" if expected else "rejected") in recorder.decisions


def test_an_approver_answers_without_pausing(pdp):
    agent, ran = _agent(_config(approver=lambda r: ApprovalDecision(True)))
    assert agent("pay INV-43").stop_reason == "end_turn"
    assert ran == ["INV-43"]


def test_event_without_interrupt_support_denies(pdp):
    recorder = DecisionRecorder()
    engine = _StrandsAuthZ(_config(recorder=recorder))
    reason = engine.before_tool(FakeToolEvent("pay_invoice", {"invoice_id": "INV-43"}, invocation_state={"username": "alice"}))
    assert reason == "'pay_invoice' was not approved, so it did not run."
    assert recorder.decisions == ["approval_required", "rejected", "deny"]


def test_resume_does_not_ask_the_pdp_again(pdp, monkeypatch):
    # The resumed run runs the hook again; it must not re-ask the PDP before the answer.
    monkeypatch.setattr(AuthZConfig, "build_cache", lambda self: None)   # every check is a PDP call
    recorder = DecisionRecorder()
    agent, ran = _agent(_config(recorder=recorder))
    paused = agent("pay INV-43")
    agent([{"interruptResponse": {"interruptId": paused.interrupts[0].id, "response": {"approved": True}}}])
    assert ran == ["INV-43"]
    assert pdp.environments == [{}, {"approval_granted": True}]
    assert recorder.decisions == ["approval_required", "approved", "permit"]
    assert len({e.correlation_id for e in recorder.events}) == 1
