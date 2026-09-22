"""
Human approval on the Google ADK path: with no approver configured, a tool call the policy
wants approved pauses with ADK tool confirmation (``adk_request_confirmation``); the app
answers with a FunctionResponse. Runs a real ADK runner with a scripted test model; the PDP
is stubbed.
"""
from __future__ import annotations

import asyncio

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.core.pdp import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig
from axiomatics_ai.google_adk import authorized_adk_agent
from axiomatics_ai.google_adk.authz import _AdkAuthZ
from tests.approval_fakes import ApprovalPdp, DecisionRecorder

from .conftest import FakeContext, FakeTool


class _ScriptedModel(BaseLlm):
    """Calls pay_invoice, then answers "done" once a function response is back."""

    model: str = "scripted"

    async def generate_content_async(self, llm_request, stream=False):
        last = llm_request.contents[-1]
        if any(part.function_response for part in last.parts or []):
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="done")]))
        else:
            call = types.FunctionCall(name="pay_invoice", args={"invoice_id": "INV-43"})
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))


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


def _run(config, answer):
    """Run until the confirmation request, answer it (unless ``answer`` is None), and return
    (the confirmation call or None, the invoice ids paid)."""
    ran: list[str] = []

    def pay_invoice(invoice_id: str) -> str:
        """Pay an invoice."""
        ran.append(invoice_id)
        return "paid"

    agent = authorized_adk_agent(LlmAgent(name="payments", model=_ScriptedModel(), tools=[pay_invoice]), config)

    async def main():
        runner = InMemoryRunner(agent=agent, app_name="app")
        session = await runner.session_service.create_session(app_name="app", user_id="alice", state={"username": "alice"})
        confirmation = None
        message = types.Content(role="user", parts=[types.Part(text="pay INV-43")])
        async for event in runner.run_async(user_id="alice", session_id=session.id, new_message=message):
            for part in (event.content.parts if event.content else None) or []:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    confirmation = part.function_call
        if confirmation is not None and answer is not None:
            reply = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                name="adk_request_confirmation", id=confirmation.id, response=answer,
            ))])
            async for _ in runner.run_async(user_id="alice", session_id=session.id, new_message=reply):
                pass
        return confirmation, ran

    return asyncio.run(main())


def test_pauses_with_the_approval_request(pdp):
    confirmation, ran = _run(_config(), answer=None)
    payload = confirmation.args["toolConfirmation"]["payload"]
    assert payload["type"] == "axiomatics.approval_request" and payload["tool_name"] == "pay_invoice"
    assert confirmation.args["toolConfirmation"]["hint"] == "Payments over 100 need approval"
    assert ran == []


def test_confirmed_runs_the_tool_after_a_second_pdp_check(pdp):
    recorder = DecisionRecorder()
    _, ran = _run(_config(recorder=recorder), answer={"confirmed": True, "payload": {"approver": "manager"}})
    assert ran == ["INV-43"]
    assert pdp.environments[-1] == {"approval_granted": True}
    approved = next(e for e in recorder.events if e.decision == "approved")
    assert approved.detail["approver"] == "manager" and approved.detail["source"] == "interrupt"


def test_declined_does_not_run_the_tool(pdp):
    recorder = DecisionRecorder()
    _, ran = _run(_config(recorder=recorder), answer={"confirmed": False})
    assert ran == []
    assert "rejected" in recorder.decisions


def test_an_approver_answers_without_pausing(pdp):
    confirmation, ran = _run(_config(approver=lambda r: ApprovalDecision(True)), answer=None)
    assert confirmation is None and ran == ["INV-43"]


def test_context_without_confirmation_support_denies(pdp):
    # A tool context that can't request confirmation (no function_call_id): denied, logged.
    recorder = DecisionRecorder()
    engine = _AdkAuthZ(_config(recorder=recorder))
    result = engine.before_tool(FakeTool("pay_invoice"), {"invoice_id": "INV-43"}, FakeContext({"username": "alice"}))
    assert result["error"] == "'pay_invoice' was not approved, so it did not run."
    assert recorder.decisions == ["approval_required", "rejected", "deny"]


def test_confirmation_does_not_ask_the_pdp_again(pdp, monkeypatch):
    monkeypatch.setattr(AuthZConfig, "build_cache", lambda self: None)   # every check is a PDP call
    recorder = DecisionRecorder()
    _, ran = _run(_config(recorder=recorder), answer={"confirmed": True, "payload": {}})
    assert ran == ["INV-43"]
    assert pdp.environments == [{}, {"approval_granted": True}]
    assert recorder.decisions == ["approval_required", "approved", "permit"]
    assert len({e.correlation_id for e in recorder.events}) == 1
