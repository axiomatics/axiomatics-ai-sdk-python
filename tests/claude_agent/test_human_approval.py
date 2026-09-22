"""
Human approval on the Claude Agent SDK path: there is no framework interrupt to pause on, so
only the configured approver can approve (can_use_tool awaits it); without one the call is
denied and the rejection logged. PDP stubbed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("claude_agent_sdk")

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.claude_agent.authz import _ClaudeAuthZ
from axiomatics_ai.core.pdp import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig
from tests.approval_fakes import ApprovalPdp, DecisionRecorder


@pytest.fixture
def pdp(monkeypatch):
    fake = ApprovalPdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
    return fake


def _engine(approver=None, recorder=None) -> _ClaudeAuthZ:
    return _ClaudeAuthZ(AuthZConfig(
        enforcement=standard_tool_enforcement(),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
        hooks=[recorder] if recorder else [],
    ), {"username": "alice"})


def test_approver_approves(pdp):
    assert _engine(lambda r: ApprovalDecision(True, approver="manager")).invocation_denial(
        "pay_invoice", {"invoice_id": "INV-43"}) is None


def test_async_approver(pdp):
    async def approver(request):
        return ApprovalDecision(True)
    assert _engine(approver).invocation_denial("pay_invoice", {"invoice_id": "INV-43"}) is None


def test_approver_rejects(pdp):
    assert _engine(lambda r: ApprovalDecision(False)).invocation_denial("pay_invoice", {"invoice_id": "INV-43"})


def test_without_approver_denied_and_logged(pdp):
    recorder = DecisionRecorder()
    assert _engine(recorder=recorder).invocation_denial("pay_invoice", {"invoice_id": "INV-43"})
    assert recorder.decisions == ["approval_required", "rejected", "deny"]
    assert recorder.events[-2].detail["source"] == "none"
    assert recorder.events[-1].detail["not_approved"] is True
