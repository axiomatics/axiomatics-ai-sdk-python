"""
Human approval on the FastMCP server path: with no approver configured, a tool call the
policy wants approved asks the MCP client's user with an elicitation (approve / reject). The
client must support elicitation; one that doesn't is a deny. In-memory client; PDP stubbed.
"""
from __future__ import annotations

import asyncio
import secrets

import jwt
import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.elicitation import ElicitResult

import axiomatics_ai.core.enforcement as enforcement_module
import axiomatics_ai.fastmcp.fastmcp_middleware as fmw_module
from axiomatics_ai import AuthZConfig, standard_tool_enforcement
from axiomatics_ai.core.pdp import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig
from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware
from tests.approval_fakes import ApprovalPdp, DecisionRecorder

# A throwaway signing key for the demo token (the server reads it without verifying).
_SECRET = secrets.token_hex(32)


@pytest.fixture
def pdp(monkeypatch):
    fake = ApprovalPdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
    token = jwt.encode({"username": "alice"}, _SECRET, algorithm="HS256")
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: {"authorization": f"Bearer {token}"})
    return fake


def _server(approver=None, recorder=None) -> FastMCP:
    mcp = FastMCP("payments")

    @mcp.tool
    def pay_invoice(invoice_id: str) -> str:
        return f"paid {invoice_id}"

    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
        hooks=[recorder] if recorder else [],
    )))
    return mcp


def _call(mcp, answer="__no_handler__"):
    """Call pay_invoice; the client answers the elicitation with ``answer``. Returns the
    tool's text, or the error message, and the prompts the client was shown."""
    prompts: list[str] = []

    async def handler(message, response_type, params, context):
        prompts.append(message)
        return answer

    async def main():
        kwargs = {} if answer == "__no_handler__" else {"elicitation_handler": handler}
        async with Client(mcp, **kwargs) as client:
            try:
                result = await client.call_tool("pay_invoice", {"invoice_id": "INV-43"})
                return result.content[0].text
            except Exception as exc:  # the ToolError the middleware raises on deny
                return f"error: {exc}"
    return asyncio.run(main()), prompts


def test_approve_by_elicitation(pdp):
    recorder = DecisionRecorder()
    text, prompts = _call(_server(recorder=recorder), "approve")
    assert text == "paid INV-43"
    assert "pay_invoice" in prompts[0] and "Payments over 100 need approval" in prompts[0]
    assert pdp.environments[-1] == {"approval_granted": True}
    assert recorder.decisions == ["approval_required", "approved", "permit"]


@pytest.mark.parametrize("answer", ["reject", ElicitResult(action="decline"), ElicitResult(action="cancel")])
def test_reject_decline_or_cancel_denies(pdp, answer):
    recorder = DecisionRecorder()
    text, _ = _call(_server(recorder=recorder), answer)
    assert text == "error: 'pay_invoice' was not approved, so it did not run."
    assert recorder.decisions == ["approval_required", "rejected", "deny"]
    assert recorder.events[1].detail["source"] == "interrupt"     # a human answered


def test_client_without_elicitation_support_denies(pdp):
    recorder = DecisionRecorder()
    text, _ = _call(_server(recorder=recorder))
    assert text == "error: 'pay_invoice' was not approved, so it did not run."
    rejected = recorder.events[-2]
    assert rejected.detail["note"] == "the client could not be asked (elicitation)"
    # Nobody was asked: logged like a missing approver, not like a human's rejection.
    assert rejected.detail["source"] == "none"
    assert recorder.decisions[-1] == "deny"


def test_an_approver_answers_without_eliciting(pdp):
    text, prompts = _call(_server(approver=lambda r: ApprovalDecision(True)), "reject")
    assert text == "paid INV-43" and prompts == []


def test_elicitation_does_not_ask_the_pdp_again(pdp, monkeypatch):
    monkeypatch.setattr(AuthZConfig, "build_cache", lambda self: None)   # every check is a PDP call
    recorder = DecisionRecorder()
    text, _ = _call(_server(recorder=recorder), "approve")
    assert text == "paid INV-43"
    assert pdp.environments == [{}, {"approval_granted": True}]
    assert len({e.correlation_id for e in recorder.events}) == 1
