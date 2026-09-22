"""
Human approval on the LangChain path: with no approver configured, a tool call the policy
wants approved pauses the graph with LangGraph's ``interrupt()``; the app resumes with
``Command(resume=...)``. The model is a scripted test model; the PDP is stubbed.
"""
from __future__ import annotations

import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, LangChainAgentAuthZMiddleware, PDPConnection, standard_tool_enforcement
from axiomatics_ai.core.pdp import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig
from tests.approval_fakes import ApprovalPdp, DecisionRecorder


class _Model(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _agent(approver=None, recorder=None, cache_ttl_seconds=300.0):
    ran: list[str] = []

    @tool
    def pay_invoice(invoice_id: str) -> str:
        """Pay an invoice."""
        ran.append(invoice_id)
        return "paid"

    messages = iter([
        AIMessage(content="", tool_calls=[{"name": "pay_invoice", "args": {"invoice_id": "INV-43"}, "id": "c1"}]),
        AIMessage(content="done"),
    ])
    config = AuthZConfig(
        connection=PDPConnection(cache_ttl_seconds=cache_ttl_seconds),
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
        hooks=[recorder] if recorder else [],
    )
    agent = create_agent(
        _Model(messages=messages), tools=[pay_invoice],
        middleware=[LangChainAgentAuthZMiddleware(config)], checkpointer=InMemorySaver(),
    )
    return agent, ran


THREAD = {"configurable": {"thread_id": "t1"}}
ALICE = {"username": "alice"}


@pytest.fixture
def pdp(monkeypatch):
    fake = ApprovalPdp()
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", fake)
    return fake


def test_pauses_with_the_approval_request(pdp):
    agent, ran = _agent()
    result = agent.invoke({"messages": [("user", "pay INV-43")]}, THREAD, context=ALICE)
    (pause,) = result["__interrupt__"]
    assert pause.value["type"] == "axiomatics.approval_request"
    assert pause.value["tool_name"] == "pay_invoice" and pause.value["args"] == {"invoice_id": "INV-43"}
    assert ran == []


@pytest.mark.parametrize(("answer", "expected"), [
    ({"approved": True, "approver": "manager"}, ["INV-43"]),
    ({"approved": False}, []),
    ("yes please", []),   # only an explicit True approves
])
def test_resume(pdp, answer, expected):
    recorder = DecisionRecorder()
    agent, ran = _agent(recorder=recorder)
    agent.invoke({"messages": [("user", "pay INV-43")]}, THREAD, context=ALICE)
    result = agent.invoke(Command(resume=answer), THREAD, context=ALICE)
    assert ran == expected
    assert "__interrupt__" not in result
    # One approval_required: the resumed run re-runs the step, but doesn't check the call again.
    assert recorder.decisions == (
        ["approval_required", "approved", "permit"] if expected else ["approval_required", "rejected", "deny"]
    )
    assert len({e.correlation_id for e in recorder.events}) == 1


def test_async_path_pauses_and_resumes(pdp):
    agent, ran = _agent()

    async def run():
        await agent.ainvoke({"messages": [("user", "pay INV-43")]}, THREAD, context=ALICE)
        await agent.ainvoke(Command(resume={"approved": True}), THREAD, context=ALICE)
    asyncio.run(run())
    assert ran == ["INV-43"]


def test_an_approver_answers_without_pausing(pdp):
    agent, ran = _agent(approver=lambda request: ApprovalDecision(True, approver="bot"))
    result = agent.invoke({"messages": [("user", "pay INV-43")]}, THREAD, context=ALICE)
    assert "__interrupt__" not in result
    assert ran == ["INV-43"]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_resume_does_not_ask_the_pdp_again(pdp, mode):
    # With the cache off, every check is a PDP call: the ask, then the one with the approval.
    agent, ran = _agent(cache_ttl_seconds=0)
    if mode == "sync":
        agent.invoke({"messages": [("user", "pay INV-43")]}, THREAD, context=ALICE)
        agent.invoke(Command(resume={"approved": True}), THREAD, context=ALICE)
    else:
        async def run():
            await agent.ainvoke({"messages": [("user", "pay INV-43")]}, THREAD, context=ALICE)
            await agent.ainvoke(Command(resume={"approved": True}), THREAD, context=ALICE)
        asyncio.run(run())
    assert ran == ["INV-43"]
    assert pdp.environments == [{}, {"approval_granted": True}]
