"""
Example: human approval of tool calls, asked for by the policy — LangChain.

The policy (domain_human_approval.yaml) lets a payment up to 100 EUR through, but wants a
human to approve a larger one or any cancellation. The agent's model is real; the human is
you, at the console (or --approve / --reject).

  1. "Pay invoice INV-42" (40 EUR)   → permitted, no approval needed
  2. "Pay invoice INV-43" (500 EUR)  → Deny + advice require-approval: the graph PAUSES with
                                       LangGraph's interrupt(); you answer; the run resumes
                                       with Command(resume=...), and the PDP is asked again
                                       with approval_granted = true
  3. "Cancel invoice INV-43"         → Permit + obligation require-approval: same pause; your
                                       approval fulfils the obligation (one PDP call)

The policy decides on the invoice's REAL amount: InvoiceContextProvider (approval_console.py)
looks it up, rather than trusting an amount the model typed in.

Every step is logged by the audit hook: approval_required → approved / rejected → permit / deny.
With --approver, an approver callback asks you instead, inside the tool call, with no pause.

Run with ADS at http://localhost:8080 and examples/policies/domain_human_approval.yaml loaded.
Set LLM_MODEL and your API key in a .env file or environment.

    uv run python examples/langchain_agents/human_approval.py [--approve | --reject] [--approver]
"""
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for approval_console
from approval_console import (
    INVOICES, SYSTEM_PROMPT, AuditHook, InvoiceContextProvider, ask_human, list_invoices_text,
)

from axiomatics_ai import (
    ApprovalConfig,
    ApprovalDecision,
    AuthZConfig,
    LangChainAgentAuthZMiddleware,
    ObligationsAndAdviceConfig,
    PDPConnection,
    standard_tool_enforcement,
)

LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/google/gemini-3.5-flash-lite")


@tool
def list_invoices() -> str:
    """List the open invoices."""
    return list_invoices_text()


@tool
def pay_invoice(invoice_id: str) -> str:
    """Pay an invoice."""
    invoice = INVOICES[invoice_id]
    return f"paid {invoice_id} ({invoice['amount']:.2f} EUR to {invoice['creditor']})"


@tool
def cancel_invoice(invoice_id: str) -> str:
    """Cancel an invoice."""
    return f"cancelled {invoice_id}"


def build_agent(use_approver: bool):
    approver = (lambda request: ApprovalDecision.from_answer(ask_human(request.to_payload()))) if use_approver else None
    config = AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        context_provider=InvoiceContextProvider(),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
        hooks=[AuditHook()],
    )
    return create_agent(
        ChatLiteLLM(model=LLM_MODEL, temperature=0),
        tools=[list_invoices, pay_invoice, cancel_invoice],
        system_prompt=SYSTEM_PROMPT,
        middleware=[LangChainAgentAuthZMiddleware(config)],
        checkpointer=InMemorySaver(),   # interrupt() needs a checkpointer to resume
    )


def ask(agent, request: str) -> None:
    print(f"\n[alice] {request}")
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    context = {"username": "alice"}
    result = agent.invoke({"messages": [("user", request)]}, thread, context=context)
    while "__interrupt__" in result:   # paused for a human: answer, then resume
        answer = ask_human(result["__interrupt__"][0].value)
        result = agent.invoke(Command(resume=answer), thread, context=context)
    print(f"[agent] {result['messages'][-1].content}")


if __name__ == "__main__":
    agent = build_agent(use_approver="--approver" in sys.argv)
    for request in ("Pay invoice INV-42.", "Pay invoice INV-43.", "Cancel invoice INV-43."):
        ask(agent, request)
