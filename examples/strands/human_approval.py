"""
Example: human approval of tool calls, asked for by the policy — Strands.

The Strands counterpart of examples/langchain_agents/human_approval.py, on the same policy
(domain_human_approval.yaml): payments up to 100 EUR go through; a larger payment or any
cancellation needs a human's approval. The model is real; the human is you, at the console
(or --approve / --reject).

A call that needs approval pauses the agent with a Strands interrupt named
``axiomatics-approval``: the result's ``stop_reason`` is "interrupt" and each interrupt's
``reason`` is the approval request. The app resumes by calling the agent with an
``interruptResponse`` per interrupt — the SDK then checks the answer with the PDP.

Run with ADS at http://localhost:8080 and examples/policies/domain_human_approval.yaml loaded.
Set LLM_MODEL (or STRANDS_LLM_MODEL) and your API key in a .env file or environment.

    uv run python examples/strands/human_approval.py [--approve | --reject]
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from strands import Agent, tool
from strands.models.litellm import LiteLLMModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for approval_console
from approval_console import (
    INVOICES, SYSTEM_PROMPT, AuditHook, InvoiceContextProvider, ask_human, list_invoices_text,
)

from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.strands import authorized_strands_agent

MODEL = os.getenv("STRANDS_LLM_MODEL", os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna"))


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


def build_agent() -> Agent:
    config = AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        context_provider=InvoiceContextProvider(),
        hooks=[AuditHook()],
    )
    agent = Agent(
        model=LiteLLMModel(model_id=MODEL), system_prompt=SYSTEM_PROMPT,
        tools=[list_invoices, pay_invoice, cancel_invoice],
        state={"username": "alice"}, callback_handler=None,
    )
    return authorized_strands_agent(agent, config)


def ask(request: str) -> None:
    print(f"\n[alice] {request}")
    agent = build_agent()   # a fresh conversation per request
    result = agent(request)
    while result.stop_reason == "interrupt":   # paused for a human: answer each, then resume
        responses = [
            {"interruptResponse": {"interruptId": pause.id, "response": ask_human(pause.reason)}}
            for pause in result.interrupts
        ]
        result = agent(responses)
    print(f"[agent] {str(result).strip()}")


if __name__ == "__main__":
    for request in ("Pay invoice INV-42.", "Pay invoice INV-43.", "Cancel invoice INV-43."):
        ask(request)
