"""
Example: human approval of tool calls, asked for by the policy — OpenAI Agents SDK.

The OpenAI counterpart of examples/langchain_agents/human_approval.py, on the same policy
(domain_human_approval.yaml): payments up to 100 EUR go through; a larger payment or any
cancellation needs a human's approval. The model is real; the human is you, at the console
(or --approve / --reject).

A call that needs approval goes through the SDK's own approval flow: the tool's
``needs_approval`` also asks the PDP, so the run returns with a ``ToolApprovalItem`` in
``result.interruptions``. The app answers with ``state.approve(item)`` / ``state.reject(item)``
and runs again from the state — the SDK then checks an approval with the PDP, and logs a
rejection on the next turn. The approval item carries the tool call, not the policy's reason.

Run with ADS at http://localhost:8080 and examples/policies/domain_human_approval.yaml loaded.
Set LLM_MODEL (or OPENAI_LLM_MODEL) and your API key in a .env file or environment (routed
through OpenRouter by default).

    uv run python examples/openai_agents/human_approval.py [--approve | --reject]
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool
from agents.tracing import set_tracing_disabled

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for approval_console
from approval_console import (
    INVOICES, SYSTEM_PROMPT, AuditHook, InvoiceContextProvider, ask_human, list_invoices_text,
)

from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.openai_agents import authorized_openai_agent

set_tracing_disabled(True)  # tracing would post the tool arguments to OpenAI's platform

MODEL = os.getenv("OPENAI_LLM_MODEL", os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")).removeprefix("openrouter/")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")


@function_tool
def list_invoices() -> str:
    """List the open invoices."""
    return list_invoices_text()


@function_tool
def pay_invoice(invoice_id: str) -> str:
    """Pay an invoice."""
    invoice = INVOICES[invoice_id]
    return f"paid {invoice_id} ({invoice['amount']:.2f} EUR to {invoice['creditor']})"


@function_tool
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
    model = OpenAIChatCompletionsModel(model=MODEL, openai_client=AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY))
    agent = Agent(
        name="finance_assistant", instructions=SYSTEM_PROMPT, model=model,
        tools=[list_invoices, pay_invoice, cancel_invoice],
    )
    return authorized_openai_agent(agent, config)


async def ask(agent: Agent, request: str) -> None:
    print(f"\n[alice] {request}")
    result = await Runner.run(agent, request, context={"username": "alice"})
    while result.interruptions:   # paused for a human: answer each item, then run again
        state = result.to_state()
        for item in result.interruptions:
            answer = ask_human({"tool_name": item.tool_name, "args": json.loads(item.arguments or "{}")})
            (state.approve if answer["approved"] else state.reject)(item)
        result = await Runner.run(agent, state)
    print(f"[agent] {str(result.final_output).strip()}")


async def main() -> None:
    agent = build_agent()
    for request in ("Pay invoice INV-42.", "Pay invoice INV-43.", "Cancel invoice INV-43."):
        await ask(agent, request)


if __name__ == "__main__":
    asyncio.run(main())
