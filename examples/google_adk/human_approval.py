"""
Example: human approval of tool calls, asked for by the policy — Google ADK.

The ADK counterpart of examples/langchain_agents/human_approval.py, on the same policy
(domain_human_approval.yaml): payments up to 100 EUR go through; a larger payment or any
cancellation needs a human's approval. The model is real; the human is you, at the console
(or --approve / --reject).

A call that needs approval pauses the run with ADK tool confirmation: the event stream carries
an ``adk_request_confirmation`` function call whose payload is the approval request. The app
answers with a FunctionResponse ({"confirmed": bool, "payload": {...}}), and ADK runs the tool
call again — the SDK then checks the answer with the PDP.

Run with ADS at http://localhost:8080 and examples/policies/domain_human_approval.yaml loaded.
Set LLM_MODEL (or GOOGLE_ADK_LLM_MODEL) and your API key in a .env file or environment.

    uv run python examples/google_adk/human_approval.py [--approve | --reject]
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for approval_console
from approval_console import (
    INVOICES, SYSTEM_PROMPT, AuditHook, InvoiceContextProvider, ask_human, list_invoices_text,
)

from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.google_adk import authorized_adk_agent

MODEL = os.getenv("GOOGLE_ADK_LLM_MODEL", os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna"))


def list_invoices() -> str:
    """List the open invoices."""
    return list_invoices_text()


def pay_invoice(invoice_id: str) -> str:
    """Pay an invoice."""
    invoice = INVOICES[invoice_id]
    return f"paid {invoice_id} ({invoice['amount']:.2f} EUR to {invoice['creditor']})"


def cancel_invoice(invoice_id: str) -> str:
    """Cancel an invoice."""
    return f"cancelled {invoice_id}"


def build_agent() -> LlmAgent:
    config = AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        context_provider=InvoiceContextProvider(),
        hooks=[AuditHook()],
    )
    agent = LlmAgent(
        name="finance_assistant", model=LiteLlm(model=MODEL), instruction=SYSTEM_PROMPT,
        tools=[list_invoices, pay_invoice, cancel_invoice],
    )
    return authorized_adk_agent(agent, config)


async def ask(runner: InMemoryRunner, request: str) -> None:
    print(f"\n[alice] {request}")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="alice", state={"username": "alice"},
    )
    message = types.Content(role="user", parts=[types.Part(text=request)])
    answer_text = ""
    while message is not None:
        confirmations, reply = [], None
        async for event in runner.run_async(user_id="alice", session_id=session.id, new_message=message):
            for part in (event.content.parts if event.content else None) or []:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    confirmations.append(part.function_call)
                elif part.text and event.author == "finance_assistant":
                    answer_text = part.text
        if confirmations:   # paused for a human: answer each request, then resume
            parts = []
            for call in confirmations:
                answer = ask_human(call.args["toolConfirmation"]["payload"])
                parts.append(types.Part(function_response=types.FunctionResponse(
                    name="adk_request_confirmation", id=call.id,
                    response={"confirmed": answer["approved"], "payload": {"approver": answer["approver"]}},
                )))
            reply = types.Content(role="user", parts=parts)
        message = reply
    print(f"[agent] {answer_text.strip()}")


async def main() -> None:
    runner = InMemoryRunner(agent=build_agent(), app_name="human_approval")
    for request in ("Pay invoice INV-42.", "Pay invoice INV-43.", "Cancel invoice INV-43."):
        await ask(runner, request)


if __name__ == "__main__":
    asyncio.run(main())
