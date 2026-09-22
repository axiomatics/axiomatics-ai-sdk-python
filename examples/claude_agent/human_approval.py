"""
Example: human approval of tool calls, asked for by the policy — Claude Agent SDK.

The Claude counterpart of examples/langchain_agents/human_approval.py, on the same policy
(domain_human_approval.yaml): payments up to 100 EUR go through; a larger payment or any
cancellation needs a human's approval. The model is real; the human is you, at the console
(or --approve / --reject).

The Claude Agent SDK has no interrupt to pause a tool call on, so approval comes from an
APPROVER callback (ObligationsAndAdviceConfig.approval.approver): the PreToolUse hook simply
waits for it, in process. Without an approver, a call that needs approval is denied.

The tools run in an in-process SDK MCP server.

Run with ADS at http://localhost:8080 and examples/policies/domain_human_approval.yaml loaded.
Set OPENROUTER_API_KEY (or ANTHROPIC_API_KEY), and optionally ANTHROPIC_LLM_MODEL.

    uv run python examples/claude_agent/human_approval.py [--approve | --reject]
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Route the Claude Agent SDK through OpenRouter (same provider as the other examples), unless
# the environment is already pointed at a real Anthropic endpoint.
if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ.setdefault("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", os.environ["OPENROUTER_API_KEY"])

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, create_sdk_mcp_server, tool,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for approval_console
from approval_console import (
    INVOICES, SYSTEM_PROMPT, AuditHook, InvoiceContextProvider, ask_human, list_invoices_text,
)

from axiomatics_ai import (
    ApprovalConfig,
    ApprovalDecision,
    AuthZConfig,
    ObligationsAndAdviceConfig,
    PDPConnection,
    standard_tool_enforcement,
)
from axiomatics_ai.claude_agent import authorized_claude_options

MODEL = os.getenv("ANTHROPIC_LLM_MODEL", os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")).removeprefix("openrouter/")


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


@tool("list_invoices", "List the open invoices.", {})
async def list_invoices(args: dict) -> dict:
    return _text(list_invoices_text())


@tool("pay_invoice", "Pay an invoice.", {"invoice_id": str})
async def pay_invoice(args: dict) -> dict:
    invoice = INVOICES[args["invoice_id"]]
    return _text(f"paid {args['invoice_id']} ({invoice['amount']:.2f} EUR to {invoice['creditor']})")


@tool("cancel_invoice", "Cancel an invoice.", {"invoice_id": str})
async def cancel_invoice(args: dict) -> dict:
    return _text(f"cancelled {args['invoice_id']}")


async def approver(request) -> ApprovalDecision:
    """The human: the PreToolUse hook waits for this; input() runs on a thread so the loop keeps going."""
    return ApprovalDecision.from_answer(await asyncio.to_thread(ask_human, request.to_payload()))


def options_for(user: str) -> ClaudeAgentOptions:
    config = AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        enforcement=standard_tool_enforcement(visibility=False, post_invocation=False),
        context_provider=InvoiceContextProvider(),
        obligations_and_advice=ObligationsAndAdviceConfig(approval=ApprovalConfig(approver=approver)),
        hooks=[AuditHook()],
    )
    finance = create_sdk_mcp_server("finance", tools=[list_invoices, pay_invoice, cancel_invoice])
    # The PreToolUse hook from authorized_claude_options governs every tool call.
    base = ClaudeAgentOptions(model=MODEL, system_prompt=SYSTEM_PROMPT, mcp_servers={"finance": finance}, max_turns=6)
    return authorized_claude_options(base, config, identity={"username": user})


async def ask(request: str) -> None:
    print(f"\n[alice] {request}")
    parts: list[str] = []
    async with ClaudeSDKClient(options=options_for("alice")) as client:
        await client.query(request)
        async for message in client.receive_response():
            if isinstance(message, ResultMessage) and message.result:
                parts = [message.result]
            elif isinstance(message, AssistantMessage) and not parts:
                parts += [b.text for b in message.content if getattr(b, "text", None)]
    print(f"[agent] {' '.join(' '.join(parts).split())[:400]}")


async def main() -> None:
    for request in ("Pay invoice INV-42.", "Pay invoice INV-43.", "Cancel invoice INV-43."):
        await ask(request)


if __name__ == "__main__":
    asyncio.run(main())
