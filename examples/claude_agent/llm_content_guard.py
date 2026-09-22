"""
Example: BEFORE_MODEL input content guard on the Claude Agent SDK path.

The Claude counterpart of examples/langchain_agents/llm_content_guard.py, on the SAME policy
(domain_llm_content_guard.yaml) — but INPUT-ONLY. The Claude Agent SDK exposes no hook that can
see or redact the assistant's final answer, so only the BEFORE_MODEL guard is expressible here
(see axiomatics_ai/claude_agent/authz.py for why). AFTER_MODEL is not supported on this path.

BEFORE_MODEL (a UserPromptSubmit hook) — the input guard labels the prompt's `topic`; the policy
denies restricted topics (acquisitions, layoffs) for regular employees but exempts alice (an
executive). A block is a HARD block: the CLI never sends the turn to the model and the refusal
is surfaced as the block reason (there is no model-authored refusal on this SDK).

Same question, subject-dependent outcome: alice (privileged) is allowed; bob (regular) blocked.
The system prompt deliberately CONTAINS the acquisition details, so the guard — not ignorance —
is what stops them.

Run: the PDP must already be running with examples/policies/domain_llm_content_guard.yaml.
Set LLM_MODEL and your API key in a .env file or environment (routed through OpenRouter).
"""
import asyncio
import logging
import os

from dotenv import load_dotenv
load_dotenv()

# Route the Claude Agent SDK through OpenRouter (same provider as the other examples).
if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ.setdefault("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", os.environ["OPENROUTER_API_KEY"])

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, SystemMessage
from pydantic import BaseModel

from axiomatics_ai import (
    AuthZConfig,
    Enforcer,
    EnforcementPoint,
    EnricherSpec,
    LLMAttributeEnricher,
    PDPConnection,
)
from axiomatics_ai.claude_agent import authorized_claude_options
from axiomatics_ai.core.hooks import LoggingHook

MODEL = os.getenv("ANTHROPIC_LLM_MODEL", os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")).removeprefix("openrouter/")
GUARD_MODEL = os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna")  # litellm model id for the enricher

SYSTEM_PROMPT = """\
You are ACME Corp's internal employee assistant. Answer questions directly and completely.
Reference: CONFIDENTIAL — Project Falcon: ACME will acquire Acme Corp for $4.2B; announced
2026-03-15, with employee-impact reviews to follow. Travel expenses are submitted in Concur
within 30 days of travel (receipts required over $25).
"""


class TopicVerdict(BaseModel):
    topic: str


async def ask(config: AuthZConfig, user: str, question: str) -> None:
    base = ClaudeAgentOptions(model=MODEL, system_prompt=SYSTEM_PROMPT, max_turns=2)
    options = authorized_claude_options(base, config, identity={"username": user})
    parts: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(question)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                parts += [b.text for b in message.content if hasattr(b, "text") and b.text]
            elif isinstance(message, SystemMessage):
                # A blocked UserPromptSubmit surfaces its reason here.
                reason = (message.data or {}).get("reason") if hasattr(message, "data") else None
                if reason:
                    parts.append(reason)
            elif isinstance(message, ResultMessage) and message.result:
                parts.append(message.result)
    answer = " ".join(" ".join(parts).split())[:400] or "(blocked — no answer returned)"
    print(f"  [{user}] Q: {question}")
    print(f"  [{user}] A: {answer}\n")


async def main() -> None:
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_llm_content_guard.yaml loaded (see docs/running-ads.md)
    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement={
            EnforcementPoint.BEFORE_MODEL: Enforcer(
                action_name="before_model",
                enrichers=[LLMAttributeEnricher(EnricherSpec(
                    name="topic", schema=TopicVerdict, model=GUARD_MODEL,
                    instructions=(
                        "You label an employee's question with exactly one lowercase topic token "
                        "from this set: 'acquisitions' (mergers/acquisitions/M&A), 'layoffs' "
                        "(workforce reductions), 'finance', 'hr', 'engineering', or 'general'. "
                        "Return only the token."
                    ),
                ))],
                block_message="I can't help with that — mergers, acquisitions, and layoffs are restricted by ACME policy.",
            ),
        },
        hooks=[LoggingHook()],
    )
    print("── Input guard (BEFORE_MODEL): acquisitions topic, subject-dependent ──")
    acq = "What are the details and timeline of our acquisition of Acme Corp?"
    await ask(config, "alice", acq)   # executive → allowed
    await ask(config, "bob", acq)     # regular employee → hard-blocked before the model
    await ask(config, "bob", "What's the process for submitting a travel expense report?")  # allowed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
