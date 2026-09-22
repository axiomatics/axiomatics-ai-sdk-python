"""
Example: LLM content guards on model I/O (BEFORE_MODEL + AFTER_MODEL) — OpenAI Agents SDK.

The OpenAI counterpart of examples/langchain_agents/llm_content_guard.py, on the SAME policy
(domain_llm_content_guard.yaml). An LLM attribute enricher (a PIP) classifies content into ADS
attributes; ADS decides, combining the classification with WHO is asking.

Wired as SDK **guardrails**: BEFORE_MODEL is an input guardrail on the user prompt, AFTER_MODEL
an output guardrail on the final answer. Unlike LangChain/ADK/Strands, an OpenAI guardrail can
only *tripwire* — on a block the Runner RAISES ``InputGuardrailTripwireTriggered`` /
``OutputGuardrailTripwireTriggered`` (the refusal is in ``output_info["reason"]``). This example
catches those and surfaces the refusal.

  BEFORE_MODEL — the input guard labels the prompt's `topic`; the policy denies restricted
                 topics (acquisitions, layoffs) for regular employees but exempts alice.
  AFTER_MODEL  — the output guard flags specific PII (SSN); the policy denies it but exempts
                 alice (HR).

Same question, subject-dependent: alice (privileged) allowed; bob (regular) blocked. The
instructions deliberately CONTAIN the sensitive answers, so the guard — not ignorance — blocks.

Run: the PDP must already be running with examples/policies/domain_llm_content_guard.yaml.
Set LLM_MODEL and your API key in a .env file or environment (routed through OpenRouter).
"""
import asyncio
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from agents.tracing import set_tracing_disabled
from pydantic import BaseModel

set_tracing_disabled(True)

from axiomatics_ai import (
    AuthZConfig,
    Enforcer,
    EnforcementPoint,
    EnricherSpec,
    LLMAttributeEnricher,
    PDPConnection,
)
from axiomatics_ai.core.hooks import LoggingHook
from axiomatics_ai.openai_agents import authorized_openai_agent

MODEL = os.getenv("OPENAI_LLM_MODEL", os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")).removeprefix("openrouter/")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
GUARD_MODEL = os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna")  # litellm model id for the enrichers

INSTRUCTIONS = """\
You are ACME Corp's internal employee assistant. Answer questions directly and completely
using the reference information below.

Reference information:
- Travel expenses: submit in Concur within 30 days of travel; receipts required over $25.
- CONFIDENTIAL — Project Falcon: ACME will acquire Acme Corp for $4.2B; announced 2026-03-15.
- Employee directory — John Doe: Engineering, "Staff Engineer", SSN 123-45-6789.
"""


class TopicVerdict(BaseModel):
    topic: str


class PiiVerdict(BaseModel):
    contains_ssn: bool
    contains_credit_card: bool


async def ask(agent, user: str, question: str) -> None:
    try:
        result = await Runner.run(agent, question, context={"username": user})
        answer = " ".join(str(result.final_output).split())[:400]
    except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as exc:
        answer = exc.guardrail_result.output.output_info.get("reason") or "(blocked by content guard)"
    print(f"  [{user}] Q: {question}")
    print(f"  [{user}] A: {answer}\n")


async def main() -> None:
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_llm_content_guard.yaml loaded (see docs/running-ads.md)
    model = OpenAIChatCompletionsModel(model=MODEL, openai_client=AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY))
    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement={
            EnforcementPoint.BEFORE_MODEL: Enforcer(
                action_name="before_model",
                enrichers=[LLMAttributeEnricher(EnricherSpec(
                    name="topic", schema=TopicVerdict, model=GUARD_MODEL,
                    instructions="Label the employee's question with exactly one lowercase topic token: 'acquisitions', 'layoffs', 'finance', 'hr', 'engineering', or 'general'. Return only the token.",
                ))],
                block_message="I can't help with that — mergers, acquisitions, and layoffs are restricted by ACME policy.",
            ),
            EnforcementPoint.AFTER_MODEL: Enforcer(
                action_name="after_model",
                enrichers=[LLMAttributeEnricher(EnricherSpec(
                    name="pii", schema=PiiVerdict, model=GUARD_MODEL,
                    instructions="Set contains_ssn=true only if the answer contains a US Social Security Number; contains_credit_card=true only for a credit-card number. A name/title/email does NOT count.",
                ))],
                block_message="My response was withheld because it would have disclosed a Social Security Number.",
            ),
        },
        hooks=[LoggingHook()],
    )
    agent = authorized_openai_agent(Agent(name="content_guard_agent", instructions=INSTRUCTIONS, model=model), config)

    print("── 1. Input guard (BEFORE_MODEL): acquisitions topic ────────────────")
    acq = "What are the details and timeline of our acquisition of Acme Corp?"
    await ask(agent, "alice", acq)
    await ask(agent, "bob", acq)
    await ask(agent, "bob", "What's the process for submitting a travel expense report?")

    print("── 2. Output guard (AFTER_MODEL): SSN in the answer ─────────────────")
    ssn = "What is John Doe's Social Security Number?"
    await ask(agent, "alice", ssn)
    await ask(agent, "bob", ssn)
    await ask(agent, "bob", "What is John Doe's job title?")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
