"""
Example: LLM content guards on model I/O (BEFORE_MODEL + AFTER_MODEL) — Google ADK.

The ADK counterpart of examples/langchain_agents/llm_content_guard.py, on the SAME policy
(domain_llm_content_guard.yaml). An LLM attribute enricher (a PIP) classifies content into ADS
attributes; ADS decides, combining the classification with WHO is asking:

  BEFORE_MODEL (before_model_callback) — the input guard labels the prompt's `topic`; the
                 policy denies restricted topics (acquisitions, layoffs) for regular employees
                 but exempts alice. A blocked prompt returns a refusal LlmResponse — the model
                 is never called.
  AFTER_MODEL (after_model_callback)   — the output guard flags specific PII (SSN/credit-card)
                 in the answer; the policy denies those but exempts alice (HR). A blocked
                 answer is replaced by a refusal LlmResponse.

Same question, subject-dependent outcome: alice (privileged) is allowed; bob (regular) blocked.
The agent's instruction deliberately CONTAINS the sensitive answers, so the guard — not
ignorance — is what stops them. Enrichment is observable via the on_enrichment hook.

Run: the PDP must already be running with examples/policies/domain_llm_content_guard.yaml.
Set LLM_MODEL (or GOOGLE_ADK_LLM_MODEL) and your API key in a .env file or environment.
"""
import asyncio
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from axiomatics_ai import (
    AuthZConfig,
    Enforcer,
    EnforcementPoint,
    EnricherSpec,
    LLMAttributeEnricher,
    PDPConnection,
)
from axiomatics_ai.core.hooks import LoggingHook
from axiomatics_ai.google_adk import authorized_adk_agent

APP = "adk-content-guard-demo"
MODEL = os.getenv("GOOGLE_ADK_LLM_MODEL", os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna"))
GUARD_MODEL = os.getenv("LLM_MODEL", MODEL)  # litellm model id for the enrichers


# The agent KNOWS the sensitive answers — the guards, not ignorance, are what block them.
INSTRUCTION = """\
You are ACME Corp's internal employee assistant. Answer questions directly and completely
using the reference information below.

Reference information:
- Travel expenses: submit expense reports in Concur within 30 days of travel. Receipts are
  required for anything over $25; your manager approves, then Finance reimburses.
- CONFIDENTIAL — Project Falcon: ACME will acquire Acme Corp for $4.2B; announced 2026-03-15,
  with employee-impact reviews to follow.
- Employee directory — John Doe: Engineering, title "Staff Engineer",
  email john.doe@acme.example, SSN 123-45-6789.
"""


class TopicVerdict(BaseModel):
    topic: str


class PiiVerdict(BaseModel):
    contains_ssn: bool
    contains_credit_card: bool


def _input_guard() -> LLMAttributeEnricher:
    return LLMAttributeEnricher(EnricherSpec(
        name="topic", schema=TopicVerdict, model=GUARD_MODEL,
        instructions=(
            "You label an employee's question with exactly one lowercase topic token from this "
            "set: 'acquisitions' (mergers/acquisitions/M&A), 'layoffs' (workforce reductions), "
            "'finance', 'hr', 'engineering', or 'general'. Return only the token."
        ),
    ))


def _output_guard() -> LLMAttributeEnricher:
    return LLMAttributeEnricher(EnricherSpec(
        name="pii", schema=PiiVerdict, model=GUARD_MODEL,
        instructions=(
            "You inspect an assistant's answer for specific PII. Set contains_ssn=true only if "
            "it contains a US Social Security Number (e.g. 123-45-6789). Set "
            "contains_credit_card=true only if it contains a credit-card number. A plain name, "
            "job title, or email does NOT count."
        ),
    ))


def build_agent(ads_url: str):
    agent = Agent(model=LiteLlm(model=MODEL), name="content_guard_agent", instruction=INSTRUCTION)
    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement={
            EnforcementPoint.BEFORE_MODEL: Enforcer(
                action_name="before_model", enrichers=[_input_guard()],
                block_message="I can't help with that — mergers, acquisitions, and layoffs are restricted by ACME policy.",
            ),
            EnforcementPoint.AFTER_MODEL: Enforcer(
                action_name="after_model", enrichers=[_output_guard()],
                block_message="My response was withheld because it would have disclosed a Social Security Number.",
            ),
        },
        hooks=[LoggingHook()],
    )
    return authorized_adk_agent(agent, config)


async def ask(runner, user: str, session: str, question: str) -> str:
    await runner.session_service.create_session(
        app_name=APP, user_id=user, session_id=session, state={"username": user},
    )
    message = types.Content(role="user", parts=[types.Part(text=question)])
    final = ""
    async for event in runner.run_async(user_id=user, session_id=session, new_message=message):
        if event.is_final_response() and event.content:
            final = event.content.parts[0].text
    print(f"  [{user}] Q: {question}")
    print(f"  [{user}] A: {' '.join(final.split())[:400]}\n")


async def main() -> None:
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_llm_content_guard.yaml loaded (see docs/running-ads.md)
    runner = Runner(agent=build_agent(ads_url), app_name=APP, session_service=InMemorySessionService())

    # 1. Input guard — restricted topic, subject-dependent (alice allowed, bob blocked).
    print("── 1. Input guard (BEFORE_MODEL): acquisitions topic ────────────────")
    acq = "What are the details and timeline of our acquisition of Acme Corp?"
    await ask(runner, "alice", "a-acq", acq)
    await ask(runner, "bob", "b-acq", acq)
    await ask(runner, "bob", "b-exp", "What's the process for submitting a travel expense report?")

    # 2. Output guard — specific PII (SSN), subject-dependent (alice allowed, bob blocked).
    print("── 2. Output guard (AFTER_MODEL): SSN in the answer ─────────────────")
    ssn = "What is John Doe's Social Security Number?"
    await ask(runner, "alice", "a-ssn", ssn)
    await ask(runner, "bob", "b-ssn", ssn)
    await ask(runner, "bob", "b-title", "What is John Doe's job title?")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
