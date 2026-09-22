"""
Example: LLM content guards on model I/O (BEFORE_MODEL + AFTER_MODEL) — Strands.

The Strands counterpart of examples/langchain_agents/llm_content_guard.py, on the SAME policy
(domain_llm_content_guard.yaml). An LLM attribute enricher (a PIP) classifies content into ADS
attributes; ADS decides, combining the classification with WHO is asking:

  BEFORE_MODEL (before_model_call) — the input guard labels the prompt's `topic`; the policy
                 denies restricted topics (acquisitions, layoffs) for regular employees but
                 exempts alice. A block returns Deny(reason) — Strands cancels the model call
                 and surfaces the reason as the answer.
  AFTER_MODEL (after_model_call)   — the output guard flags specific PII (SSN/credit-card) in
                 the answer; the policy denies those but exempts alice (HR). A block returns a
                 Transform that rewrites the answer with the refusal.

Same question, subject-dependent outcome: alice (privileged) allowed; bob (regular) blocked.
The agent's system prompt deliberately CONTAINS the sensitive answers, so the guard — not
ignorance — is what stops them. Enrichment is observable via the on_enrichment hook.

Run: the PDP must already be running with examples/policies/domain_llm_content_guard.yaml.
Set LLM_MODEL and your API key in a .env file or environment.
"""
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from strands import Agent
from strands.models.litellm import LiteLLMModel

from axiomatics_ai import (
    AuthZConfig,
    Enforcer,
    EnforcementPoint,
    EnricherSpec,
    LLMAttributeEnricher,
    PDPConnection,
)
from axiomatics_ai.core.hooks import LoggingHook
from axiomatics_ai.strands import authorized_strands_agent

MODEL = os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna")


SYSTEM_PROMPT = """\
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
        name="topic", schema=TopicVerdict, model=MODEL,
        instructions=(
            "You label an employee's question with exactly one lowercase topic token from this "
            "set: 'acquisitions' (mergers/acquisitions/M&A), 'layoffs' (workforce reductions), "
            "'finance', 'hr', 'engineering', or 'general'. Return only the token."
        ),
    ))


def _output_guard() -> LLMAttributeEnricher:
    return LLMAttributeEnricher(EnricherSpec(
        name="pii", schema=PiiVerdict, model=MODEL,
        instructions=(
            "You inspect an assistant's answer for specific PII. Set contains_ssn=true only if "
            "it contains a US Social Security Number (e.g. 123-45-6789). Set "
            "contains_credit_card=true only if it contains a credit-card number. A plain name, "
            "job title, or email does NOT count."
        ),
    ))


def _config(ads_url: str) -> AuthZConfig:
    return AuthZConfig(
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


def ask(config: AuthZConfig, user: str, question: str) -> None:
    # Fresh agent per turn so identity (bound in state) and history stay clean.
    agent = Agent(model=LiteLLMModel(model_id=MODEL), system_prompt=SYSTEM_PROMPT, state={"username": user})
    authorized_strands_agent(agent, config)
    answer = " ".join(str(agent(question)).split())[:400]
    print(f"  [{user}] Q: {question}")
    print(f"  [{user}] A: {answer}\n")


def main() -> None:
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_llm_content_guard.yaml loaded (see docs/running-ads.md)
    config = _config(ads_url)

    print("── 1. Input guard (BEFORE_MODEL): acquisitions topic ────────────────")
    acq = "What are the details and timeline of our acquisition of Acme Corp?"
    ask(config, "alice", acq)
    ask(config, "bob", acq)
    ask(config, "bob", "What's the process for submitting a travel expense report?")

    print("── 2. Output guard (AFTER_MODEL): SSN in the answer ─────────────────")
    ssn = "What is John Doe's Social Security Number?"
    ask(config, "alice", ssn)
    ask(config, "bob", ssn)
    ask(config, "bob", "What is John Doe's job title?")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
