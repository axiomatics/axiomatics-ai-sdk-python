"""
Example: LLM content guards on model I/O (BEFORE_MODEL + AFTER_MODEL).

An LLM attribute enricher (a PIP) classifies content into ADS attributes; ADS decides,
combining the classification with WHO is asking. Two points, each with its own enricher and a
custom refusal message:

  BEFORE_MODEL — the input guard labels the user prompt's `topic`. The policy
                 (domain_llm_content_guard.yaml) denies restricted business topics
                 (acquisitions, layoffs) — but exempts alice (an executive). A blocked prompt
                 never reaches the model.
  AFTER_MODEL  — the output guard flags *specific* PII (SSN / credit-card) in the final
                 answer. The policy denies those — but exempts alice (HR). A blocked answer is
                 replaced by the refusal.

So the same question resolves differently by subject: alice (privileged) is allowed; bob
(regular employee) is blocked. The enricher only classifies — the policy decides.

The agent's system prompt deliberately CONTAINS the answers to the restricted questions (the
acquisition details and an employee SSN), so you can see the guard is what stops them: the
model knows the answer and would reveal it, but the enricher + policy block it. Enrichment is
observable via the on_enrichment hook (LoggingHook here).

Run with ADS at http://localhost:8080 and domain YAML loaded:
  examples/policies/domain_llm_content_guard.yaml
Set LLM_MODEL and your API key in a .env file or environment variables.
"""
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from axiomatics_ai import (
    AuthZConfig,
    Enforcer,
    EnforcementPoint,
    EnricherSpec,
    LLMAttributeEnricher,
    LangChainAgentAuthZMiddleware,
    PDPConnection,
)
from axiomatics_ai.core.hooks import LoggingHook


# The assistant KNOWS the sensitive answers — the guards, not ignorance, are what block them.
SYSTEM_PROMPT = """\
You are ACME Corp's internal employee assistant. Answer questions directly and completely
using the reference information below.

Reference information:
- Travel expenses: submit expense reports in Concur within 30 days of travel. Receipts are
  required for anything over $25; your manager approves, then Finance reimburses on the next
  payroll cycle.
- CONFIDENTIAL — Project Falcon: ACME will acquire Acme Corp for $4.2B; the deal is announced
  on 2026-03-15, with employee-impact reviews to follow.
- Employee directory — John Doe: Engineering, title "Staff Engineer",
  email john.doe@acme.example, SSN 123-45-6789.
"""


# ── Enricher verdict schemas — their field names ARE the ADS attribute vocabulary ──

class TopicVerdict(BaseModel):
    """Input guard: one lowercase topic label for the user's question."""
    topic: str


class PiiVerdict(BaseModel):
    """Output guard: which specific PII types the answer contains."""
    contains_ssn: bool
    contains_credit_card: bool


def _input_guard(model: str) -> LLMAttributeEnricher:
    return LLMAttributeEnricher(EnricherSpec(
        name="topic",
        schema=TopicVerdict,
        instructions=(
            "You label an employee's question with exactly one lowercase topic token from this "
            "set: 'acquisitions' (mergers/acquisitions/M&A), 'layoffs' (workforce reductions/"
            "redundancies), 'finance', 'hr', 'engineering', or 'general'. Return only the token."
        ),
        model=model,
    ))


def _output_guard(model: str) -> LLMAttributeEnricher:
    return LLMAttributeEnricher(EnricherSpec(
        name="pii",
        schema=PiiVerdict,
        instructions=(
            "You inspect an assistant's answer for specific PII. Set contains_ssn=true only if "
            "it contains a US Social Security Number (e.g. 123-45-6789). Set "
            "contains_credit_card=true only if it contains a credit-card number. A plain name, "
            "job title, or email does NOT count."
        ),
        model=model,
    ))


def ask(agent, prompt, *, username):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context={"username": username, "session_id": f"s-{username}"},
    )
    print(f"  [{username}] Q: {prompt}")
    print(f"  [{username}] A: {result['messages'][-1].content}\n")


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_llm_content_guard.yaml loaded (see docs/running-ads.md)
    model = os.getenv("LLM_MODEL")

    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement={
            EnforcementPoint.BEFORE_MODEL: Enforcer(
                action_name="before_model",
                enrichers=[_input_guard(model)],
                block_message=(
                    "I can't help with that — questions about mergers, acquisitions, or "
                    "layoffs are restricted by ACME policy."
                ),
            ),
            EnforcementPoint.AFTER_MODEL: Enforcer(
                action_name="after_model",
                enrichers=[_output_guard(model)],
                block_message=(
                    "My response was withheld because it would have disclosed a Social Security "
                    "Number or other restricted personal information."
                ),
            ),
        },
        hooks=[LoggingHook()],  # observe every enrichment + decision
    )

    agent = create_agent(
        model=ChatLiteLLM(model=model),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # ── 1. Input guard (BEFORE_MODEL): restricted topic, subject-dependent ───────
    # The assistant KNOWS Project Falcon, but the 'acquisitions' topic is denied for bob
    # before the model is even called — while alice (executive) is allowed.
    print("── 1. Input guard (BEFORE_MODEL): acquisitions topic ────────────────")
    ask(agent, "What are the details and timeline of our acquisition of Acme Corp?", username="alice")
    ask(agent, "What are the details and timeline of our acquisition of Acme Corp?", username="bob")
    # Baseline: an allowed topic is answered for everyone.
    ask(agent, "What's the process for submitting a travel expense report?", username="bob")

    # ── 2. Output guard (AFTER_MODEL): specific PII (SSN), subject-dependent ──────
    # The model answers with the SSN it knows; the output guard flags it and blocks it for
    # bob, while alice (HR) receives it. A non-PII field is allowed for everyone.
    print("── 2. Output guard (AFTER_MODEL): SSN in the answer ─────────────────")
    ask(agent, "What is John Doe's Social Security Number?", username="alice")
    ask(agent, "What is John Doe's Social Security Number?", username="bob")
    ask(agent, "What is John Doe's job title?", username="bob")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
