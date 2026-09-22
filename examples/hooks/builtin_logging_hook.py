"""
Example: the built-in LoggingHook — "log every ADS call and every decision".

Attaching ``LoggingHook`` to ``AuthZConfig(hooks=[...])`` makes the SDK log, for every
path, two levels of events:

  - transport: on_pdp_request / on_pdp_response (one per ADS wire call — with latency,
    status, and errors)
  - decision:  on_decision (the semantic outcome at each enforcement point, tagged with
    the ``surface`` it came from and a ``cached`` flag)

LoggingHook emits through the standard ``logging`` module (logger name
"axiomatics_ai.hooks", level INFO), so you just configure logging to see it. Hooks are
observe-only — they never change a decision — and add no overhead when the list is empty.

Prerequisites:
  - A running ADS loaded with examples/policies/domain_all_authz_enabled.yaml
    (set PDP_BASE_URL, or it defaults to http://localhost:8080).
  - LLM_MODEL and your API key in a .env file or environment variables.
"""
import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain.tools import tool

from axiomatics_ai import LangChainAgentAuthZMiddleware, AuthZConfig, LoggingHook
from axiomatics_ai import PDPConnection, standard_tool_enforcement

# Show the hook output. LoggingHook logs at INFO under "axiomatics_ai.hooks".
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


@tool
def search_docs(query: str) -> str:
    """Search the documentation and return matching articles."""
    return json.dumps([
        {"id": "d1", "title": "Getting Started", "doc_type": "public"},
        {"id": "d2", "title": "Internal Spec",   "doc_type": "classified"},
    ])


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}"


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")

    # The only change needed to get full observability: hooks=[LoggingHook()].
    config = AuthZConfig(connection=PDPConnection(base_url=ads_url), enforcement=standard_tool_enforcement(), hooks=[LoggingHook()])

    model = ChatLiteLLM(model=os.getenv("LLM_MODEL"))
    agent = create_agent(
        model=model,
        tools=[search_docs, send_email],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # Watch the logs: each turn emits tool_visibility, then tool_invocation, then
    # post_invocation decisions, plus the ADS transport calls behind them. bob's
    # send_email is hidden (visibility) and his classified record is filtered
    # (post-invocation) — both visible in the decision events.
    print("── alice ───────────────────────────────────────────────────────────")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Search the docs and show everything."}]},
        context={"username": "alice", "session_id": "s1"},
    )
    print("\n── bob ─────────────────────────────────────────────────────────────")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Search the docs and show everything."}]},
        context={"username": "bob", "session_id": "s2"},
    )


if __name__ == "__main__":
    main()
