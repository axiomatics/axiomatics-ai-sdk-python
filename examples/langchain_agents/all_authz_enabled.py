"""
Example: all three enforcement features enabled simultaneously.

  TOOL_VISIBILITY — filter the tool list before each model call
  BEFORE_TOOL     — authorize each tool call before execution
  AFTER_TOOL      — filter records in JSON tool responses

This is the most restrictive configuration. Every model turn, every tool call,
and every record returned by a tool is checked against ADS policy.

Run with ADS at http://localhost:8080 and domain YAML loaded:
  examples/policies/domain_all_authz_enabled.yaml
Set LLM_MODEL and your API key in a .env file or environment variables.
"""
import json
import os

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain.tools import tool

from axiomatics_ai import LangChainAgentAuthZMiddleware, AuthZConfig
from axiomatics_ai import PDPConnection, standard_tool_enforcement


@tool
def search_docs(query: str) -> str:
    """Search the documentation and return matching articles."""
    return json.dumps([
        {"id": "d1", "title": "Getting Started", "doc_type": "public"},
        {"id": "d2", "title": "Internal Spec",   "doc_type": "classified"},
    ])


@tool
def get_account_balance(account_id: str) -> str:
    """Get the balance for an account."""
    return f"Balance for {account_id}: $1,000"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}"


def invoke(agent, prompt, *, username, session_id):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context={"username": username, "session_id": session_id},
    )
    print(result["messages"][-1].content)


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_all_authz_enabled.yaml loaded (see docs/running-ads.md)
    config = AuthZConfig(connection=PDPConnection(base_url=ads_url),
        enforcement=standard_tool_enforcement(),
    )

    model = ChatLiteLLM(model=os.getenv("LLM_MODEL"))
    agent = create_agent(
        model=model,
        tools=[search_docs, get_account_balance, send_email],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # ── 1. Tool visibility ─────────────────────────────────────────────────────
    # Alice is permitted to see all three tools.
    # Bob is only permitted to see search_docs and get_account_balance;
    # send_email is hidden from the model before the first LLM call.
    print("── 1. Tool visibility ──────────────────────────────────────────────")
    print("alice:")
    invoke(agent, "List every tool you have available right now.", username="alice", session_id="s1a")
    print("\nbob:")
    invoke(agent, "List every tool you have available right now.", username="bob", session_id="s1b")

    # ── 2. Tool invocation ─────────────────────────────────────────────────────
    # Alice is permitted to call get_account_balance for any account.
    # Bob is only permitted to call it for ACC-001; the call for ACC-002 is
    # intercepted by ADS before the tool executes and returns an access-denied
    # message to the model.
    print("\n── 2. Tool invocation ──────────────────────────────────────────────")
    print("alice (ACC-002):")
    invoke(agent, "Get the balance for account ACC-002.", username="alice", session_id="s2a")
    print("\nbob (ACC-002):")
    invoke(agent, "Get the balance for account ACC-002.", username="bob", session_id="s2b")

    # ── 3. Record filtering (post-invocation) ──────────────────────────────────
    # Both users can call search_docs. After the tool runs, ADS evaluates each
    # record individually. Alice receives all records. Bob's classified record is
    # removed from the response before the model ever sees it.
    print("\n── 3. Record filtering (post-invocation) ───────────────────────────")
    print("alice:")
    invoke(agent, "Search the docs and show me everything you find.", username="alice", session_id="s3a")
    print("\nbob:")
    invoke(agent, "Search the docs and show me everything you find.", username="bob", session_id="s3b")



if __name__ == "__main__":
    main()
