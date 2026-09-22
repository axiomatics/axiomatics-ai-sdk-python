"""
Example: passing LangChainAgentAuthZMiddleware explicitly to create_agent.

Use this pattern when you want fine-grained control over which agents get the
middleware, or when you prefer not to use the global activate() patch.

Run with ADS at http://localhost:8080 and domain YAML loaded:
  examples/policies/domain_tool_visibility_only.yaml
Set LLM_MODEL and your API key in a .env file or environment variables.
"""
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
    """Search the documentation."""
    return f"Results for: {query}"


@tool
def get_account_balance(account_id: str) -> str:
    """Get the balance for an account."""
    return f"Balance for {account_id}: $1,000"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}"


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_tool_visibility_only.yaml loaded (see docs/running-ads.md)
    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
    )

    model = ChatLiteLLM(model=os.getenv("LLM_MODEL"))
    agent = create_agent(
        model=model,
        tools=[search_docs, get_account_balance, send_email],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # Alice is permitted to see all tools, so the agent lists all three.
    print("=== alice ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "List every tool you have available to you right now."}]},
        context={"username": "alice", "session_id": "session-001"},
    )
    print(result["messages"][-1].content)

    # Bob is only permitted to see search_docs, so get_account_balance and
    # send_email are hidden from the model entirely.
    print("\n=== bob ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "List every tool you have available to you right now."}]},
        context={"username": "bob", "session_id": "session-002"},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
