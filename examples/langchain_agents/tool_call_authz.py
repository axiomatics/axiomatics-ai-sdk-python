"""
Example: authorizing individual tool calls via a BEFORE_TOOL Enforcer.

With this flag on, the middleware sends a single XACML request to ADS before
each tool execution. The request includes the tool name and the tool's arguments,
so your policy can make decisions based on specific argument values (e.g. only
allow get_account_balance for account_id=ACC-001).

If ADS denies the call, the tool never executes and the model receives an
"Access denied" ToolMessage instead of the real tool output.

Run with ADS at http://localhost:8080 and domain YAML loaded:
  examples/policies/domain_tool_invocation.yaml
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
def get_account_balance(account_id: str) -> str:
    """Get the balance for an account."""
    return f"Balance for {account_id}: $1,000"


@tool
def search_docs(query: str) -> str:
    """Search the documentation."""
    return f"Results for: {query}"


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_tool_invocation.yaml loaded (see docs/running-ads.md)
    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement=standard_tool_enforcement(post_invocation=False),
    )

    model = ChatLiteLLM(model=os.getenv("LLM_MODEL"))
    agent = create_agent(
        model=model,
        tools=[get_account_balance, search_docs],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # Alice is permitted to call get_account_balance for any account.
    print("=== alice (ACC-002) ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Get the balance for ACC-002"}]},
        context={"username": "alice", "session_id": "session-001"},
    )
    print(result["messages"][-1].content)

    # Bob is only permitted to call get_account_balance for ACC-001.
    # The call for ACC-002 is denied before the tool executes.
    print("\n=== bob (ACC-002) ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Get the balance for ACC-002"}]},
        context={"username": "bob", "session_id": "session-002"},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
