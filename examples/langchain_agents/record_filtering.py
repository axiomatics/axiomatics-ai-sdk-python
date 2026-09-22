"""
Example: filtering records returned by tools via an AFTER_TOOL Enforcer.

After a tool executes and returns a JSON array, the middleware sends a XACML
MultiRequest to ADS — one sub-request per record. Records whose sub-request
receives a Deny are removed from the array before the model sees the result.

Tools that return non-JSON or a JSON object (not an array) are passed through
unchanged — record filtering only applies to JSON arrays.

Run with ADS at http://localhost:8080 and domain YAML loaded:
  examples/policies/domain_post_invocation.yaml
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
def list_customer_records(query: str) -> str:
    """Return a JSON list of customer records."""
    records = [
        {"id": "r1", "name": "Alice Johnson", "doc_type": "profile"},
        {"id": "r2", "name": "Bob Smith",     "doc_type": "classified"},
        {"id": "r3", "name": "Carol Davis",   "doc_type": "profile"},
    ]
    return json.dumps(records)


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_post_invocation.yaml loaded (see docs/running-ads.md)
    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement=standard_tool_enforcement(invocation=False),
    )

    model = ChatLiteLLM(model=os.getenv("LLM_MODEL"))
    agent = create_agent(
        model=model,
        tools=[list_customer_records],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # Alice is permitted to read all records, including classified ones.
    print("=== alice ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "List all customer records"}]},
        context={"username": "alice", "session_id": "session-001"},
    )
    print(result["messages"][-1].content)

    # Bob's policy denies records with doc_type=classified.
    # The model receives only the two non-classified records.
    print("\n=== bob ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "List all customer records"}]},
        context={"username": "bob", "session_id": "session-002"},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
