"""
Example: implementing a custom ContextProvider.

The default provider reads username / user_id / tenant_id / session_id from the
context object. If your app uses different field names, implement the
ContextProvider protocol to map your context to XACML categories.

The first key returned by get_subject() is used as the primary subject
identifier for ADS MultiRequests and the decision cache key, so put the
most identifying attribute first.

Run with ADS at http://localhost:8080 and domain YAML loaded:
  examples/policies/domain_custom_context_provider.yaml
Set LLM_MODEL and your API key in a .env file or environment variables.
"""
from dataclasses import dataclass
from typing import Any
import os

from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain.tools import tool

from dotenv import load_dotenv
load_dotenv()

from axiomatics_ai import LangChainAgentAuthZMiddleware, AuthZConfig
from axiomatics_ai import PDPConnection, standard_tool_enforcement


# ── Custom context dataclass ──────────────────────────────────────────────────

@dataclass
class EmployeeContext:
    email: str        # primary identifier
    department: str
    clearance_level: str
    request_id: str   # used as session_id for cache scoping


# ── Custom provider mapping EmployeeContext to XACML categories ───────────────

class EmployeeContextProvider:
    def get_subject(self, ctx: EmployeeContext) -> dict[str, Any]:
        # First key (email) becomes the primary subject identifier.
        return {
            "email": ctx.email,
            "department": ctx.department,
            "clearance_level": ctx.clearance_level,
        }

    def get_action(self, action_name: str, ctx: EmployeeContext) -> dict[str, Any]:
        return {"name": action_name}

    def get_resource(
        self,
        resource_name: str,
        ctx: EmployeeContext,
        *,
        args: dict | None = None,
        resource_name_attribute: str | None = None,
    ) -> dict[str, Any]:
        # Arguments first, identifier last: a tool argument that happens to share the
        # identifier's name must never replace the tool name the policy sees.
        resource = dict(args or {})
        if resource_name_attribute:
            resource[resource_name_attribute] = resource_name
        return resource

    def get_environment(self, ctx: EmployeeContext) -> dict[str, Any]:
        return {"session_id": ctx.request_id}


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_docs(query: str) -> str:
    """Search the documentation."""
    return f"The docs for {query} will never be found"


@tool
def access_classified_report(report_id: str) -> str:
    """Access a classified internal report."""
    return f"Report {report_id}: Project Aurora — Q3 budget approved at $42M, variance -3%."


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_custom_context_provider.yaml loaded (see docs/running-ads.md)
    config = AuthZConfig(
        connection=PDPConnection(base_url=ads_url),
        enforcement=standard_tool_enforcement(),
        context_provider=EmployeeContextProvider(),
    )

    model = ChatLiteLLM(model=os.getenv("LLM_MODEL"))
    agent = create_agent(
        model=model,
        tools=[search_docs, access_classified_report],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # Alice has top-secret clearance, so both tools are visible and executable.
    print("=== alice (top-secret) ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Access the classified report R-42"}]},
        context=EmployeeContext(
            email="alice@example.com",
            department="engineering",
            clearance_level="top-secret",
            request_id="req-abc123",
        ),
    )
    print(result["messages"][-1].content)

    # Bob has standard clearance: access_classified_report is hidden, so the
    # model can only offer search_docs.
    print("\n=== bob (standard) ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Access the classified report R-42"}]},
        context=EmployeeContext(
            email="bob@example.com",
            department="engineering",
            clearance_level="standard",
            request_id="req-def456",
        ),
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
