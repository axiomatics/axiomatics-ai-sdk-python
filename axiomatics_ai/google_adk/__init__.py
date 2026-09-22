"""Client-side PDP authorization for Google ADK agents.

    from axiomatics_ai import AuthZConfig
    from axiomatics_ai.google_adk import authorized_adk_agent

    agent = authorized_adk_agent(
        Agent(model=LiteLlm(...), name="ads_agent", tools=[mcp_toolset]),
        AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement()),
    )
    # identity via session state (bind once) or run_async(state_delta=...) per invocation

Enforces tool visibility, tool invocation, and post-invocation via ADK callbacks — no
wrapping of the agent or its toolsets. See authz.py and README.md.
"""
from axiomatics_ai.google_adk.authz import authorized_adk_agent

__all__ = ["authorized_adk_agent"]
