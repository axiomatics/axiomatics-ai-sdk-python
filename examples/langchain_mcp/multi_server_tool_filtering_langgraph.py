"""
Example: PDP-driven tool filtering for LangGraph with multiple MCP servers.

``MultiServerMCPClient.get_tools()`` loads the combined tool list from all configured MCP
servers; ``authorized_mcp_tools(tools, config, context=...)`` then filters it per user with
one batched PDP request before it is bound to the model.

Prerequisites:
  - PDP running at PDP_BASE_URL (default http://localhost:8080) with domain_mcp_tool_visibility.yaml loaded
  - .env file (or env vars) with LLM_MODEL and the appropriate API key

Run:
    uv run python examples/langchain_mcp/multi_server_tool_filtering_langgraph.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.langchain_mcp import authorized_mcp_tools

load_dotenv()

_LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")
_LLM_API_BASE = os.getenv("LLM_API_BASE") or None

_CONFIG = AuthZConfig(
    connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
    enforcement=standard_tool_enforcement(),
)

_HERE = Path(__file__).parent
_MCP_CLIENT = MultiServerMCPClient(
    {
        "docs": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(_HERE / "mcp_docs_server.py")],
        },
        "accounts": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(_HERE / "mcp_account_server.py")],
        },
    }
)


async def build_graph_for_user(user_context: dict):
    """
    Build a LangGraph graph whose multi-server tool list is filtered for this user.
    """
    all_tools = await _MCP_CLIENT.get_tools()
    tools = authorized_mcp_tools(all_tools, _CONFIG, context=user_context)
    print(f"  permitted tools: {[t.name for t in tools]}")

    kwargs: dict = {"model": _LLM_MODEL}
    if _LLM_API_BASE:
        kwargs["api_base"] = _LLM_API_BASE
    model = ChatLiteLLM(**kwargs)
    model_with_tools = model.bind_tools(tools)

    def model_node(state: MessagesState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", model_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue)
    graph.add_edge("tools", "model")
    return graph.compile()


async def run_for_user(username: str, session_id: str, question: str):
    print(f"\n-- {username} ------------------------------------------")
    ctx = {"username": username, "session_id": session_id}
    graph = await build_graph_for_user(ctx)
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})
    print(f"  answer: {result['messages'][-1].content}")


async def main():
    question = "What tools do you have available? Briefly list them."

    await run_for_user("alice", "session-alice-001", question)
    await run_for_user("bob", "session-bob-001", question)


if __name__ == "__main__":
    asyncio.run(main())
