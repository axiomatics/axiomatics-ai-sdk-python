"""
Example: PDP-driven tool filtering for a custom LangGraph graph with MCP tools.

Tools are loaded with the MCP adapters as usual, then filtered per user with
``authorized_mcp_tools(tools, config, context=...)`` — one batched PDP request — before
being bound to the model. The LangGraph graph itself is unchanged.

The example builds a graph for alice (all tools) and bob (search_docs only)
and sends the same question to both, showing how policy shapes what each user
can do.

Prerequisites:
  - PDP running at PDP_BASE_URL (default http://localhost:8080) with domain_mcp_tool_visibility.yaml loaded
  - .env file (or env vars) with LLM_MODEL and the appropriate API key

Run:
    uv run python examples/langchain_mcp/tool_filtering_langgraph.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.tools import load_mcp_tools
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

_MCP_CONNECTION = {
    "transport": "stdio",
    "command": sys.executable,
    "args": [str(Path(__file__).with_name("mcp_server.py"))],
}


async def build_graph_for_user(user_context: dict):
    """
    Build a LangGraph graph whose tool list is already filtered for this user.
    Call once per user session; invoke the returned graph as many times as needed.
    """
    all_tools = await load_mcp_tools(None, connection=_MCP_CONNECTION)
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
    print(f"\n── {username} ──────────────────────────────────────────")
    ctx = {"username": username, "session_id": session_id}
    graph = await build_graph_for_user(ctx)
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})
    print(f"  answer: {result['messages'][-1].content}")


async def main():
    question = "What tools do you have available? Briefly list them."

    await run_for_user("alice", "session-alice-001", question)
    await run_for_user("bob",   "session-bob-001",   question)


if __name__ == "__main__":
    asyncio.run(main())
