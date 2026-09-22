"""
Example: explicit MCP tool-visibility filtering via `authorized_mcp_tools` (no monkeypatching).

Load MCP tools with the adapters as usual, then filter the list per user through ADS in one
call:

    tools = await client.get_tools()
    tools = authorized_mcp_tools(tools, config, context={"username": "alice"})
    agent = create_agent(model, tools)   # the agent only ever sees the permitted tools

This path is **tool-visibility only** (invocation / post-invocation aren't enforced client-side
here); the tools are filtered before the agent is built, so build one agent per user.

Prerequisites: ADS running at PDP_BASE_URL with a tool-visibility domain loaded. Set LLM_MODEL and
your API key in a .env file or environment.
"""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient

from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.langchain_mcp import authorized_mcp_tools

_PDP_BASE_URL = os.getenv("PDP_BASE_URL", "http://localhost:8080")
_CONFIG = AuthZConfig(
    connection=PDPConnection(base_url=_PDP_BASE_URL),
    enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False),
)
_MCP = {"docs-server": {
    "transport": "stdio",
    "command": sys.executable,
    "args": [str(Path(__file__).with_name("mcp_server.py"))],
}}


async def run_for_user(username: str, question: str) -> None:
    client = MultiServerMCPClient(_MCP)

    # Load, then filter to the tools this user may see — the one explicit call.
    tools = authorized_mcp_tools(await client.get_tools(), _CONFIG, context={"username": username})
    print(f"── {username}: permitted tools = {[t.name for t in tools]}")

    agent = create_agent(model=ChatLiteLLM(model=os.getenv("LLM_MODEL")), tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print(f"   {result['messages'][-1].content}\n")


async def main() -> None:
    q = "What tools do you have? List them, then use one."
    await run_for_user("alice", q)
    await run_for_user("bob", q)


if __name__ == "__main__":
    asyncio.run(main())
