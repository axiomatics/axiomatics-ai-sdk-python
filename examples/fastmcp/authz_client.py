"""
Example: LangChain agent connecting to the Axiomatics-protected FastMCP server.

The server (authz_server.py) enforces authorization via middleware — the client
is completely unaware of it. All filtering happens transparently on the server
side based on the JWT in the Authorization header.

alice → allowed all tools (search_docs, get_account_balance, send_email)
bob   → allowed search_docs only (per domain_mcp_tool_visibility.yaml)

Prerequisites:
  - authz_server.py running: uv run python examples/fastmcp/authz_server.py
  - ADS running at http://localhost:8080 with domain_mcp_tool_visibility.yaml loaded
  - .env file (or env vars) with LLM_MODEL and the appropriate API key

Run:
    uv run python examples/fastmcp/authz_client.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import jwt
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

load_dotenv()

_LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")
_LLM_API_BASE = os.getenv("LLM_API_BASE") or None
_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# JWTs are unsigned here — the server decodes without signature verification.
# In production, sign them with your IdP's private key.
_ALICE_JWT = jwt.encode({"username": "alice", "sub": "alice"}, "demo-secret", algorithm="HS256")  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring
_BOB_JWT = jwt.encode({"username": "bob", "sub": "bob"}, "demo-secret", algorithm="HS256")  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring


async def run_for_user(username: str, jwt_token: str, question: str) -> None:
    print(f"\n── {username} ──────────────────────────────────────────")
    client = MultiServerMCPClient(
        {
            "axio-fastmcp": {
                "transport": "streamable_http",
                "url": _SERVER_URL,
                "headers": {"Authorization": f"Bearer {jwt_token}"},
            }
        }
    )
    tools = await client.get_tools()
    print(f"  permitted tools: {[t.name for t in tools]}")

    kwargs: dict = {"model": _LLM_MODEL}
    if _LLM_API_BASE:
        kwargs["api_base"] = _LLM_API_BASE

    agent = create_agent(model=ChatLiteLLM(**kwargs), tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print(f"  answer: {result['messages'][-1].content}")


async def main() -> None:
    question = "What tools do you have available? Briefly list them."
    await run_for_user("alice", _ALICE_JWT, question)
    await run_for_user("bob", _BOB_JWT, question)


if __name__ == "__main__":
    asyncio.run(main())
