"""
Example: LangChain agent connecting to the authorization-filtered RAG FastMCP server
(rag_server.py).

The agent is completely unaware of authorization. It connects with a JWT in the
``Authorization`` header and is given the server's ``search_documentation`` tool; when it
decides to call that tool, the server binds the caller's identity into the ambient authz
context, so the wrapped vector store returns only the documents ADS permits for that user.
Same agent, same question, different retrieved documents:

  alice (engineering, level 2) → engineering L1/L2 + public L1   (not engineering L3, not finance)
  bob   (finance, level 3)     → finance L1/L3 + public L1

Prerequisites:
  - rag_server.py running: uv run python examples/fastmcp/rag_server.py
  - ADS running at http://localhost:8080 with domain_rag_retrieval.yaml loaded
  - .env file (or env vars) with LLM_MODEL and the appropriate API key

Run:
    uv run python examples/fastmcp/rag_client.py
"""
from __future__ import annotations

import asyncio
import os

import jwt
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

_LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")
_LLM_API_BASE = os.getenv("LLM_API_BASE") or None
_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8002/mcp")

# JWTs are unsigned here — the server decodes without signature verification.
# In production, sign them with your IdP's private key.
_ALICE_JWT = jwt.encode(  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring
    {"username": "alice", "department": "engineering", "level": 2}, "demo-secret", algorithm="HS256"
)
_BOB_JWT = jwt.encode(  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring
    {"username": "bob", "department": "finance", "level": 3}, "demo-secret", algorithm="HS256"
)


async def run_for_user(username: str, jwt_token: str, question: str) -> None:
    print(f"\n── {username} ──────────────────────────────────────────")
    client = MultiServerMCPClient(
        {
            "axio-rag": {
                "transport": "streamable_http",
                "url": _SERVER_URL,
                "headers": {"Authorization": f"Bearer {jwt_token}"},
            }
        }
    )
    tools = await client.get_tools()  # includes search_documentation, filtered server-side

    kwargs: dict = {"model": _LLM_MODEL}
    if _LLM_API_BASE:
        kwargs["api_base"] = _LLM_API_BASE

    agent = create_agent(model=ChatLiteLLM(**kwargs), tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print(f"  answer: {result['messages'][-1].content}")


async def main() -> None:
    question = (
        "Search the internal documentation for anything you can find and list the exact "
        "titles of the documents you retrieved."
    )
    await run_for_user("alice", _ALICE_JWT, question)
    await run_for_user("bob", _BOB_JWT, question)


if __name__ == "__main__":
    asyncio.run(main())
