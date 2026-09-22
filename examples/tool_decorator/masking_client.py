"""
Example: LangChain agent calling the @protected_tool masking server.

Demonstrates ADS-driven post-processing: both users can call both tools,
but the data they receive differs based on the ADS decision:

  alice → Permit → full records (dob, email, phone visible; raw note text)
  bob   → Deny   → masked records (dob/email/phone = ***; emails in notes redacted)

The masking is transparent to the agent — it simply receives whatever the
server returns and answers based on that.

Prerequisites:
  - masking_server.py running:
      uv run python examples/tool_decorator/masking_server.py
  - ADS running at http://localhost:8080 with domain_tool_decorator_masking.yaml loaded
  - .env file (or env vars) with LLM_MODEL and the appropriate API key

Run:
    uv run python examples/tool_decorator/masking_client.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

load_dotenv()

_LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")
_LLM_API_BASE = os.getenv("LLM_API_BASE") or None
_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# JWTs are unsigned here — the server decodes without signature verification.
# In production, sign them with your IdP's private key.
_ALICE_JWT = jwt.encode({"username": "alice"}, "demo-secret", algorithm="HS256")  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring
_BOB_JWT = jwt.encode({"username": "bob"}, "demo-secret", algorithm="HS256")  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring


async def run_for_user(username: str, jwt_token: str, question: str) -> None:
    print(f"\n── {username} ──────────────────────────────────────────")
    client = MultiServerMCPClient(
        {
            "patient-records-server": {
                "transport": "streamable_http",
                "url": _SERVER_URL,
                "headers": {"Authorization": f"Bearer {jwt_token}"},
            }
        }
    )
    tools = await client.get_tools()
    print(f"  tools: {[t.name for t in tools]}")

    kwargs: dict = {"model": _LLM_MODEL}
    if _LLM_API_BASE:
        kwargs["api_base"] = _LLM_API_BASE

    agent = create_agent(model=ChatLiteLLM(**kwargs), tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print(f"  answer: {result['messages'][-1].content}")


async def main() -> None:
    # Both users call the same tools with the same question.
    # alice receives full data; bob receives masked/redacted data.
    question = (
        "Get all patient records and all clinical notes. "
        "Summarise what contact details and email addresses you can see."
    )
    await run_for_user("alice", _ALICE_JWT, question)
    await run_for_user("bob", _BOB_JWT, question)


if __name__ == "__main__":
    asyncio.run(main())
