"""
MCP client for the gateway demo — a LangChain create_agent whose MCP tools are loaded
THROUGH the gateway. No client-side authorization: the gateway filters the tool list.

Identity model (note vs the client-side `agents` path): here the create_agent has NO
authorization middleware, so we do NOT pass identity to agent.invoke(context=...) — that
`context=` mechanism only matters when LangChainAgentAuthZMiddleware is enforcing in-process.
Instead, identity travels as a JWT header on the MCP transport connection to the GATEWAY,
which is the enforcement point. Because the gateway enforces tool visibility, the tool set
returned by get_tools() already reflects the caller's permissions — alice gets all three,
bob gets only search_docs — before create_agent ever sees them.

Multi-user consequence: the visible tools are identity-scoped AT LOAD TIME, so a
multi-user server must load tools and build the agent PER USER (as this example does in the
loop below) — a single long-lived agent would be frozen to whoever loaded its tools.
(A future example will show the alternative "Model B": one long-lived shared agent that
loads all tools once and injects the end-user identity per tool CALL, so the gateway
enforces tool_invocation per user — at the cost of every user seeing all tool names.)

  --user alice|bob|both   whose identity to load tools as (default: both)

Run after mcp_server.py + gateway.py are up (and ADS with domain_gateway.yaml).
Config in .env: PDP_BASE_URL, LLM_MODEL, OPENROUTER_API_KEY
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

GATEWAY = "http://127.0.0.1:9210"


def bearer(username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {jwt.encode({'username': username}, 'demo-secret', algorithm='HS256')}"}  # nosemgrep: jwt-python-hardcoded-secret — demo identity for a runnable example; the server side decodes without verifying, see the module docstring


async def run_as(user: str) -> None:
    # Per user: point the MCP client at the GATEWAY with this caller's JWT, load the
    # (already-filtered) tools, and build an agent around them.
    client = MultiServerMCPClient({
        "acme": {"url": f"{GATEWAY}/tools/", "transport": "streamable_http", "headers": bearer(user)},
    })
    tools = await client.get_tools()
    print(f"[{user}] tools visible via gateway: {sorted(t.name for t in tools)}")

    agent = create_agent(model=ChatLiteLLM(model=os.getenv("LLM_MODEL")), tools=tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Search the internal docs for the refund policy."}]}
    )
    # A message's .content may be a plain string or a list of content blocks (e.g. when the
    # model streams reasoning/tool blocks) — normalise to text before printing.
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    print(f"[{user}] reply: {content.strip()[:200]}")


async def main(users: list[str]) -> None:
    for user in users:
        await run_as(user)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", choices=["alice", "bob", "both"], default="both")
    args = parser.parse_args()
    asyncio.run(main(["alice", "bob"] if args.user == "both" else [args.user]))
