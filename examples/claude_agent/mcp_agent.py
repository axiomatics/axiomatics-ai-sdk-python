"""
Client-side ADS authorization for a Claude Agent SDK agent that consumes MCP tools.

The Claude Agent SDK has no Agent object — you pass a ClaudeAgentOptions to query().
`authorized_claude_options(options, config, identity=...)` returns a copy with ADS enforcement wired
into the options:
  - tool invocation   — a PreToolUse hook blocks a disallowed call before it runs (ARG-LEVEL)
  - post-invocation   — a PostToolUse hook filters records the caller may not read

Tool visibility is NOT supported on this path (the SDK exposes no per-turn tool-list filter),
so write policies at the invocation point. Identity is captured by closure — one options
object per user (built per user below).

The hook runs for every tool call, whatever allowed_tools or permission_mode say, with
ClaudeSDKClient (used here) or a one-shot query(), and decides it: a permitted call is allowed,
so no allowed_tools or permission_mode is needed.

Under domain_all_authz_enabled.yaml:
  - alice: may call any tool; reads all records.
  - bob:   may call get_account_balance only for ACC-001; his 'classified' records are dropped.

Prerequisites (each in its own process, like ADS):
  1. ADS with domain_all_authz_enabled.yaml on PDP_BASE_URL
  2. the MCP backend:  python examples/claude_agent/mcp_backend.py
Config in .env: PDP_BASE_URL, MCP_URL (default http://127.0.0.1:9340/mcp), OPENROUTER_API_KEY,
and optionally ANTHROPIC_LLM_MODEL (default anthropic/claude-sonnet-5). Like the other
examples this routes the SDK through OpenRouter (set below from OPENROUTER_API_KEY); to use
Anthropic directly instead, set ANTHROPIC_API_KEY in your environment.

Run:  python examples/claude_agent/mcp_agent.py
"""
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

# Route the Claude Agent SDK through OpenRouter (same LLM provider as the other examples),
# unless the environment is already pointed at a real Anthropic endpoint.
if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ.setdefault("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", os.environ["OPENROUTER_API_KEY"])

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage

from axiomatics_ai import AuthZConfig
from axiomatics_ai.claude_agent import authorized_claude_options
from axiomatics_ai import PDPConnection, standard_tool_enforcement

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:9340/mcp")
# Routed through OpenRouter, so strip any litellm-style "openrouter/" prefix (no-op if absent).
MODEL = os.getenv("ANTHROPIC_LLM_MODEL", os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")).removeprefix("openrouter/")
MCP = {"acme": {"type": "http", "url": MCP_URL}}
SYSTEM = "You are a helpful assistant. Use the available tools when relevant."


async def ask_as(user: str, question: str) -> str:
    # Identity is baked into the options via closure — build one options object per user.
    # The PreToolUse hook from authorized_claude_options decides every MCP call: the policy's
    # Permit allows it (no allowed_tools needed), its Deny blocks it whatever the mode says.
    base = ClaudeAgentOptions(model=MODEL, system_prompt=SYSTEM, mcp_servers=MCP, max_turns=6)
    options = authorized_claude_options(base, AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")), enforcement=standard_tool_enforcement()),
                                 identity={"username": user})
    parts: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(question)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        parts.append(block.text)
            elif isinstance(message, ResultMessage) and message.result:
                parts.append(message.result)
    return " ".join(" ".join(parts).split())[:600]


async def main() -> None:
    print("(tool visibility is not enforced on this path — policies apply at invocation/post)\n")

    print("── 1. Tool invocation — arg-level (bob may read ACC-001, not ACC-002) ──")
    for user, acct in (("alice", "ACC-002"), ("bob", "ACC-001"), ("bob", "ACC-002")):
        print(f"[{user} {acct}] {await ask_as(user, f'Get the balance for account {acct}.')}")

    post_q = "Search the docs (query 'spec') and list the id and doc_type of EVERY result you receive."
    print("\n── 2. Post-invocation (alice: both docs; bob: 'classified' doc filtered out) ──")
    for user in ("alice", "bob"):
        print(f"[{user}] {await ask_as(user, post_q)}")


if __name__ == "__main__":
    asyncio.run(main())
