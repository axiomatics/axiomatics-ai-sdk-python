"""
Client-side ADS authorization for an OpenAI Agents SDK agent that consumes MCP tools — all
three enforcement points at once, on domain_all_authz_enabled.yaml.

A normal Agents-SDK agent connects to an MCP server via MCPServerStreamableHttp (the SDK's
client-side handle for a remote MCP server; wired as in ads-mcp-server's example).
`authorized_openai_agent(...)` attaches ADS enforcement by wrapping the one seam the Runner calls
every turn, `agent.get_all_tools`:
  1. tool visibility     — the model only sees tools the caller may use
  2. tool invocation     — a disallowed call is blocked before the tool runs
  3. post-invocation     — records the caller may not read are filtered from the result

Identity is supplied per run via the Runner's `context` (reuse the same object to bind once).
Under domain_all_authz_enabled.yaml:
  - alice: sees all tools; may call any; reads all records.
  - bob:   sees search_docs + get_account_balance; may call get_account_balance only for
           ACC-001; his 'classified' records are dropped from search_docs results.

Prerequisites (each in its own process, like ADS):
  1. ADS with domain_all_authz_enabled.yaml on PDP_BASE_URL
  2. the MCP backend:  python examples/openai_agents/mcp_backend.py
Config in .env: PDP_BASE_URL, MCP_URL (default http://127.0.0.1:9330/mcp), LLM_MODEL (or
OPENAI_LLM_MODEL), and an OpenAI-compatible endpoint — OPENAI_BASE_URL + OPENAI_API_KEY, or
OPENROUTER_API_KEY (defaults to OpenRouter).

Run:  python examples/openai_agents/mcp_agent.py
"""
import asyncio
import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.tracing import set_tracing_disabled
from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(True)  # tracing would post to OpenAI's platform regardless of base_url

from axiomatics_ai import AuthZConfig
from axiomatics_ai.openai_agents import authorized_openai_agent
from axiomatics_ai import PDPConnection, standard_tool_enforcement

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:9330/mcp")
# The Agents SDK sends the model id verbatim to the OpenAI-compatible endpoint, so strip any
# litellm-style "openrouter/" prefix the shared LLM_MODEL may carry (a no-op if absent).
MODEL = os.getenv("OPENAI_LLM_MODEL", os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")).removeprefix("openrouter/")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
SYSTEM = "You are a helpful assistant. Use the available tools when relevant."


def build_agent(mcp_server) -> Agent:
    model = OpenAIChatCompletionsModel(
        model=MODEL,
        openai_client=AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY),
    )
    agent = Agent(name="ads_agent", instructions=SYSTEM, model=model, mcp_servers=[mcp_server])
    # THE one-line integration — all three enforcement points (defaults are all True).
    return authorized_openai_agent(
        agent,
        AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")), enforcement=standard_tool_enforcement()),
    )


async def ask_as(agent: Agent, user: str, question: str) -> str:
    # identity is per-run: passed as the Runner context (reuse the object to bind once)
    result = await Runner.run(agent, question, context={"username": user})
    return " ".join(str(result.final_output).split())[:600]


async def main() -> None:
    # MCPServerStreamableHttp is the SDK's client-side handle to the (external) MCP backend.
    async with MCPServerStreamableHttp(name="acme", params={"url": MCP_URL}, cache_tools_list=False) as mcp_server:
        agent = build_agent(mcp_server)

        vis_q = "List the exact names of every tool available to you, comma-separated. Names only."
        print("── 1. Tool visibility (alice: all 3; bob: search_docs + get_account_balance) ──")
        for u in ("alice", "bob"):
            print(f"[{u}] {await ask_as(agent, u, vis_q)}")

        print("\n── 2. Tool invocation — arg-level (bob may read ACC-001, not ACC-002) ──")
        for user, acct in (("alice", "ACC-002"), ("bob", "ACC-001"), ("bob", "ACC-002")):
            print(f"[{user} {acct}] {await ask_as(agent, user, f'Get the balance for account {acct}.')}")

        post_q = "Search the docs (query 'spec') and list the id and doc_type of EVERY result you receive."
        print("\n── 3. Post-invocation (alice: both docs; bob: 'classified' doc filtered out) ──")
        for u in ("alice", "bob"):
            print(f"[{u}] {await ask_as(agent, u, post_q)}")


if __name__ == "__main__":
    asyncio.run(main())
