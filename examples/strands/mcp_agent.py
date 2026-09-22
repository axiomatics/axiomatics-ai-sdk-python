"""
Client-side ADS authorization for a Strands agent that consumes MCP tools — all three
enforcement points at once, on domain_all_authz_enabled.yaml.

A normal Strands agent connects to an MCP server via MCPClient (wired as in ads-mcp-server's
example). `authorized_strands_agent(...)` attaches ADS enforcement via a Strands intervention handler:
  1. tool visibility     — the model only sees tools the caller may use
  2. tool invocation     — a disallowed call is blocked before the tool runs
  3. post-invocation     — records the caller may not read are filtered from the result

Identity is bound once per user via the agent's state (no per-invocation plumbing).
Under domain_all_authz_enabled.yaml:
  - alice: sees all tools; may call any; reads all records.
  - bob:   sees search_docs + get_account_balance; may call get_account_balance only for
           ACC-001; his 'classified' records are dropped from search_docs results.

Prerequisites (each in its own process, like ADS):
  1. ADS with domain_all_authz_enabled.yaml on PDP_BASE_URL
  2. the MCP backend:  python examples/strands/mcp_backend.py
Config in .env: PDP_BASE_URL, MCP_URL (default http://127.0.0.1:9320/mcp), LLM_MODEL (or
STRANDS_LLM_MODEL), OPENROUTER_API_KEY

Run:  python examples/strands/mcp_agent.py
"""
import os

from dotenv import load_dotenv
from mcp.client.streamable_http import streamable_http_client
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.tools.mcp import MCPClient

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.strands import authorized_strands_agent
from axiomatics_ai import PDPConnection, standard_tool_enforcement

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:9320/mcp")
MODEL = os.getenv("STRANDS_LLM_MODEL", os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna"))
SYSTEM = "You are a helpful assistant. Use the available tools when relevant."


def build_agent(user: str, mcp_client: MCPClient):
    agent = Agent(
        model=LiteLLMModel(model_id=MODEL),
        system_prompt=SYSTEM,
        tools=[mcp_client],
        state={"username": user},          # bind identity ONCE — no per-invocation plumbing
        callback_handler=None,             # silence Strands' streamed tool-call prints in the demo
    )
    # THE one-line integration — all three enforcement points (defaults are all True).
    return authorized_strands_agent(
        agent,
        AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")), enforcement=standard_tool_enforcement()),
    )


def ask_as(user: str, question: str) -> str:
    # Fresh MCP client + agent per call: isolates each scenario's conversation history and
    # avoids reusing an already-running MCP session across agents. Strands starts/manages the
    # client lifecycle when it is passed in ``tools=[...]`` (no explicit ``with`` needed).
    mcp_client = MCPClient(lambda: streamable_http_client(MCP_URL))
    agent = build_agent(user, mcp_client)
    return " ".join(str(agent(question)).split())[:600]


def main() -> None:
    vis_q = "List the exact names of every tool available to you, comma-separated. Names only."
    print("── 1. Tool visibility (alice: all 3; bob: search_docs + get_account_balance) ──")
    for u in ("alice", "bob"):
        print(f"[{u}] {ask_as(u, vis_q)}")

    print("\n── 2. Tool invocation — arg-level (bob may read ACC-001, not ACC-002) ──")
    for user, acct in (("alice", "ACC-002"), ("bob", "ACC-001"), ("bob", "ACC-002")):
        print(f"[{user} {acct}] {ask_as(user, f'Get the balance for account {acct}.')}")

    post_q = "Search the docs (query 'spec') and list the id and doc_type of EVERY result you receive."
    print("\n── 3. Post-invocation (alice: both docs; bob: 'classified' doc filtered out) ──")
    for u in ("alice", "bob"):
        print(f"[{u}] {ask_as(u, post_q)}")


if __name__ == "__main__":
    main()
