"""
Client-side ADS authorization for a Google ADK agent that consumes MCP tools — all three
enforcement points at once, on domain_all_authz_enabled.yaml.

A normal ADK agent connects to an MCP server via McpToolset (wired as in ads-mcp-server's
example). `authorized_adk_agent(...)` attaches ADS enforcement via ADK callbacks:
  1. tool visibility     — the model only sees tools the caller may use
  2. tool invocation     — a disallowed call is blocked before the tool runs
  3. post-invocation     — records the caller may not read are filtered from the result

Identity is bound once per user via the ADK session's state (no per-invocation plumbing).
Under domain_all_authz_enabled.yaml:
  - alice: sees all tools; may call any; reads all records.
  - bob:   sees search_docs + get_account_balance; may call get_account_balance only for
           ACC-001; his 'classified' records are dropped from search_docs results.

Prerequisites (each in its own process, like ADS):
  1. ADS with domain_all_authz_enabled.yaml on PDP_BASE_URL
  2. the MCP backend:  python examples/google_adk/mcp_backend.py
Config in .env: PDP_BASE_URL, MCP_URL (default http://127.0.0.1:9310/mcp), LLM_MODEL (or
GOOGLE_ADK_LLM_MODEL), OPENROUTER_API_KEY

Run:  python examples/google_adk/mcp_agent.py
"""
import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.genai import types

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.google_adk import authorized_adk_agent
from axiomatics_ai import PDPConnection, standard_tool_enforcement

APP = "adk-mcp-demo"
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:9310/mcp")
MODEL = os.getenv("GOOGLE_ADK_LLM_MODEL", os.getenv("LLM_MODEL", "openrouter/openai/gpt-5.6-luna"))


def build_agent():
    toolset = McpToolset(connection_params=StreamableHTTPConnectionParams(url=MCP_URL))
    agent = Agent(
        model=LiteLlm(model=MODEL),
        name="ads_agent",
        instruction="You are a helpful assistant. Use the available tools when relevant.",
        tools=[toolset],
    )
    # THE one-line integration — all three enforcement points (defaults are all True).
    return authorized_adk_agent(
        agent,
        AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")), enforcement=standard_tool_enforcement()),
    )


async def ask_as(runner, user: str, session: str, question: str) -> str:
    # bind identity ONCE, in the session state — no per-invocation plumbing
    await runner.session_service.create_session(
        app_name=APP, user_id=user, session_id=session, state={"username": user},
    )
    message = types.Content(role="user", parts=[types.Part(text=question)])
    final = ""
    async for event in runner.run_async(user_id=user, session_id=session, new_message=message):
        if event.is_final_response() and event.content:
            # Collect text across every part rather than indexing parts[0]: when a tool call
            # is denied, ADK ends the turn with content whose `parts` list is EMPTY, and some
            # parts carry a function_call/function_response with no text at all.
            texts = [p.text for p in (event.content.parts or []) if p.text]
            if texts:
                final = " ".join(texts)
    answer = " ".join(final.split())[:600]
    # Empty means the model ended the turn without saying anything — e.g. every tool call was
    # denied, or ADS was unreachable and enforcement failed closed. Report it instead of
    # crashing, so the run continues and the log shows which.
    return answer or "(no answer text — see the log for denials or ADS errors)"


async def main() -> None:
    runner = Runner(agent=build_agent(), app_name=APP, session_service=InMemorySessionService())

    vis_q = "List the exact names of every tool available to you, comma-separated. Names only."
    print("── 1. Tool visibility (alice: all 3; bob: search_docs + get_account_balance) ──")
    for u in ("alice", "bob"):
        print(f"[{u}] {await ask_as(runner, u, f'{u}-vis', vis_q)}")

    print("\n── 2. Tool invocation — arg-level (bob may read ACC-001, not ACC-002) ──")
    for user, acct in (("alice", "ACC-002"), ("bob", "ACC-001"), ("bob", "ACC-002")):
        q = f"Get the balance for account {acct}."
        print(f"[{user} {acct}] {await ask_as(runner, user, f'{user}-inv-{acct}', q)}")

    post_q = "Search the docs (query 'spec') and list the id and doc_type of EVERY result you receive."
    print("\n── 3. Post-invocation (alice: both docs; bob: 'classified' doc filtered out) ──")
    for u in ("alice", "bob"):
        print(f"[{u}] {await ask_as(runner, u, f'{u}-post', post_q)}")


if __name__ == "__main__":
    asyncio.run(main())
