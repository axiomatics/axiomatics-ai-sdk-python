"""
Example: the `create_authorized_agent` factory — the recommended, call-site-explicit way to
add ADS authorization to a LangChain agent.

`create_authorized_agent(*args, config=..., **kwargs)` is a thin wrapper over
`langchain.agents.create_agent` that injects `LangChainAgentAuthZMiddleware(config)` as the
outermost middleware, then forwards everything else (model, tools, system_prompt, …) verbatim.
Any middleware you pass is preserved after ours (and an already-present one isn't duplicated).

Equivalent to doing it by hand (the "direct middleware" way, also supported):

    from langchain.agents import create_agent
    from axiomatics_ai import LangChainAgentAuthZMiddleware
    agent = create_agent(model, tools, middleware=[LangChainAgentAuthZMiddleware(config), *my_mw])

Run with ADS at http://localhost:8080 and domain_all_authz_enabled.yaml loaded.
Set LLM_MODEL and your API key in a .env file or environment variables.
"""
import json
import os

from dotenv import load_dotenv
load_dotenv()

from langchain.agents.middleware import AgentMiddleware
from langchain_litellm import ChatLiteLLM
from langchain.tools import tool

from axiomatics_ai import AuthZConfig, create_authorized_agent
from axiomatics_ai import PDPConnection, standard_tool_enforcement


@tool
def search_docs(query: str) -> str:
    """Search the documentation and return matching articles."""
    return json.dumps([
        {"id": "d1", "title": "Getting Started", "doc_type": "public"},
        {"id": "d2", "title": "Internal Spec",   "doc_type": "classified"},
    ])


@tool
def get_account_balance(account_id: str) -> str:
    """Get the balance for an account."""
    return f"Balance for {account_id}: $1,000"


class LoggingMiddleware(AgentMiddleware):
    """A trivial user middleware — shows that your own middleware composes with the injected
    authorization one (ours runs outermost, yours after)."""
    def wrap_model_call(self, request, handler):
        print(f"    [logging middleware] model turn with {len(request.tools)} tool(s) visible")
        return handler(request)


def ask(agent, prompt, *, username):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context={"username": username, "session_id": f"s-{username}"},
    )
    print(f"  [{username}] {result['messages'][-1].content}")


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")
    # ADS must already be running with domain_all_authz_enabled.yaml loaded (see docs/running-ads.md)
    config = AuthZConfig(connection=PDPConnection(base_url=ads_url), enforcement=standard_tool_enforcement())

    # THE one-line integration — the factory injects the authz middleware; LoggingMiddleware is
    # preserved after it. Everything else (model, tools) is forwarded to create_agent.
    agent = create_authorized_agent(
        model=ChatLiteLLM(model=os.getenv("LLM_MODEL")),
        tools=[search_docs, get_account_balance],
        config=config,
        middleware=[LoggingMiddleware()],
    )

    # alice sees both tools; bob (per domain_all_authz_enabled.yaml) doesn't see send_email and
    # can't read the 'classified' doc — enforced by the injected middleware.
    print("── Tool visibility + record filtering via the factory ──")
    ask(agent, "List every tool you can use, then search the docs and show all results.", username="alice")
    print()
    ask(agent, "List every tool you can use, then search the docs and show all results.", username="bob")



if __name__ == "__main__":
    main()
