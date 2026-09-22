import os
from functools import lru_cache
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient

from axiomatics_ai import LangChainAgentAuthZMiddleware, AuthZConfig, PDPConnection, standard_tool_enforcement
from async_utils import run_async
from data_models import AppConfig, AppContext

_LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")
_LLM_API_BASE = os.getenv("LLM_API_BASE") or None


class AppContextProvider:
    """Maps AppContext fields to XACML request categories."""

    def get_subject(self, context: AppContext) -> dict:
        # First key is used as the primary subject identifier by the middleware
        return {
            "username": context.username,
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
        }

    def get_action(self, action_name: str, context: AppContext) -> dict:
        return {"name": action_name}

    def get_resource(
        self,
        resource_name: str,
        context: AppContext,
        *,
        args: dict | None = None,
        resource_name_attribute: str | None = None,
    ) -> dict:
        # Arguments first, SDK-supplied values last, so a tool argument can't overwrite them.
        resource = dict(args or {})
        resource["agent_name"] = context.agent_name
        if resource_name_attribute:
            resource[resource_name_attribute] = resource_name
        return resource

    def get_environment(self, context: AppContext) -> dict:
        return {"session_id": context.session_id}


def _build_middleware(config: AppConfig) -> list:
    mw_config = AuthZConfig(
        connection=PDPConnection(
            base_url=config.ads_base_url,
            timeout=config.request_timeout_seconds,
        ),
        context_provider=AppContextProvider(),
        enforcement=standard_tool_enforcement(),
    )
    return [LangChainAgentAuthZMiddleware(config=mw_config)]


def _build_tools(config: AppConfig) -> list:
    tools = [search_docs, create_ticket, send_email]
    if config.invocation_mode == "async":
        tools.extend(_load_mcp_tools())
    return tools


@lru_cache(maxsize=1)
def _load_mcp_tools():
    client = MultiServerMCPClient(
        {
            "demo": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(Path(__file__).with_name("mcp_server.py"))],
            }
        }
    )
    return tuple(run_async(client.get_tools()))


def _build_model() -> ChatLiteLLM:
    kwargs: dict = {"model": _LLM_MODEL}
    if _LLM_API_BASE:
        kwargs["api_base"] = _LLM_API_BASE
    return ChatLiteLLM(**kwargs)


def build_agent(config: AppConfig, app_context: AppContext):
    model = _build_model()
    tools = _build_tools(config)
    middleware = _build_middleware(config)

    if config.agent_runtime == "deep-agents":
        return create_deep_agent(model=model, tools=tools, middleware=middleware)
    return create_agent(model=model, tools=tools, middleware=middleware, context_schema=None)


@tool
def search_docs(query: str) -> str:
    """Search internal documentation. Returns a JSON list of matching documents."""
    import json
    return json.dumps([
        {"id": "doc-001", "title": "Getting Started Guide",      "summary": f"Introduction to the platform. Query: {query}", "doc_type": "public"},
        {"id": "doc-002", "title": "API Reference",              "summary": "Full API reference for all endpoints.",          "doc_type": "public"},
        {"id": "doc-003", "title": "Troubleshooting FAQ",        "summary": "Common issues and how to resolve them.",         "doc_type": "public"},
        {"id": "doc-004", "title": "Internal Security Policies", "summary": "Confidential security guidelines.",              "doc_type": "classified"},
        {"id": "doc-005", "title": "Q3 Financial Projections",   "summary": "Confidential financial forecast for Q3.",        "doc_type": "classified"},
    ])


@tool
def create_ticket(title: str, description: str) -> str:
    """Create a support ticket."""
    return f"Created ticket '{title}'"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to} with subject '{subject}'"
