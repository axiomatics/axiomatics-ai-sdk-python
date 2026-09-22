"""Authorization for LangChain agents (``create_agent``) and deep agents.

``LangChainAgentAuthZMiddleware`` is the agent middleware; ``create_authorized_agent`` and
``create_authorized_deep_agent`` build an agent with it wired in.
"""
from axiomatics_ai._extras import requires_extra
from axiomatics_ai.core.config import AuthZConfig, ContextProvider, DefaultClientContextProvider

with requires_extra("langchain-agents", "axiomatics_ai.langchain_agents"):
    from axiomatics_ai.langchain_agents.middleware import LangChainAgentAuthZMiddleware
    from axiomatics_ai.langchain_agents.factory import create_authorized_agent, create_authorized_deep_agent

__all__ = [
    "AuthZConfig",
    "ContextProvider",
    "DefaultClientContextProvider",
    "LangChainAgentAuthZMiddleware",
    "create_authorized_agent",
    "create_authorized_deep_agent",
]
