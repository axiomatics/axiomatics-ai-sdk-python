"""Tool-visibility filtering for MCP tools loaded with the LangChain MCP adapters.

``authorized_mcp_tools(tools, config, *, context)`` returns the subset of already-loaded MCP
tools that the subject in ``context`` may see, per the config's ``TOOL_VISIBILITY`` enforcer
(one batched PDP request)::

    tools = await client.get_tools()                      # or: await load_mcp_tools(session)
    tools = authorized_mcp_tools(tools, config, context={"username": "alice"})
    agent = create_authorized_agent(model=model, tools=tools, config=config)

Tool loading stays with the MCP adapters; this only filters the list PDP returns. If the config
has no ``TOOL_VISIBILITY`` enforcer (or ``tools`` is empty), the list is returned unchanged. A
``None`` context (no caller identity) returns no tools, unless ``fail_open`` is set.
"""
from __future__ import annotations

import logging
from typing import Any

from axiomatics_ai.core.config import AuthZConfig

logger = logging.getLogger(__name__)


def authorized_mcp_tools(tools: list, config: AuthZConfig | None = None, *, context: Any) -> list:
    """Filter loaded MCP tools to those the caller may see (one batched PDP request).

    Args:
        tools (list): The tools, as the LangChain MCP adapters loaded them.
        config (AuthZConfig, optional): Must enforce ``TOOL_VISIBILITY``; without it the list is
            returned unchanged.
        context (Any): The caller's identity, handed to the config's context provider
            (``DefaultClientContextProvider`` by default).

    Returns:
        list: The permitted tools. With ``context=None`` (no identity), no tools, or every tool
        under ``fail_open``.
    """
    from axiomatics_ai.core.pdp import strip_none
    from axiomatics_ai.core.config import DefaultClientContextProvider, EnforcementPoint
    from axiomatics_ai.core.enforcement import authorized_tools

    config = config or AuthZConfig()
    enforcer = config.enforcer(EnforcementPoint.TOOL_VISIBILITY)
    if not tools or enforcer is None:
        return tools
    if context is None:
        # No caller identity to authorize: showing every tool would fail open.
        logger.warning(
            "authorized_mcp_tools: context is None (no caller identity); %s.",
            "returning all tools (fail_open)" if config.connection.fail_open else "returning no tools",
        )
        return tools if config.connection.fail_open else []

    provider = config.context_provider or DefaultClientContextProvider()
    permitted = authorized_tools(
        pdp=config.build_pdp_client(),   # PdpClient is stateless — cheap to build per call
        cache=None,
        config=config,
        surface="langchain_mcp",
        subject=strip_none(provider.get_subject(context)),
        action=provider.get_action(enforcer.action_name, context),
        tool_names=[t.name for t in tools],
        environment=strip_none(provider.get_environment(context)),
    )
    return [t for t in tools if t.name in permitted]
