"""Explicit factories for an PDP-authorized LangChain agent / deep agent.

``create_authorized_agent(*args, config, **kwargs)`` and
``create_authorized_deep_agent(*args, config, **kwargs)`` are thin wrappers over
``langchain.agents.create_agent`` and ``deepagents.create_deep_agent`` that inject
``LangChainAgentAuthZMiddleware(config)`` into the agent's middleware chain. They are the
one-call, call-site-explicit equivalent of::

    from langchain.agents import create_agent
    from axiomatics_ai import LangChainAgentAuthZMiddleware
    agent = create_agent(model, tools, middleware=[LangChainAgentAuthZMiddleware(config)])

Because a LangChain agent's enforcement seam is *middleware supplied at construction* (a
compiled graph has no post-hoc attach point), these are factories that mirror the two agent
creators rather than an ``authorized_<framework>_agent(agent)`` that mutates an existing agent — unlike the
ADK / Strands / OpenAI paths, whose frameworks expose a live attach seam.

Every argument other than ``config`` and ``middleware`` is forwarded verbatim to the underlying
creator (``model``, ``tools``, ``system_prompt``, ``response_format``, …), so they track its
signature.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

from axiomatics_ai._extras import requires_extra
from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.langchain_agents.middleware import LangChainAgentAuthZMiddleware

logger = logging.getLogger(__name__)


def _authz_middleware_chain(
    config: AuthZConfig, middleware: list | None, agent_name: str | None = None,
) -> list:
    """Put ``LangChainAgentAuthZMiddleware(config)`` first in the caller's middleware.

    It runs outermost. If one is already present (your own, e.g. with another config), the list
    is left as it is (no duplicate).
    """
    caller = list(middleware or [])
    if any(isinstance(m, LangChainAgentAuthZMiddleware) for m in caller):
        return caller
    return [LangChainAgentAuthZMiddleware(config, agent_name=agent_name), *caller]


# Set on the graphs these factories build, so a deep agent can tell an authorized compiled
# sub-agent from one that would run its tools unchecked.
_AUTHORIZED_MARK = "_axiomatics_authorized"


def _mark_authorized(graph: Any) -> Any:
    with contextlib.suppress(AttributeError, TypeError):
        setattr(graph, _AUTHORIZED_MARK, True)
    return graph


def create_authorized_agent(*args: Any, config: AuthZConfig, middleware: list | None = None, **kwargs: Any):
    """Create a LangChain agent (``langchain.agents.create_agent``) with authorization wired in.

    Args:
        *args (Any): Passed to ``create_agent`` (e.g. the model).
        config (AuthZConfig): What to enforce and where the PDP is.
        middleware (list, optional): Your own middleware; the authorization middleware runs
            before it.
        **kwargs (Any): Passed to ``create_agent`` (e.g. ``tools``, ``checkpointer``).

    Returns:
        CompiledStateGraph: Whatever ``create_agent`` returns. It can be a deep agent's compiled
        sub-agent (``{"runnable": ...}``).
    """
    from langchain.agents import create_agent

    chain = _authz_middleware_chain(config, middleware, kwargs.get("name"))
    return _mark_authorized(create_agent(*args, middleware=chain, **kwargs))


def create_authorized_deep_agent(*args: Any, config: AuthZConfig, middleware: list | None = None, **kwargs: Any):
    """Create a deep agent (``deepagents.create_deep_agent``) with authorization wired in.

    Requires the ``deepagents`` package (imported lazily). Sub-agents are authorized with the
    same config: each declarative sub-agent in ``subagents``, and the general-purpose one
    deepagents adds, gets the authorization middleware. The PDP sees the ``task`` call that
    starts a sub-agent, then the sub-agent's own calls, with the caller's identity (the run's
    ``context`` reaches the sub-agent) and the sub-agent's name as the subject's
    ``config.attribute_names.acting_agent``, so a policy can treat sub-agents differently.
    To give a sub-agent another config, put your own ``LangChainAgentAuthZMiddleware`` in its
    spec's ``middleware``; it is kept as it is.

    A compiled sub-agent (``{"runnable": ...}``) runs its own graph, so it must be built with
    ``create_authorized_agent`` or ``create_authorized_deep_agent``. A remote sub-agent
    (``{"graph_id": ...}``) runs elsewhere and must enforce there; a warning says so.

    The ``task`` tool's output (the sub-agent's answer) is not post-processed: the sub-agent's
    own tool results already were, and its text is nothing a post-invocation strategy reads.

    Args:
        *args (Any): Passed to ``create_deep_agent``.
        config (AuthZConfig): What to enforce and where the PDP is, for the main agent and every
            sub-agent.
        middleware (list, optional): Your own middleware; the authorization middleware runs
            before it.
        **kwargs (Any): Passed to ``create_deep_agent`` (e.g. ``tools``, ``subagents``,
            ``skills``, ``name``).

    Returns:
        CompiledStateGraph: Whatever ``create_deep_agent`` returns.

    Raises:
        ValueError: A compiled sub-agent was not built with these factories.
    """
    with requires_extra("deepagents", "create_authorized_deep_agent"):
        from deepagents import create_deep_agent

    kwargs["subagents"] = _authorized_subagents(kwargs.get("subagents"), config, kwargs.get("skills"))
    chain = _authz_middleware_chain(config, middleware, kwargs.get("name"))
    for m in chain:
        if isinstance(m, LangChainAgentAuthZMiddleware):
            m._unchecked_output_tools = _TASK_TOOL
    return _mark_authorized(create_deep_agent(*args, middleware=chain, **kwargs))


# deepagents' tool that runs a synchronous sub-agent; every one is authorized here.
_TASK_TOOL = frozenset({"task"})


def _authorized_subagents(subagents: list | None, config: AuthZConfig, skills: list | None) -> list:
    """Give every sub-agent spec the authorization middleware, adding the general-purpose one."""
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    specs = list(subagents or [])
    if not any(spec.get("name") == GENERAL_PURPOSE_SUBAGENT["name"] for spec in specs):
        # deepagents' own general-purpose sub-agent takes no middleware; this one is built the
        # same way (main model, tools, permissions and interrupt_on inherited) with it.
        default = {**GENERAL_PURPOSE_SUBAGENT}
        if skills is not None:
            default["skills"] = skills
        specs.insert(0, default)

    authorized = []
    for spec in specs:
        name = spec.get("name")
        if "graph_id" in spec:
            logger.warning(
                "Remote sub-agent '%s' runs in its own deployment: enforce authorization there.", name,
            )
            authorized.append(spec)
        elif "runnable" in spec:
            if not getattr(spec["runnable"], _AUTHORIZED_MARK, False):
                raise ValueError(
                    f"Compiled sub-agent '{name}' would run its tools unauthorized: build it with "
                    "create_authorized_agent or create_authorized_deep_agent."
                )
            authorized.append(spec)
        else:
            chain = _authz_middleware_chain(config, spec.get("middleware"), name)
            authorized.append({**spec, "middleware": chain})
    return authorized
