"""Shared A2A invocation enforcement: the one PDP-decision path of the client and server guards.

Mirrors the ``fastmcp`` ``on_call_tool`` flow: build the resource through the context
provider, consult the decision cache, and fall back to a single PDP request. Keeping it
here means client and server share one authorization semantics.
"""
from __future__ import annotations

from typing import Any

from axiomatics_ai.core.pdp import PdpClient
from axiomatics_ai.core.authz_helpers import resolve_authz_context
from axiomatics_ai.core.cache import AuthzDecisionCacheWithTTL
from axiomatics_ai.core.config import AuthZConfig, EnforcementPoint
from axiomatics_ai.core.enforcement import authorized_call


class A2AAuthorizationError(PermissionError):
    """Raised when an outbound A2A call is denied by the PDP (client side)."""


def authorize_invocation(
    *,
    pdp: PdpClient,
    cache: AuthzDecisionCacheWithTTL,
    config: AuthZConfig,
    provider: Any,
    context: Any,
    agent_name: str,
    method: str,
    skill_hint: str | None = None,
    message_text: str | None = None,
    surface: str = "a2a",
) -> bool:
    """Decide whether an A2A call is permitted.

    The resource is the target (or served) agent's identity, with the RPC method and, when
    present, a skill hint and the message text as extra attributes, so a policy can condition
    on any of them.

    Args:
        pdp (PdpClient): The PDP client.
        cache (AuthzDecisionCacheWithTTL): The decision cache.
        config (AuthZConfig): The config (the A2A_INVOCATION action name, attribute names).
        provider (ContextProvider): Builds the attributes from ``context``.
        context (Any): The subject dict on the client, the HTTP headers on the server.
        agent_name (str): The agent's identity, sent as
            ``attribute_names.agent``.
        method (str): The JSON-RPC method, e.g. ``"message/send"``.
        skill_hint (str, optional): A skill id the caller put in the message. Caller-supplied:
            don't base a permit on it alone.
        message_text (str, optional): The message text, when the guard includes it.
        surface (str): The label of the decision event (``"a2a_client"``, ``"a2a_server"``).

    Returns:
        bool: Whether the call is permitted.
    """
    resolved = resolve_authz_context(provider, context)
    # A2A opts in by instantiating the client auth / server middleware, so it always enforces;
    # the action name comes from the A2A_INVOCATION enforcer if configured, else the default.
    a2a_enforcer = config.enforcer(EnforcementPoint.A2A_INVOCATION)
    action_name = a2a_enforcer.action_name if a2a_enforcer else "a2a_invocation"
    action = provider.get_action(action_name, context)

    args: dict[str, Any] = {"method": method}
    if skill_hint:
        args["skill"] = skill_hint
    if message_text:
        args["message"] = message_text
    resource = provider.get_resource(
        agent_name, context, args=args, resource_name_attribute=config.attribute_names.agent,
    )

    return authorized_call(
        pdp=pdp,
        cache=cache,
        config=config,
        surface=surface,
        enforcement_point="a2a_invocation",
        subject=resolved.subject,
        action=action,
        resource=resource,
        environment=resolved.environment,
    )
