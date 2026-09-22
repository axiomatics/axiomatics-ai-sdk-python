"""Post-invocation strategy protocol and context.

Post-invocation is the enforcement point applied to a tool's result after it
executes. A strategy receives the raw result and a context object carrying the
PDP client, subject, and environment, and returns the (possibly transformed) result.

This module holds the strategy **protocol** (``PostInvocationStrategy``) and the
**context** (``PostInvocationContext``). The built-in strategy presets —
``RecordFilterStrategy`` (drop denied records) and ``ColumnMaskStrategy`` (mask denied
fields) — live in ``core.presets`` (re-exported from the top-level ``axiomatics_ai``).

Custom strategies: implement ``apply(result, context)`` and pass an instance via
the ``AFTER_TOOL`` Enforcer's ``strategy`` or ``@protected_tool(post_processor=...)``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from axiomatics_ai.core.pdp import PdpClient

logger = logging.getLogger(__name__)


@dataclass
class PostInvocationContext:
    """What a post-invocation strategy gets alongside the tool result.

    Attributes:
        pdp_client (PdpClient): The configured PDP client.
        subject (dict): The subject attributes.
        environment (dict): The environment attributes.
        tool_name (str): The tool that ran.
        args (dict): Its call arguments (empty if not applicable).
        resource (dict): The resource for a single post-invocation request, built by the
            context provider exactly as for tool invocation (by default ``{"tool_name":
            tool_name, **args}``).
        action (dict): The action attributes, e.g. ``{"name": "post_invocation"}``.
        fail_open (bool): What to do when the PDP is unreachable: True permits, False
            restricts.
    """

    pdp_client: PdpClient
    subject: dict
    environment: dict
    tool_name: str
    args: dict
    resource: dict
    action: dict
    fail_open: bool

    @property
    def action_name(self) -> str:
        """The action name for this point, ``action["name"]``.

        A convenience for custom strategies that build their own action dict; pass
        ``context.action`` to send the whole dict.

        Returns:
            str: The action name.
        """
        return self.action.get("name", "")


@runtime_checkable
class PostInvocationStrategy(Protocol):
    """The protocol for a post-invocation transform.

    The strategy does its own PDP interaction. Built in: ``RecordFilterStrategy`` (one batched
    request, one sub-request per record; denied records are removed) and ``ColumnMaskStrategy``
    (one request for the whole result; named fields are masked on Deny). Example::

        class MyStrategy:
            def apply(self, result: Any, context: PostInvocationContext) -> Any:
                # use context.pdp_client, context.subject, context.resource, ...
                return result

        config = AuthZConfig(enforcement={EnforcementPoint.AFTER_TOOL: Enforcer(
            action_name="post_invocation", strategy=MyStrategy())})
    """

    def apply(self, result: Any, context: PostInvocationContext) -> Any:
        """Transform the tool's data according to the PDP's decision.

        Args:
            result (Any): The tool's data, with the framework's wrapper removed: a list or a dict
                on the middleware paths, the tool's own return value on ``@protected_tool``.
            context (PostInvocationContext): The call's context.

        Returns:
            Any: The transformed data, or ``result`` itself when nothing changes.
        """
        ...
