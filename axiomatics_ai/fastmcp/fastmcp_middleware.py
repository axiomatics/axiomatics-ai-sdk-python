"""Server-side authorization middleware for FastMCP.

Intercepts ``tools/list`` and ``tools/call`` and enforces the PDP's decisions before
execution. Usage::

    from fastmcp import FastMCP
    from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
    from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware

    mcp = FastMCP("my-server")

    @mcp.tool()
    def search_docs(query: str) -> str: ...

    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),
    )))
    mcp.run(transport="http")  # stdio is not supported

**Custom attributes.** By default every JWT claim becomes a subject attribute. Set
``AuthZConfig(context_provider=...)`` to control each category; every method receives the
HTTP headers dict::

    class MyProvider(DefaultServerContextProvider):
        def get_environment(self, headers):
            return {"session_id": headers.get("x-session-id", "")}

To verify tokens here, use ``AgentIdentityContextProvider`` with a ``JwtVerification``.

**Ambient identity for RAG.** ``on_call_tool`` binds the caller's identity into the ambient
authz context for the duration of the tool call (``bind_ambient_identity``), so a vector
store wrapped with ``to_authorized_vectorstore`` filters inside the tool. This happens
whichever points are enforced: behind the gateway, a RAG backend runs this middleware with
``enforcement=None`` only to re-establish the identity (FastMCP's proxy forwards the
caller's JWT when the route sets ``forward_authorization``).

**Post-invocation** rewrites ``structured_content`` and the text blocks rendering it
together, so ``structuredContent`` never carries records the text omits. Output the
strategy can't inspect passes through as a ``skipped`` decision, or is withheld under
``on_unrecognized_output="fail_closed"``.

Notes: stdio is not supported (use the client-side ``authorized_mcp_tools()``); the default
provider reads the bearer token without verifying it (authentication is assumed upstream);
JWT decoding requires PyJWT.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from typing import Any

from fastmcp.exceptions import FastMCPError, ToolError
from fastmcp.server.dependencies import get_http_headers, get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import ToolResult
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from axiomatics_ai.core.authz_helpers import resolve_authz_context
from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider, EnforcementPoint
from axiomatics_ai.core.enforcement import (
    authorized_tool_invocation,
    authorized_tools,
    denial_message,
    post_process_tool_output,
)
from axiomatics_ai.core.pdp import ApprovalDecision, ApprovalRequest, PdpRequestError
from axiomatics_ai.core.post_invocation import PostInvocationContext
from axiomatics_ai.rag.wrap import ambient_identity_bound, authz_context

logger = logging.getLogger(__name__)

_APPROVE, _REJECT = "approve", "reject"


async def _ask_by_elicitation(fastmcp_context: Any, request: ApprovalRequest) -> ApprovalDecision:
    """Ask the MCP client's user to approve the call with an elicitation (approve or reject).

    The client must support elicitation: one that doesn't is a refusal nobody was asked about
    (``asked=False``); a decline or cancel is a rejection.
    """
    reason = f" ({request.reason})" if request.reason else ""
    message = f"Approve the call to '{request.tool_name}' with {request.args}?{reason}"
    try:
        result = await fastmcp_context.elicit(message, response_type=[_APPROVE, _REJECT])
    except (McpError, RuntimeError) as exc:
        # The client can't elicit (no elicitation support, or no live session).
        logger.warning("Could not ask for approval of %r by elicitation: %s", request.tool_name, exc)
        return ApprovalDecision(
            approved=False, note="the client could not be asked (elicitation)", asked=False,
        )
    except Exception as exc:  # noqa: BLE001 — any invalid answer (fails validation) is a rejection
        logger.warning("Invalid elicitation answer for %r: %s", request.tool_name, exc)
        return ApprovalDecision(approved=False, note="invalid elicitation answer")
    if getattr(result, "action", None) != "accept":
        return ApprovalDecision(approved=False, note=f"elicitation {getattr(result, 'action', 'failed')}")
    return ApprovalDecision(approved=getattr(result, "data", None) == _APPROVE)


# ── Ambient identity bridge ─────────────────────────────────────────────────────


@contextlib.contextmanager
def bind_ambient_identity(headers: dict[str, str] | None = None):
    """Bind the request's HTTP headers as the ambient authz identity for the block.

    The bridge that lets RAG stores wrapped with ``to_authorized_vectorstore``, which read the
    caller from the ``authz_context`` contextvar, resolve the caller inside a FastMCP tool,
    where nothing else populates it. Re-entrant: if an identity is already bound (e.g. a tool
    set its own with ``authz_context``), it yields without re-binding.

    Args:
        headers (dict[str, str], optional): Headers a caller already resolved; read from the
            current request otherwise.

    Yields:
        None: While the identity is bound.
    """
    if ambient_identity_bound():
        yield
        return
    if headers is None:
        headers = caller_headers()
    with authz_context(headers):
        yield


def caller_headers() -> dict[str, str]:
    """The current request's headers, ``Authorization`` included, for the context provider.

    A request with more than one ``Authorization`` header gets none: authentication in front
    (MCP's bearer auth) checks the first, a header dict keeps the last, so a caller could pass
    with their own token and be decided as whoever the second one names.

    Returns:
        dict[str, str]: The headers.
    """
    return without_duplicate_authorization(get_http_headers(include={"authorization"}) or {})


def _caused_by_pdp_error(exc: BaseException | None) -> bool:
    """Whether a PdpRequestError is ``exc`` or anywhere in its cause / context chain."""
    seen = 0
    while exc is not None and seen < 10:
        if isinstance(exc, PdpRequestError):
            return True
        exc, seen = exc.__cause__ or exc.__context__, seen + 1
    return False


def without_duplicate_authorization(headers: dict[str, str]) -> dict[str, str]:
    """``headers`` without ``Authorization`` if the current request carries more than one.

    Args:
        headers (dict[str, str]): The request's headers, as a dict (one value per name).

    Returns:
        dict[str, str]: The headers, anonymous when the request was ambiguous.
    """
    with contextlib.suppress(RuntimeError):   # no HTTP request (stdio, tests)
        if len(get_http_request().headers.getlist("authorization")) > 1:
            logger.warning("Request with more than one Authorization header: treated as anonymous")
            return {k: v for k, v in headers.items() if k.lower() != "authorization"}
    return headers


# ── Post-invocation helper ─────────────────────────────────────────────────────


def _apply_post_invocation_to_tool_result(
    result: ToolResult,
    *,
    post_ctx: PostInvocationContext,
    enforcer: Any,
    config: Any,
    surface: str,
) -> ToolResult:
    """Run the post-invocation strategy on a ``ToolResult`` and repack it.

    The ``structured_content`` and the text blocks are one MCP result (the text renders the
    structured data), so both go through the shared MCP handling and are rewritten together
    — a client reading ``structuredContent`` never sees records the text omits. Without
    ``structured_content``, each text block holding JSON data is filtered. Every other block
    (an image, an embedded resource) can restate that data, so it is treated like any block
    next to the data: reported, and withheld under ``on_unrecognized_output="fail_closed"``.

    Output the strategy cannot inspect is returned untouched and reported as a ``skipped``
    decision, or withheld under ``on_unrecognized_output="fail_closed"``.
    """
    blocks = [
        {"type": "text", "text": b.text} if isinstance(b, TextContent) else b.model_dump(mode="json")
        for b in result.content
    ]
    originals = {id(d): b for d, b in zip(blocks, result.content, strict=True)}
    payload = {"structuredContent": result.structured_content, "content": blocks}
    outcome = post_process_tool_output(
        payload, enforcer=enforcer, post_ctx=post_ctx, config=config, surface=surface,
    )
    if outcome.is_block:
        raise ToolError(outcome.reason)
    if outcome.content is None:
        return result
    return result.model_copy(update={
        "content": [
            # A block kept as it was goes back as the original object; text is rebuilt.
            originals.get(id(b)) or TextContent(type="text", text=b["text"])
            for b in outcome.content["content"]
        ],
        "structured_content": outcome.content["structuredContent"],
    })


# ── Middleware class ───────────────────────────────────────────────────────────


class FastMCPAuthZMiddleware(Middleware):
    """FastMCP middleware enforcing a PDP's decisions on tool listing and tool calls.

    ``on_list_tools`` filters the tool list (TOOL_VISIBILITY). ``on_call_tool`` blocks denied
    calls (BEFORE_TOOL, asking the client by elicitation when the policy wants approval), binds
    the caller's identity for wrapped RAG stores, and post-processes the result (AFTER_TOOL).
    Every attribute comes from the config's context provider, applied to the HTTP headers
    (``DefaultServerContextProvider`` by default: token claims, unverified). Only tools are
    guarded: resources and prompts pass through. stdio is not supported.

    Args:
        config (AuthZConfig, optional): What to enforce and where the PDP is;
            ``AuthZConfig()`` (nothing enforced) by default.
        surface (str, optional): The label of the decision events (the gateway passes
            ``"gateway_fastmcp"``).
    """

    # Default only — the constructor's `surface` argument overrides it into self._surface
    # (the gateway passes "gateway_fastmcp"). Named _DEFAULT_SURFACE rather than _SURFACE so
    # it cannot be confused with that instance attribute.
    _DEFAULT_SURFACE = "fastmcp"

    def __init__(
        self,
        config: AuthZConfig | None = None,
        *,
        surface: str | None = None,
    ) -> None:
        self.config = config or AuthZConfig()
        self._provider = self.config.context_provider or DefaultServerContextProvider()
        self._ads = self.config.build_pdp_client()
        self._cache = self.config.build_cache()
        # Tags emitted decision events; the gateway overrides it to "gateway_fastmcp".
        self._surface = surface or self._DEFAULT_SURFACE
        before_tool = self.config.enforcer(EnforcementPoint.BEFORE_TOOL)
        if before_tool is not None and before_tool.enrichers:
            logger.warning(
                "BEFORE_TOOL has enrichers (e.g. intent_enricher), but a server-side path never "
                "sees the user's messages: they are always unavailable here, and with "
                "on_unavailable='fail_closed' every tool call is denied. Enforce intent on the "
                "client-side agent instead."
            )

    def _get_headers(self) -> dict[str, str]:
        return caller_headers()

    async def on_request(self, context: MiddlewareContext, call_next) -> Any:
        """Keep the PDP's details on the server when it rejects a request.

        The error is logged here and the caller gets a generic one: its message names the PDP's
        URL and echoes its answer.

        Args:
            context (MiddlewareContext): The FastMCP request context.
            call_next (Callable): The rest of the chain (this middleware's handlers, the tool).

        Returns:
            Any: What the chain returns.

        Raises:
            ToolError: The PDP rejected a request (misconfiguration; details in the log).
        """
        try:
            return await call_next(context)
        except (PdpRequestError, FastMCPError) as exc:
            # FastMCP wraps an error raised inside a tool, resource or prompt (a wrapped RAG
            # store's PDP call) in its own error, with the PDP's error somewhere in the chain.
            if not _caused_by_pdp_error(exc):
                raise
            logger.exception("MCP %s: the PDP rejected the authorization request", context.method)
            generic = type(exc) if isinstance(exc, FastMCPError) else ToolError
            raise generic("Authorization is unavailable.") from None

    async def on_list_tools(self, context: MiddlewareContext, call_next) -> list:
        """Filter ``tools/list`` to the tools the caller may see.

        Args:
            context (MiddlewareContext): The FastMCP request context.
            call_next (Callable): Produces the full tool list.

        Returns:
            list: The permitted tools.
        """
        tools = await call_next(context)

        visibility_enforcer = self.config.enforcer(EnforcementPoint.TOOL_VISIBILITY)
        if visibility_enforcer is None:
            return tools

        headers = self._get_headers()
        tool_names = [t.name for t in tools]
        if not tool_names:
            return tools
        # The identity (a token may need verifying against fetched keys) and the PDP call
        # block, so they run on a worker thread: a slow PDP must not stall every session.
        permitted = await asyncio.to_thread(self._permitted_tools, headers, visibility_enforcer, tool_names)
        return [t for t in tools if t.name in permitted]

    def _permitted_tools(self, headers: dict, visibility_enforcer: Any, tool_names: list[str]) -> set[str]:
        ctx = resolve_authz_context(self._provider, headers)
        action = self._provider.get_action(visibility_enforcer.action_name, headers)
        return authorized_tools(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._surface,
            subject=ctx.subject,
            action=action,
            tool_names=tool_names,
            environment=ctx.environment,
        )

    async def on_call_tool(self, context: MiddlewareContext, call_next) -> Any:
        """Authorize ``tools/call``, run the tool with the caller's identity bound, and post-process.

        Args:
            context (MiddlewareContext): The FastMCP request context.
            call_next (Callable): Runs the tool.

        Returns:
            ToolResult: The (filtered) result.

        Raises:
            ToolError: The call is denied or not approved, or its output is withheld.
        """
        headers = self._get_headers()
        ctx = await asyncio.to_thread(resolve_authz_context, self._provider, headers)
        tool_name = context.message.name
        arguments = context.message.arguments or {}

        invocation_enforcer = self.config.enforcer(EnforcementPoint.BEFORE_TOOL)
        if invocation_enforcer is not None:
            action = self._provider.get_action(invocation_enforcer.action_name, headers)
            resource = self._provider.get_resource(
                tool_name, headers, args=arguments,
                resource_name_attribute=self.config.attribute_names.tool,
            )
            fastmcp_context = getattr(context, "fastmcp_context", None)
            # MCP has no tool-call id; one per call ties the elicitation's events together.
            call_id = uuid.uuid4().hex

            def decide(approval: ApprovalDecision | None = None):
                return authorized_tool_invocation(
                    pdp=self._ads,
                    cache=self._cache,
                    config=self.config,
                    surface=self._surface,
                    subject=ctx.subject,
                    action=action,
                    resource=resource,
                    environment=ctx.environment,
                    user_messages=[],  # a server never sees the conversation (see __init__)
                    approval=approval,
                    can_interrupt=fastmcp_context is not None,
                    args=arguments,
                    call_id=call_id,
                )

            decision = await asyncio.to_thread(decide)
            if decision.pending is not None:
                answer = await _ask_by_elicitation(fastmcp_context, decision.pending)
                decision = await asyncio.to_thread(decide, answer)
            if not decision.permitted:
                raise ToolError(denial_message(tool_name, decision))

        # Bind the caller identity into the ambient authz context for the duration of the
        # tool call, so a RAG store wrapped with to_authorized_vectorstore filters with no
        # per-tool wiring — the single-process topology needs only this middleware.
        with bind_ambient_identity(headers):
            result = await call_next(context)

        post_enforcer = self.config.enforcer(EnforcementPoint.AFTER_TOOL)
        if post_enforcer is None or post_enforcer.strategy is None:
            return result

        action_name = post_enforcer.action_overrides.get(tool_name, post_enforcer.action_name)
        action = self._provider.get_action(action_name, headers)
        post_ctx = PostInvocationContext(
            pdp_client=self._ads,
            subject=ctx.subject,
            environment=ctx.environment,
            tool_name=tool_name,
            args=arguments,
            resource=self._provider.get_resource(
                tool_name, headers, args=arguments,
                resource_name_attribute=self.config.attribute_names.tool,
            ),
            action=action,
            fail_open=self.config.connection.fail_open,
        )
        return await asyncio.to_thread(
            _apply_post_invocation_to_tool_result,
            result,
            post_ctx=post_ctx,
            enforcer=post_enforcer,
            config=self.config,
            surface=self._surface,
        )
