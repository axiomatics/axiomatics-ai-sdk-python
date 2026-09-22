"""Per-tool authorization and post-invocation via function decorators.

The tool **always executes**. If a ``post_processor`` is provided it is always
called with the result and a ``PostInvocationContext``. The post_processor is
fully responsible for any PDP interaction and transformation:

- **No post_processor** — tool runs and result is returned as-is. PDP is not
  called.
- **post_processor provided** — called unconditionally. Use ``ColumnMaskStrategy``
  or ``RecordFilterStrategy`` for PDP-driven masking / filtering, or provide
  any async callable for custom logic.

Setup::

    from axiomatics_ai.tool_decorator import configure_authz_decorator, protected_tool
    from axiomatics_ai import AuthZConfig, ColumnMaskStrategy

    configure_authz_decorator(config=AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080")))

    @mcp.tool()
    @protected_tool(post_processor=ColumnMaskStrategy("dob", "email"))
    def get_patient_records(patient_id: str) -> list[dict]:
        # alice (Permit) → full records
        # bob   (Deny)   → records with dob and email masked
        ...

Notes:
-----
- **Tool visibility is not supported by this path.** Use ``FastMCPAuthZMiddleware``
  alongside this path if you also need to hide tools from the LLM's tool list.
- **Tool invocation blocking is not provided by this path.** Combine with
  ``FastMCPAuthZMiddleware`` (with a ``BEFORE_TOOL`` Enforcer) if you need
  to hard-block tool calls before execution.
- Identity is extracted from the ``Authorization: Bearer`` JWT header via
  ``get_http_headers(include={"authorization"})``. HTTP transport required; stdio is not supported.
- When PDP is unreachable, ``fail_open`` in ``AuthZConfig`` is forwarded to the
  post_processor via ``PostInvocationContext``.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Awaitable, Callable

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers as _get_http_headers

from axiomatics_ai.core.pdp import PdpClient, PdpRequestError, strip_none
from axiomatics_ai.fastmcp.fastmcp_middleware import without_duplicate_authorization
from axiomatics_ai.core.cache import AuthzDecisionCacheWithTTL
from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider, EnforcementPoint
from axiomatics_ai.core.enforcement import apply_post_invocation
from axiomatics_ai.core.post_invocation import PostInvocationContext

_SURFACE = "tool_decorator"

logger = logging.getLogger(__name__)


# ── Module-level config ────────────────────────────────────────────────────────


@dataclass
class _Config:
    pdp: PdpClient
    provider: DefaultServerContextProvider
    config: AuthZConfig
    cache: AuthzDecisionCacheWithTTL


_decorator_config: _Config | None = None


def configure_authz_decorator(
    *,
    config: AuthZConfig | None = None,
) -> None:
    """Set up ``@protected_tool`` once, at server start-up, before any protected tool runs.

    Args:
        config (AuthZConfig, optional): The PDP connection and context provider (the
            ``AFTER_TOOL`` Enforcer's action name is used when one is configured).
    """
    global _decorator_config
    cfg = config or AuthZConfig()
    _decorator_config = _Config(
        pdp=cfg.build_pdp_client(),
        provider=cfg.context_provider or DefaultServerContextProvider(),
        config=cfg,
        cache=cfg.build_cache(),
    )


# ── Decorator ─────────────────────────────────────────────────────────────────


def protected_tool(
    post_processor: Callable[[Any, PostInvocationContext], Awaitable[Any]] | None = None,
) -> Callable:
    """Decorate a tool so its result goes through a post-invocation strategy.

    The tool always runs; the decorator makes no invocation check (combine with
    ``FastMCPAuthZMiddleware`` for that). The post-processor gets the tool's own return value
    and a ``PostInvocationContext`` with the caller's identity (from the request headers) and
    the call's arguments, positional ones included. Example::

        @mcp.tool()
        @protected_tool(post_processor=ColumnMaskStrategy("ssn", "dob"))
        def get_employee_records(department: str) -> list[dict]:
            ...

    Args:
        post_processor (PostInvocationStrategy | Callable, optional): A strategy (an object
            with ``apply(result, context)``, e.g. ``ColumnMaskStrategy`` or
            ``RecordFilterStrategy``) or an async callable ``(result, context) -> Any``. None
            returns the result as it is, without asking the PDP.

    Returns:
        Callable: The decorator.

    Raises:
        RuntimeError: When a decorated tool runs before ``configure_authz_decorator()``.
    """

    def decorator(fn: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(fn)
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            result = await _call_tool(fn, is_async=is_async, args=args, kwargs=kwargs)
            if post_processor is None:
                return result  # no post_processor — skip PDP entirely
            cfg, post_ctx = _build_post_invocation_context(
                fn.__name__, _call_arguments(signature, args, kwargs),
            )
            try:
                return await _apply_post_processor(post_processor, result, post_ctx, cfg)
            except PdpRequestError:
                # Its message names the PDP's URL and echoes its answer: keep that in the log.
                logger.exception("protected_tool [%s]: the PDP rejected the request", fn.__name__)
                raise ToolError("Authorization is unavailable.") from None

        return wrapper

    return decorator


def _call_arguments(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    """Get the call's arguments by parameter name, positional ones included.

    So the policy sees the same arguments however the tool was called.
    """
    try:
        return dict(signature.bind_partial(*args, **kwargs).arguments)
    except TypeError:
        return dict(kwargs)  # arguments the signature rejects: the tool call itself fails


async def _call_tool(fn: Callable, *, is_async: bool, args: tuple, kwargs: dict) -> Any:
    """Run the decorated tool: awaited when it is a coroutine function, else on a worker thread.

    ``asyncio.to_thread`` copies the context, so the caller's identity (the ambient authz
    context a wrapped RAG store reads, FastMCP's request headers) reaches the tool.
    """
    if is_async:
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


def _build_post_invocation_context(tool_name: str, tool_args: dict):
    """Resolve the caller's identity and build this call's ``PostInvocationContext``.

    Returns ``(cfg, post_ctx)``. Raises if ``configure_authz_decorator()`` has not been called.
    """
    if _decorator_config is None:
        raise RuntimeError(
            "configure_authz_decorator() must be called before using @protected_tool."
        )
    cfg = _decorator_config
    try:
        headers = without_duplicate_authorization(_get_http_headers(include={"authorization"}) or {})
    except RuntimeError:
        headers = {}
        logger.warning("protected_tool [%s]: no HTTP context available", tool_name)

    # @protected_tool opts in per-decorator (not via config.enforcement), so read the
    # AFTER_TOOL enforcer if one is configured, else fall back to standard defaults.
    after_tool = cfg.config.enforcer(EnforcementPoint.AFTER_TOOL)
    action_name = after_tool.action_name if after_tool else "post_invocation"

    post_ctx = PostInvocationContext(
        pdp_client=cfg.pdp,
        subject=strip_none(cfg.provider.get_subject(headers)),
        environment=strip_none(cfg.provider.get_environment(headers)),
        tool_name=tool_name,
        args=tool_args,
        resource=cfg.provider.get_resource(
            tool_name, headers, args=tool_args,
            resource_name_attribute=cfg.config.attribute_names.tool,
        ),
        action=cfg.provider.get_action(action_name, headers),
        fail_open=cfg.config.connection.fail_open,
    )
    return cfg, post_ctx


async def _apply_post_processor(post_processor: Any, result: Any, post_ctx: Any, cfg: Any) -> Any:
    """Hand the tool result to the post-processor.

    Strategies expose a sync ``apply()`` and run through the shared helper so they emit a
    post-invocation decision event; a plain async callable is opaque, so it is awaited as-is.
    """
    if hasattr(post_processor, "apply"):
        return await asyncio.to_thread(
            lambda: apply_post_invocation(
                strategy=post_processor,
                records=result,
                post_ctx=post_ctx,
                config=cfg.config,
                surface=_SURFACE,
            ),
        )
    return await post_processor(result, post_ctx)
