"""Client-side A2A authorization — ``A2AClientAuthZ``.

An ``httpx.Auth`` that authorizes every **outbound** A2A ``message/send`` before it
leaves the process. Despite plugging into httpx's generically-named ``auth=`` slot, this
is authoriZation (PDP Permit/Deny), not authentication. Because it hooks httpx (below the
a2a-sdk client) it works with any framework that lets you influence the httpx client —
which is all of them:

    authz = A2AClientAuthZ(config=AuthZConfig(...), context={"username": "alice", "tenant_id": "acme"})

    # Google ADK:
    RemoteA2aAgent(..., httpx_client=httpx.AsyncClient(auth=authz))
    # AWS Strands:
    A2AClientToolProvider(..., httpx_client_args={"auth": authz})
    # self-wired a2a-sdk:
    ClientConfig(httpx_client=httpx.AsyncClient(auth=authz))

On a Deny the request is not sent — ``A2AAuthorizationError`` is raised. Non-invocation
traffic (agent-card fetches, ``tasks/get``, etc.) passes through untouched; only
``message/send`` / ``SendMessage`` (and their streaming variants) are authorized.

The four XACML categories
-------------------------
- **subject** and **environment** are derived by the provider from ``context`` (the same
  runtime object ``agent.invoke(context=...)`` takes — the default provider reads
  ``username``/``user_id``/``tenant_id`` into subject and ``session_id`` into environment;
  put region/session/etc. here and a custom provider's ``get_environment`` picks them up).
- **action** comes from the ``A2A_INVOCATION`` Enforcer's ``action_name`` (default
  ``"a2a_invocation"``; override via a custom provider's ``get_action``).
- **resource** is the target agent identity plus the RPC method and any skill hint parsed
  off the wire (override via a custom provider's ``get_resource``).

Identity — the per-invocation problem
--------------------------------------
The context *provider* is fixed at construction, but the *context* itself may vary per
call. ``httpx.Auth`` only ever receives the outbound ``request``; it has no seam to accept
application-level per-call context (the a2a-sdk ``ClientCallContext`` does not reach this
layer). So supply ``context`` in one of two ways:

- ``context=`` — a static object. Use when one client serves one caller (the common case).
- ``context_source=`` — a zero-arg callable resolved on *every* request. Use for a
  multi-user process (e.g. a server calling agents on behalf of different users): set a
  ``contextvars.ContextVar`` around each agent turn and have the callable read it.

(The server-side ``A2AAuthZMiddleware`` has no such limitation — the context arrives in the
request headers per invocation.)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Callable

import httpx

from axiomatics_ai.a2a import _wire
from axiomatics_ai.a2a._enforce import A2AAuthorizationError, authorize_invocation
from axiomatics_ai.core.config import AuthZConfig, DefaultClientContextProvider

logger = logging.getLogger(__name__)


class A2AClientAuthZ(httpx.Auth):
    """An ``httpx.Auth`` that blocks unauthorized outbound A2A message sends.

    Only ``message/send`` / ``SendMessage`` (and their streaming variants) on the JSON-RPC
    binding are checked; other requests pass through. On a Deny the request is not sent.

    Args:
        config (AuthZConfig, optional): The PDP connection, the A2A_INVOCATION action name,
            fail_open and cache settings.
        context (Any, optional): A static runtime context for the provider's ``get_subject`` /
            ``get_environment`` (default keys: ``username``, ``user_id``, ``tenant_id``,
            ``session_id``). For a single-caller client.
        context_source (Callable[[], Any], optional): Called on every request for the context,
            e.g. reading a ``contextvars.ContextVar`` your app sets around each call. For
            per-invocation identity; takes precedence over ``context``.
        agent_name (str, optional): The target agent's identity for the PDP; the request URL's
            host by default.
        include_message_text (bool): Add the outbound message text to the resource, so a
            policy can match on content. Off by default (privacy, cache-key churn).
    """

    # httpx must buffer the request body before calling auth_flow so we can read it.
    requires_request_body = True

    def __init__(
        self,
        config: AuthZConfig | None = None,
        *,
        context: Any | None = None,
        context_source: Callable[[], Any] | None = None,
        agent_name: str | None = None,
        include_message_text: bool = False,
    ) -> None:
        self.config = config or AuthZConfig()
        self._provider = self.config.context_provider or DefaultClientContextProvider()
        self._context = context if context is not None else {}
        self._context_source = context_source
        self._agent_name = agent_name
        self._include_message_text = include_message_text
        self._ads = self.config.build_pdp_client()
        self._cache = self.config.build_cache()

    def _resolve_context(self) -> Any:
        """Resolve the runtime context for this request (per-invocation if ``context_source`` set)."""
        return self._context_source() if self._context_source is not None else self._context

    def auth_flow(self, request: httpx.Request):
        """Authorize the request (sync client), then send it.

        Args:
            request (httpx.Request): The outbound request.

        Yields:
            httpx.Request: The request, if permitted.

        Raises:
            A2AAuthorizationError: The message send is denied.
        """
        self._authorize(request)
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        # With an AsyncClient: the PDP call blocks, so it runs on a worker thread rather than
        # stalling the event loop.
        """Authorize the request (async client) on a worker thread, then send it.

        Args:
            request (httpx.Request): The outbound request.

        Yields:
            httpx.Request: The request, if permitted.

        Raises:
            A2AAuthorizationError: The message send is denied.
        """
        await asyncio.to_thread(self._authorize, request)
        yield request

    def _authorize(self, request: httpx.Request) -> None:
        """Raise ``A2AAuthorizationError`` when an outbound message send is denied."""
        rpc = _wire.parse_jsonrpc(request.content) if request.method == "POST" else None
        if rpc is None or not _wire.is_invocation_method(rpc.get("method")):
            return

        params = rpc.get("params") if isinstance(rpc.get("params"), dict) else {}
        agent_name = self._agent_name or request.url.host
        message_text = _wire.extract_message_text(params) if self._include_message_text else None

        permitted = authorize_invocation(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            provider=self._provider,
            context=self._resolve_context(),
            agent_name=agent_name,
            method=rpc["method"],
            skill_hint=_wire.extract_skill_hint(params),
            message_text=message_text,
            surface="a2a_client",
        )
        logger.debug(
            "A2A outbound %s → %s: %s",
            rpc["method"], agent_name, "Permit" if permitted else "Deny",
        )
        if not permitted:
            raise A2AAuthorizationError(
                f"Access denied: not authorized to send an A2A message to '{agent_name}'."
            )
