"""``build_gateway``: assemble the gateway ASGI app from a ``GatewayConfig``.

The config holds the routes and the shared ``AuthZConfig``. **Experimental**: not
production grade; the gateway will move to its own package.

The gateway is a Starlette router that mounts one *enforcing* sub-app per route. Both
protocols produce an ASGI app that is mounted identically (``Mount(prefix, app=sub)``);
they differ only in how that app is built, because their enforcement lives at different
layers:

  - A2A route → ``A2AAuthZMiddleware( HttpForwarder(target_url) )``
        Enforcement is an ASGI/HTTP-wire middleware (request-phase invocation check); since
        no off-the-shelf A2A proxy exists, we supply the transport (HttpForwarder). A
        middleware wrapping an app is itself an ASGI app.

  - MCP route → ``FastMCP.as_proxy(target_url)`` + ``FastMCPAuthZMiddleware`` → ``.http_app()``
        FastMCP already bundles the proxy/transport; enforcement is added as a *FastMCP*
        middleware (operates on parsed MCP messages — needed for response-phase tools/list
        filtering and post-invocation). ``.http_app()`` is the resulting ASGI app.

Both reuse existing, tested middleware — the gateway adds no new enforcement code. Identity
comes from the caller's headers (``Authorization: Bearer`` JWT), through a context provider
that verifies the token (see ``GatewayConfig``). One gateway can carry many MCP routes and
many A2A routes.

    cfg = GatewayConfig(
        routes=[Route(...), ...],
        authz=AuthZConfig(connection=PDPConnection(base_url=...), enforcement=standard_tool_enforcement()),
    )
    app = build_gateway(cfg)
"""
from __future__ import annotations

import contextlib
import copy
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Mount

from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider
from axiomatics_ai.gateway.config import GatewayConfig, Route
from axiomatics_ai.gateway.forwarder import HttpForwarder

# Protocol middleware is imported lazily inside each builder so a gateway only pulls in the
# dependencies for the protocols it actually routes (e.g. an A2A-only gateway needs no
# fastmcp, an MCP-only gateway needs no a2a).


def _route_authz(route: Route, authz: AuthZConfig) -> AuthZConfig:
    """Get the config one route enforces with.

    The shared one, or, when the route is its own resource server (``route.audience``), a copy
    whose provider verifies tokens for that audience. The copy shares the parent's cache
    registry, so ``invalidate_decisions`` on the parent still reaches this route's cache.
    """
    if route.audience is None:
        return authz
    for_audience = getattr(authz.context_provider, "for_audience", None)
    if for_audience is None:
        raise ValueError(
            f"Route {route.path_prefix!r} sets audience, which needs a verifying context provider "
            "(AgentIdentityContextProvider with a JwtVerification) on the gateway's AuthZConfig."
        )
    route_config = copy.copy(authz)          # shallow: shares _state (caches), hooks, connection
    route_config.context_provider = for_audience(route.audience)
    return route_config


def _mcp_upstream(route: Route):
    """Get what the FastMCP proxy connects to.

    FastMCP's proxy client forwards the caller's ``Authorization`` header upstream by default;
    unless the route opts in, a transport that never does is used. A non-URL target (e.g. an
    in-process FastMCP server) has no HTTP hop, so there is nothing to strip.
    """
    target = route.target_url
    if route.forward_authorization or not (isinstance(target, str) and target.startswith("http")):
        return target
    from fastmcp.client.transports.http import StreamableHttpTransport

    class _NoTokenPassthroughTransport(StreamableHttpTransport):
        forward_incoming_headers = property(lambda self: False, lambda self, value: None)

    return _NoTokenPassthroughTransport(target)


def _build_mcp_subapp(route: Route, authz: AuthZConfig):
    from fastmcp.server import create_proxy

    from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware

    proxy = create_proxy(_mcp_upstream(route))  # FastMCP proxy → the backend MCP server
    proxy.add_middleware(
        FastMCPAuthZMiddleware(_route_authz(route, authz), surface="gateway_fastmcp")
    )
    if not route.mcp_resources_and_prompts:
        proxy.add_middleware(_tools_only_middleware())
    return proxy.http_app(path="/")  # serve MCP at the root of this route's mount


def _tools_only_middleware():
    """A FastMCP middleware that hides the target's resources and prompts, and refuses them.

    The gateway authorizes tool calls only; a resource read or a prompt would reach the target
    with no policy check.
    """
    from fastmcp.exceptions import PromptError, ResourceError
    from fastmcp.server.middleware import Middleware

    class _ToolsOnly(Middleware):
        async def on_list_resources(self, context, call_next):
            return []

        async def on_list_resource_templates(self, context, call_next):
            return []

        async def on_list_prompts(self, context, call_next):
            return []

        async def on_read_resource(self, context, call_next):
            raise ResourceError("This gateway route serves tools only.")

        async def on_get_prompt(self, context, call_next):
            raise PromptError("This gateway route serves tools only.")

    return _ToolsOnly()


def _build_a2a_subapp(route: Route, authz: AuthZConfig, forwarder: HttpForwarder, public_url: str | None):
    from axiomatics_ai.a2a import A2AAuthZMiddleware
    from axiomatics_ai.gateway.rewrite import A2ACardRewriter

    # enforce (middleware) → rewrite the card url to the gateway (so clients route back
    # through us) → forward to the target.
    return A2AAuthZMiddleware(
        A2ACardRewriter(forwarder, public_url=public_url),
        _route_authz(route, authz),
        agent_name=route.agent_name,
        surface="gateway_a2a_server",
    )


def _require_verified_tokens(config: GatewayConfig) -> None:
    """Refuse a gateway whose identity comes from tokens it never verifies.

    Unless the config says authentication happens in front of it.
    """
    provider = config.authz.context_provider
    if config.allow_unverified_tokens or (provider is not None and type(provider) is not DefaultServerContextProvider):
        return
    raise ValueError(
        "build_gateway: the gateway reads identity from bearer tokens, and the default context "
        "provider only decodes them, so a forged token would be trusted. Set "
        "authz.context_provider to a verifying provider (AgentIdentityContextProvider with a "
        "JwtVerification), or GatewayConfig(allow_unverified_tokens=True) if authentication "
        "already happens in front of the gateway."
    )


def build_gateway(config: GatewayConfig) -> Starlette:
    """Build the gateway ASGI app from a ``GatewayConfig``.

    Mounts one enforcing sub-app per route: A2A routes get ``A2AAuthZMiddleware`` in front of a
    forwarder, MCP routes a FastMCP proxy with ``FastMCPAuthZMiddleware``. The app's lifespan
    runs the MCP sub-apps' lifespans and closes the forwarders' HTTP clients on shutdown.

    Args:
        config (GatewayConfig): The routes and the shared ``AuthZConfig``.

    Returns:
        Starlette: The gateway app.

    Raises:
        ValueError: There are no routes, a route's audience needs a verifying provider, or
            identity would come from unverified tokens (see ``GatewayConfig``).
    """
    if not config.routes:
        raise ValueError("build_gateway requires GatewayConfig(routes=[...]) with at least one route")

    mounts: list[Mount] = []
    mcp_apps: list[Any] = []  # FastMCP http_apps whose lifespans the parent must run
    forwarders: list[HttpForwarder] = []  # their HTTP clients are closed on shutdown

    for route in config.routes:
        if route.protocol == "mcp":
            sub = _build_mcp_subapp(route, config.authz)
            mcp_apps.append(sub)
        elif route.protocol == "a2a":
            forwarder = HttpForwarder(route.target_url, forward_authorization=route.forward_authorization)
            forwarders.append(forwarder)
            sub = _build_a2a_subapp(route, config.authz, forwarder, config.public_url)
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"Unsupported protocol for route {route.path_prefix!r}: {route.protocol!r}")
        mounts.append(Mount(route.path_prefix, app=sub))
    # After the routes, so a route's own problem (an audience without a verifying provider)
    # is the error reported.
    _require_verified_tokens(config)

    # A mounted FastMCP app has its own lifespan (its session manager). Starlette does not
    # run a mounted app's lifespan automatically, so we chain them into the parent's.
    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with contextlib.AsyncExitStack() as stack:
            for forwarder in forwarders:
                stack.push_async_callback(forwarder.aclose)
            for mcp_app in mcp_apps:
                await stack.enter_async_context(mcp_app.lifespan(app))
            yield

    return Starlette(routes=mounts, lifespan=lifespan)
