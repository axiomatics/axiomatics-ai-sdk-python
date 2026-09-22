"""Gateway configuration: the routes and the shared authorization settings.

**Experimental**: not production grade; the gateway will move to its own package.

``GatewayConfig`` is the routing layer: which target servers this gateway sits in front of
and what protocol each speaks. Its ``authz`` field is the ``AuthZConfig`` that supplies the
authorization core (PDP endpoint, action names, fail-open, cache, context provider) shared
by every route. A gateway can front **many** MCP servers and **many** A2A servers at once —
one ``Route`` per target server.

    GatewayConfig(
        routes=[Route(...), Route(...)],
        authz=AuthZConfig(connection=PDPConnection(base_url=...), enforcement=standard_tool_enforcement()),
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from axiomatics_ai.core.config import AuthZConfig

Protocol = Literal["mcp", "a2a"]  # "openai" planned for a later cut


@dataclass
class Route:
    """One target server the gateway protects, mounted at ``path_prefix``.

    There is intentionally no MCP ``server_name``: for MCP the PDP resource is already the tool
    name. Disambiguating same-named tools across servers is deferred (a composite
    ``{server}___{tool}`` name is the plan).

    Attributes:
        path_prefix (str): The mount point, e.g. ``/billing``; clients call
            ``http://gateway{path_prefix}/...``.
        target_url (str): The real server: for ``a2a`` the agent's base URL, for ``mcp`` the
            MCP endpoint URL (e.g. ``http://host:8000/mcp``).
        protocol (str): ``"a2a"`` or ``"mcp"``; selects the enforcement mechanism.
        agent_name (str | None): A2A only: the PDP resource label for the agent. Set it: without
            it the resource falls back to the mount sub-path.
        audience (str | list[str] | None): The audience this route's tokens must carry, when the
            target is its own resource server. Needs a verifying context provider.
        forward_authorization (bool): Forward the caller's ``Authorization`` header (and, on
            A2A routes, ``Cookie``) to the target. Off by default: passing a token on to another
            service is an antipattern (AIMS §10.8, MCP's no-token-passthrough rule).
        mcp_resources_and_prompts (bool): MCP only: serve the target's resources and prompts.
            The gateway authorizes tool calls only, so they would pass with no policy check.
            Off by default: they are hidden, and reading one is refused.
    """

    path_prefix: str
    target_url: str
    protocol: Protocol
    agent_name: str | None = None
    audience: str | list[str] | None = None
    forward_authorization: bool = False
    mcp_resources_and_prompts: bool = False


@dataclass
class GatewayConfig:
    """The gateway's route table and the authorization settings every route shares.

    Pass it to ``build_gateway``. ``authz.context_provider`` must verify the callers' tokens:
    the gateway is usually the edge, where a forged token must not pass.

    Attributes:
        routes (list[Route]): The target servers.
        authz (AuthZConfig): The shared authorization config (PDP, enforcement, provider).
        allow_unverified_tokens (bool): Accept the default provider, which only decodes tokens,
            for a gateway behind an ingress that authenticates first. Off by default.
        public_url (str | None): The gateway's address as clients reach it (e.g.
            ``https://gateway.example.com``), used to rewrite A2A agent cards; without it the
            request's ``Host`` header, which a client controls, is used.
    """

    routes: list[Route]
    authz: AuthZConfig = field(default_factory=AuthZConfig)
    allow_unverified_tokens: bool = False
    public_url: str | None = None
