"""The gateway: a reverse proxy that authorizes A2A and MCP traffic at the wire.

It sits in front of the agents and servers you run; one gateway can front many A2A agents
and many MCP servers.

**Experimental** — not production grade. The gateway will move to its own package; import
it from ``axiomatics_ai.gateway`` (it is not re-exported from ``axiomatics_ai``).

    from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
    from axiomatics_ai.gateway import GatewayConfig, Route, build_gateway

    cfg = GatewayConfig(
        routes=[
            Route("/billing", "http://billing:9000", protocol="a2a", agent_name="billing-agent"),
            Route("/tools",   "http://mcp:8000/mcp", protocol="mcp"),
        ],
        authz=AuthZConfig(
            connection=PDPConnection(base_url="http://localhost:8080"),
            enforcement=standard_tool_enforcement(),
        ),
    )
    app = build_gateway(cfg)     # an ASGI app — serve with uvicorn

See README.md for the architecture (per-route enforcing sub-apps reusing the existing
A2A / FastMCP middleware).
"""
from axiomatics_ai._extras import requires_extra

with requires_extra("gateway", "axiomatics_ai.gateway"):
    from axiomatics_ai.gateway.app import build_gateway
    from axiomatics_ai.gateway.config import GatewayConfig, Route

__all__ = ["GatewayConfig", "Route", "build_gateway"]
