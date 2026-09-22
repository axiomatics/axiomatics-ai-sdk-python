"""
The gateway — one reverse proxy fronting your A2A agents and MCP servers.
(Experimental — not production grade; the gateway will move to its own package.)

Routes are DATA, not hardcoded: they come from a JSON spec so the same module runs locally
or in a container fronting any set of servers. Resolution order:
  1. GATEWAY_ROUTES_FILE — path to a JSON file (list of route objects), or
  2. GATEWAY_ROUTES      — inline JSON (list of route objects), or
  3. the demo defaults below (/billing → :9211 A2A, /tools → :9212 MCP).

Each route object: {"path_prefix", "target_url", "protocol": "a2a"|"mcp", "agent_name"?}.
Example (container, reaching host services):
  GATEWAY_ROUTES='[
    {"path_prefix":"/billing","target_url":"http://host.docker.internal:9211","protocol":"a2a","agent_name":"billing-agent"},
    {"path_prefix":"/tools","target_url":"http://host.docker.internal:9212/mcp","protocol":"mcp"}
  ]'

Clients talk only to the gateway; it rewrites A2A agent-card urls to itself. The MCP route
is configured visibility-only here (invocation/post-invocation disabled) to keep the demo
policy focused on tool filtering.

Config in .env / env: PDP_BASE_URL, GATEWAY_ROUTES / GATEWAY_ROUTES_FILE, GATEWAY_PUBLIC_URL.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from axiomatics_ai import AuthZConfig
from axiomatics_ai.gateway import GatewayConfig, Route, build_gateway
from axiomatics_ai import PDPConnection, standard_tool_enforcement

PORT = 9210

_DEMO_ROUTES = [
    {"path_prefix": "/billing", "target_url": "http://127.0.0.1:9211", "protocol": "a2a", "agent_name": "billing-agent"},
    {"path_prefix": "/tools", "target_url": "http://127.0.0.1:9212/mcp", "protocol": "mcp"},
]


def _load_route_specs() -> list[dict]:
    if path := os.getenv("GATEWAY_ROUTES_FILE"):
        return json.loads(Path(path).read_text())
    if raw := os.getenv("GATEWAY_ROUTES"):
        return json.loads(raw)
    return _DEMO_ROUTES


config = GatewayConfig(
    routes=[Route(**spec) for spec in _load_route_specs()],
    authz=AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),  # MCP route: visibility only
    ),
    # The demo clients send unsigned tokens, so this gateway trusts them unverified. A real
    # gateway sets authz.context_provider to AgentIdentityContextProvider(JwtVerification(...))
    # instead — build_gateway refuses to start with unverified tokens otherwise.
    allow_unverified_tokens=True,
    public_url=os.getenv("GATEWAY_PUBLIC_URL"),   # e.g. https://gateway.example.com; else the Host header
)

app = build_gateway(config)

if __name__ == "__main__":
    print(f"Gateway on :{PORT}  (routes: {[r.path_prefix for r in config.routes]})")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
