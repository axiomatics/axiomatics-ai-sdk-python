# gateway — a reverse proxy that authorizes A2A and MCP traffic

> **Preview.** The gateway is not production grade yet (see
> [Scope](#scope--not-yet-supported)), and it will move to its own package. Import it from `axiomatics_ai.gateway`; it is not re-exported from `axiomatics_ai`.

The gateway is the out-of-process form factor of the SDK's wire-level enforcement. It is a
reverse proxy you put in front of the agents and MCP servers you run, so authorization applies
with no changes to those servers or their clients. One gateway can front many A2A agents and
many MCP servers at once, making it the single enforcement point for a fleet. It enforces
`TOOL_VISIBILITY`, `BEFORE_TOOL` and `AFTER_TOOL` on MCP routes and `A2A_INVOCATION` on A2A
routes. Identity comes from the caller's request headers (`Authorization: Bearer <JWT>`),
verified via `AgentIdentityContextProvider` — `build_gateway` refuses the unverified default
unless you set `allow_unverified_tokens=True`. The caller's token is not forwarded to the
targets unless a route opts in.

The gateway reuses the exact middleware from the in-process paths, so it adds no new
enforcement logic:

- **MCP route** → `create_proxy(target)` (FastMCP) + `FastMCPAuthZMiddleware`. Tool
  visibility, invocation and post-invocation, at the MCP-message layer. See [fastmcp](fastmcp.md).
- **A2A route** → `A2AAuthZMiddleware` + `A2ACardRewriter` in front of an `HttpForwarder`.
  Invocation enforcement at the HTTP/JSON-RPC wire, card-url rewriting so clients route back
  through the gateway, then a plain reverse-proxy hop to the target. See [a2a](a2a.md).

## Attach

`build_gateway(config)` takes a `GatewayConfig` and returns a Starlette ASGI app. Serve it
with uvicorn.

```python
from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.gateway import GatewayConfig, Route, build_gateway

cfg = GatewayConfig(
    routes=[
        Route("/billing", "http://billing:9000",   protocol="a2a", agent_name="billing-agent"),
        Route("/support", "http://support:9000",   protocol="a2a", agent_name="support-agent"),
        Route("/docs",    "http://docs:8000/mcp",  protocol="mcp"),
        Route("/crm",     "http://crm:8000/mcp",   protocol="mcp"),
    ],
    authz=AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),
    ),
)

app = build_gateway(cfg)          # a Starlette ASGI app — serve with uvicorn
```

Clients point at the gateway: `http://gateway/billing/...` (A2A) or `http://gateway/docs`
(MCP). Each request is authorized against the PDP before being forwarded.

`GatewayConfig` holds the routes and, in `authz`, the `AuthZConfig` that supplies the shared
authorization settings for every route: PDP endpoint, action names, fail-open, cache, and
context provider. `authz` defaults to `AuthZConfig()`. `build_gateway` raises `ValueError` if
there are no routes, or if identity would come from unverified tokens:

| `GatewayConfig` field | Meaning |
|---|---|
| `routes` | The `Route`s below. |
| `authz` | The shared `AuthZConfig`. Its `context_provider` must verify tokens (`AgentIdentityContextProvider` with a `JwtVerification`, or your own provider). |
| `allow_unverified_tokens` | `True` accepts the default provider, which only decodes tokens — for a gateway behind an ingress that authenticates first. Default `False`. |
| `public_url` | The gateway's address as clients reach it (e.g. `https://gateway.example.com`), used to rewrite A2A agent cards. Default `None`: taken from the request's `Host` header, which the client controls. |

| `Route` field | Meaning |
|---|---|
| `path_prefix` | Mount point on the gateway (e.g. `/billing`). Clients call `http://gateway{path_prefix}/...`. |
| `target_url` | The real server to forward to. For `a2a` the agent's base URL; for `mcp` the MCP endpoint URL (e.g. `http://host:8000/mcp`). |
| `protocol` | `"a2a"` or `"mcp"`. Selects the enforcement mechanism for this route. |
| `agent_name` | A2A only. The PDP resource label (policy-facing agent identity) used in the `a2a_invocation` request. Required for A2A routes; without it the resource would fall back to the meaningless mount sub-path. |
| `audience` | Tokens for this route must carry this audience, when the target is its own resource server. Needs `AgentIdentityContextProvider` on the config. Default `None`: the provider's own audience. See [Identity](#identity). |
| `forward_authorization` | Forward the caller's `Authorization` header to the target. Default `False`. See [Identity](#identity). |

There is intentionally no MCP `server_name` field. For MCP the PDP resource is already the
tool name, threaded into both visibility and invocation by the FastMCP middleware.

## Enforcement points

| Point | Hook used | On Deny |
|---|---|---|
| `TOOL_VISIBILITY` (MCP routes) | `FastMCPAuthZMiddleware` on the FastMCP proxy, `tools/list` response phase | Hidden tools do not appear in `tools/list`. |
| `BEFORE_TOOL` (MCP routes) | `FastMCPAuthZMiddleware`, `tools/call` request phase | The client receives an MCP error. |
| `AFTER_TOOL` (MCP routes) | `FastMCPAuthZMiddleware`, `tools/call` response phase | Records filtered or fields masked per the post-invocation strategy. |
| `A2A_INVOCATION` (A2A routes) | `A2AAuthZMiddleware` on the HTTP/JSON-RPC wire, every JSON-RPC method | The client receives a JSON-RPC error `-32001` (`-32600` for a POST that is not a single JSON-RPC request). |

`BEFORE_MODEL` and `AFTER_MODEL` are not available. The gateway sees only MCP and A2A wire
traffic, never the model's prompts or answers. Retrieval constraint injection over MCP is not
yet supported; see [Scope](#scope--not-yet-supported). For the same reason
[intent-based access control](../intent-based-access.md) is not available: an `intent_enricher`
on `BEFORE_TOOL` is always unavailable here and, failing closed, denies every tool call.

MCP routes serve **tools only** by default: the gateway authorizes tool calls, so the target's
resources and prompts would reach the client with no policy check. They are left out of
`resources/list`, `resources/templates/list` and `prompts/list`, and reading one is refused.
`Route(..., mcp_resources_and_prompts=True)` serves them, unchecked.

A request path with a `.` or `..` segment (also percent-encoded, `%2e%2e`) gets HTTP 400 on A2A
routes, so it can't reach a path on the target other than the one the route authorizes.

Which MCP points are active is set by `GatewayConfig.authz.enforcement`, exactly as on the
in-process [fastmcp](fastmcp.md) path. The runnable example configures its MCP route
visibility-only with `standard_tool_enforcement(invocation=False, post_invocation=False)`.

## Identity

The gateway is a server-side surface. The caller's request headers reach
`config.authz.context_provider`, which is `DefaultServerContextProvider` by default: it decodes the
`Authorization: Bearer` JWT and forwards every claim as a subject attribute. This applies to
both route types.

For MCP routes the identity rides in the JWT header on the transport connection. A LangChain
agent that loads tools through the gateway needs no authorization middleware of its own; the
gateway is the enforcement point. Because MCP tools are identity-scoped at load time, an agent
that loads tools through the gateway sees the tool list for the identity it connected with.

**Trust boundary.** The gateway is usually your edge, so `build_gateway` requires a context
provider that verifies the bearer token: set `AgentIdentityContextProvider` with a
`JwtVerification` on `authz` (see [Deployment](#deployment) and
[Agent identity](../agent-identity.md#verify-the-token)). Behind an ingress that authenticates
first, `GatewayConfig(allow_unverified_tokens=True)` accepts the default provider, which decodes
the token without checking its signature. A request with no token, or one that fails
verification, has an empty subject and is denied without asking the PDP.

**No token passthrough.** The caller's token was issued for the gateway, not for the servers
behind it. Passing it on is an antipattern (AIMS §10.8) and the MCP authorization spec forbids
it, so the gateway strips the `Authorization` header before forwarding: on MCP routes (FastMCP's
proxy would otherwise forward it) and on A2A routes, where `Cookie` is stripped with it.

**What the A2A forwarder changes.** Hop-by-hop headers (`Connection`, `Keep-Alive`, `TE`,
`Upgrade`, …, and any a `Connection` header names) are never forwarded, either way. The
client's own `X-Forwarded-*` / `Forwarded` headers are dropped and replaced by what the gateway
saw: `X-Forwarded-For` (the peer address), `X-Forwarded-Proto`, `X-Forwarded-Host`. Every other
header passes through unchanged (a2a-sdk 1.x needs `A2A-Version`). On the way back, repeated
`Set-Cookie` headers stay separate, and `Content-Encoding` is not relayed because the body was
already decoded. The forwarders' HTTP clients are closed when the gateway shuts down. A
target that must see the token, such as a [RAG backend that re-establishes identity](fastmcp.md#two-topologies),
needs `Route(..., forward_authorization=True)`. An MCP route whose target is an in-process
FastMCP server rather than a URL has no HTTP hop, so there is nothing to strip.

**One audience per upstream.** When the gateway is a single resource server, one
`JwtVerification(audience=...)` covers every route. When each upstream is its own resource
server, and callers get a token per upstream, give each route its audience:

```python
GatewayConfig(
    routes=[
        Route("/billing", BILLING_URL, protocol="a2a", agent_name="billing-agent", audience="https://billing.example.com"),
        Route("/docs",    DOCS_URL,    protocol="mcp", audience="https://docs.example.com"),
    ],
    authz=AuthZConfig(
        context_provider=AgentIdentityContextProvider(
            verification=JwtVerification(issuer="https://idp.example.com", audience="https://gateway.example.com"),
        ),
    ),
)
```

Each such route gets a copy of the provider that verifies for its audience. The copies share the
`authz` config's cache registry, so `config.authz.invalidate_decisions()` still reaches every route.
`build_gateway` raises `ValueError` if a route sets `audience` and the provider can't verify.

Why not one `audience` list of every upstream? A token is accepted if its audience is in the
list, so a token issued for billing would be accepted on the docs route too. A per-route
audience keeps each token to the route it was issued for.

## Client requirement (A2A)

An A2A client must be pointed at the gateway URL for each agent, not the agent's own address.
The gateway rewrites the served agent card's `url` to itself, so the client routes every
`message/send` back through the gateway and never learns the agent's real address. A client
pointed directly at a target bypasses enforcement entirely.

## Configuration notes

Only `GatewayConfig` and `Route` are specific to this path. `GatewayConfig.authz` is an
ordinary `AuthZConfig`; full field reference in [Configuration](../configuration.md#authzconfig-field-reference).

The gateway needs the `gateway` extra: `pip install "axiomatics-ai-sdk[gateway]"`. Protocol
middleware is imported lazily per route, so an A2A-only gateway needs no `fastmcp` and an
MCP-only gateway needs no `a2a`. `build_gateway()` returns an ASGI app; the server is a
deployment choice, so uvicorn is not part of the extra.

## What the PDP receives

The gateway adds nothing of its own. Each route sends exactly what the in-process middleware
it wraps would send.

- MCP routes: the subject is the JWT claims (or the [named attributes](../agent-identity.md#what-the-pdp-receives)
  with `AgentIdentityContextProvider`); the resource `tool_name` is the bare tool name
  (visibility and invocation, plus the arguments at invocation) or the record fields
  (post-invocation). Shapes under
  [Tool visibility](../enforcement-points.md#1-tool-visibility),
  [Tool invocation](../enforcement-points.md#2-tool-invocation) and
  [Post-invocation](../enforcement-points.md#3-post-invocation).
- A2A routes: the subject is the JWT claims; the resource `agent_name` is the route's `agent_name`,
  plus `method` and `skill`. Shape under
  [A2A invocation](../enforcement-points.md#5-a2a-invocation).

Identically named tools on two MCP servers are not disambiguated today. The plan is an
ARN-style composite convention, where the gateway prefixes the tool so the PDP sees
`{server_name}___{tool_name}` and policy authors key on that, with no new PDP attribute.

## Deployment

The gateway is a standard stateless ASGI app. Deploy it like any web service.

**Your gateway module** (`mygateway.py`): build the app from a `GatewayConfig` whose routes
point at your real services. Drive everything from the environment (12-factor):

```python
# mygateway.py
import os
from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement
from axiomatics_ai.gateway import GatewayConfig, Route, build_gateway
from axiomatics_ai import AgentIdentityContextProvider, JwtVerification

cfg = GatewayConfig(
    routes=[
        Route("/billing", os.environ["BILLING_URL"],  protocol="a2a", agent_name="billing-agent"),
        Route("/docs",    os.environ["DOCS_MCP_URL"], protocol="mcp"),
    ],
    authz=AuthZConfig(
        connection=PDPConnection(base_url=os.environ["PDP_BASE_URL"]),
        enforcement=standard_tool_enforcement(),
        context_provider=AgentIdentityContextProvider(verification=JwtVerification(
            issuer=os.environ["IDP_ISSUER"], audience=os.environ["GATEWAY_AUDIENCE"],
        )),
    ),
)
app = build_gateway(cfg)
```

| Env var | Purpose |
|---|---|
| `PDP_BASE_URL` | PDP endpoint the gateway queries for decisions |
| `BILLING_URL`, `DOCS_MCP_URL`, … | your real backend service URLs (one per route) |
| `IDP_ISSUER`, `GATEWAY_AUDIENCE` | only if the gateway is your edge and verifies tokens itself: the trusted issuer, and the gateway's audience at that issuer |

**Docker.** [`examples/gateway/Dockerfile`](../../examples/gateway/Dockerfile) builds a
production image that installs the SDK with the `gateway` extra plus uvicorn, runs as an
unprivileged user, exposes `9210`, and declares a TCP health check on the listen port (the
gateway mounts only the routes you configure, so there is no unauthenticated `/health`
endpoint). Its `CMD` runs the example module `examples.gateway.gateway:app`, which reads its
routes as data from `GATEWAY_ROUTES_FILE` (path to a JSON list of route objects) or
`GATEWAY_ROUTES` (inline JSON). Point the `CMD` at your own module for production.

```bash
docker build -f examples/gateway/Dockerfile -t axio-gateway .
docker run -p 9210:9210 \
  -e PDP_BASE_URL=http://host.docker.internal:8080 \
  -e GATEWAY_ROUTES_FILE=/config/routes.json \
  -v "$PWD/routes.json:/config/routes.json:ro" \
  axio-gateway
```

Terminate TLS at an ingress or load balancer in front of the gateway. Scale with
`uvicorn --workers N` or multiple replicas. The gateway is stateless; the PDP decision cache
is per-process, so replicas simply warm their own.

## Scope / not yet supported

- **OpenAI-compatible routes** (chat/completions tool filtering): planned. The protocol set
  is designed to extend to `"openai"`.
- **Transparent forward-proxy mode**: v1 is a path-prefix registry only.
- **SSE / streaming passthrough**: v1 handles request/response JSON-RPC. Responses are
  buffered so MCP response-phase enforcement can run.
- **RAG retrieval constraint injection over MCP**: reverse-query the PDP and inject a metadata
  filter into retrieval-tool calls via an out-of-band channel (MCP `_meta`) the LLM cannot see.
- **Composite tool names across MCP servers**: see [What the PDP receives](#what-the-pdp-receives).
- **MCP resources and prompts**: refused by default (above); authorizing them with the PDP is
  planned.
- **The A2A REST (HTTP+JSON) binding**: only JSON-RPC is authorized; other requests get 403.

## Examples

All scripts in [`examples/gateway/`](../../examples/gateway/) run against one domain,
[`examples/policies/domain_gateway.yaml`](../../examples/policies/domain_gateway.yaml): **alice** may
invoke `billing-agent` and sees all MCP tools; **bob** may not invoke billing and sees only
`search_docs`.

| Script | What it is | Port | Domain |
|---|---|---|---|
| `a2a_server.py` | A plain ADK billing agent (A2A), no authorization; the gateway enforces. | 9211 | `domain_gateway.yaml` |
| `mcp_server.py` | A plain FastMCP server (3 tools), no authorization; the gateway enforces. | 9212 | `domain_gateway.yaml` |
| `gateway.py` | The gateway: `/billing` → the A2A agent, `/tools` → the MCP server. Routes from `GATEWAY_ROUTES_FILE` / `GATEWAY_ROUTES`, or demo defaults. | 9210 | `domain_gateway.yaml` |
| `a2a_client.py` | An ADK agent that calls `/billing` through the gateway (alice/bob). | — | `domain_gateway.yaml` |
| `mcp_langchain.py` | A `create_agent` that loads its MCP tools through `/tools` (alice/bob). | — | `domain_gateway.yaml` |

Config in `.env`: `PDP_BASE_URL`, `LLM_MODEL`, `OPENROUTER_API_KEY`. Run order: start the PDP
with the domain, then the two backends, then the gateway, then the clients. The clients talk
only to the gateway. Expected output:

```
[alice] → The company billing address is 1200 Market Street, Suite 400, San Francisco, CA 94103.
[bob]   → I was unable to retrieve the billing information.
[alice] tools visible via gateway: ['get_account_balance', 'search_docs', 'send_email']
[bob]   tools visible via gateway: ['search_docs']
```

Notes from the example:

- The A2A client uses `AgentTool`, not `sub_agents`. With `sub_agents` (transfer), a denied
  call does not return control to the orchestrator, so it cannot relay the denial. `AgentTool`
  (call) gets the result or denial back and relays it.
- The MCP example builds the agent per user because MCP tools are identity-scoped at load
  time. A future example will show a long-lived shared agent with per-call invocation
  enforcement.
- Identity in the example is a demo *unsigned* JWT. See [Identity](#identity) for production.

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
