# a2a — authorization for the Agent2Agent protocol

Attaches to the A2A **HTTP/JSON-RPC wire**, on either side of the call. On the server side it
is a pure-ASGI middleware in front of an agent you expose; on the client side it is an
`httpx.Auth` on the client that calls a remote agent. It enforces one point, `A2A_INVOCATION`,
and works with Google ADK, AWS Strands, LangChain, or a self-wired a2a-sdk app, because it hooks
the transport rather than framework types. Identity comes from the request headers (server) or
from a context you supply (client).

The runtime imports nothing from `a2a-sdk`, so it handles both the 0.3.x (Pydantic) and 1.x
(protobuf) lines identically. The only version difference is the RPC method name
(`message/send` vs `SendMessage`), which is a simple check.

| | `A2AAuthZMiddleware` (server) | `A2AClientAuthZ` (client) |
|---|---|---|
| Type | pure-ASGI middleware | `httpx.Auth` |
| Direction | inbound `message/send` | outbound `message/send` |
| Role | **authoritative** (recommended) | complementary egress control |
| Identity | request headers / `Bearer` JWT (`DefaultServerContextProvider`) | local caller (`DefaultClientContextProvider`) |
| On deny | JSON-RPC error `-32001`, agent never runs | raises `A2AAuthorizationError` before the request leaves |

Both sides make the same PDP decision. The resource is the target/served **agent** identity plus
the RPC method and, when present in the message metadata, a **skill** hint.

## Attach

### Server — protect an agent you expose

```python
from axiomatics_ai import A2AAuthZMiddleware, AuthZConfig, PDPConnection

# `app` is any ASGI app hosting an A2A agent — ADK to_a2a(), Strands to_starlette_app(),
# or a self-wired A2AStarletteApplication().build().
app.add_middleware(
    A2AAuthZMiddleware,
    config=AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080")),
    agent_name="billing-agent",   # PDP resource label (policy-facing; explicit by design)
)
```

Because ASGI middleware composes, adding it to a single parent app that mounts many agents
enforces on all of them at once. When `agent_name` is unset it falls back to the request path,
so the PDP still distinguishes each mounted agent.

Every JSON-RPC method is authorized, not only `message/send` / `SendMessage`: `tasks/get`,
`tasks/cancel`, `tasks/resubscribe`, push-notification config, and any method a later protocol
version adds. The method is sent as the `method` resource attribute, so a policy tells them
apart. A policy that only permits message sends therefore denies a client that polls its task —
add a rule for the task methods your agents use.

A POST that is not a single JSON-RPC request is rejected with a JSON-RPC Invalid Request error
(`-32600`) and never reaches the agent: a batch array, a body with no `method`, invalid JSON,
and the HTTP+JSON REST binding (`POST /v1/message:send`), which this middleware does not
support.

Of non-POST requests, only a `GET` or `HEAD` of the public agent card
(`/.well-known/agent-card.json`, and the legacy `/.well-known/agent.json`, also under a mounted
agent's path) passes through. Everything else gets HTTP 403 without reaching the agent: the REST
binding's task reads, SSE subscriptions, the authenticated extended card, and anything else the
app serves. Add your own public paths, such as a health check, with `public_paths`:

```python
app.add_middleware(A2AAuthZMiddleware, config=config, public_paths=("/health",))
```

The server side is the authoritative place to enforce. It sits at the resource, protects the
agent whichever client calls, and cannot be bypassed by a misconfigured or hostile client.

### Client — gate outbound calls to remote agents

> **Preview.** The client guard covers the JSON-RPC transport only (see
> [Known limitations](../security.md#known-limitations)); the server middleware is the
> authoritative control.

```python
import httpx
from axiomatics_ai import A2AClientAuthZ, AuthZConfig, PDPConnection

authz = A2AClientAuthZ(
    AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080")),
    context={"username": "alice", "tenant_id": "acme"},  # or context_source=callable, per request
    agent_name="billing-agent",
)
http = httpx.AsyncClient(auth=authz)   # pass this client into ADK / Strands / a2a-sdk
```

Despite plugging into httpx's `auth=` slot, this is authori**z**ation (a PDP Permit/Deny), not
authentication. When `agent_name` is unset it defaults to the request URL host. It gates
only the message sends the client originates; other requests pass through, and the server
middleware authorizes them.

How each framework takes the two pieces:

| Framework | Server | Client |
|---|---|---|
| Google ADK | `to_a2a(agent)` → `app.add_middleware(A2AAuthZMiddleware, ...)` | `RemoteA2aAgent(httpx_client=httpx.AsyncClient(auth=A2AClientAuthZ(...)))` |
| AWS Strands | `A2AServer(agent).to_starlette_app()` → `app.add_middleware(...)` | `A2AClientToolProvider(httpx_client_args={"auth": A2AClientAuthZ(...)})` |
| LangChain | `create_agent` in an a2a-sdk `AgentExecutor` → `add_middleware(...)` | a `@tool` calling the a2a client, httpx `auth=A2AClientAuthZ(...)` |
| self-wired a2a-sdk | `A2AStarletteApplication(...).build()` → `add_middleware(...)` | `ClientConfig(httpx_client=httpx.AsyncClient(auth=A2AClientAuthZ(...)))` |

The [gateway](gateway.md) reuses the same server middleware to protect A2A servers out of
process.

## Enforcement points

| Point | Hook used | On Deny |
|---|---|---|
| `A2A_INVOCATION` | server: pure-ASGI middleware on inbound `message/send`; client: `httpx.Auth` on outbound `message/send` | server: JSON-RPC error `-32001` returned, the agent never runs; client: `A2AAuthorizationError` raised, the request never leaves |

No other point is available on this path. The middleware and the client guard see only the
JSON-RPC envelope on the wire, not the model or the tool calls inside the remote agent. To
enforce `BEFORE_TOOL`, `TOOL_VISIBILITY`, or the model guards inside the agent, attach the
integration for that agent's framework as well.

The Deny is sent with HTTP 200, following the JSON-RPC convention, so a2a-sdk clients parse it
as a structured error rather than a transport failure. `-32001` is in the
implementation-defined server-error range.

When both sides are on, the client blocks first (egress), so the server never sees the denied
call — defense in depth.

## Identity

**Server.** `DefaultServerContextProvider` reads the caller's `Authorization: Bearer <JWT>`
header on every request and forwards every claim as a subject attribute. No per-call wiring is
needed. The default provider reads the claims without verifying the signature, and logs a
one-time warning saying so. In production the token is verified by your auth layer upstream of
the agent, or, if the agent is the edge, by `AgentIdentityContextProvider`:

```python
from axiomatics_ai import A2AAuthZMiddleware, AgentIdentityContextProvider, AuthZConfig, JwtVerification, PDPConnection

app.add_middleware(
    A2AAuthZMiddleware,
    config=AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        context_provider=AgentIdentityContextProvider(
            verification=JwtVerification(issuer="https://idp.example.com", audience="https://billing-agent.example.com"),
        ),
    ),
    agent_name="billing-agent",
)
```

The subject is then the calling agent (`agent_id`, from `client_id`), the user it acts for
(`user_id`) and the current actor (`actor_id`); see [Agent identity](../agent-identity.md).

**Client.** `httpx.Auth` only ever receives the outbound request; it has no seam for
application-level per-call context (the a2a-sdk `ClientCallContext` does not reach this layer).
So the context is supplied at construction, in one of two ways:

- `context=` — a static object. Use when one client serves one caller (the common case).
- `context_source=` — a zero-arg callable resolved on every request. Use for a multi-user
  process, e.g. a server calling agents on behalf of different users: set a
  `contextvars.ContextVar` around each agent turn and have the callable read it.
  `context_source` takes precedence over `context` when both are given.

`DefaultClientContextProvider` reads `username`, `user_id`, `tenant_id`, and `agent_id` (the
calling agent's own identifier, e.g. a SPIFFE ID) into the subject and `session_id` into the
environment. Pass a custom provider via
`AuthZConfig(context_provider=...)` to change the mapping. See
[Default attribute names](../identity.md#default-attribute-names) and
[Custom provider, client side](../identity.md#custom-provider-client-side).

## Configuration notes

- **Install extra:** `a2a`. Its only extra runtime dependency is PyJWT (with `crypto`, for
  token verification). `a2a-sdk` itself is not required at runtime.
- **Always on.** Instantiating the middleware or the client guard is the opt-in. Unlike the
  tool points, enforcement does not depend on an `Enforcer` being present in
  `AuthZConfig.enforcement`.
- **Action name.** The `A2A_INVOCATION` `Enforcer`'s `action_name` (default `"a2a_invocation"`)
  is the action name sent to the PDP. To change it:
  ```python
  from axiomatics_ai import AuthZConfig, Enforcer, EnforcementPoint

  config = AuthZConfig(enforcement={EnforcementPoint.A2A_INVOCATION: Enforcer(action_name="send_message")})
  ```
  `standard_tool_enforcement()` configures the tool points only; it does not add an
  `A2A_INVOCATION` entry, and the default action name applies without one.
- **`include_message_text`** (both primitives, default `False`). When `True`, the message text
  is added to the resource as `message` so policy can match on content. Off by default for
  privacy and because it churns the decision cache key.
- Connection, `fail_open`, cache, and `context_provider` are shared with every other path; see
  [Configuration](../configuration.md).

## What the PDP receives

One request per inbound (server) or outbound (client) `message/send`.

| Category | Attribute | Example |
|---|---|---|
| subject | JWT claims (server) or context (client) | `username` = `alice` |
| action | `name` | `a2a_invocation` |
| resource | `agent_name` (`AuthZConfig.attribute_names.agent`) | the `agent_name` you pass when attaching, e.g. `billing-agent` |
| resource | `method` | `message/send` |
| resource | `skill` | from the message metadata, when present, e.g. `refund` |
| resource | `message` | the message text, only with `include_message_text=True` |

The skill hint is read from the message metadata under `skillId`, `skill_id`, or `skill`.
`agent_name` is the policy-facing identity, decoupled from each framework's internal agent or
card name (ADK, for instance, forbids hyphens in agent names). Shapes and the
XACML JSON / AuthZEN wire formats are on [Enforcement points](../enforcement-points.md#5-a2a-invocation).
Example policy: `examples/policies/domain_a2a.yaml` — **alice** may send to `billing-agent`;
**bob** may not, unless the message carries `skill=refund`.

## Examples

Runnable server/client pairs for ADK, Strands, and LangChain, plus a no-LLM self-wired demo,
are in [`examples/a2a/`](../../examples/a2a/). All use the Axiomatics ADS domain
`examples/policies/domain_a2a.yaml`.

| Script | What it shows | Port |
|---|---|---|
| `self_wired_a2a.py` | a2a-sdk server + client in one process, no LLM, no open ports; both directions, including `skill=refund` gating | — |
| `adk_server.py` / `adk_client.py` | Google ADK billing agent behind the middleware; ADK `RemoteA2aAgent` calling it | `9101` |
| `strands_server.py` / `strands_client.py` | AWS Strands `A2AServer` behind the middleware; `A2AClientToolProvider` calling it | `9102` |
| `langchain_server.py` / `langchain_client.py` | LangChain `create_agent` in an a2a-sdk `AgentExecutor`; a `@tool` calling it | `9103` |

**Prerequisites**

1. Install the example dependencies: `uv sync --extra all` (brings in a2a-sdk, ADK, Strands,
   and the LiteLLM providers).
2. Start Axiomatics ADS with the A2A domain. In `examples/policies/`, point `deployment.yaml` at
   `domain_a2a.yaml`, then run the jar:
   ```bash
   cd examples/policies && java -jar access-decision-service-<version>.jar
   ```
   See [Running ADS](../running-ads.md).
3. `.env` at the repo root:
   ```
   PDP_BASE_URL=http://localhost:8080
   LLM_MODEL=openrouter/<your-model>
   OPENROUTER_API_KEY=<key>
   ```

**Run**

```bash
python examples/a2a/self_wired_a2a.py                 # zero-dependency demo first

# terminal 1 — the protected agent server (ADK shown; same shape for strands/langchain)
python examples/a2a/adk_server.py                 # add --no-authz to disable the middleware

# terminal 2 — the calling agent
python examples/a2a/adk_client.py                 # runs as alice then bob
python examples/a2a/adk_client.py --user bob      # just bob
python examples/a2a/adk_client.py --no-authz      # disable client-side A2AClientAuthZ
```

The billing agent only knows one fact — Acme's billing address, **"1200 Market Street"**. If
that address appears in the client output, the remote agent was actually reached; if it is
missing, the call was blocked.

**Toggling both switches — the truth table**

`--no-authz` on the server disables `A2AAuthZMiddleware`; `--no-authz` on the client disables
`A2AClientAuthZ`. Run bob under each combination:

| server authz | client authz | alice | bob | who blocked bob |
|:---:|:---:|:---:|:---:|---|
| on | on | reaches | blocked | client (before send) |
| on | off | reaches | blocked | server (`-32001`) |
| off | on | reaches | blocked | client (before send) |
| off | off | reaches | reaches | nobody — no enforcement |

To watch the server enforce on its own, run the client with `--no-authz`.

**Notes**

- Identity in the examples is a demo Bearer JWT (`{"username": "alice"|"bob"}`), read
  unverified by the default server provider.
- Skill-level gating (`skill=refund`) is demonstrated from `self_wired_a2a.py`, which sets the
  message metadata directly. The framework clients gate at the agent level, which is what a
  policy typically keys on.

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
