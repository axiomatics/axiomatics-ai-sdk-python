# fastmcp — server-side authorization via FastMCP middleware

This path attaches to a FastMCP server as middleware. It runs server side: every `tools/list`
and `tools/call` request is checked against the PDP before execution, and the client needs no
changes. It enforces `TOOL_VISIBILITY`, `BEFORE_TOOL`, and `AFTER_TOOL`, and binds the caller's
identity for authorization-filtered retrieval inside tools. Identity comes from the
`Authorization: Bearer <JWT>` header of each request.

> **stdio not supported.** There is no per-request HTTP object in stdio mode. Use the
> client-side [langchain_mcp](langchain_mcp.md) path for stdio setups.

## Attach

`to_authorized_fastmcp_server(server, config)` attaches the middleware and returns the same
server:

```python
from fastmcp import FastMCP
from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement, to_authorized_fastmcp_server

mcp = FastMCP("my-server")

@mcp.tool()
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return "..."

# One call — attaches the authorization middleware and returns the server.
mcp = to_authorized_fastmcp_server(
    mcp,
    AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement()),
)

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

Prefer to hold the middleware list yourself, for example to add other FastMCP middleware?
Attach it directly. This is exactly what the wrapper does:

```python
from axiomatics_ai import FastMCPAuthZMiddleware
mcp.add_middleware(FastMCPAuthZMiddleware(config))
```

The client carries no SDK code. It passes the JWT in the `Authorization` header and the server
filters transparently:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "my-server": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",
        "headers": {"Authorization": "Bearer <jwt>"},
    }
})
tools = await client.get_tools()  # already filtered server-side
```

## Enforcement points

| Point | FastMCP hook used | On Deny |
|---|---|---|
| `TOOL_VISIBILITY` | `on_list_tools` | The tool is removed from the `tools/list` response. |
| `BEFORE_TOOL` | `on_call_tool`, before the tool runs | The call fails with a `ToolError` (`Access denied: not authorized to call '<tool>'.`). |
| `AFTER_TOOL` | `on_call_tool`, after the tool returns | The configured strategy filters or masks the records in the result. Default: `RecordFilterStrategy()`. |
| Retrieval (`RetrievalConfig`) | `on_call_tool` binds the caller's identity around the call | A vector store wrapped with `to_authorized_vectorstore` filters per user. See [RAG inside a tool](#rag-inside-a-tool-ambient-identity). |

Each point is active only when its `Enforcer` is configured. The middleware intercepts only
`tools/list` and `tools/call`, so `BEFORE_MODEL`, `AFTER_MODEL`, and `A2A_INVOCATION` are not
available on this path: there is no model call or agent-to-agent message on an MCP server.

[Intent-based access control](../intent-based-access.md) does not work here: the server never
sees the user's messages, so an `intent_enricher` on `BEFORE_TOOL` is always unavailable and,
with `on_unavailable="fail_closed"`, denies every call (a warning is logged at start-up).
Enforce intent on the client-side agent.

For post-invocation, the first `TextContent` block of the tool result is inspected. Bare JSON
arrays, JSON strings, and records nested under a dict key are all filtered row by row. Output
holding no records passes through unchanged and is reported to hooks as a `skipped`
post-invocation decision. See [Post-invocation](../post-invocation.md) for the strategies,
`PostInvocationContext` fields, and the custom strategy guide.

```python
from axiomatics_ai import AuthZConfig, ColumnMaskStrategy, Enforcer, EnforcementPoint, FastMCPAuthZMiddleware

mcp.add_middleware(FastMCPAuthZMiddleware(
    AuthZConfig(enforcement={EnforcementPoint.AFTER_TOOL: Enforcer(action_name="post_invocation", strategy=ColumnMaskStrategy("ssn", "dob"))})
))
```

## Human approval

When the policy asks for a human's approval ([Obligations and human approval](../human-approval.md))
and no `approver` is configured, the middleware asks the **MCP client's user** with an
elicitation during the tool call (form mode: approve or reject), then checks the answer with
the PDP. **The client must support elicitation** (FastMCP `Client(elicitation_handler=...)`,
langchain-mcp-adapters `Callbacks(on_elicitation=...)`); a client that doesn't, a decline or a
cancel is a rejection. The elicitation needs a live session with the client for the length of
the call.

## Identity

By default `DefaultServerContextProvider` reads the claims from the `Authorization: Bearer` JWT
and forwards all of them as subject attributes. Policy can reference any claim (for example
`username`, `tenant_id`, `sub`). The `fastmcp` extra brings in PyJWT.

Every provider method receives the raw HTTP headers dict. Override only what you need; the rest
keeps its default behaviour.

> **Trust boundary.** The default reads claims **without signature verification** and logs a
> one-time warning saying so. Run the server behind an ingress, API gateway, or auth middleware
> that verifies the token before the request arrives. If this process is itself the edge, or you
> want defence in depth, verify here with `AgentIdentityContextProvider` (shown below). See
> [Identity: trust boundary](../identity.md#trust-boundary).

**Client side** — create and send the JWT with each request:

```python
import jwt  # pip install PyJWT
from langchain_mcp_adapters.client import MultiServerMCPClient

# In a real application the JWT would come from your IdP (OAuth2 / OIDC flow).
# For testing you can create one manually:
token = jwt.encode(
    {"sub": "alice", "tid": "acme"},
    PRIVATE_KEY,
    algorithm="RS256",
)

client = MultiServerMCPClient({
    "my-server": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",
        "headers": {"Authorization": f"Bearer {token}"},
    }
})
tools = await client.get_tools()
```

**Server side** — verifying the token (when this process is the edge, or for defence in depth):

```python
from axiomatics_ai import (
    AgentIdentityContextProvider, AuthZConfig, FastMCPAuthZMiddleware, JwtVerification,
    PDPConnection, standard_tool_enforcement,
)

mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement=standard_tool_enforcement(),
    context_provider=AgentIdentityContextProvider(
        verification=JwtVerification(issuer="https://idp.example.com", audience="https://tools.example.com"),
    ),
)))
```

The PDP then receives `user_id`, `agent_id`, `actor_id`, `scopes` and the other
[named attributes](../agent-identity.md#what-the-pdp-receives) instead of raw claims; a token
that fails verification gives an empty subject. The token is signed by your identity provider,
not by a key you hold. See [Agent identity](../agent-identity.md) for the settings, and
[Identity](../identity.md) for the full provider interface.

## Configuration notes

The standard tool points, spelled out:

```python
AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement={
        EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="tool_visibility"),
        EnforcementPoint.BEFORE_TOOL:     Enforcer(action_name="tool_invocation"),
        EnforcementPoint.AFTER_TOOL:      Enforcer(
            action_name="post_invocation",
            # strategy=ColumnMaskStrategy("ssn", "dob"),  # default: RecordFilterStrategy()
        ),
    },
)
```

- `FastMCPAuthZMiddleware(config)` takes the context provider from `config.context_provider`,
  falling back to `DefaultServerContextProvider()`.
- Transport must be `http` or `sse`. stdio is not supported.
- To set the internal `surface` label used to tag decision events (default `"fastmcp"`), use
  `FastMCPAuthZMiddleware` directly rather than the wrapper.

See [Configuration](../configuration.md#authzconfig-field-reference) for every field.

## What the PDP receives

The request shapes are the standard ones in [Enforcement points](../enforcement-points.md).
This path fills them as follows.

**Tool listing (`TOOL_VISIBILITY`)** — one XACML JSON MultiRequest per `tools/list` call, one
sub-request per tool:

- **Subject** — from `get_subject(headers)`
- **Action** — from `get_action(action_name, headers)` (the visibility Enforcer's `action_name`)
- **Resource** — `{"tool_name": "<tool_name>"}` (`AuthZConfig.attribute_names.tool`)
- **Environment** — from `get_environment(headers)`

**Tool invocation (`BEFORE_TOOL`)** — one XACML JSON request per `tools/call`, before the tool
runs:

- **Subject** — from `get_subject(headers)`
- **Action** — from `get_action(action_name, headers)` (the `BEFORE_TOOL` Enforcer's `action_name`)
- **Resource** — from `get_resource(tool_name, headers, args=arguments, resource_name_attribute=config.attribute_names.tool)`; by default `{"tool_name": "<tool_name>", **arguments}`
- **Environment** — from `get_environment(headers)`

**Post-invocation (`AFTER_TOOL`)** — depends on the active strategy. For the built-in
strategies, the action name is the `AFTER_TOOL` Enforcer's `action_name`, or the per-tool
override from that Enforcer's `action_overrides`.

With `protocol="authzen"` the same attributes go out as AuthZEN requests instead; see
[Wire formats](../enforcement-points.md#wire-formats).

## RAG inside a tool (ambient identity)

If a tool searches a vector store wrapped with [`to_authorized_vectorstore`](rag.md), that
wrapper resolves the caller from the *ambient* authorization context — which nothing populates
on a FastMCP server. `FastMCPAuthZMiddleware` bridges this automatically: on every `tools/call`
it binds the request's `Authorization` header into the ambient context for the duration of the
call, so a wrapped store filters per-user with **no per-tool wiring**. The tool body stays the
ordinary `vector_store.similarity_search(query, k=...)`.

This binding happens **regardless of which enforcement points are configured** — so RAG
filtering works whether or not the other enforcement points are on.

```python
from axiomatics_ai import (
    AuthZConfig, DefaultServerContextProvider, FastMCPAuthZMiddleware, PDPConnection,
    RetrievalConfig, to_authorized_vectorstore,
)

config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"),
                     context_provider=DefaultServerContextProvider(),
                     retrieval=RetrievalConfig())
vector_store = to_authorized_vectorstore(vector_store, config)  # wrap once

@mcp.tool()
def search_documentation(query: str) -> str:
    """Search internal documentation."""
    docs = vector_store.similarity_search(query, k=10)  # unchanged — filtered by the PDP
    return "\n".join(d.page_content for d in docs)

mcp.add_middleware(FastMCPAuthZMiddleware(config))
```

### Two topologies

- **Single process (no gateway)** — `FastMCPAuthZMiddleware` alone covers tool visibility,
  invocation, post-invocation, *and* RAG filtering. Populate `enforcement` with the tool points.
- **Behind the [gateway](gateway.md)** — the gateway carries `FastMCPAuthZMiddleware` (it owns
  visibility / invocation / post-invocation). The backend RAG server runs
  `FastMCPAuthZMiddleware` too, but with `enforcement=None`, so it re-enforces nothing and only
  re-establishes the caller's identity for its wrapped stores. The gateway does not pass the
  caller's token on by default ([no token passthrough](../agent-identity.md#gateway-no-token-passthrough)),
  so this topology needs `Route(..., forward_authorization=True)` on the backend's route.

```python
# Backend RAG server behind the gateway: identity only, no re-enforcement.
FastMCPAuthZMiddleware(AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    context_provider=DefaultServerContextProvider(),
    retrieval=RetrievalConfig(),   # RAG-only backend: retrieval filtering, no tool enforcement
))
```

> **⚠️ An absent or partial subject is handled by the SDK, not by your policy.** When a request
> arrives with no (or an unverified) `Authorization` header, the bound subject is empty; when the
> subject is missing some attributes the policy needs, those attributes are unbound. In both
> cases retrieval **fails closed** (no documents) unless you set
> `AuthZConfig(connection=PDPConnection(fail_open=True))` — the SDK never leaves the search
> unfiltered on a missing/partial identity.
>
> A subject-presence check in policy (e.g. `stringBagSize(user.name) == 0`) works for the
> *forward-evaluated* enforcement points — `tool_visibility`, `tool_invocation`,
> `post_invocation`, and `a2a_invocation` — where an absent subject attribute is an empty value
> the policy can test. It is **inert for retrieval**: a CAQ reverse query treats unprovided
> subject attributes as *unbound free variables*, not absent values, so the check never
> evaluates true during partial evaluation — Axiomatics ADS returns a condition rather than a
> `Deny`. (Verified on Axiomatics ADS 26.1.6: the identical rule denies anonymous
> visibility/invocation but has no effect on retrieval.) So don't rely on policy to deny an
> anonymous *retrieval* — that is the SDK's job, via `fail_open`. Either way, the subject you
> *do* build is only as trustworthy as the layer that authenticated the token — see the
> trust-boundary note above.

Runnable example: [`examples/fastmcp/rag_server.py`](../../examples/fastmcp/rag_server.py) +
[`rag_client.py`](../../examples/fastmcp/rag_client.py).

## Examples

Runnable scripts are in [`examples/fastmcp/`](../../examples/fastmcp/). Each assumes the PDP is
already running with the listed domain (see [Running Axiomatics ADS](../running-ads.md)) and
reads `PDP_BASE_URL` (default `http://localhost:8080`) plus `LLM_MODEL` and your provider key
from the environment / `.env`.

| Script | Domain | What it shows |
|---|---|---|
| `authz_server.py` | [`domain_mcp_tool_visibility.yaml`](../../examples/policies/domain_mcp_tool_visibility.yaml) | Server-side enforcement via `FastMCPAuthZMiddleware` — alice sees all tools, bob only `search_docs`. |
| `authz_client.py` | [`domain_mcp_tool_visibility.yaml`](../../examples/policies/domain_mcp_tool_visibility.yaml) | A LangChain agent connecting as alice and bob to the protected server above, showing filtered tool lists. |
| `to_authorized_server.py` | [`domain_all_authz_enabled.yaml`](../../examples/policies/domain_all_authz_enabled.yaml) | The same enforcement attached with the `to_authorized_fastmcp_server(server, config)` wrapper. |
| `rag_server.py` | [`domain_rag_retrieval.yaml`](../../examples/policies/domain_rag_retrieval.yaml) | Authorization-filtered RAG inside a FastMCP tool (ambient identity). |
| `rag_client.py` | [`domain_rag_retrieval.yaml`](../../examples/policies/domain_rag_retrieval.yaml) | A LangChain agent connecting as alice and bob, retrieving different documents from the same tool. |
| `payments_server.py` | [`domain_agent_identity.yaml`](../../examples/policies/domain_agent_identity.yaml) | A payments server that verifies the caller's OAuth access token (`AgentIdentityContextProvider`) and allows a payment only within the user's approved RAR grant. |
| `payments_agent.py` | [`domain_agent_identity.yaml`](../../examples/policies/domain_agent_identity.yaml) | A LangChain agent paying invoices through that server with tokens from a local test issuer: the approved payment goes through, a larger one and one without a user are denied. |
| `approval_server.py` + `approval_client.py` | [`domain_human_approval.yaml`](../../examples/policies/domain_human_approval.yaml) | Human approval by MCP elicitation: the server asks the client's user to approve a large payment or a cancellation. |

Run a server in one terminal and its client in another:

```bash
python examples/fastmcp/authz_server.py     # terminal 1
python examples/fastmcp/authz_client.py      # terminal 2
```

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
