# tool_decorator — per-tool post-invocation on a FastMCP server

The `tool_decorator` path attaches to individual tools on a FastMCP server through the
`@protected_tool` decorator. It is server side. It enforces one point, `AFTER_TOOL`: the tool
always executes, and the `post_processor` you pass is always called with the result and is
responsible for the PDP call and any transformation. Identity comes from the
`Authorization: Bearer <JWT>` header of the incoming MCP request.

## Attach

Call `configure_authz_decorator()` once at server startup, before any protected tool is
invoked. Then decorate each tool you want post-processed.

```python
from fastmcp import FastMCP
from axiomatics_ai import (
    AuthZConfig, ColumnMaskStrategy, PDPConnection, standard_tool_enforcement,
    configure_authz_decorator, protected_tool,
)

# One-time setup — call before any protected tool is invoked
configure_authz_decorator(
    config=AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),
    )
)

mcp = FastMCP("my-server")

@mcp.tool()
@protected_tool(post_processor=ColumnMaskStrategy("dob", "email", "phone"))
def get_patient_records(patient_id: str) -> list[dict]:
    """ColumnMaskStrategy asks the PDP: Permit → full records; Deny → dob, email, phone masked."""
    return [...]

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

`configure_authz_decorator` takes a single keyword argument, `config`. With no argument it
uses `AuthZConfig()`. Sync tools run off the event loop thread; async tools are awaited.

The client needs no SDK code. It passes the JWT in the `Authorization` header:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "my-server": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",
        "headers": {"Authorization": "Bearer <jwt>"},
    }
})
tools = await client.get_tools()
```

## Enforcement points

| Point | Framework hook used | On Deny |
|---|---|---|
| `AFTER_TOOL` | `@protected_tool(post_processor=...)` wrapping the tool function | Decided by the strategy: `ColumnMaskStrategy` masks the named fields, `RecordFilterStrategy` removes the denied records |

Opt-in is per tool, through the decorator, not through `config.enforcement`:

| `post_processor` | Behaviour |
|---|---|
| `None` | Tool runs, result returned as-is. The PDP is not called. |
| Strategy or async callable | Tool runs, `post_processor` called unconditionally. PDP interaction is owned by the strategy. |

Built-in strategies implement `apply(result, context)` and are run through the SDK's shared
post-invocation helper, so they emit a post-invocation decision event. A plain async callable
`(result, ctx: PostInvocationContext) -> Any` is awaited as-is.

Points that are **not** available on this path:

- **`TOOL_VISIBILITY`** — tools decorated with `@protected_tool` are always visible in the MCP
  tool list. Add [`FastMCPAuthZMiddleware`](fastmcp.md) alongside if you need to hide tools
  from the LLM.
- **`BEFORE_TOOL`** — this path never blocks a call before execution. Add
  `FastMCPAuthZMiddleware` with a `BEFORE_TOOL` Enforcer if you need to hard-block tool calls.

## Identity

Identity is read from the `Authorization: Bearer <JWT>` header via FastMCP's
`get_http_headers(include={"authorization"})`. **HTTP transport is required; stdio is not
supported.** If no HTTP context is available when the tool runs, the headers are treated as
empty and a warning is logged.

By default `DefaultServerContextProvider` decodes the JWT and forwards every claim as a subject
attribute. To verify the token and send named attributes (`user_id`, `agent_id`, `actor_id`,
...), set `AgentIdentityContextProvider` ([Agent identity](../agent-identity.md)):

```python
from axiomatics_ai import AgentIdentityContextProvider, AuthZConfig, JwtVerification, PDPConnection, configure_authz_decorator

configure_authz_decorator(config=AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    context_provider=AgentIdentityContextProvider(
        verification=JwtVerification(issuer="https://idp.example.com", audience="https://records.example.com"),
    ),
))
```

Set a custom provider on `AuthZConfig.context_provider` to pick claims, read other headers, or
add resource attributes:

```python
from axiomatics_ai import (
    AuthZConfig, DefaultServerContextProvider, PDPConnection,
    configure_authz_decorator, decode_bearer_jwt_unverified, standard_tool_enforcement,
)

class MyProvider(DefaultServerContextProvider):
    def get_environment(self, headers: dict) -> dict:
        return {"session_id": headers.get("x-session-id", "")}

    def get_resource(self, resource_name: str, headers: dict, *, args=None, resource_name_attribute=None) -> dict:
        resource = dict(args or {})
        if resource_name_attribute:
            resource[resource_name_attribute] = resource_name  # set last: the identifier wins
        claims = decode_bearer_jwt_unverified(headers)
        if claims.get("tenant_id"):
            resource["tenant_id"] = claims["tenant_id"]
        return resource

configure_authz_decorator(
    config=AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),
        context_provider=MyProvider(),
    )
)
```

**Trust boundary.** The default provider (and `decode_bearer_jwt_unverified` above) reads
claims **without signature verification**, and logs a one-time warning saying so.
Authentication is assumed to have happened upstream (ingress, API gateway, auth middleware). If
this process is the edge, or you want defence in depth, use `AgentIdentityContextProvider`. See
[Identity](../identity.md#trust-boundary).

## Configuration notes

```python
configure_authz_decorator(
    config=AuthZConfig(
        connection=PDPConnection(
            base_url="http://localhost:8080",
            fail_open=False,          # forwarded to the strategy via PostInvocationContext
            cache_ttl_seconds=300.0,
            cache_maxsize=1024,
        ),
        enforcement={
            EnforcementPoint.AFTER_TOOL: Enforcer(action_name="post_invocation"),
        },
        context_provider=MyProvider(),  # optional, see Identity above
    ),
)
```

- The `TOOL_VISIBILITY`, `BEFORE_TOOL`, and `AFTER_TOOL` entries in `enforcement` do not gate
  anything on this path; opt-in is per tool via `@protected_tool`.
- The `AFTER_TOOL` Enforcer's `action_name` is read and used to build the request. If no
  `AFTER_TOOL` Enforcer is configured, the default is `post_invocation`. The tool name is sent
  as `AuthZConfig.attribute_names.tool` (default `tool_name`).
- `fail_open` on `PDPConnection` is forwarded to the strategy via `PostInvocationContext`; the
  strategy applies it when the PDP is unreachable.

Full field reference in [Configuration](../configuration.md#authzconfig-field-reference).

## What the PDP receives

The PDP call is made by the strategy after the tool executes. The request shape depends on the
strategy:

- **`ColumnMaskStrategy`** — one XACML JSON request per tool call:
  - **Subject** — from `get_subject(headers)`
  - **Action** — `{"name": action_name}` (from the `AFTER_TOOL` Enforcer's `action_name`)
  - **Resource** — `{"tool_name": tool_name, **call_args}` (from `get_resource`, via `context.resource`); the call's arguments by parameter name, positional ones included
  - **Environment** — from `get_environment(headers)`

- **`RecordFilterStrategy`** — one XACML JSON MultiRequest per tool call, one sub-request per
  record in the result. Same subject, action, and environment; the resource carries a stable
  `record_hash` (SDK-generated SHA-256 key, `RecordFilterStrategy(key_attribute=...)`) plus the
  record's fields as extra attributes, and no `tool_name`.

Attribute tables and a worked policy are on [Enforcement points](../enforcement-points.md#3-post-invocation)
and [Post-invocation](../post-invocation.md).

## Post-invocation strategies

`ColumnMaskStrategy` and `RecordFilterStrategy` can be passed directly as `post_processor`:

```python
from axiomatics_ai import ColumnMaskStrategy, RecordFilterStrategy, protected_tool

@protected_tool(post_processor=ColumnMaskStrategy("dob", "ssn"))
def get_employee() -> dict: ...

@protected_tool(post_processor=RecordFilterStrategy())
def get_records() -> list[dict]: ...
```

Any custom strategy implementing `apply(result, context)`, or any async callable, also works.
See [Post-invocation](../post-invocation.md#custom-strategy) for the custom strategy guide.

## Combining with `fastmcp` middleware

If you need both `list_tools` filtering and per-tool post-invocation, add
`FastMCPAuthZMiddleware` alongside `@protected_tool`. Leave `BEFORE_TOOL` off the middleware
so the PDP is not called twice for the same tool:

```python
from axiomatics_ai import AuthZConfig, EnforcementPoint, Enforcer, PDPConnection
from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware

mcp.add_middleware(
    FastMCPAuthZMiddleware(AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement={EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="tool_visibility")},  # visibility only; the decorator handles post-invocation
    ))
)
```

See [fastmcp](fastmcp.md) for the middleware page.

## Examples

Runnable scripts are in [`examples/tool_decorator/`](../../examples/tool_decorator/). They
assume a PDP is already running with `examples/policies/domain_tool_decorator_masking.yaml` loaded
(see [Running ADS](../running-ads.md)), and read `PDP_BASE_URL`, `LLM_MODEL`, and your provider
key from the environment or `.env`.

| Script | What it shows | Domain file |
|---|---|---|
| `masking_server.py` | A FastMCP server whose tools are decorated with `@protected_tool(post_processor=...)`. `ColumnMaskStrategy` masks known PII fields on Deny; a custom `EmailRedactionStrategy` redacts emails in free text. | `examples/policies/domain_tool_decorator_masking.yaml` |
| `masking_client.py` | A LangChain agent calling the masking server. alice sees full data; bob gets masked and redacted results. | same |

```bash
python examples/tool_decorator/masking_server.py    # terminal 1
python examples/tool_decorator/masking_client.py     # terminal 2
```

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
