# Identity

A PDP decides on attributes. The **context provider** is the piece that turns whatever your
application knows about the caller into the four attribute categories the request carries:
subject, action, resource, environment. Override only the methods you need; the rest keep
their defaults.

## Two base classes

| Base class | Identity source | Used by |
|---|---|---|
| `DefaultClientContextProvider` | a dict or object your code passes at the call site (`agent.invoke(..., context=...)`, session state, run context) | `langchain_agents`, `langchain_mcp`, `rag`, `google_adk`, `strands`, `openai_agents`, `claude_agent`, `direct` |
| `DefaultServerContextProvider` | the HTTP headers of the incoming request (`Authorization: Bearer <JWT>`, or custom headers) | `fastmcp`, `tool_decorator`, `a2a`, `gateway` |

On the server paths, `AgentIdentityContextProvider` (a `DefaultServerContextProvider`) verifies
the bearer token and maps its claims to named attributes. It has its own page:
[Agent identity](agent-identity.md).

Both expose the same four methods:

```python
class ContextProvider(Protocol):
    def get_subject(self, context) -> dict: ...
    def get_action(self, action_name: str, context) -> dict: ...
    def get_resource(self, resource_name: str, context, *, args: dict | None = None,
                     resource_name_attribute: str | None) -> dict: ...
    def get_environment(self, context) -> dict: ...
```

`context` is the runtime object for client paths and the headers dict for server paths.
`action_name` is the `Enforcer.action_name` for the point being enforced; `resource_name` is
the tool name (or agent name, document, ...); `resource_name_attribute` is the attribute the SDK
suggests sending it as (`AuthZConfig.attribute_names.tool` at the tool points,
`.agent` for A2A, the caller's choice for `Authorizer`, `None` for the
content guards, whose resource has no identifier). A provider may use a different id, as long as
the policy uses the same one. `args` are the tool-call arguments on invocation and
post-invocation. Return `None` for an attribute to omit it. Pass an instance via
`AuthZConfig(context_provider=MyProvider())`.

## Default attribute names

If you supply no provider, the defaults look for these keys (dict keys or object attributes):

| Provider | Category | Attributes sent |
|---|---|---|
| `DefaultClientContextProvider` | subject | `username`, `user_id`, `tenant_id`, `agent_id` |
| `DefaultClientContextProvider` | environment | `session_id` |
| `DefaultServerContextProvider` | subject | every claim in the bearer JWT, unverified |
| `DefaultServerContextProvider` | environment | none |
| `AgentIdentityContextProvider` | subject, environment | `user_id`, `agent_id`, `actor_id`, `scopes`, ..., and `transaction_id`; see [Agent identity](agent-identity.md#what-the-pdp-receives) |
| both | action | `{"name": <action_name>}` |
| both | resource | `{<resource_name_attribute>: <resource_name>, **args}`, e.g. `{"tool_name": "search_docs", "query": "..."}` |

An argument whose name equals `resource_name_attribute` is dropped with a warning: the
identifier always wins, so a model-chosen argument value cannot impersonate another tool. Keep
that guarantee in a custom provider too.

`agent_id` is the acting agent's own identifier, for example its SPIFFE ID. It is read from the
context on each call, so one application can run several agents.

Whatever a provider returns, a value that is an object, a nested list or a mixed-type list is
dropped before the request is sent, with one warning per attribute
([Agent identity](agent-identity.md#values-the-pdp-cannot-carry)).

## Custom provider, client side

Map your application's field names, add attributes, or change the action and resource shape:

```python
from axiomatics_ai import AuthZConfig, DefaultClientContextProvider, create_authorized_agent, standard_tool_enforcement

class MyProvider(DefaultClientContextProvider):
    def get_subject(self, context) -> dict:
        return {"employee_id": context["emp_id"], "department": context["dept"]}

    def get_environment(self, context) -> dict:
        return {"region": context.get("region", "")}

    def get_resource(self, resource_name, context, *, args=None, resource_name_attribute=None) -> dict:
        resource = {**(args or {}), "classification": context.get("classification", "")}
        if resource_name_attribute:
            resource[resource_name_attribute] = resource_name  # set last: the identifier wins
        return resource

agent = create_authorized_agent(model=model, tools=tools, config=AuthZConfig(
    context_provider=MyProvider(), enforcement=standard_tool_enforcement(),
))
result = agent.invoke({"messages": [...]}, context={"emp_id": "E42", "dept": "eng", "region": "eu"})
```

The same dict reaches every provider method, so one provider serves every enforcement point.

## Custom provider, server side

Server paths read the `Authorization: Bearer` header. To verify the token and send named
attributes, use `AgentIdentityContextProvider`:

```python
from axiomatics_ai import AgentIdentityContextProvider, AuthZConfig, JwtVerification, PDPConnection, standard_tool_enforcement

config = AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement=standard_tool_enforcement(),
    context_provider=AgentIdentityContextProvider(
        verification=JwtVerification(issuer="https://idp.example.com", audience="https://tools.example.com"),
        claims={"realm_access.roles": "roles"},
    ),
)
```

[Agent identity](agent-identity.md) covers the verification settings, the attributes it sends,
extra claims, and RAR. Subclass it, or `DefaultServerContextProvider`, to read other headers:

```python
class MyProvider(AgentIdentityContextProvider):
    def get_environment(self, headers: dict) -> dict:
        return {**super().get_environment(headers), "session_id": headers.get("x-session-id", "")}
```

## Trust boundary

**By default the SDK authorizes; it does not authenticate.** `DefaultServerContextProvider`
reads the bearer JWT's claims **without verifying its signature** and forwards them as subject
attributes, and logs a one-time warning saying so at the first request. The helper is named
`decode_bearer_jwt_unverified` so the assumption is visible at the call site.

This default suits a deployment where authentication already happened at the edge; redoing it
here would mean this process needs your identity provider's keys for a job already done. The
requirement it places on you is a deployment one:

> Run every server-side surface (`fastmcp`, `tool_decorator`, `a2a`, `gateway`) behind an
> ingress, API gateway, or auth middleware that authenticates the request before it arrives.
> A token that reaches the provider unverified yields whatever subject it asserts.

If this process **is** your edge, or you want defence in depth, use
`AgentIdentityContextProvider` as shown above. See [Security](security.md) for the full list of assumptions.

## Ambient identity for RAG inside a tool

A vector store wrapped with `to_authorized_vectorstore` resolves the caller from an *ambient*
context, so retrieval code inside an agent tool needs no per-call wiring. Client paths populate
it from the agent's context; on a FastMCP server the middleware binds the request's headers for
the duration of each tool call; for a plain (non-agent) pipeline use `authz_context(user)`.
See [rag](integrations/rag.md).

---

← Previous: [Configuration](configuration.md) · [Docs index](README.md) · Next: [Agent identity](agent-identity.md) →
