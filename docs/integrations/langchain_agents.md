# langchain_agents — authorization middleware for LangChain agents

This path attaches to `langchain.agents.create_agent` (and `deepagents.create_deep_agent`) as
an agent middleware. It runs client side, inside the agent process. It can enforce
`BEFORE_MODEL`, `TOOL_VISIBILITY`, `BEFORE_TOOL`, `AFTER_TOOL`, and `AFTER_MODEL`. Identity comes
from the `context=` object you pass to `agent.invoke(...)`. All decisions come from the PDP's
policy; no policy logic lives in application code.

Install the `langchain-agents` extra, and `deepagents` too for `create_authorized_deep_agent`.

## Attach

`create_authorized_agent()` is a one-call factory over `create_agent`. It injects
`LangChainAgentAuthZMiddleware(config)` for you and forwards everything else (`model`, `tools`,
`system_prompt`, …) verbatim.

```python
from langchain_litellm import ChatLiteLLM
from axiomatics_ai import create_authorized_agent, AuthZConfig, PDPConnection, standard_tool_enforcement

model = ChatLiteLLM(model="openai/gpt-5.6-luna")

agent = create_authorized_agent(
    model=model,
    tools=tools,
    config=AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),
    ),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What can you help me with?"}]},
    context={"username": "alice"},
)
```

Any `middleware=` you pass is preserved: the authorization middleware runs outermost, yours
after. An already-present `LangChainAgentAuthZMiddleware` is not duplicated.

For **deep agents**, use `create_authorized_deep_agent(...)`. Same signature; it wraps
`deepagents.create_deep_agent` and requires the `deepagents` package.

The factory is sugar over passing the middleware directly. Do it by hand when you want full
control of the middleware list:

```python
from langchain.agents import create_agent
from axiomatics_ai import LangChainAgentAuthZMiddleware, AuthZConfig, PDPConnection, standard_tool_enforcement

config = AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement=standard_tool_enforcement(),
)
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[LangChainAgentAuthZMiddleware(config)],   # add your own middleware alongside as needed
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What can you help me with?"}]},
    context={"username": "alice"},
)
```

## Enforcement points

| Point | Framework hook used | On Deny |
|---|---|---|
| `BEFORE_MODEL` | `wrap_model_call` / `awrap_model_call`, before the handler | prompt blocked; a refusal is substituted in-band and the model is never called |
| `TOOL_VISIBILITY` | `wrap_model_call` / `awrap_model_call`, `request.override(tools=...)` | tool removed from the list before the model call |
| `BEFORE_TOOL` | `wrap_tool_call` / `awrap_tool_call`, before the tool runs | an access-denied `ToolMessage` is returned to the model; the tool is not invoked |
| `AFTER_TOOL` | `wrap_tool_call` / `awrap_tool_call`, on the tool result | result filtered (records removed) or masked (fields replaced), per the strategy |
| `AFTER_MODEL` | `wrap_model_call` / `awrap_model_call`, after the handler | answer replaced by a refusal |

The guards use the Enforcer's `block_message` or a per-point default for the refusal. Only
real user prompts and the model's text are inspected, including text sent alongside a tool call
(a block keeps the tool call); reasoning traces are skipped. Streamed tokens reach the user
before the output guard decides ([Content guards](../content-guards.md)).

`BEFORE_TOOL` can also check each call against the user's intent, classified from the
`HumanMessage`s in the agent state (deepagents included); see
[Intent-based access control](../intent-based-access.md).

`A2A_INVOCATION` and retrieval (`RetrievalConfig`) are not part of this path. They belong to
[a2a](a2a.md) and [rag](rag.md).

## Sub-agents

`create_authorized_deep_agent` authorizes every deepagents sub-agent with the same config: each
spec in `subagents` and the general-purpose one get the middleware, and a compiled sub-agent must
come from `create_authorized_agent` / `create_authorized_deep_agent`. The subject carries
`agent_name`: the agent's `name=`, or the sub-agent's. To give a sub-agent another config, put
your own `LangChainAgentAuthZMiddleware(other_config)` in its spec's `middleware`. See
[Sub-agents](../enforcement-points.md#sub-agents).

## Human approval

When the policy asks for a human's approval ([Obligations and human approval](../human-approval.md))
and no `approver` is configured, the tool call pauses the graph with LangGraph's `interrupt()`.
The run returns `__interrupt__` carrying the approval request; resume with
`Command(resume={"approved": True})` (or `False`), and the call is checked with the PDP again.
`interrupt()` needs the agent to have a **checkpointer** (`create_agent(..., checkpointer=...)`)
and a `thread_id`:

```python
result = agent.invoke({"messages": [("user", "Pay invoice INV-43.")]}, config, context=ctx)
while "__interrupt__" in result:
    request = result["__interrupt__"][0].value        # tool_name, args, reason, ...
    result = agent.invoke(Command(resume={"approved": ask_the_user(request)}), config, context=ctx)
```

With an `approver` configured, it answers inside the tool call and the graph never pauses.

## Identity

The middleware reads the context object you pass to `agent.invoke(..., context=ctx)`.
`DefaultClientContextProvider` looks for these attributes (as dict keys or object attributes):

| Key | XACML category | Notes |
|---|---|---|
| `username` | Subject | |
| `user_id` | Subject | |
| `tenant_id` | Subject | |
| `session_id` | Environment | Forwarded as an environment attribute |

All four are sent to the PDP on every request. The decision cache is keyed on the full set of
inputs (subject, action, resource, environment), so any change in these values produces a
distinct cache entry.

Implement a custom provider when your runtime context uses different field names, or when you
need extra attributes (e.g. `region`, `classification`) taken from the context:

```python
from axiomatics_ai import AuthZConfig, DefaultClientContextProvider, create_authorized_agent, standard_tool_enforcement

class MyProvider(DefaultClientContextProvider):
    def get_subject(self, context):
        return {"employee_id": context["emp_id"], "department": context["dept"]}

    def get_environment(self, context):
        env = super().get_environment(context)
        env["region"] = context.get("region", "")
        return env

    def get_resource(self, resource_name, context, *, args=None, resource_name_attribute=None):
        resource = {**(args or {}), "classification": context.get("classification", "")}
        if resource_name_attribute:
            resource[resource_name_attribute] = resource_name  # set last: the identifier wins
        return resource

agent = create_authorized_agent(model=model, tools=tools,
    config=AuthZConfig(context_provider=MyProvider(), enforcement=standard_tool_enforcement()))
```

All four methods receive the same runtime context object. Override only what you need; the
rest keeps its default behaviour. See [Identity](../identity.md#custom-provider-client-side)
for the full `ContextProvider` contract and the trust boundary.

## Configuration notes

A configuration that enables every point this path supports:

```python
AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement={
        # Input content guard (optional) — an LLM enricher classifies the prompt; the PDP decides.
        EnforcementPoint.BEFORE_MODEL:    Enforcer(action_name="before_model",
                                                   enrichers=[prompt_injection_enricher(model="openai/gpt-5.6-luna")]),
        EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="tool_visibility"),
        EnforcementPoint.BEFORE_TOOL:     Enforcer(action_name="tool_invocation"),
        EnforcementPoint.AFTER_TOOL:      Enforcer(
            action_name="post_invocation",
            # strategy=ColumnMaskStrategy("ssn", "dob"),  # default: RecordFilterStrategy()
        ),
        # Output content guard (optional) — classifies the final answer.
        EnforcementPoint.AFTER_MODEL:     Enforcer(action_name="after_model", enrichers=[pii_enricher()]),
    },
)
```

`standard_tool_enforcement()` is the shorthand for the three tool points. Decisions are cached
within the session for the configured TTL. See [Configuration](../configuration.md) for every
`AuthZConfig` field, protocols, and PDP compatibility.

## What the PDP receives

The request shapes are the shared ones described in
[Enforcement points](../enforcement-points.md). This path's specifics, per point:

### Content guards (`BEFORE_MODEL` / `AFTER_MODEL`)

One request per model turn (input) or final answer (output). The point's LLM enrichers
classify the prompt or answer into resource attributes (e.g. `injection_detected`,
`contains_ssn`); the PDP decides.

- **Subject** — from `get_subject()`
- **Action** — `{"name": "before_model"}` / `{"name": "after_model"}` (configurable via each Enforcer's `action_name`)
- **Resource** — **only the enricher attributes** and each `<name>_available` sentinel (no identifier; the action names the point)
- **Environment** — from `get_environment()`

See [Content guards](../content-guards.md) for the enrichers and the policy.

### Tool listing (`TOOL_VISIBILITY`)

One batch request per model turn (XACML MultiRequest, or one AuthZEN `evaluations` request),
one sub-request per tool:

- **Subject** — from `get_subject()`
- **Action** — `{"name": "tool_visibility"}` (configurable via the visibility Enforcer's `action_name`)
- **Resource** — `{"tool_name": "<tool_name>"}` (`AuthZConfig.attribute_names.tool`)

Tools that receive `Deny` (or no matching rule when `fail_open=False`) are removed before the
model call.

### Tool invocation (`BEFORE_TOOL`)

One request per tool invocation, before the tool runs:

- **Subject** — from `get_subject()`
- **Action** — `{"name": "tool_invocation"}` (configurable via the `BEFORE_TOOL` Enforcer's `action_name`)
- **Resource** — `{"tool_name": "<tool_name>"}` plus all tool call arguments (e.g. `{"account_id": "ACC-001"}`)
- **Environment** — from `get_environment()`, plus the [intent attributes](../intent-based-access.md#what-the-pdp-receives) when an `intent_enricher` is configured

### Post-invocation (`AFTER_TOOL`)

The PDP call depends on the active strategy (the `AFTER_TOOL` Enforcer's `strategy`):

- **`RecordFilterStrategy`** (default) — one batch request, one sub-request per record (record fields plus the `record_hash` correlation key). Records that receive `Deny` are removed.
- **`ColumnMaskStrategy`** — one request for the whole result, with the tool-invocation resource (`tool_name` plus arguments). On `Deny`, named fields are replaced with `***`.

The action name is the `AFTER_TOOL` Enforcer's `action_name`, or the per-tool override from
that Enforcer's `action_overrides`.

## Post-invocation strategies

See [Post-invocation](../post-invocation.md) for the full reference, `PostInvocationContext`
fields, and the custom strategy guide.

```python
from axiomatics_ai import AuthZConfig, ColumnMaskStrategy, Enforcer, EnforcementPoint, PDPConnection, create_authorized_agent

agent = create_authorized_agent(model=model, tools=tools, config=AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement={EnforcementPoint.AFTER_TOOL: Enforcer(
        action_name="post_invocation",
        strategy=ColumnMaskStrategy("ssn", "dob"),
    )},
))
```

## Examples

Runnable scripts live in [`examples/langchain_agents/`](../../examples/langchain_agents/). Each
assumes a PDP is already running with the listed domain (see
[Running Axiomatics ADS](../running-ads.md)) and reads `PDP_BASE_URL` (default
`http://localhost:8080`), `LLM_MODEL`, and your provider key from the environment or `.env`.

| Script | Domain (`examples/policies/`) | What it shows |
|---|---|---|
| `create_authorized_agent.py` | `domain_all_authz_enabled.yaml` | The **recommended** entry point: `create_authorized_agent(...)`, plus composing your own middleware |
| `all_authz_enabled.py` | `domain_all_authz_enabled.yaml` | All tool enforcement points at once (visibility + invocation + post-invocation) |
| `explicit_middleware.py` | `domain_tool_visibility_only.yaml` | Passing `LangChainAgentAuthZMiddleware` yourself in `middleware=[...]` |
| `tool_call_authz.py` | `domain_tool_invocation.yaml` | Blocking individual tool calls via a `BEFORE_TOOL` Enforcer |
| `record_filtering.py` | `domain_post_invocation.yaml` | Filtering records from a tool result via an `AFTER_TOOL` Enforcer |
| `custom_context_provider.py` | `domain_custom_context_provider.yaml` | A custom `ContextProvider` mapping app fields to XACML categories |
| `llm_content_guard.py` | `domain_llm_content_guard.yaml` | Input/output LLM content guards (`BEFORE_MODEL` + `AFTER_MODEL`) |
| `intent_based_access.py` | `domain_intent.yaml` | Intent-based access control: a tool call the user did not ask for (an injected "send") is denied at `BEFORE_TOOL` |
| `agent_behind_api.py` | `domain_agent_identity.yaml` | A LangChain agent behind a `/chat` endpoint that verifies the user's access token client-side (`AgentIdentityContextProvider(token_from=...)`) and pays only within the user's approved grant. |
| `simple-ui/` | `domain_simple_ui.yaml` | Streamlit demo with `alice`/`bob` users and a live PDP driving visibility, invocation, and post-invocation |
| `human_approval.py` | `domain_human_approval.yaml` | Human approval: a large payment or a cancellation pauses the graph (`interrupt()`) for your answer; `--approver` asks through an approver callback instead |

```bash
python examples/langchain_agents/create_authorized_agent.py
```

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
