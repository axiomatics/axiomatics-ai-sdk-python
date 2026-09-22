# google_adk — client-side PDP authorization for Google ADK agents

This path attaches to a Google ADK `Agent` that consumes tools (for example MCP tools via
`McpToolset`). It is client-side: enforcement runs in the agent process, before and after the
model and tool calls. It can enforce `BEFORE_MODEL`, `TOOL_VISIBILITY`, `BEFORE_TOOL`,
`AFTER_TOOL`, and `AFTER_MODEL` by attaching ADK callbacks. Identity comes from the ADK session
state.

## Attach

`authorized_adk_agent(agent, config)` is the one-call entry point. It adds callbacks to the
agent and returns the same instance. It does not wrap or subclass the agent or its toolsets.

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from axiomatics_ai import AuthZConfig, PDPConnection, authorized_adk_agent, standard_tool_enforcement

agent = authorized_adk_agent(
    Agent(model=LiteLlm(model="openai/gpt-5.6-luna"), name="pdp_agent", tools=[mcp_toolset]),
    AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),
    ),
)
```

Only the points you configure in `enforcement` are wired. Callbacks already set on the agent are
preserved: the SDK appends its own after them.

Importing the module needs no ADK. It duck-types the objects ADK passes to its callbacks, so the
SDK adds no dependency — you bring your own ADK agent. The only ADK import happens lazily, when
a content guard has to build a refusal `LlmResponse`.

## Enforcement points

| Point | ADK hook | On Deny |
|---|---|---|
| `BEFORE_MODEL` (input guard) | `before_model_callback` — the point's LLM enrichers classify the prompt; the PDP decides | returns a refusal `LlmResponse`; the model is never called |
| `TOOL_VISIBILITY` | `before_model_callback` — one batched `get_permitted_tools` call (cached), then denied tools are removed from the `LlmRequest` | the model never sees the tool |
| `BEFORE_TOOL` (invocation) | `before_tool_callback` — `authorized_tool_invocation` | returns a denial dict; the tool never runs |
| `AFTER_TOOL` (post-invocation) | `after_tool_callback` — the configured strategy filters the result (handles the MCP `CallToolResult` shape) | denied records are removed before the model sees them |
| `AFTER_MODEL` (output guard) | `after_model_callback` — enrichers classify the final answer; the PDP decides | returns a replacement `LlmResponse` carrying the refusal |

`BEFORE_MODEL` and `TOOL_VISIBILITY` share one `before_model_callback`. The input guard runs
first; a block short-circuits the model. Visibility then prunes both the request's tool map and
the function declarations the model sees.

The two content guards inspect only real user prompts and the model's complete responses,
including text sent alongside a function call (a block keeps the call). Function-response
continuation turns and streaming partials are skipped; partials reach the user before the
output guard decides.

Our callbacks are placed **before** any already on the agent. ADK stops at the first callback
that returns a value, so behind a user callback that does, the answer check or result filtering
would never run. The
guards need `enrichers=[...]` on the point's `Enforcer` and the `llm-enrichment` extra. See
[Content guards](../content-guards.md) for the presets (`prompt_injection_enricher`,
`pii_enricher`) and [per-framework behaviour on a block](../content-guards.md#per-framework-behaviour-on-a-block).

`BEFORE_TOOL` can also check each call against the user's intent, classified from the
user-authored session events (function responses skipped); see
[Intent-based access control](../intent-based-access.md).

Post-invocation returns the tool result unchanged when it contains nothing filterable, or when
no `AFTER_TOOL` strategy is configured. See [Post-invocation](../post-invocation.md).

`A2A_INVOCATION` and retrieval (`RetrievalConfig`) are not wired by this path.

## Sub-agents

`authorized_adk_agent` also authorizes every agent below the one you pass: `sub_agents` (also
through `SequentialAgent` and other workflow agents) and the agent an `AgentTool` wraps, with the
same config. The subject carries `agent_name`, the ADK agent's `name`. To give one another
config, authorize it first with that config. See
[Sub-agents](../enforcement-points.md#sub-agents).

## Human approval

When the policy asks for a human's approval ([Obligations and human approval](../human-approval.md))
and no `approver` is configured, `before_tool_callback` pauses the call with ADK tool
confirmation (`tool_context.request_confirmation`). The event stream carries an
`adk_request_confirmation` function call; its `args["toolConfirmation"]["payload"]` is the
approval request. Answer with a `FunctionResponse` of the same name and id,
`{"confirmed": True, "payload": {"approver": "..."}}`; ADK runs the callback again with the
answer, and the call is checked with the PDP again.

## Identity

The PDP subject is read from the ADK **session state** (`context.state`) of the callback or
tool context. ADK's `State` is dict-like but not a dict, so the SDK coerces it with `to_dict()`
and hands the result to the shared `DefaultClientContextProvider`. Set `config.context_provider`
to change the mapping.

Populate the state either way — your choice, not forced:

- **Bind once per user:** `session_service.create_session(..., state={"username": "alice"})`
- **Per invocation:** `runner.run_async(..., state_delta={"username": "alice"})`

If the context has no state, the provider receives an empty dict. See
[Identity](../identity.md) for the base classes, the
[default attribute names](../identity.md#default-attribute-names), and how to write a
[custom client-side provider](../identity.md#custom-provider-client-side).

## Configuration notes

This path uses the shared `AuthZConfig` (PDP endpoint, action names, fail-open, cache,
post-invocation strategy), the same as every other path. Nothing is specific to ADK. See
[Configuration](../configuration.md) for the full reference.

`fail_open` for post-invocation is taken from `config.connection.fail_open`.

## What the PDP receives

- Tool names are the ADK tool's `name`, as ADK exposes it to the model (for MCP tools, the MCP
  tool name).
- For `BEFORE_TOOL`, the resource is built from the tool name and the call's `args`.
- For `AFTER_TOOL`, the strategy's `action_name` can be overridden per tool via
  `action_overrides`.
- Content guards send the enrichers' classification of the prompt or answer.

See [Enforcement points](../enforcement-points.md) for the request shapes and
[attribute conventions](../enforcement-points.md#attribute-conventions).

## Examples

All scripts read `PDP_BASE_URL` (default `http://localhost:8080`), `LLM_MODEL` (or
`GOOGLE_ADK_LLM_MODEL`), and `OPENROUTER_API_KEY` from `.env`. Start the PDP with the listed
domain first (see [Running Axiomatics ADS](../running-ads.md)).

| Script | What it shows | Domain file |
|---|---|---|
| `mcp_agent.py` | Tool visibility, tool invocation, and post-invocation record filtering on an agent consuming MCP tools. Identity bound once per user via `create_session(state=...)`. | `examples/policies/domain_all_authz_enabled.yaml` |
| `mcp_backend.py` | The standalone MCP backend for `mcp_agent.py`; serves `http://127.0.0.1:9310/mcp` (override with `MCP_URL`). | — |
| `llm_content_guard.py` | Input guard (`BEFORE_MODEL`) and output guard (`AFTER_MODEL`) classifying prompt and answer with an LLM enricher. | `examples/policies/domain_llm_content_guard.yaml` |
| `human_approval.py` | Human approval: a large payment or a cancellation pauses the run with ADK tool confirmation for your answer | `examples/policies/domain_human_approval.yaml` |

Run `mcp_agent.py` as three processes, each in its own terminal:

```bash
# 1. the PDP with the all-enabled domain (point deployment.yaml at domain_all_authz_enabled.yaml)
cd examples/policies && java -jar access-decision-service-<version>.jar

# 2. the MCP backend
python examples/google_adk/mcp_backend.py

# 3. the agent
python examples/google_adk/mcp_agent.py
```

Expected output (alice is unrestricted; bob is limited by `domain_all_authz_enabled.yaml`):

```
── 1. Tool visibility ──
[alice] search_docs, get_account_balance, send_email
[bob]   search_docs, get_account_balance                     # send_email hidden

── 2. Tool invocation (get_account_balance ACC-002) ──
[alice] The balance for account ACC-002 is $1,000.
[bob]   I do not have permission to access the balance for ACC-002.   # blocked before the tool ran

── 3. Post-invocation record filtering (search_docs) ──
[alice] d1 (public), d2 (classified)
[bob]   d1 (public)                                          # the 'classified' record was filtered out
```

Sources: [examples/google_adk/](../../examples/google_adk/).

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
