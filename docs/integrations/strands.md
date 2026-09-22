# strands — client-side authorization for Strands agents

This path attaches to a Strands `Agent` that consumes tools, for example MCP tools via
`MCPClient` or native `@tool` functions. It runs on the client side, in the agent process. It
can enforce the input guard (`BEFORE_MODEL`), `TOOL_VISIBILITY`, `BEFORE_TOOL`, `AFTER_TOOL`, and
the output guard (`AFTER_MODEL`). Identity comes from the agent's persistent state merged with
the per-invocation `invocation_state`, read by the client-side context provider.

## Attach

`authorized_strands_agent(agent, config)` attaches one Strands *intervention handler* (the
newer, higher-level control API) to an already-constructed `Agent` and returns the same
instance. It does **not** wrap or subclass the agent or its toolsets.

```python
from strands import Agent
from strands.models.litellm import LiteLLMModel

from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement, authorized_strands_agent

agent = authorized_strands_agent(
    Agent(model=LiteLLMModel(model_id="openai/gpt-5.6-luna"), tools=[mcp_client], state={"username": "alice"}),
    AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement()),
)
```

Only the points you configure in `enforcement` are wired. Each handler method gates itself on
the config, so the single handler is registered onto the agent's existing hook registry
regardless. Any handlers already configured are preserved.

## Enforcement points

| Enforcement point | Strands hook | On Deny |
|---|---|---|
| `BEFORE_MODEL` (input guard) | `before_model_call` — the point's LLM enrichers classify the prompt; the PDP decides | returns `Deny(reason=...)`; Strands cancels the model call and surfaces the reason as the answer |
| `TOOL_VISIBILITY` | `before_model_call` runs one batched `get_permitted_tools` (cached); a non-destructive filter on the agent's tool registry then drops denied tools from the specs the model receives (read live before every model call) | the model never sees the tool |
| `BEFORE_TOOL` | `before_tool_call` — `authorized_tool_invocation` | returns `Deny(reason=...)`; the tool never runs |
| `AFTER_TOOL` | `after_tool_call` — the configured strategy filters the result via `Transform` (records in a native `@tool`'s `text` / `json` content blocks, or an MCP tool's `structuredContent`) | denied records removed before the model sees them |
| `AFTER_MODEL` (output guard) | `after_model_call` — enrichers classify the model's text, including text sent alongside a tool call; the PDP decides | returns `Transform` rewriting the text with the refusal; `toolUse` blocks are kept (there is no `Deny` at this point) |

Visibility is **true and dynamic**, not just invocation-time blocking. It is also
**source-agnostic**: it works for MCP tools and native `@tool` functions alike, because it
filters the registry the event loop reads, not the tool source.

The content guards need `enrichers=[...]` on the `Enforcer` and the `llm-enrichment` extra. See
[Content guards](../content-guards.md). Shapes for every point are in
[Enforcement points](../enforcement-points.md).

`BEFORE_TOOL` can also check each call against the user's intent, classified from the
`user`-role messages in the agent history (tool-result messages skipped); see
[Intent-based access control](../intent-based-access.md).

## Sub-agents

`authorized_strands_agent` also authorizes the agents it uses as tools (`agent.as_tool()`), with
the same config and the caller's identity (Strands starts that agent without the caller's
invocation state; the SDK passes the identity on). The subject carries `agent_name`, the agent's
`name`. To give one another config, authorize it first with that config. See
[Sub-agents](../enforcement-points.md#sub-agents).

## Human approval

When the policy asks for a human's approval ([Obligations and human approval](../human-approval.md))
and no `approver` is configured, `before_tool_call` pauses the agent with a Strands interrupt
named `axiomatics-approval` (the result's `stop_reason` is `"interrupt"`; each interrupt's
`reason` is the approval request). Resume by calling the agent with an `interruptResponse` per
interrupt, `{"approved": True}` or `False`; the hook runs again with the answer, and the call is
checked with the PDP again. Strands' own `Confirm` action is not used, because it would run
the tool on a yes without asking the PDP.

## Identity

The subject is read by merging two dicts:

- **Bind once per user:** the agent's persistent state, `Agent(..., state={"username": "alice"})`.
- **Per invocation:** the `invocation_state`, `agent("prompt", username="alice")`.

The per-invocation values take precedence. The merged dict is handed to the client-side
`DefaultClientContextProvider`. Set a custom provider via `config.context_provider`. See
[Identity](../identity.md), in particular [Custom provider, client side](../identity.md#custom-provider-client-side).

## Configuration notes

Requires the `strands` extra. The adapter registers a Strands intervention handler, so it imports
Strands at runtime.

```bash
pip install "axiomatics-ai-sdk[strands]"
```

The path uses the shared `AuthZConfig` (PDP endpoint, action names, fail-open, cache,
post-invocation strategy), the same as every other path. Full reference:
[Configuration](../configuration.md).

**One `Agent` per user or session.** The handler keeps the current turn's permitted tools on
the agent, so one `Agent` instance must not serve two users at once. Strands' default
(`ConcurrentInvocationMode.THROW`) refuses a second concurrent invocation, which is safe. Under
`UNSAFE_REENTRANT`, one user's model can be shown another user's permitted tools (each call is
still checked against the right identity at `BEFORE_TOOL`).

## What the PDP receives

This path adds no deltas of its own. Requests follow the shared shapes in
[Enforcement points](../enforcement-points.md); the subject attributes come from the merged
identity dict described above.

## Post-invocation strategies

Both built-in and custom `PostInvocationStrategy` implementations are supported, identical to
every other path. The path only *locates* the record list in the tool result (via the shared
`core.records_helpers`) and hands it to the `AFTER_TOOL` Enforcer's `strategy`. See
[Post-invocation](../post-invocation.md).

## Examples

| Script | What it shows | Domain file |
|---|---|---|
| [`mcp_agent.py`](../../examples/strands/) | A Strands agent consuming MCP tools with `TOOL_VISIBILITY`, `BEFORE_TOOL`, and `AFTER_TOOL` enforced at once; identity bound once per user via agent state | `examples/policies/domain_all_authz_enabled.yaml` |
| [`mcp_backend.py`](../../examples/strands/) | The standalone MCP backend the agent example consumes (`http://127.0.0.1:9320/mcp`) | — |
| [`llm_content_guard.py`](../../examples/strands/) | Input guard (`BEFORE_MODEL`) and output guard (`AFTER_MODEL`) classifying prompt and answer text with an LLM enricher; the PDP allows, blocks, or replaces | `examples/policies/domain_llm_content_guard.yaml` |
| [`human_approval.py`](../../examples/strands/) | Human approval: a large payment or a cancellation pauses the agent with a Strands interrupt for your answer | `examples/policies/domain_human_approval.yaml` |

`mcp_agent.py` needs three processes, each in its own terminal: the PDP with the domain loaded,
`mcp_backend.py`, then `mcp_agent.py`. Config in `.env`: `PDP_BASE_URL`, `MCP_URL` (default
`http://127.0.0.1:9320/mcp`), `LLM_MODEL` (or `STRANDS_LLM_MODEL`), `OPENROUTER_API_KEY`.
Expected output under `domain_all_authz_enabled.yaml`: alice is unrestricted; bob does not see
`send_email`, is blocked on `get_account_balance` for `ACC-002`, and has the `classified` record
filtered out of `search_docs` results.

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
