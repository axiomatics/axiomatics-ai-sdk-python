# openai_agents — client-side PDP authorization for OpenAI Agents SDK agents

Attaches to an OpenAI Agents SDK `Agent` that consumes tools, for example MCP tools via
`MCPServerStreamableHttp`. It runs on the client side, inside the agent process. It enforces
`TOOL_VISIBILITY`, `BEFORE_TOOL` and `AFTER_TOOL` by wrapping the one seam the Runner calls
every turn, `agent.get_all_tools(run_context)`, and adds `BEFORE_MODEL` and `AFTER_MODEL` as SDK
guardrails. Identity comes from the `context` object you pass to `Runner.run`.

## Attach

`authorized_openai_agent(agent, config)` returns the same `Agent` instance with authorization
attached. It does not wrap or subclass the agent. It imports no `agents` module; it duck-types
the SDK objects, so the SDK adds no dependency and you bring your own agent.

```python
from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement, authorized_openai_agent

agent = authorized_openai_agent(
    Agent(name="pdp_agent", model=..., mcp_servers=[mcp_server]),
    AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement()),
)
result = await Runner.run(agent, question, context={"username": "alice"})
```

Only the points you configure in `enforcement` are wired. Passing no `config` uses
`AuthZConfig()` defaults.

## Enforcement points

| Point | Framework hook used | On Deny |
|---|---|---|
| `BEFORE_MODEL` (input guard) | an SDK `input_guardrail`; the point's LLM enrichers classify the prompt and the PDP decides | tripwire: the Runner raises `InputGuardrailTripwireTriggered`; catch it and read the refusal from `exc.guardrail_result.output.output_info["reason"]` |
| `TOOL_VISIBILITY` | `get_all_tools` returns all tools (native and MCP) every turn with the run context; one batched `get_permitted_tools` call (cached) filters the list | the model never sees the tool |
| `BEFORE_TOOL` | each surviving tool's `on_invoke_tool` is wrapped to check `authorized_tool_invocation` first | returns a denial string; the tool never runs |
| `AFTER_TOOL` | the same wrapper filters the tool's result records via the configured strategy (shared `core.records_helpers`) | denied records are removed before the model sees them |
| `AFTER_MODEL` (output guard) | an SDK `output_guardrail` on the final answer | tripwire: the Runner raises `OutputGuardrailTripwireTriggered` (same `output_info["reason"]` shape) |

Visibility is true and dynamic, and source-agnostic: `get_all_tools` yields native
`@function_tool` tools and MCP tools (converted to `FunctionTool`) alike.

**Tools that can't be checked call by call.** Hosted tools (`WebSearchTool`, `FileSearchTool`,
`CodeInterpreterTool`, `ImageGenerationTool`, `HostedMCPTool`) run on OpenAI's servers, and
`ShellTool`, `LocalShellTool`, `ApplyPatchTool` and `ComputerTool` run through the SDK's own
executors; none goes through `on_invoke_tool`. With `BEFORE_TOOL` enforced, the policy decides
them by name instead, once per turn, in one batched request with the `BEFORE_TOOL` action (the
event's `enforcement_point` is `tool_invocation`): a tool the policy doesn't permit is removed
from the list. The policy sees the tool's name, not its arguments. Per-call checks for the
tools that allow them (hosted MCP, shell, apply-patch, computer) are planned.

The input guard runs before the model (`run_in_parallel=False`), so a blocked prompt never
reaches it. A tool object shared by several agents is checked with the config and name of the
agent that calls it.

The content guards are SDK guardrails. Unlike the other client paths, they cannot substitute a
refusal in-band. A block raises a tripwire exception that the caller catches; this is the SDK's
native guardrail model. The guards need `enrichers=[...]` on the `Enforcer` and the
`llm-enrichment` extra. See [Content guards](../content-guards.md) and its
[per-framework table](../content-guards.md#per-framework-behaviour-on-a-block).

`BEFORE_TOOL` can also check each call against the user's intent, classified from the
`user`-role items of the current run's input only: for multi-turn context, pass earlier turns in
the input (e.g. `result.to_input_list()` plus the new message). See
[Intent-based access control](../intent-based-access.md).

```python
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

try:
    result = await Runner.run(agent, question, context={"username": "bob"})
except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as exc:
    refusal = exc.guardrail_result.output.output_info.get("reason")
```

Lifecycle hooks (`RunHooks` / `AgentHooks`) are intentionally not used. They are purely
observational (every method returns `None`) and cannot block a call or rewrite a result.

`A2A_INVOCATION` and retrieval are not part of this path; see [a2a](a2a.md) and [rag](rag.md).

## Sub-agents

`authorized_openai_agent` also authorizes the agents it can hand off to and the agents it uses
as tools (`agent.as_tool()`), with the same config. The subject carries `agent_name`, the agent's
`name`. To give one another config, authorize it first with that config. A `Handoff` built by
hand names no agent (a warning says so), and `agent.clone()` needs its own call. See
[Sub-agents](../enforcement-points.md#sub-agents).

## Human approval

When the policy asks for a human's approval ([Obligations and human approval](../human-approval.md))
and no `approver` is configured, the call goes through the SDK's own approval flow: each tool's
`needs_approval` also asks the PDP (combined with any `needs_approval` the tool already had), so
the run returns with a `ToolApprovalItem` in `result.interruptions`. Answer with
`state = result.to_state()` and `state.approve(item)` / `state.reject(item)`, then
`Runner.run(agent, state)`; an approved call is checked with the PDP again when it runs.
`state.approve(item, always_approve=True)` approves only the call that paused: the SDK would
mark every later call of the tool approved, whatever its arguments, so such a call is checked
without an approval, and is refused if the policy wants one. Approve each call (or give the
policy's approval a `ttl`) instead.
The `needs_approval` check is only a pre-check: it logs `approval_required` when it pauses and
nothing otherwise, and runs no obligation handlers, so each call is logged, and its handlers
run, once, by the check in `on_invoke_tool`.

A rejection is applied by the OpenAI SDK without calling this SDK, so it is logged on the next
turn, from what this process remembers. A run saved and resumed in another process is not
logged as rejected (the tool still never runs).

## Identity

The subject is read from the object passed to `Runner.run(agent, input, context=...)`. It is
reached as `run_context.context` in `get_all_tools`, and as `tool_context.context` inside a tool
call. Supply it per invocation, or reuse the same object across runs to bind once. It is coerced
to a dict and handed to `DefaultClientContextProvider`, which by default reads `username`,
`user_id`, `tenant_id` and `session_id`. Set your own provider via
`AuthZConfig(context_provider=...)`. See [Identity](../identity.md) and
[Custom provider, client side](../identity.md#custom-provider-client-side).

## Configuration notes

- Uses the shared `AuthZConfig`, same as every other path. Full reference:
  [Configuration](../configuration.md).
- There is no dedicated runtime extra for this path (as with `google_adk`), because the adapter
  imports no `agents`. Install the OpenAI Agents SDK yourself; it is included in the `examples`
  extra for the runnable examples.
- The content guards require the `llm-enrichment` extra.

## What the PDP receives

This path adds no deltas of its own. Tool names are sent as the SDK exposes them from
`get_all_tools`; MCP tools arrive already converted to `FunctionTool`, so they are not
namespaced by server. Subject and environment attributes come from the run context via
`DefaultClientContextProvider`. Request shapes per point are in
[Enforcement points](../enforcement-points.md); guard attributes are in
[Content guards](../content-guards.md#what-the-pdp-receives).

## Post-invocation strategies

Both built-in and custom `PostInvocationStrategy` implementations are supported, identical to
every other path. The path only locates the record list in the tool result (via the shared
`core.records_helpers`) and hands it to the `AFTER_TOOL` Enforcer's `strategy`. See
[Post-invocation](../post-invocation.md).

## Examples

| Script | What it shows | Domain file |
|---|---|---|
| [`mcp_agent.py`](../../examples/openai_agents/) | `TOOL_VISIBILITY`, `BEFORE_TOOL`, `AFTER_TOOL` on MCP tools; alice unrestricted, bob limited | `examples/policies/domain_all_authz_enabled.yaml` |
| [`mcp_backend.py`](../../examples/openai_agents/) | the standalone MCP backend `mcp_agent.py` talks to (`http://127.0.0.1:9330/mcp`) | none |
| [`llm_content_guard.py`](../../examples/openai_agents/) | `BEFORE_MODEL` topic guard and `AFTER_MODEL` PII guard as tripwire guardrails | `examples/policies/domain_llm_content_guard.yaml` |
| [`human_approval.py`](../../examples/openai_agents/) | Human approval: a large payment or a cancellation pauses the run (`result.interruptions`) for your `approve` / `reject` | `examples/policies/domain_human_approval.yaml` |

`mcp_agent.py` needs three processes: Axiomatics ADS with the domain loaded, the MCP backend,
then the agent. Config in `.env`: `PDP_BASE_URL`, `MCP_URL` (default `http://127.0.0.1:9330/mcp`),
`LLM_MODEL` (or `OPENAI_LLM_MODEL`), and an OpenAI-compatible endpoint: `OPENAI_BASE_URL` +
`OPENAI_API_KEY`, or `OPENROUTER_API_KEY` (defaults to OpenRouter). See
[Running Axiomatics ADS](../running-ads.md) and [examples/openai_agents/](../../examples/openai_agents/).

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
