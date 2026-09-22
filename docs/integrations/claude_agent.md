# claude_agent — client-side authorization for the Claude Agent SDK

The Claude Agent SDK has no `Agent` object. You pass a `ClaudeAgentOptions` to `query()` or
`ClaudeSDKClient`, and the SDK drives the Claude Code runtime. So the SDK attaches to the
*options*, on the client side. It enforces `BEFORE_MODEL` (input guard), `BEFORE_TOOL` (tool
invocation) and `AFTER_TOOL` (post-invocation). Identity is a dict you pass once, per user, when
you build the options.

## Attach

`authorized_claude_options(options, config, *, identity)` returns a copy of `options` with PDP
enforcement wired in. It keeps any `can_use_tool` and `hooks` already set on the options.

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from axiomatics_ai import AuthZConfig, PDPConnection, authorized_claude_options, standard_tool_enforcement

base = ClaudeAgentOptions(
    model=...,
    mcp_servers={"acme": {"type": "http", "url": MCP_URL}},
)
options = authorized_claude_options(
    base,
    AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),
    ),
    identity={"username": "alice"},
)

async with ClaudeSDKClient(options=options) as client:   # or query(prompt=..., options=options)
    await client.query("...")
    async for message in client.receive_response():
        ...
```

`authorized_claude_options` adds only what `config` enforces. A `TOOL_VISIBILITY` or
`AFTER_MODEL` `Enforcer` is ignored on this path, with a warning (see below).

## Enforcement points

| Point | Framework hook used | On Deny |
|---|---|---|
| `BEFORE_MODEL` (input guard) | a `UserPromptSubmit` hook: the point's LLM enrichers classify the prompt; the PDP decides | **hard block** — returns `{"decision": "block", "reason": ...}`; the CLI never sends the turn to the model and the reason is surfaced (no prompt rewrite, no model-authored refusal) |
| `BEFORE_TOOL` (tool invocation) | a `PreToolUse` hook runs `authorized_tool_invocation` with the tool's args as resource attributes (arg-level) | returns `permissionDecision: "deny"` with the reason; the tool never runs. On Permit it returns `permissionDecision: "allow"` |
| `AFTER_TOOL` (post-invocation) | a `PostToolUse` hook filters the result's records via the configured strategy | returns `updatedToolOutput` with denied records removed |
| `TOOL_VISIBILITY` | **not supported** | — |
| `AFTER_MODEL` (output guard) | **not supported** | — |

The input guard needs `enrichers=[...]` on the `BEFORE_MODEL` `Enforcer`. See
[Content guards](../content-guards.md); it requires the `llm-enrichment` extra. Post-invocation
strategies are described in [Post-invocation](../post-invocation.md).

`BEFORE_TOOL` can also check each call against the user's intent
([Intent-based access control](../intent-based-access.md)). The `PreToolUse` hook sees a tool
call's session but not the conversation, so the `UserPromptSubmit` hook keeps each session's
recent prompts (it is installed for this even without a `BEFORE_MODEL` guard, and then never
blocks). One options object can therefore serve several conversations without mixing their
intent; a tool call whose session is unknown is denied.

### Why a hook, not `can_use_tool`

The CLI skips the `can_use_tool` permission callback for tools listed in `allowed_tools`, under
`permission_mode="bypassPermissions"`, and for allow rules in settings files. A policy enforced
there would silently stop applying. The `PreToolUse` hook runs for every tool call, in every
permission mode, and in both `ClaudeSDKClient` and one-shot `query()`. It is placed before any
`PreToolUse` hooks already on the options.

**The policy decides.** A permitted call returns `permissionDecision: "allow"`, so it runs
without `allowed_tools` or a permission mode, and without a permission prompt. A denied call
returns `"deny"`. To add a check of your own on top of the policy, add your own `PreToolUse`
hook: a deny from any hook blocks the call. A `can_use_tool` callback on the options is not
consulted for a call the policy permitted.

**Tool visibility is not supported.** The Claude Agent SDK exposes no per-turn tool-list filter
and no runtime identity at tool-list assembly, so visibility cannot be enforced per subject.
Write your policies at the invocation point instead; they take effect in the `PreToolUse` hook.
This is unaffected by MCP tool search: `PreToolUse` fires at call time and `PostToolUse` at
result time, whether or not tool definitions were deferred.

**Output guard (`AFTER_MODEL`) is not supported.** The SDK exposes no hook that can see or
redact the assistant's final answer. It is delivered as an observe-only `AssistantMessage` in
the receive stream, and none of the hook events can intercept it: `UserPromptSubmit` is
input-only, `Pre/PostToolUse` only see tool calls, and `Stop` receives just `stop_hook_active`
(its block forces *more* generation, not redaction). If an app funnels its answer through a
designated tool, guard that tool at `AFTER_TOOL` instead.

**Deny does not cancel the run.** A denied call gets `permissionDecision: "deny"` with the
reason. The SDK blocks that tool call, tells the model why, and the agent loop continues; it is
not a hard stop. Whether the model then narrates
the denial ("I'm not authorized...") or moves on depends on the model. Smaller models sometimes
retry or wander until `max_turns` and produce no final text. The enforcement is unaffected: the
blocked tool never runs and no denied data is returned.

## Human approval

The Claude Agent SDK has no interrupt to pause a tool call on, so a call the policy wants a
human to approve ([Obligations and human approval](../human-approval.md)) is approved only by the
configured `approver`: the `PreToolUse` hook waits for it, in process. Without an approver the
call is denied, and the refusal is logged with `source: "none"`.

## Identity

The SDK's callbacks carry no app-supplied identity, so `identity` is captured by closure when
you call `authorized_claude_options`. Build **one options object per user**; the same options
cannot serve different subjects.

The `identity` dict is handed to the context provider as the runtime context. With the default
`DefaultClientContextProvider`, the keys `username`, `user_id`, `tenant_id` become subject
attributes and `session_id` becomes an environment attribute. Pass your own provider via
`AuthZConfig(context_provider=...)` to map other keys. See [Identity](../identity.md).

## Configuration notes

- Requires the `claude-agent-sdk` extra; the adapter returns SDK types.
  ```bash
  pip install "axiomatics-ai-sdk[claude-agent-sdk]"
  ```
- Uses the shared `AuthZConfig`, the same as every other path. Full reference in
  [Configuration](../configuration.md).
- Every hook works in both `ClaudeSDKClient` (streaming) and one-shot `query(prompt=<str>)`.

## What the PDP receives

- **MCP tool names are split.** Claude namespaces MCP tools as `mcp__<server>__<tool>`. The
  resource gets the bare tool name (`tool_name` = `search_docs`), so one policy is portable
  across all paths, and the server as its own attribute (`mcp_server` = `acme`,
  `AuthZConfig.attribute_names.mcp_server`). A policy can therefore tell
  `mcp__trusted__search_docs` from `mcp__other__search_docs`. Built-in tools (`Read`, `Bash`, …)
  carry no `mcp_server`.
- Tool invocation is arg-level: the tool's arguments are sent as resource attributes alongside
  `tool_name`.

Request shapes for each point are in [Enforcement points](../enforcement-points.md).

## Examples

| Script | What it shows | Domain file |
|---|---|---|
| `mcp_agent.py` | Tool invocation (arg-level) and post-invocation over MCP tools, via `ClaudeSDKClient` | `examples/policies/domain_all_authz_enabled.yaml` |
| `mcp_backend.py` | The standalone MCP backend the agent consumes (`http://127.0.0.1:9340/mcp`) | — |
| `llm_content_guard.py` | Input guard (`BEFORE_MODEL`) with an LLM enricher; input-only on this path | `examples/policies/domain_llm_content_guard.yaml` |
| `human_approval.py` | Human approval through an approver callback that asks you at the console; tools in an in-process SDK MCP server | `examples/policies/domain_human_approval.yaml` |

Run instructions, `.env` settings and expected output are in
[examples/claude_agent/](../../examples/claude_agent/).

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
