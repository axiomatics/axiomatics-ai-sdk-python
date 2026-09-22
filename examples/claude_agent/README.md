# Claude Agent SDK — client-side MCP authorization example

`mcp_agent.py` shows a Claude Agent SDK agent that consumes MCP tools, with Axiomatics ADS
enforcing invocation and post-invocation via `authorized_claude_options(...)` — which wires ADS into
the `ClaudeAgentOptions` (no agent object exists in this SDK):

1. **tool invocation** — a `PreToolUse` hook blocks a disallowed call before it runs (**arg-level**)
2. **post-invocation** — a `PostToolUse` hook filters records the caller may not read

**Tool visibility is not supported** on this path (the SDK has no per-turn tool-list filter), so
policies apply at the invocation point. See [the path README](../../axiomatics_ai/claude_agent/README.md).

Identity is captured by closure — one options object per user. The hook runs for every tool
call, whatever `allowed_tools` or `permission_mode` say.

## Run

Three processes, each in its own terminal (like the other examples):

```bash
# 1. ADS with the all-enabled domain (point deployment.yaml at domain_all_authz_enabled.yaml)
cd examples/policies && java -jar access-decision-service-<version>.jar

# 2. the MCP backend (standalone, serves http://127.0.0.1:9340/mcp)
python examples/claude_agent/mcp_backend.py

# 3. the agent
python examples/claude_agent/mcp_agent.py
```

Config in `.env`: `PDP_BASE_URL`, `MCP_URL` (default `http://127.0.0.1:9340/mcp`),
`OPENROUTER_API_KEY`, and optionally `ANTHROPIC_LLM_MODEL` / `LLM_MODEL`. Like the other examples
this routes the SDK through OpenRouter (set from `OPENROUTER_API_KEY`); to use Anthropic directly
instead, set `ANTHROPIC_API_KEY`.

Expected output (alice is unrestricted; bob is limited by `domain_all_authz_enabled.yaml`):

```
── 1. Tool invocation — arg-level (bob may read ACC-001, not ACC-002) ──
[alice ACC-002] The balance for account ACC-002 is $1,000.
[bob ACC-001]   The balance for account ACC-001 is $1,000.          # arg-level permit
[bob ACC-002]   Access denied: not authorized to call get_account_balance.   # arg-level deny

── 2. Post-invocation (search_docs) ──
[alice] d1 (public), d2 (classified)
[bob]   d1 (public)                                                 # 'classified' filtered out
```

## How it maps to the Claude Agent SDK

| PEP | Mechanism | ADS call |
|---|---|---|
| Invocation | `PreToolUse` hook → `permissionDecision: "deny"` | `authorized_tool_invocation` (arg-level) |
| Post-invocation | `PostToolUse` hook → `updatedToolOutput` | post-invocation strategy |
| Visibility | not supported (policies apply at invocation) | — |

## Content-guard example

`llm_content_guard.py` is a second, self-contained script for the Claude Agent SDK: it needs the PDP running with `domain_llm_content_guard.yaml`, then runs an input guard (`BEFORE_MODEL`) and — where the framework supports it — an output guard (`AFTER_MODEL`) that classify prompt/answer text with an LLM enricher so ADS can allow, block, or replace it.

```bash
python examples/claude_agent/llm_content_guard.py
```

`human_approval.py` shows human approval asked for by the policy: it needs the PDP running with `domain_human_approval.yaml`; a finance assistant pays a small invoice, and a large payment or a cancellation waits for your answer through an approver callback (the Claude Agent SDK has no interrupt to pause on). Pass `--approve` or `--reject` to answer without a prompt.

```bash
python examples/claude_agent/human_approval.py
```
