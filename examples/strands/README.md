# Strands — client-side MCP authorization example

`mcp_agent.py` shows a Strands agent that consumes MCP tools, with Axiomatics ADS enforcing
all three points via `authorized_strands_agent(...)` — a single intervention handler attached to an
already-built agent, **no wrapping of the agent or its toolset**:

1. **tool visibility** — the model only sees tools the caller may use
2. **tool invocation** — a disallowed call is blocked before the tool runs
3. **post-invocation** — records the caller may not read are filtered from the result

Identity is bound **once per user** via the agent's state (`Agent(state={"username": ...})`);
no per-invocation plumbing. (`agent(prompt, username=...)` per call also works — user's choice.)

## Run

Config in `.env`: `PDP_BASE_URL`, `LLM_MODEL` (or `STRANDS_LLM_MODEL`), `OPENROUTER_API_KEY`.

Three processes, each in its own terminal (like the other examples):

```bash
# 1. ADS with the all-enabled domain (point deployment.yaml at domain_all_authz_enabled.yaml)
cd examples/policies && java -jar access-decision-service-<version>.jar

# 2. the MCP backend (standalone, serves http://127.0.0.1:9320/mcp)
python examples/strands/mcp_backend.py

# 3. the agent
python examples/strands/mcp_agent.py
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

## How it maps to Strands

| PEP | Strands hook | ADS call |
|---|---|---|
| Visibility | `before_model_call` + a registry filter | `get_permitted_tools` (batched, cached) |
| Invocation | `before_tool_call` → `Deny(reason=...)` | `authorized_tool_invocation` |
| Post-invocation | `after_tool_call` → `Transform` | post-invocation strategy |

`authorized_strands_agent` uses Strands' newer *interventions* API and registers the handler onto the
agent's existing hook registry.

## Content-guard example

`llm_content_guard.py` is a second, self-contained script for Strands: it needs the PDP running with `domain_llm_content_guard.yaml`, then runs an input guard (`BEFORE_MODEL`) and — where the framework supports it — an output guard (`AFTER_MODEL`) that classify prompt/answer text with an LLM enricher so ADS can allow, block, or replace it.

```bash
python examples/strands/llm_content_guard.py
```

`human_approval.py` shows human approval asked for by the policy: it needs the PDP running with `domain_human_approval.yaml`; a finance assistant pays a small invoice, and a large payment or a cancellation waits for your answer through a Strands interrupt. Pass `--approve` or `--reject` to answer without a prompt.

```bash
python examples/strands/human_approval.py
```
