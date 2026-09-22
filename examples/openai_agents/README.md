# OpenAI Agents SDK — client-side MCP authorization example

`mcp_agent.py` shows an OpenAI Agents SDK agent that consumes MCP tools, with Axiomatics ADS
enforcing all three points via `authorized_openai_agent(...)` — which wraps the one seam the Runner
calls every turn (`agent.get_all_tools`), **no wrapping of the agent or its MCP server**:

1. **tool visibility** — the model only sees tools the caller may use
2. **tool invocation** — a disallowed call is blocked before the tool runs
3. **post-invocation** — records the caller may not read are filtered from the result

Identity is supplied per run via the Runner's `context` (reuse the same object to bind once).

## Run

Three processes, each in its own terminal (like the other examples):

```bash
# 1. ADS with the all-enabled domain (point deployment.yaml at domain_all_authz_enabled.yaml)
cd examples/policies && java -jar access-decision-service-<version>.jar

# 2. the MCP backend (standalone, serves http://127.0.0.1:9330/mcp)
python examples/openai_agents/mcp_backend.py

# 3. the agent
python examples/openai_agents/mcp_agent.py
```

Config in `.env`: `PDP_BASE_URL`, `MCP_URL` (default `http://127.0.0.1:9330/mcp`), `LLM_MODEL`
(or `OPENAI_LLM_MODEL`), and an OpenAI-compatible endpoint — `OPENAI_BASE_URL` +
`OPENAI_API_KEY`, or `OPENROUTER_API_KEY` (defaults to OpenRouter).

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

## How it maps to the Agents SDK

| PEP | Mechanism | ADS call |
|---|---|---|
| Visibility | filter the list from `agent.get_all_tools` | `get_permitted_tools` (batched, cached) |
| Invocation | wrap each tool's `on_invoke_tool` (deny before running) | `authorized_tool_invocation` |
| Post-invocation | same wrapper filters the result records | post-invocation strategy |

`authorized_openai_agent` imports no `agents` — it duck-types the SDK objects — so the SDK adds no
dependency; you bring your own agent.

## Content-guard example

`llm_content_guard.py` is a second, self-contained script for the OpenAI Agents SDK: it needs the PDP running with `domain_llm_content_guard.yaml`, then runs an input guard (`BEFORE_MODEL`) and — where the framework supports it — an output guard (`AFTER_MODEL`) that classify prompt/answer text with an LLM enricher so ADS can allow, block, or replace it.

```bash
python examples/openai_agents/llm_content_guard.py
```

`human_approval.py` shows human approval asked for by the policy: it needs the PDP running with `domain_human_approval.yaml`; a finance assistant pays a small invoice, and a large payment or a cancellation waits for your answer through the SDK's own approval flow (`result.interruptions`, `state.approve` / `state.reject`). Pass `--approve` or `--reject` to answer without a prompt.

```bash
python examples/openai_agents/human_approval.py
```
