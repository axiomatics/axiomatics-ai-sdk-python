# Google ADK — client-side MCP authorization example

`mcp_agent.py` shows a Google ADK agent that consumes MCP tools, with Axiomatics ADS
enforcing all three points via `authorized_adk_agent(...)` — **no wrapping of the agent or its
toolset**, just attached callbacks:

1. **tool visibility** — the model only sees tools the caller may use
2. **tool invocation** — a disallowed call is blocked before the tool runs
3. **post-invocation** — records the caller may not read are filtered from the result

Identity is bound **once per user** via the ADK session's state (`create_session(state=...)`);
no per-invocation plumbing. (`run_async(state_delta=...)` per call also works — user's choice.)

## Run

Config in `.env`: `PDP_BASE_URL`, `LLM_MODEL` (or `GOOGLE_ADK_LLM_MODEL`), `OPENROUTER_API_KEY`.

Three processes, each in its own terminal (like the other examples):

```bash
# 1. ADS with the all-enabled domain (point deployment.yaml at domain_all_authz_enabled.yaml)
cd examples/policies && java -jar access-decision-service-<version>.jar

# 2. the MCP backend (standalone, serves http://127.0.0.1:9310/mcp)
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

## How it maps to ADK

| PEP | ADK hook | ADS call |
|---|---|---|
| Visibility | `before_model_callback` — filter `LlmRequest` tools | `get_permitted_tools` (batched, cached) |
| Invocation | `before_tool_callback` — return a denial dict to block | `authorized_tool_invocation` |
| Post-invocation | `after_tool_callback` — filter the MCP result's records | post-invocation strategy |

`authorized_adk_agent` itself imports **no ADK** — it duck-types the objects ADK passes its
callbacks — so the SDK adds no dependency; you bring your own ADK agent.

## Content-guard example

`llm_content_guard.py` is a second, self-contained script for Google ADK: it needs the PDP running with `domain_llm_content_guard.yaml`, then runs an input guard (`BEFORE_MODEL`) and — where the framework supports it — an output guard (`AFTER_MODEL`) that classify prompt/answer text with an LLM enricher so ADS can allow, block, or replace it.

```bash
python examples/google_adk/llm_content_guard.py
```

`human_approval.py` shows human approval asked for by the policy: it needs the PDP running with `domain_human_approval.yaml`; a finance assistant pays a small invoice, and a large payment or a cancellation waits for your answer through ADK tool confirmation. Pass `--approve` or `--reject` to answer without a prompt.

```bash
python examples/google_adk/human_approval.py
```
