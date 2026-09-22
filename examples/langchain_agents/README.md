# langchain_agents examples — authorization for LangChain agents

Each script assumes **ADS is already running** with the listed domain (see the
[Running ADS](../../docs/running-ads.md)) and reads `PDP_BASE_URL` (default `http://localhost:8080`),
`LLM_MODEL`, and your provider key from the environment / `.env`.

| Script | Domain | What it shows |
|---|---|---|
| `create_authorized_agent.py` | `domain_all_authz_enabled.yaml` | The **recommended** entry point: `create_authorized_agent(...)`, a one-call wrapper over `create_agent`. |
| `all_authz_enabled.py` | `domain_all_authz_enabled.yaml` | All enforcement points at once (visibility + invocation + post-invocation). |
| `explicit_middleware.py` | `domain_tool_visibility_only.yaml` | Passing `LangChainAgentAuthZMiddleware` yourself in `middleware=[...]`. |
| `tool_call_authz.py` | `domain_tool_invocation.yaml` | Blocking individual tool calls via a `BEFORE_TOOL` Enforcer. |
| `record_filtering.py` | `domain_post_invocation.yaml` | Filtering records from a tool result via an `AFTER_TOOL` Enforcer. |
| `custom_context_provider.py` | `domain_custom_context_provider.yaml` | A custom `ContextProvider` mapping app fields to XACML categories. |
| `llm_content_guard.py` | `domain_llm_content_guard.yaml` | Input/output LLM content guards (`BEFORE_MODEL` + `AFTER_MODEL`). |
| `intent_based_access.py` | `domain_intent.yaml` | Intent-based access control: tool calls checked against the user's classified intent, blocking an injected send. |
| `human_approval.py` | `domain_human_approval.yaml` | Human approval asked for by the policy: a large payment or a cancellation pauses the graph (`interrupt()`) for your answer. `--approve` / `--reject` answer without a prompt; `--approver` asks through an approver callback instead. |
| `agent_behind_api.py` | `domain_agent_identity.yaml` | A LangChain agent behind a `/chat` endpoint that verifies the user's access token client-side (`AgentIdentityContextProvider(token_from=...)`) and pays only within the user's approved grant. |

```bash
python examples/langchain_agents/create_authorized_agent.py
```

A larger, UI-driven demo lives in [`simple-ui/`](simple-ui/). See the
[langchain_agents module README](../../axiomatics_ai/langchain_agents/README.md) for the API.
