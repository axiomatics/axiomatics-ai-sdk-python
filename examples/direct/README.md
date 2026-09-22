# direct examples — ask the PDP from your own code

For applications the SDK has no adapter for: `Authorizer(config)` builds each request from
your `AuthZConfig` (context provider, cache, hooks, fail-open, wire protocol) and returns the
answer. Enforcement is your application's job.

Each script assumes the PDP is already running with the listed domain and reads `PDP_BASE_URL`
(default `http://localhost:8080`) from the environment / `.env`. No LLM is involved.

| Script | Domain | What it shows |
|---|---|---|
| `authorizer_basic.py` | `domain_all_authz_enabled.yaml` | `is_permitted_batch(...)` (one batched request), `is_permitted(...)` (one request per call, with call arguments), and `partial_evaluation_text(...)` (the condition under which access is permitted). |

```bash
uv run python examples/direct/authorizer_basic.py
```
