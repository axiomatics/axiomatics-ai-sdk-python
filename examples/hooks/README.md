# hooks examples — observe-only observability

Hooks fire on every path via `AuthZConfig(hooks=[...])` and never change a decision. Both scripts
assume **ADS is already running** with `domain_all_authz_enabled.yaml` loaded (see the
[Running ADS](../../docs/running-ads.md)); they read `PDP_BASE_URL`, `LLM_MODEL`, and your provider key
from the environment / `.env`.

| Script | What it shows |
|---|---|
| `builtin_logging_hook.py` | The built-in `LoggingHook` — "log every ADS call and every decision". |
| `custom_hook.py` | A custom `AuthzHook` — build your own audit trail / metrics from the SDK's decision and transport events. |
| `siem_webhook_sink.py` | The built-in `AuditLogHook` (AIMS §11 audit records, hash-chained) with a non-blocking sink that batches records to a SIEM URL (`SIEM_URL`; a local collector otherwise). Needs `domain_all_authz_enabled.yaml`. |

```bash
python examples/hooks/builtin_logging_hook.py
```

See [Hooks](../../docs/hooks.md).
