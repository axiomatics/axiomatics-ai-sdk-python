# Hooks

Hooks let you observe what the SDK does, for logging, metrics, tracing, or audit, without
changing any decision. They are **observe-only** (mutation lives in context providers and
post-invocation strategies) and **path-independent**: attach them once on
`AuthZConfig(hooks=[...])` and they fire on every path. An empty list adds no overhead, and a
hook that raises is isolated and logged; it can never break enforcement.

## Events

Subclass `AuthzHook` and override only what you need:

| Method | Emitted by | When |
|---|---|---|
| `on_pdp_request(e)` / `on_pdp_response(e)` | the PDP client | every wire call: endpoint, URL, protocol, latency, status, error. Transport level; use for latency and error dashboards |
| `on_decision(e)` | each enforcement point | the semantic outcome: `surface`, `enforcement_point`, subject / action / resource / `environment`, a coarse `decision`, a `cached` flag, a `timestamp` (epoch seconds). A cache hit emits a decision with no transport event |
| `on_enrichment(e)` | content guards | one LLM enricher ran: point, enricher, content, attributes produced, availability, latency |
| `on_invalidation(e)` | `config.invalidate_decisions()` | cached decisions were dropped: the `subject` attributes that selected them (empty = all), how many were `removed`, the `reason`, a `timestamp` ([Agent identity](agent-identity.md#revocation-and-the-decision-cache)) |

Events raised while handling one decision share a `correlation_id`, so transport and decision
events can be stitched together.

`DecisionEvent.surface` names the path: `langchain_agents`, `google_adk`, `strands`,
`openai_agents`, `claude_agent`, `langchain_mcp`, `rag`, `fastmcp`, `tool_decorator`,
`a2a_client`, `a2a_server`, `gateway_fastmcp`, `gateway_a2a_server`, and `direct` (or the
label you give an `Authorizer`).

`DecisionEvent.decision` is `permit` or `deny` for single decisions, `partial` for batch
decisions (with `detail={"permitted": n, "total": m}`), `filtered` / `passthrough` for
post-invocation, and `deny_all` / `filtered` / `unfiltered` for retrieval. Human approval adds
`approval_required`, then `approved` / `rejected`, before the final `permit` / `deny`
([What gets logged](human-approval.md#what-gets-logged)). When a decision carried obligations or
advice, `detail` lists them (`obligations`, `advice`), each as `{"id": ..., "attributes": {...}}`
with its attribute assignments (for example an approval's `reason` or `ttl`), plus
`unknown_obligations` or `failed_obligations` when those turned a Permit into a Deny.

## Built-in `LoggingHook`

Logs every event through the standard `logging` module:

```python
from axiomatics_ai import AuthZConfig, LoggingHook, PDPConnection, standard_tool_enforcement

config = AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement=standard_tool_enforcement(),
    hooks=[LoggingHook()],           # LoggingHook(logger_=..., level=logging.DEBUG) to tune
)
```

## Audit trail: `AuditLogHook`

`AuditLogHook` writes one structured record per decision and one per cache invalidation, with
the fields AIMS §11 asks an audit log to record
([Agent identity](agent-identity.md#audit-trail)):

```python
from axiomatics_ai import AuditLogHook

config = AuthZConfig(..., hooks=[AuditLogHook(hash_chain=True)])
```

| Field | Decision record | Remediation record |
|---|---|---|
| `event` | `decision` | `remediation` |
| `timestamp` | ISO 8601, UTC | same |
| `correlation_id` | shared with the transport events of the same decision | |
| `agent_id`, `user_id`, `actor_id`, `agent_name` | lifted from the subject; `agent_name` is the agent or sub-agent in the app that made the call ([Sub-agents](enforcement-points.md#sub-agents)) | |
| `surface`, `enforcement_point`, `action`, `resource` | what was attempted, where | |
| `decision`, `cached`, `detail` | as on `DecisionEvent` | |
| `subject`, `environment` | every attribute sent, including `transaction_id` and any risk, posture or intent attributes | `subject`: the attributes that selected the dropped decisions |
| `remediation`, `removed`, `reason` | | `cache_invalidation`, how many, why |
| `previous_hash`, `hash` | with `hash_chain=True` | same |

- **Sink.** By default each record is logged as one JSON line on the `axiomatics_ai.audit`
  logger at INFO, so any log pipeline or SIEM agent can collect it. Give that logger a handler
  and level, or nothing is written. Pass any callable taking the record dict as `sink=` instead.
  It is called synchronously, so a slow sink should queue.
- **Sending to a SIEM URL.** Attach a `logging` handler that ships to it, or use a callable sink;
  [`examples/hooks/siem_webhook_sink.py`](../examples/hooks/siem_webhook_sink.py) posts records
  to a webhook.
- **Hash chain.** With `hash_chain=True` each record carries the previous record's hash and its
  own: SHA-256 over the record's JSON with sorted keys, including `previous_hash`, before `hash`
  is added. The first record's `previous_hash` is empty. A deleted, inserted or edited record
  breaks the chain, so the trail is tamper-evident. The chain lives in the process: a restart,
  or each worker, starts a new one.
- **Durability** belongs to the sink and the store behind it (SIEM, append-only log), not to the
  SDK.
- `agent_attribute`, `user_attribute` and `actor_attribute` name the subject attributes lifted to
  the top level, when your provider uses other names.

## Custom hook

```python
from axiomatics_ai import AuthzHook, AuthZConfig, LoggingHook

class DenialCollector(AuthzHook):
    def __init__(self):
        self.denials = []

    def on_decision(self, event):
        if event.decision == "deny":
            self.denials.append((event.surface, event.subject, event.resource))

    def on_pdp_response(self, event):
        print(f"PDP {event.endpoint} took {event.latency_ms:.1f}ms (status={event.status_code})")

config = AuthZConfig(enforcement=standard_tool_enforcement(), hooks=[DenialCollector(), LoggingHook()])  # run in order
```

Runnable examples: [`examples/hooks/`](../examples/hooks/).

---

← Previous: [Post-invocation](post-invocation.md) · [Docs index](README.md) · Next: [Security](security.md) →
