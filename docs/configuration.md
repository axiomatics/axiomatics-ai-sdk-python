# Configuration

Every path takes one object, `AuthZConfig`. It says how to reach the PDP, which enforcement
points are on and how each behaves, where identity comes from, and what to observe. Fields a
path does not use are ignored.

## `AuthZConfig` field reference

```python
from axiomatics_ai import (
    AuthZConfig, PDPConnection, PdpEndpoints, RetrievalConfig,
    EnforcementPoint, Enforcer, standard_tool_enforcement, AttributeNames,
)

AuthZConfig(
    # How to reach the PDP and cache its decisions.
    connection=PDPConnection(
        base_url="http://localhost:8080",
        timeout=10.0,                        # HTTP timeout, seconds
        protocol="xacml",                    # "xacml" (default) or "authzen" — see "Protocols"
        fail_open=False,                     # PDP unreachable or inconclusive: True → permit, False → deny
        cache_ttl_seconds=300.0,             # decision cache TTL (never past a verified token's exp); 0 disables
        cache_maxsize=1024,
        endpoints=PdpEndpoints(),            # request paths — see "Endpoints"
        headers={},                          # e.g. {"X-Api-Key": ...} — see "Authenticating to the PDP"
        auth=None,                           # ("user", "password") for Basic, or an httpx.Auth
        verify=True,                         # False, a CA bundle path, or an ssl.SSLContext
        cert=None,                           # client certificate for mTLS: path or (cert, key)
        http_client=None,                    # your own httpx.Client instead of the four above
    ),

    # Decision-based enforcement points → their Enforcer. A point that is present is enforced;
    # an absent point (or enforcement=None) is not. There are no hidden defaults.
    enforcement={
        EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="tool_visibility"),
        EnforcementPoint.BEFORE_TOOL:     Enforcer(action_name="tool_invocation"),
        EnforcementPoint.AFTER_TOOL:      Enforcer(
            action_name="post_invocation",
            strategy=None,                   # None → RecordFilterStrategy(); e.g. ColumnMaskStrategy("ssn")
            action_overrides={"search_docs": "search_post_invocation"},  # per-tool action name
        ),
        # EnforcementPoint.BEFORE_MODEL / AFTER_MODEL: content guards — see content-guards.md
        # enrichers=[intent_enricher()] on BEFORE_TOOL: intent-based access — see intent-based-access.md
    },
    # Equivalent to the three tool points above: enforcement=standard_tool_enforcement()

    # Constraint-based retrieval filtering for RAG (partial evaluation). None → off.
    retrieval=RetrievalConfig(action_name="retrieval", attribute_map=None),

    # Where identity comes from — see identity.md. None → the path's default provider.
    # On server paths, AgentIdentityContextProvider(JwtVerification(...)) verifies tokens — see agent-identity.md.
    context_provider=None,

    # Observe-only hooks — see hooks.md. Empty → no overhead.
    hooks=[],

    # What to do with obligations and advice, including human approval — see human-approval.md.
    # None → unknown obligations deny (XACML), "require-approval" is recognised, no approver.
    obligations_and_advice=None,

    # The attributes the SDK's own names are sent as — see enforcement-points.md
    # "Attribute conventions".
    attribute_names=AttributeNames(
        tool="tool_name",            # resource: tool visibility, tool invocation, post-invocation
        agent="agent_name",          # resource: A2A invocation
        mcp_server="mcp_server",     # resource: the MCP server of a Claude Agent SDK tool
        acting_agent="agent_name",   # subject: the agent or sub-agent making the call; None: off
    ),
)
```

`standard_tool_enforcement()` builds the three tool points; turn one off with
`standard_tool_enforcement(visibility=False)`, `invocation=False`, or `post_invocation=False`.
`config.enforces(point)` tells you whether a point is on. `config.validate()` returns a list of
always-wrong misconfigurations (it also runs at construction and logs them as warnings),
including an enricher placed at a point it is not meant for (see
[Content guards](content-guards.md#configure)).

`Enforcer` fields: `action_name`; `enrichers` (LLM enrichers, run at `BEFORE_MODEL`,
`AFTER_MODEL`, and `BEFORE_TOOL` only) with `on_unavailable` (`"fail_closed"` default, or
`"fail_open"`) and `block_message`; `strategy`, `action_overrides`, and `on_unrecognized_output`
(`"fail_open"` default, or `"fail_closed"`; see [Post-invocation](post-invocation.md#output-the-strategy-cannot-inspect))
(`AFTER_TOOL` only). Each
enricher's `EnricherSpec` also takes `cache_ttl_seconds` / `cache_maxsize` (classification
cache, off by default), `num_user_messages` (`BEFORE_TOOL` only), and `allowed_points`.

## Authenticating to the PDP

The PDP client sends every decision over one pooled `httpx.Client` per adapter. Configure it on
`PDPConnection`:

```python
import httpx

# API key in a header, and a private CA
PDPConnection(base_url="https://pdp.internal", headers={"X-Api-Key": key}, verify="/etc/ca.pem")

# Basic auth
PDPConnection(base_url="https://pdp.internal", auth=("svc-agent", secret))

# A bearer token that expires: an httpx.Auth that sets (and refreshes) it on each request
class PdpToken(httpx.Auth):
    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {token_source.current()}"
        yield request

PDPConnection(base_url="https://pdp.internal", auth=PdpToken())

# mTLS
PDPConnection(base_url="https://pdp.internal", cert=("/certs/client.pem", "/certs/client.key"))

# or bring your own client (proxies, transports, limits): then don't set the four above
PDPConnection(base_url="https://pdp.internal", http_client=httpx.Client(...))
```

`headers`, `auth`, `cert` and `http_client` are left out of `PDPConnection`'s `repr`. Put
credentials there, never in `base_url` (`https://user:pass@host`): the URL appears in logs, error
messages and `repr`.
`PdpClient.close()` (or `with config.build_pdp_client() as pdp:`) releases a client the SDK built;
one you pass in is yours to close.

## Enforcement points

| Point | Fires | On Deny |
|---|---|---|
| `BEFORE_MODEL` | before the model sees the prompt | prompt blocked, refusal returned |
| `TOOL_VISIBILITY` | before the model sees the tool list | tool hidden |
| `BEFORE_TOOL` | before a tool call runs | call blocked |
| `AFTER_TOOL` | after a tool returns | result filtered or masked |
| `AFTER_MODEL` | before the answer reaches the user | answer blocked, refusal returned |
| `A2A_INVOCATION` | on an agent-to-agent message | message blocked |
| retrieval (`RetrievalConfig`) | on a vector-store search | only permitted documents returned |

Which points each integration can enforce depends on where its framework offers a hook; see
the matrix in the [project README](../README.md#what-it-covers) and each integration page.
[Enforcement points](enforcement-points.md) describes what the PDP receives at each one.

## Protocols

`connection.protocol` selects the wire protocol for decisions:

- **`"xacml"`** (default): XACML 3.0 JSON profile. Single decisions are one request; batch
  decisions (tool visibility, record filtering, `is_permitted_batch`) use the Multiple Decision
  Profile, one sub-request per resource. Lenient: attributes the policy does not declare are
  ignored.
- **`"authzen"`**: [OpenID AuthZEN](https://openid.net/wg/authzen/). Single decisions go to the
  evaluation endpoint, batches to the evaluations endpoint. PDPs typically map attributes
  strictly, so every attribute you send must be declared on the PDP side.

Both return identical decisions for the same policy and request; pick the one your PDP exposes.
Retrieval filtering is protocol-independent: it uses the partial-evaluation reverse query,
which is XACML-based (see compatibility below).

**Error handling.** A request the PDP rejects with HTTP 4xx (undeclared attribute, endpoint not
enabled, wrong path, bad credentials) raises `PdpRequestError` and is never masked by
`fail_open`. `fail_open` applies only to transient failures: PDP unreachable, timeouts, HTTP 5xx.

## Endpoints

`PdpEndpoints` holds the request path for each kind of call, appended to `base_url`. The
defaults are the paths Axiomatics ADS serves; override only the fields your PDP differs on:

```python
PDPConnection(
    base_url="https://pdp.example.com",
    endpoints=PdpEndpoints(decision="/domains/agents/pdp", health="/healthz"),
)
```

| Field | Default | Used by |
|---|---|---|
| `decision` | `/authorize` | XACML single and batch decisions |
| `health` | `/health` | `check_connection()` |
| `authzen_evaluation` | `/access/v1/evaluation` | AuthZEN single decision |
| `authzen_evaluations` | `/access/v1/evaluations` | AuthZEN batch decision |
| `partial_evaluation` | `/authorization/constraints` | retrieval filtering, `Authorizer.partial_evaluation` |
| `partial_evaluation_text` | `/authorization/simplified-constraints` | `Authorizer.partial_evaluation_text` |

## PDP compatibility

The SDK is tested against **Axiomatics ADS 26.1.x** over both protocols (AuthZEN needs
`authzen.enabled: true` in the ADS deployment file). It is designed to work with any PDP that
speaks the same standards, and single and batch decisions have been exercised against other
open-source PDPs over both AuthZEN and XACML JSON. Retrieval filtering has no standard
equivalent and works only with ADS. What each feature needs:

| Feature | Needs | Works with |
|---|---|---|
| Single decisions (`BEFORE_TOOL`, `BEFORE_MODEL`, `AFTER_MODEL`, `A2A_INVOCATION`, `ColumnMaskStrategy`, `Authorizer.is_permitted`) | XACML JSON single request, or AuthZEN evaluation | any XACML 3.0 JSON or AuthZEN PDP |
| Batch decisions (`TOOL_VISIBILITY`, `RecordFilterStrategy`, `Authorizer.is_permitted_batch`) | XACML Multiple Decision Profile, or AuthZEN evaluations | PDPs implementing the MDP or the AuthZEN batch endpoint |
| Retrieval filtering (`rag`, `Authorizer.partial_evaluation`) | partial evaluation: a reverse query returning the condition under which access is permitted | Axiomatics ADS (CAQ) only; no standard equivalent |
| Health probe | a GET endpoint returning 200 | any; path configurable |

On the AuthZEN path the SDK reads the standard boolean `decision`; when the PDP also echoes an
underlying XACML decision (ADS does) it uses that, so NotApplicable and Indeterminate keep
their `fail_open` semantics.

## Diagnostics

| Call | Returns |
|---|---|
| `config.check_connection()` | `HealthCheckResult` (`ok`, `base_url`, `protocol`, `latency_ms`, `status_code`, `error`); truthy when reachable; never raises |
| `config.summary()` | one-line description of what the config enforces |
| `config.validate()` | list of always-wrong misconfigurations, empty when fine |
| `config.build_pdp_client()` | the configured `PdpClient`, for the [direct path](integrations/direct.md) or tests |
| `config.invalidate_decisions(reason="", **subject)` | drops cached decisions for a subject (`user_id="alice"`, `agent_id=...`), or all with no attributes, in every cache built from this config, and the remembered human approvals; emits `on_invalidation`; returns how many ([Revocation](agent-identity.md#revocation-and-the-decision-cache)) |

---

← Previous: [Getting started](getting-started.md) · [Docs index](README.md) · Next: [Identity](identity.md) →
