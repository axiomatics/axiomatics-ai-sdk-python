# direct — ask the PDP from your own code

For applications the SDK has no adapter for. Every other integration hooks a framework and
decides *where* to ask the PDP; `Authorizer` is the same machinery with no framework. You decide
where to ask, it builds the request from your `AuthZConfig` and returns the answer. Enforcement
is yours: what you do with a `False` is up to your application. Works on the client or server
side, with any identity source, and needs no extra beyond the core package.

What you get over calling `PdpClient` yourself: the configured [context provider](../identity.md)
turns your application context into attributes; the decision cache is honoured; [hooks](../hooks.md)
fire with `surface="direct"`; and `fail_open`, `protocol`, and `endpoints` come from the
connection instead of call-site defaults.

## Attach

```python
from axiomatics_ai import Authorizer, AuthZConfig, PDPConnection

authz = Authorizer(AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080")))

# One decision. action / resource are names; args become resource attributes.
# The resource name is sent as "resource_name" unless you pass resource_name_attribute=.
if not authz.is_permitted(action="read_document", resource="doc-42",
                          args={"classification": "internal"},
                          context={"username": "alice"}):
    raise PermissionError("not allowed")

# One batched request: which of these may alice see?
visible = authz.is_permitted_batch(action="tool_visibility", resource_name_attribute="tool_name",
                                   resources=["search_docs", "send_email"],
                                   context={"username": "alice"})
```

No `enforcement` entry is needed: the action name is the one you pass. `Authorizer(config,
surface="billing-api")` labels this instance's decision events if you have several call sites.

## Methods

| Method | Request | Returns |
|---|---|---|
| `is_permitted(*, action, resource, args=None, resource_name_attribute="resource_name", context=None, subject=None, environment=None)` | single decision | `bool`; Permit → True, Deny → False, otherwise `connection.fail_open` |
| `is_permitted_batch(*, action, resources, resource_name_attribute="resource_name", context=None, subject=None, environment=None)` | batch decision, one sub-request per name | `set[str]` of permitted names; an empty subject resolves to `fail_open` without a PDP call |
| `partial_evaluation(*, action, context=None, subject=None, environment=None, mocked_attributes=None, decision_set=None, exclude_indeterminate=False)` | partial evaluation (reverse query) | the condition as a constraint tree (`dict`), or `None` on transient failure |
| `partial_evaluation_text(...)` | same | the PDP's human-readable rendering (`str`), display only |
| `check_connection()` | `GET` health | `HealthCheckResult`, never raises |
| `clear_cache()` | | drops cached decisions, e.g. after a policy change |
| `authz.pdp` | | the underlying `PdpClient` for anything the façade does not cover |

`context` is whatever your context provider understands (a dict with `username` for the default
provider). Pass `subject=` / `environment=` dicts to bypass the provider for those categories;
`None` values are dropped.

## Partial evaluation

`partial_evaluation` asks "under what condition would this be permitted?" rather than "is it
permitted?". The subject, action, and environment are sent as provided attributes; everything
else the policy references is left unbound, and the PDP returns the condition on those unbound
attributes as a Boolean Expression Tree. This is what the [rag](rag.md) path uses to build a
vector-store filter; here you can translate it into your own database query.

```python
tree = authz.partial_evaluation(action="read_document", context={"username": "alice"})
text = authz.partial_evaluation_text(action="read_document", context={"username": "alice"})
# text → e.g. 'resource.department == "engineering" AND resource.level <= 2'
```

Only PDPs that implement partial evaluation serve these calls (Axiomatics ADS via CAQ, at
`PdpEndpoints.partial_evaluation`); another PDP answers HTTP 404, which raises
`PdpRequestError`. See [PDP compatibility](../configuration.md#pdp-compatibility). Not cached.

## Identity

The default `DefaultClientContextProvider` reads `username`, `user_id`, `tenant_id`, and
`session_id` from the `context` dict or object. Set `AuthZConfig(context_provider=...)` to map
your own fields, or pass `subject=` explicitly. On a server, hand the request headers as
`context` together with a `DefaultServerContextProvider` to get the JWT claims as the subject;
the [trust boundary](../identity.md#trust-boundary) applies.

## What the PDP receives

Exactly the single and batch shapes on [Enforcement points](../enforcement-points.md), with your
action name in `action.name`, your resource name in `resource_name` (or the
`resource_name_attribute` you pass), and `args` merged into the resource. Write policy the same way as for tool invocation or tool visibility; the example below
reuses the tool policy the agent examples run against.

## Hooks

Every answer emits a `DecisionEvent` with `surface="direct"` (or your label) and
`enforcement_point=<action>`; `decision` is `permit` / `deny` for single checks and `partial`
with `detail={"permitted": n, "total": m}` for batches. Cache hits emit the event with
`cached=True` and no transport event.

## Examples

| Script | Domain | What it shows |
|---|---|---|
| [`authorizer_basic.py`](../../examples/direct/authorizer_basic.py) | [`domain_all_authz_enabled.yaml`](../../examples/policies/domain_all_authz_enabled.yaml) | batch visibility, per-call invocation with arguments, and the partial-evaluation text for two users; no LLM involved |

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
