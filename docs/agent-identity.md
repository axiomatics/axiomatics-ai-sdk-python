# Agent identity and AIMS alignment

An agent that acts for a user carries two identities: the user whose authority it uses, and the
agent itself. The IETF WIMSE working group's AI Identity Management System draft
([AIMS](https://datatracker.ietf.org/doc/draft-ietf-wimse-aims/), `-00`) describes how to handle
them with standards that already exist. The agent is the OAuth client (`client_id`), the user is
the token's subject (`sub`), and delegation is recorded in `act`. Resource servers decide on
these identifiers "along with other claims", and every decision is logged. This page shows how
the SDK does its part of that: it verifies tokens on server paths, turns their claims into named
PDP attributes, makes sure a cached decision never outlives the token or a revocation, and
writes an audit trail. It ends with what the SDK covers and what it leaves to other components.

Runnable example: a FastMCP payments server that verifies tokens and enforces an approved RAR
grant, called by a LangChain agent —
[`examples/fastmcp/payments_server.py`](../examples/fastmcp/payments_server.py) +
[`payments_agent.py`](../examples/fastmcp/payments_agent.py), with
[`domain_agent_identity.yaml`](../examples/policies/domain_agent_identity.yaml).

## Verify the token

On the server paths (`fastmcp`, `tool_decorator`, the `a2a` server, `gateway`) the caller sends
`Authorization: Bearer <JWT>`. `AgentIdentityContextProvider` checks the token the way a resource
server must (RFC 9068 §4) before any of its claims reach the PDP:

```python
from axiomatics_ai import AgentIdentityContextProvider, AuthZConfig, JwtVerification, PDPConnection, standard_tool_enforcement

config = AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement=standard_tool_enforcement(),
    context_provider=AgentIdentityContextProvider(
        verification=JwtVerification(
            issuer="https://idp.example.com",               # the identity provider you trust
            audience="https://payments-mcp.example.com",    # this server, as registered there
        ),
    ),
)
```

Install the path's extra (`fastmcp`, `tool-decorator`, `a2a` or `gateway`); they pull in
`pyjwt[crypto]`.

**Why you supply `issuer` and `audience`.** Whoever sends a token controls everything in it, so
the values that make it trustworthy have to come from your configuration, never from the token:

- **`issuer`**: the only identity provider whose signatures are accepted. Without this check,
  anyone could stand up their own issuer, sign tokens with their own keys, and publish those keys.
  The token's `iss` is compared with your list *before* any key is fetched, so keys only ever
  come from an issuer you trust.
- **`audience`**: this server's own identifier. A token issued for another API (a calendar
  service, say) is validly signed by the same issuer. The audience check stops it being replayed
  here.

Either one can be a list: several trusted issuers, or a server known by several names.

| `JwtVerification` field | Default | Meaning |
|---|---|---|
| `issuer`, `audience` | required | see above |
| `jwks_url` | `None` | where the signing keys are. By default they are discovered from the issuer: `/.well-known/openid-configuration`, then `/.well-known/oauth-authorization-server`, reading `jwks_uri` |
| `algorithms` | `("RS256", "PS256", "ES256", "EdDSA")` | accepted signature algorithms. Only asymmetric ones (RS, PS, ES, EdDSA) are allowed; `HS*` and `none` raise `ValueError`, because a resource server verifies with the issuer's public keys |
| `require_typ` | `True` | the header `typ` must be `at+jwt` (RFC 9068 §4), so an ID token can't be passed off as an access token. Some identity providers issue access tokens typed plain `JWT`; set `require_typ=False` for those |
| `leeway_seconds` | `30` | clock skew tolerated on `exp` and `nbf` |
| `http_timeout` | `5.0` | seconds, for discovery and key fetches |

The token must also carry `iss`, `aud`, `exp` and `sub`, and must not have expired.

**A token that fails verification** gives an empty subject and environment, so the request goes
to the PDP with no identity. The reason is logged as a warning (`Caller identity not verified,
so the request is denied: ...`), once per token, never the token itself. A request with no bearer token gives an empty subject without a warning. Write
permit rules that require the identity attributes (`user_id`, `agent_id`) so that an empty
subject is denied; see [Security](security.md#values-the-pdp-cannot-carry) for why permit-style
rules are the safe direction.

## What the PDP receives

The provider sends named attributes, not every raw claim:

| Attribute | Category | From | Notes |
|---|---|---|---|
| `user_id` | subject | `sub` | only when a user is present. With Client Credentials the client is its own subject (RFC 9068 §2.2), so `sub == client_id` means no user and `user_id` is left out. Override `is_user_present(claims)` for an identity provider that marks this differently |
| `agent_id` | subject | `client_id` | the agent (AIMS §10.3) |
| `actor_id` | subject | `act.sub`, the outermost one | the current actor. Earlier actors nested inside `act` are informational only (RFC 8693 §4.1): they are logged at debug level and never sent |
| `scopes` | subject | `scope` (space-separated) or `scp` (list) | a list |
| `acr`, `amr`, `auth_time` | subject | same-named claims | how and when the user authenticated |
| `transaction_id` | environment | `txn` | a transaction id for tracing across services (RFC 9396 §9.1) |

Rename any of them, or set one to `None` to not send it, with `AgentIdentityAttributes`:

```python
from axiomatics_ai import AgentIdentityAttributes

AgentIdentityContextProvider(verification, attributes=AgentIdentityAttributes(user_id="username", amr=None))
```

**More claims** are mapped explicitly, by dotted path, into subject attributes. A string names the
attribute. A `Claim` also converts the value to `string`, `integer`, `double` or `boolean`:

```python
from axiomatics_ai import Claim

AgentIdentityContextProvider(
    verification,
    claims={
        "realm_access.roles": "roles",                     # Keycloak roles, a list of strings
        "tenant": Claim("tenant_id"),
        "clearance": Claim("clearance_level", type="integer"),
    },
)
```

Only mapped values reach the PDP. That keeps the request valid (a JSON object claim such as
`address` can't be an attribute) and makes AuthZEN usable, since strict PDPs reject attributes
the domain doesn't declare.

**Derived attributes:** `extra_subject=lambda claims: {...}` adds your own subject attributes
computed from the claims (e.g. `{"is_admin": "admin" in claims.get("groups", [])}`). To change
the Client Credentials rule, subclass and override `is_user_present(claims)`.

**`preview(claims)`** shows what a claim set turns into, without verifying it or calling the PDP.
It returns the `subject` and `environment` attributes, and `dropped`, each claim or mapping that
was left out with the reason. Use it in a test, or to check a mapping against a real token's
claims.

### Worked example

A travel agent (`spiffe://example.com/agents/travel`) calls a payments MCP server for Alice. The
token has already been verified; its claims:

```json
{
  "iss": "https://idp.example.com",
  "aud": "https://payments-mcp.example.com",
  "sub": "alice@example.com",
  "client_id": "spiffe://example.com/agents/travel",
  "scope": "payments.initiate flights.book",
  "acr": "urn:example:mfa", "amr": ["pwd", "otp"], "auth_time": 1790000000,
  "txn": "8b4729cc-32e4-4370-8cf0-5796154d1296",
  "act": {
    "sub": "spiffe://example.com/agents/travel",
    "act": { "sub": "spiffe://example.com/agents/concierge" }
  },
  "realm_access": { "roles": ["user", "traveller"] },
  "address": { "country": "SE", "locality": "Malmö" },
  "authorization_details": [
    {
      "type": "payment_initiation",
      "actions": ["initiate"],
      "locations": ["https://payments-mcp.example.com"],
      "instructedAmount": { "currency": "EUR", "amount": "123.50" },
      "creditorName": "Merchant A",
      "creditorAccount": { "iban": "DE02100100109307118603" }
    },
    {
      "type": "account_information",
      "actions": ["list_accounts"],
      "locations": ["https://accounts.example.com"]
    }
  ],
  "iat": 1790000000, "exp": 1790000600, "jti": "t1-8f2a"
}
```

The provider:

```python
provider = AgentIdentityContextProvider(
    JwtVerification(issuer="https://idp.example.com", audience="https://payments-mcp.example.com"),
    claims={"realm_access.roles": "roles", "address": "address"},
    rar=RarMapping(type="payment_initiation", fields={
        "instructedAmount.amount": Claim("approved_amount", type="double"),
        "instructedAmount.currency": "approved_currency",
        "creditorName": "approved_creditor",
    }),
)
```

What the PDP receives:

| Attribute | Category | Value |
|---|---|---|
| `user_id` | subject | `alice@example.com` |
| `agent_id` | subject | `spiffe://example.com/agents/travel` |
| `actor_id` | subject | `spiffe://example.com/agents/travel` |
| `scopes` | subject | `["payments.initiate", "flights.book"]` |
| `acr` / `amr` / `auth_time` | subject | `urn:example:mfa` / `["pwd", "otp"]` / `1790000000` |
| `roles` | subject | `["user", "traveller"]` |
| `approved_available` | subject | `true` |
| `approved_type` / `approved_actions` / `approved_locations` | subject | `payment_initiation` / `["initiate"]` / `["https://payments-mcp.example.com"]` |
| `approved_amount` / `approved_currency` / `approved_creditor` | subject | `123.5` (a double) / `EUR` / `Merchant A` |
| `transaction_id` | environment | `8b4729cc-32e4-4370-8cf0-5796154d1296` |

What is not sent, and why:

- `iss`, `aud`, `exp`, `iat`, `jti`: verification has already used them; they are not mapped.
- The prior actor `concierge`: informational only (RFC 8693 §4.1); logged at debug level.
- `creditorAccount` and the `account_information` entry: not mapped. The second entry is not for
  this server anyway (its `locations` name another one).

What is rejected: `address` is mapped but is an object, which no PDP attribute can hold.
`preview()` reports it in `dropped` as `can't be sent as 'address': not a scalar or a list of
scalars of one type`. A value that can't be converted to its `Claim` type is rejected the same
way.

## Rich Authorization Requests

With RFC 9396 (RAR) the user approves something specific, such as "pay 123.50 EUR to Merchant
A", and the identity provider records the approved version in the token's
`authorization_details` claim (§9.1). Checking that a tool call stays inside the approval is
left to the resource server, and it is an attribute comparison. `RarMapping` turns one approved
entry into attributes:

```python
from axiomatics_ai import Claim, RarMapping

rar = RarMapping(
    type="payment_initiation",
    fields={
        "instructedAmount.amount": Claim("approved_amount", type="double"),   # RAR amounts are strings
        "creditorName": "approved_creditor",
    },
)
AgentIdentityContextProvider(verification, rar=rar)
```

- **Which entry.** The entry must have this `type`, and its `locations` must include `location`,
  which defaults to the verification audience (the first one, if it is a list). An entry without
  `locations` counts only when the token has a single audience, i.e. it was issued for this
  server alone; with several audiences it might be meant for another API, so it must name this
  server. With `action=` set, its `actions` must include that action.
- **Exactly one.** When exactly one entry matches, its common fields (RFC 9396 §2.2) are sent as
  `approved_type`, `approved_actions`, `approved_locations`, `approved_datatypes`,
  `approved_identifier` and `approved_privileges` (whichever are present), and each path in
  `fields` as its attribute. `approved_available` says whether exactly one entry matched, and it
  is always sent. When none or several match, it is `false` and no other `approved_*` attribute
  is sent.
- **Separate grants are never merged.** If the user approved 10 EUR to Merchant A and, separately,
  5000 EUR to Merchant B, merging the two could give "5000 EUR to Merchant A", which nobody
  approved. So two matching entries give `approved_available = false` and nothing else, and
  `preview()` says `2 entries match; separate grants are never merged`. Narrow the match with
  `location` or `action` if one server legitimately receives several entries.
- `prefix` (default `approved_`) changes the attribute names.

A policy compares the tool call's arguments (resource attributes) with the approval. The ALFA
attribute names below are illustrative, for `approved_available`, `approved_amount`,
`approved_creditor` and the tool's `amount` and `creditor` arguments:

```alfa
policy initiatePayment {
    target clause toolName == "initiate_payment"
    apply firstApplicable

    rule withinApprovedPayment {
        condition approvedAvailable == true
              && amount <= approvedAmount
              && creditor == approvedCreditor
        permit
    }

    rule otherwise {
        deny
    }
}
```

Not covered: RAR delivered through token introspection (opaque tokens, RFC 9396 §9.2), and
building RAR objects for a CIBA or other human-approval request. The SDK does not request tokens.

## Values the PDP cannot carry

A PDP attribute holds one scalar or a list of scalars of one type. Every request builder drops,
on every path, any value that is an object, a nested list or a mixed-type list, and logs one
warning per attribute id. That covers tool arguments, token claims, record fields and enricher
output. One malformed value, such as a model-chosen argument, can't make the whole request fail,
and a policy never receives a value it can't evaluate. If a policy needs something inside an
object, map that field: with `claims=` for tokens, or in a custom `get_resource` for arguments.

Because an argument can be dropped, check arguments with permit rules, as in the RAR policy
above. A deny rule such as "deny when `amount` > 1000" never fires when `amount` arrived as an
object and was dropped. A permit rule that requires `amount` fails closed. See
[Security](security.md#values-the-pdp-cannot-carry).

## The agent's identity on client paths

On client paths your code passes the context. `DefaultClientContextProvider` reads `agent_id`
from it, next to `username`, `user_id` and `tenant_id`, and sends it as a subject attribute.
It comes from the runtime context on each call, not from configuration, so one application can
run several agents, each with its own identifier (AIMS §6 asks for one stable identifier per
agent, such as a SPIFFE ID):

```python
agent.invoke({"messages": [...]}, context={"user_id": "alice", "agent_id": "spiffe://example.com/agents/travel"})
```

## An agent that receives the user's token

An agent often runs behind your own API: the frontend calls a `/chat` endpoint with the signed-in
user's access token, issued for your agent API. Your backend is then the resource server for that
token, so it verifies it, and the same provider works on the client path. `token_from` finds the
token in the runtime context you pass to the agent:

```python
provider = AgentIdentityContextProvider(
    verification=JwtVerification(issuer="https://idp.example.com", audience="https://travel-agent.example.com"),
    token_from=lambda context: context.get("access_token"),
)
agent = create_authorized_agent(model=..., tools=[...], config=AuthZConfig(context_provider=provider, ...))

# in the /chat endpoint:
agent.invoke({"messages": [...]}, context={"access_token": bearer_token_from_the_request})
```

Everything else is the same: the attribute names, `claims=`, `RarMapping`, caching and expiry.
Runnable example: [`examples/langchain_agents/agent_behind_api.py`](../examples/langchain_agents/agent_behind_api.py).

**When the platform already verified the token** (LangGraph Platform's auth handler, AgentCore
Runtime's inbound JWT authorizer), don't verify it twice: pass the verified claims with
`claims_from`, and they go straight to the same mapping. An `exp` in them is still honoured.

```python
AgentIdentityContextProvider(claims_from=lambda context: context.get("verified_claims"))
```

Use `claims_from` only when something trustworthy did verify the claims; the provider logs at
start-up that it does not verify tokens itself. With it, `RarMapping` needs `location=` (there is
no verification audience to default to), and `for_audience()` is not available.

**Only for tokens addressed to the agent.** A token the agent *obtained* to call another API is
for that API, not for the agent (RFC 9068 §6): don't read it. What was granted comes back in the
token response (`scope`, and `authorization_details` for RAR), which is meant for the client.

## Server default: tokens read unverified

Without `AgentIdentityContextProvider`, the server paths use `DefaultServerContextProvider`,
which still reads the bearer token **without verifying it** and forwards every claim. That is
correct only behind an ingress or API gateway that has already authenticated the request. It now
logs a one-time warning at the first request saying so. See [Identity](identity.md#trust-boundary).

## Revocation and the decision cache

AIMS §11: "Cached authorization decisions and tokens that are no longer valid MUST NOT continue
to be used" after a revocation or risk notification. The SDK does two things:

- **A cached decision never outlives the credential.** With `AgentIdentityContextProvider`, a
  decision is cached until the earlier of `cache_ttl_seconds` and the token's `exp`.
- **`config.invalidate_decisions(...)`** drops cached decisions when a signal arrives:

  ```python
  config.invalidate_decisions(reason="session revoked", user_id="alice")          # one user's decisions
  config.invalidate_decisions(reason="agent quarantined", agent_id="spiffe://example.com/agents/travel")
  config.invalidate_decisions(reason="risk level raised")                          # everything
  ```

  The keywords are subject attributes as they were sent (all given ones must match). The call
  reaches every decision cache built from that config, on every surface and every gateway route,
  returns the number dropped, and emits an `on_invalidation` hook event (`InvalidationEvent`),
  which `AuditLogHook` records as a remediation.

Invalidation forces the next call back to the PDP; it does not make the token invalid. A JWT
still verifies until it expires, so for the next decision to be a Deny the PDP has to know about
the revocation, for example through an attribute it looks up. The cache is per process: every
replica has to receive the signal.

Receiving the signal is out of scope. The SDK has no Shared Signals (SSF/CAEP) receiver. A
revocation webhook of your own would call it like this:

```python
from starlette.requests import Request
from starlette.responses import Response

async def on_revocation(request: Request) -> Response:
    # Authenticate the sender (signature, mTLS, shared secret) before acting on it.
    event = await request.json()
    config.invalidate_decisions(reason=event["reason"], user_id=event["user_id"])
    return Response(status_code=202)
```

## Audit trail

AIMS §11 asks for durable, tamper-evident audit logs recording the agent, the delegated user,
the resource or tool, the action and decision, a timestamp and correlation id, the risk state,
and any remediation. `AuditLogHook` writes one structured record per decision and per cache
invalidation, with those fields:

```python
from axiomatics_ai import AuditLogHook

config = AuthZConfig(..., hooks=[AuditLogHook(hash_chain=True)])
```

Record fields, sinks and the hash chain are on [Hooks](hooks.md#audit-trail-auditloghook).
Durability belongs to the sink and the SIEM behind it, not to the SDK.

## Gateway: no token passthrough

AIMS §10.8 calls forwarding the agent's token onward an antipattern, and the MCP authorization
spec forbids token passthrough. The gateway therefore strips the caller's `Authorization` header
before forwarding, on MCP routes (FastMCP's proxy would otherwise forward it) and on A2A routes.
`Route(forward_authorization=True)` opts a route back in, for a target that must see the token.

When each upstream is its own resource server, `Route(audience=...)` makes that route accept only
tokens issued for that upstream. See [gateway](integrations/gateway.md#identity).

## Tokens never reach the model

AIMS §8: the LLM must not have access to the agent's credentials or to credentials for tools, so
a prompt injection can't leak them. The SDK keeps to this. On server paths the token is read from
the HTTP headers, never from tool arguments or the conversation. Verification failures log the
reason, never the token, and hooks receive the mapped attributes, not the token. Keep to it in
your own code too: attach tokens in the HTTP client (for example the MCP client's `headers`),
never put them in the prompt, a tool argument, or state the model can read.

## What the SDK covers

| Requirement | Where | How the SDK does it |
|---|---|---|
| Validate access tokens at the resource server | RFC 9068 §4 | `AgentIdentityContextProvider` + `JwtVerification`: signature, `iss`, `aud`, `exp`, `typ`, asymmetric algorithms |
| Agent in `client_id`, user in `sub`, decide on both | AIMS §10.3, RFC 9068 §2.2 | `agent_id`, and `user_id` only when a user is present |
| Decide on the current actor only | RFC 8693 §4.1 | `actor_id` from the outermost `act`; prior actors logged |
| One stable identifier per agent | AIMS §6 | `agent_id` from the token, or from the runtime context on client paths |
| Enforce approved authorization details | RFC 9396 §9 | `RarMapping`; exactly one entry, never merged |
| No cached decision after revocation or expiry | AIMS §11 | cache capped at `exp`; `invalidate_decisions()` |
| Durable, tamper-evident audit | AIMS §11 | `AuditLogHook` with `hash_chain=True`; durability in your sink |
| Correlation id across services | AIMS §11 | `transaction_id` from `txn`, plus the SDK's `correlation_id` |
| Don't pass tokens on | AIMS §10.8, MCP | gateway strips `Authorization` unless `forward_authorization=True` |
| Credentials out of the model's reach | AIMS §8 | tokens read from headers only, never logged |

## What it doesn't cover

These belong to other components of an AIMS deployment:

- **Token issuance and token exchange** (AIMS §10.3 to §10.6, RFC 8693). The SDK consumes tokens;
  your identity provider issues and exchanges them.
- **Human approval** such as CIBA (AIMS §10.7). The SDK does not obtain an approval or a token for
  one.
- **Shared Signals transport** (SSF, CAEP, RISC). `invalidate_decisions()` is what a receiver calls;
  the receiver is yours.
- **WIMSE proof tokens and mTLS** (AIMS §9). Proof of possession is checked, if at all, before the
  request reaches the SDK.
- **Transaction-token issuance** (AIMS §10.5). The SDK reads `txn`; it does not mint transaction
  tokens.
- **Signed decision receipts.** Decisions are logged, with an optional hash chain, but not signed.
- **Token introspection** for opaque tokens. Only JWT access tokens are verified.

---

← Previous: [Identity](identity.md) · [Docs index](README.md) · Next: [Enforcement points](enforcement-points.md) →
