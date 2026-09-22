# Troubleshooting

Start with `config.check_connection()` and `config.summary()`. The first tells you whether the
PDP is reachable at all; the second tells you what you actually enabled.

## Nothing is being enforced

`enforcement` is empty or `None`. Every point is opt-in; `standard_tool_enforcement()` is the
usual three tool points. Check `config.summary()` in your start-up log: `enforces=[none]` means
exactly that. On `tool_decorator`, a tool without `@protected_tool` is never checked.

## Everything is denied

- **No rule matched.** The PDP returned NotApplicable and `fail_open=False` maps that to Deny.
  Compare the action name in your policy target with the `Enforcer.action_name` you configured,
  and the resource attribute id (`tool_name` at every tool point, `agent_name` for A2A,
  `resource_name` for `Authorizer`; see [Attribute conventions](enforcement-points.md#attribute-conventions)).
- **Wrong subject attribute.** The default client provider sends `username`, `user_id`,
  `tenant_id`, `agent_id`; the default server provider sends every JWT claim;
  `AgentIdentityContextProvider` sends only the [named attributes](agent-identity.md#what-the-pdp-receives)
  and the claims you map. If your policy expects
  `employee_id`, write a context provider ([Identity](identity.md)).
- **Empty subject.** No check calls the PDP for a subject with no attributes; it resolves to
  `fail_open` (denied by default). On a server that means the `Authorization` header did not
  arrive, was not a bearer JWT, or failed verification.
- **Retrieval returns no documents.** See [Security](security.md#retrieval-and-missing-identity):
  missing or partial identity fails closed by design.

## Every call is denied with `AgentIdentityContextProvider`

The token fails verification, so the subject is empty. Look for the warning
`Caller identity not verified, so the request is denied: <reason>` in the log (the
`axiomatics_ai.core.identity` logger). It is logged once per token: a rejected token is
remembered for 30 seconds, and a failed issuer discovery is retried after 30 seconds, so fixing
the cause takes effect within half a minute:

- **`token type 'JWT' is not an access token (at+jwt)`.** Your identity provider types its access
  tokens plain `JWT`. Set `JwtVerification(require_typ=False)`.
- **`not signed by a trusted issuer (...)`.** No trusted issuer's published keys verify the token.
  Nothing in the token is read before verification, so the SDK tries each trusted `issuer`'s
  keys; the parenthesis lists why each failed. Either the token comes from another issuer, or
  `issuer` doesn't equal the token's `iss` exactly (including any trailing slash), or the key it
  was signed with isn't (yet) in the issuer's JWKS.
- **`Audience doesn't match` / `Invalid audience`.** `audience` must be this server's identifier
  as registered at the identity provider, as it appears in the token's `aud`.
- **`The token is not yet valid` / `Signature has expired`.** Clock skew between this host and
  the identity provider, or a genuinely expired token. Sync the clock; `leeway_seconds` (default
  30) tolerates small skew.
- **`could not discover the signing keys (jwks_uri) of issuer ...`.** Neither
  `/.well-known/openid-configuration` nor `/.well-known/oauth-authorization-server` under the
  issuer answered with a `jwks_uri`, or the host can't reach it. Set `jwks_url` explicitly.
- **`The specified alg value is not allowed`.** Add it to `algorithms` if it is asymmetric; `HS*`
  and `none` are never accepted.
- **`Token is missing the "sub" claim`** (or `exp`, `aud`, `iss`): the token lacks a required
  claim.

No warning at all means no `Authorization: Bearer` header arrived. `preview(claims)` shows what a
verified token would turn into, without the network.

## `PdpRequestError: PDP rejected the request ... HTTP 400`

Over `protocol="authzen"` the PDP maps attributes strictly; you sent an attribute the domain
does not declare. Declare it, or drop it from the provider. Over XACML the same attribute would
be silently ignored, so switching protocols changes this behaviour. With
`AgentIdentityContextProvider`, declare every attribute it sends (the named ones, your mapped
claims, and with a `RarMapping` `approved_available` and the `approved_*` fields), or set the
unused ones to `None` in `AgentIdentityAttributes`. With an `intent_enricher`,
every intent attribute must be declared, including one `intent_<action>_recipients` per
configured action even if the policy never uses it
([Intent-based access control](intent-based-access.md#what-the-pdp-receives)).

## A list attribute never matches over AuthZEN

A rule that tests membership in a list attribute (`"send" == intentActions`, multi-valued
`roles`) denies even though the value is in the list. ADS 26.1.6's AuthZEN mapping does not
turn a JSON array into a multi-valued attribute; a single value works, a list of any length
does not. Use `protocol="xacml"` for list-valued attributes, including
[intent-based access control](intent-based-access.md).

## `PdpRequestError: ... HTTP 404`

The endpoint path is wrong for your PDP. Defaults are ADS paths; set
`PDPConnection(endpoints=PdpEndpoints(...))` ([Configuration](configuration.md#endpoints)). Over
AuthZEN on ADS, `authzen.enabled: true` must be set in the deployment file. A 404 on
`partial_evaluation` means the PDP does not support partial evaluation at all.

## `could not reach PDP at ...`

`check_connection()` could not connect. Wrong `base_url`, PDP not started, or a container that
must use `host.docker.internal` instead of `localhost`. This is the transient class of failure
that `fail_open` governs.

## AuthZConfig warnings at start-up

`AuthZConfig` validates itself on construction and logs, without raising, misconfigurations that
are always wrong: an empty `attribute_names.tool`, `.agent` or `.mcp_server`,
`strategy`, `action_overrides` or `on_unrecognized_output` on a point other than `AFTER_TOOL`,
`TOOL_VISIBILITY` without `BEFORE_TOOL` (a hidden tool can still be called by name), retrieval
configured together with
`protocol="authzen"` (retrieval always uses the XACML-based reverse query), or an enricher at
the wrong point: at a point that never runs them (`TOOL_VISIBILITY`,
`AFTER_TOOL`, `A2A_INVOCATION`), or outside the enricher's `allowed_points` (for example
`intent_enricher` anywhere but `BEFORE_TOOL`, or `prompt_injection_enricher` at `BEFORE_TOOL`).
`config.validate()` returns the same list for a test.

## Cached decision after a policy change

Decisions live in the cache for `cache_ttl_seconds` (default 300). Restart the process, lower
the TTL, set it to 0 while developing policy, or call `config.invalidate_decisions()` (every
cache built from the config) or `Authorizer.clear_cache()`.

## The model still sees a hidden tool (`claude_agent`)

The Claude Agent SDK exposes no per-turn tool-list filter, so this path has no
`TOOL_VISIBILITY`. Write the rule at `BEFORE_TOOL`; the call is denied when the model attempts
it.

## A tool argument does not appear in the request

Arguments are forwarded as resource attributes on `BEFORE_TOOL` and post-invocation. If your
provider overrides `get_resource`, it must merge `args` itself (see the example in
[Identity](identity.md#custom-provider-client-side)). An argument named like the identifier
attribute (by default `tool_name`) is dropped with a warning; rename the argument or set
`AuthZConfig.attribute_names.tool`. An argument, claim or record field whose value is an object,
a nested list or a mixed-type list is dropped too, with the warning `Attribute '...' is not sent
to the PDP` (once per attribute); map a field inside it
([Agent identity](agent-identity.md#values-the-pdp-cannot-carry)).

## Audit records don't appear

`AuditLogHook` logs to the `axiomatics_ai.audit` logger at INFO. Python's default level is
WARNING, so give that logger a handler and set it to INFO, or pass a `sink=` callable.

## The backend behind the gateway sees no identity

The gateway no longer forwards the caller's `Authorization` header. Set
`Route(..., forward_authorization=True)` on the route whose target needs it
([gateway](integrations/gateway.md#identity)).

## Content guard blocks everything

The enricher could not classify (bad model string, missing API key, provider error) and
`on_unavailable="fail_closed"` blocked without calling the PDP. Check the `EnrichmentEvent`
error via a hook or `LoggingHook`, and that the model is a fully qualified LiteLLM string with
its provider key set, or `AXIOMATICS_ENRICHER_MODEL` is set.

## Every tool call is denied with intent enrichers

- **Server-side path.** `fastmcp` and `gateway` never see the conversation, so intent is always
  unavailable and `on_unavailable="fail_closed"` denies every call; the middleware logs a
  warning at start-up. Configure `intent_enricher` on the client-side agent instead.
- **Classification fails.** Same causes as a content guard that blocks everything (above); look
  for `intent_available=false` in the `EnrichmentEvent`.
- **The policy denies.** Log the request with `LoggingHook` and compare `intent_actions` and
  `intent_<action>_recipients` with the tool call. Recipients are copied as the user wrote them,
  and a "yes, do it" confirmation carries no intent (see
  [Limits](intent-based-access.md#limits)).

## A Permit is enforced as a Deny

The decision carried an **obligation** the SDK has no handler for, and unknown obligations deny
by default (XACML: a PEP that can't fulfil an obligation must not permit). The decision event's
`detail` lists them as `unknown_obligations`. Register a handler (or `None` to accept the id)
in `ObligationsAndAdviceConfig(handlers=...)`, or set `on_unknown="ignore"`
([Obligations and human approval](human-approval.md)). A `require-approval` obligation outside
a tool call is also a Deny: only a tool call can wait for approval.

## Human approval never happens, or every approval-required call is denied

- **No way to ask.** With no `approver` and a path that can't pause (Claude Agent SDK, an ADK
  tool context without confirmation support, a FastMCP client without elicitation), the call is
  denied and a `rejected` event with `source: "none"` is logged. Configure an approver.
- **LangChain doesn't resume.** `interrupt()` needs a checkpointer and the same `thread_id` on
  the resume call.
- **Approved, but still denied.** On the Deny + advice form the PDP is asked again with
  `approval_granted = true`; the policy must permit then. If it asks for approval again, the SDK
  denies.
- **The tool is hidden.** An approval rule on tool *visibility* hides the tool; put it on
  tool invocation.

## A numeric rule matches some calls and not others

Tool arguments are sent with the type they arrive in, and a model may send `40` (a JSON integer,
XACML `integer`) or `40.0` (a `double`). A rule on one type doesn't match the other. Send the
value from a context provider with a fixed type, e.g. `float(...)`, or better, look the value up
yourself instead of trusting the model's (see `examples/approval_console.py`).

## Running the examples

- **The LLM hallucinates tools or the wrong server answers.** Another process is squatting on
  the example MCP ports. Stop it before running.
- **ADS serves the wrong policy.** ADS loads one domain at a time; each example names the domain
  it needs. Stop ADS, change `domain.path`, restart ([Running ADS](running-ads.md)).
- **`FastMCP` sees no `Authorization` header.** Read headers with
  `get_http_headers(include={"authorization"})`; the default excludes it.

---

← Previous: [Security](security.md) · [Docs index](README.md) · Next: [Performance](performance.md) →
