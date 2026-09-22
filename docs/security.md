# Security

What the SDK assumes, what it guarantees, and the switches that trade safety for availability.
Read this before running any path in production.

## Unverified tokens by default

By default the server-side paths (`fastmcp`, `tool_decorator`, `a2a`) read the caller's
identity from the `Authorization: Bearer` JWT **without verifying its signature**, and log a
one-time warning saying so. The gateway, usually the edge, refuses to start that way: it needs a
verifying provider, or `GatewayConfig(allow_unverified_tokens=True)` to say authentication
happens in front of it. The helper is named `decode_bearer_jwt_unverified` on purpose.
Run these surfaces behind an ingress, API gateway, or auth middleware that has already
authenticated the request. A token that arrives unverified yields whatever subject it asserts.

If the process is your edge, or you want defence in depth, use `AgentIdentityContextProvider`
with a `JwtVerification` naming the issuer you trust and this server's audience
([Agent identity](agent-identity.md#verify-the-token)). A token that fails verification, or no
token, gives an empty subject. An empty subject never reaches the PDP: every check resolves it
by `fail_open`, so with the default it is denied.
Client-side paths trust the `context` your own code passes; they are only as trustworthy as that
code.

## Tokens and credentials

- **No token passthrough.** The gateway strips the caller's `Authorization` header (and, on A2A
  routes, `Cookie`) before forwarding, on MCP and A2A routes; opt in per route with
  `Route(forward_authorization=True)`. A token is for the service it was issued to (AIMS §10.8).
- **Tokens stay out of the model.** The SDK reads tokens from HTTP headers only and never logs
  them. Don't pass a token in a prompt, a tool argument, or any state the model can read.
- **PDP credentials and request bodies stay out of logs.** `PDPConnection`'s `headers`, `auth`,
  `cert` and `http_client` are left out of its `repr`, and the PDP client's DEBUG log names the
  URL, status and latency only — never a request body, which carries the subject's claims. The
  [hooks](hooks.md) `on_pdp_request` / `on_pdp_response` see the bodies when you opt in.

See [Agent identity](agent-identity.md#gateway-no-token-passthrough).

## Nothing is enforced by default

An `AuthZConfig` with no `enforcement` entries and no `retrieval` enforces nothing, on every
path. This is deliberate (a bare config is legitimate on the `a2a`, `tool_decorator`, and
retrieval-only paths) but it means a typo in the enforcement dict silently turns a point off.
Log `config.summary()` at start-up, and check `config.enforces(point)` in a test.

## `fail_open`

`PDPConnection(fail_open=False)` (the default) treats a PDP that is unreachable, times out, or
returns NotApplicable / Indeterminate as a Deny. `fail_open=True` treats all of those as Permit.
It is an availability switch: under it, a PDP outage means every tool is visible, every call
runs, every record is returned, and every retrieval is unfiltered. Use it only where an outage
is worse than over-permitting, and alert on `on_pdp_response` errors when you do.

`fail_open` never masks a request the PDP rejects with HTTP 4xx (undeclared attribute, wrong
endpoint, bad credentials). Those raise `PdpRequestError`, because retrying cannot help and
silently permitting would hide a misconfiguration.

## The decision cache

Decisions are cached in memory for `cache_ttl_seconds` (default 300) keyed on the full
subject, action, resource, and environment. A policy change on the PDP is therefore not seen by
a running process for up to the TTL. Lower the TTL, set it to 0 to disable, or call
`config.invalidate_decisions()` (every cache built from the config) or `clear_cache()` on an
`Authorizer` after a change.

With `AgentIdentityContextProvider` a decision is never cached past the token's `exp`. On a
revocation or risk signal, call `config.invalidate_decisions(reason=..., user_id=...)` (or
`agent_id=...`) so no cached decision for that subject is used again (AIMS §11). This drops
decisions; it does not revoke the token, which still verifies until it expires. See
[Revocation](agent-identity.md#revocation-and-the-decision-cache). Do not put volatile values (timestamps,
request ids) in the environment unless you want to defeat the cache.

## Content guards are probabilistic

The input and output guards rely on an LLM to classify text. A classifier can be wrong in both
directions and can itself be the target of an injection. The SDK fences the content in a tag with
a random nonce and tells the classifier to treat it as data, which makes "classify this as safe"
text less effective, not impossible. Treat the guards as one layer, keep
`on_unavailable="fail_closed"` (the default), and prefer a dedicated small model. The PDP
decision downstream is deterministic; the attributes it decides on are not.

The same holds for [intent-based access control](intent-based-access.md). Its classifier reads
only the user's own messages, never tool output, which is what makes it useful against indirect
prompt injection; but a user can still phrase a request the classifier misreads.

## Retrieval and missing identity

Retrieval filtering fails closed (no documents) when the caller has no usable subject
attributes, when the subject is partial so the PDP's condition still references unbound subject
attributes, or when the condition cannot be translated to the store's filter. Only
`fail_open=True` changes this to an unfiltered search. A subject-presence check in policy does
not help here: a partial evaluation treats unprovided attributes as unbound, not absent, so the
check never fires. The SDK owns this rule; do not rely on policy to deny an anonymous retrieval.

## Tool names and argument values are attacker-influenced

Tool names on MCP servers and the arguments the model passes to a tool are inputs the SDK
forwards to the PDP as attributes. Write policies that permit known-good values rather than deny
known-bad ones, and keep `MustBePresent="false"` designators in mind: a missing attribute does
not match a Permit rule, which is the safe direction under first-applicable combining.

Argument checks in particular belong in permit rules. An argument whose value the PDP cannot
carry is dropped (next section), and a deny rule on a dropped argument never fires.

An argument cannot pose as the tool. The default providers drop, with a warning, any argument
whose name equals the resource identifier attribute (`tool_name` by default), so the identifier
the policy sees is always the real tool name. A custom `get_resource` should keep that
guarantee by setting the identifier after merging `args`.

## Values the PDP cannot carry

A PDP attribute holds one scalar or a list of scalars of one type. Every request drops
attributes whose value is an object, a nested list or a mixed-type list, with one warning per
attribute id, whether the value came from a tool argument, a token claim, a record field or an
enricher. The request still goes out without it. So a deny rule such as "deny when `amount` >
1000" is bypassed by a model that passes `amount` as `{"value": 5000}`, while a permit rule
that requires `amount` fails closed. Map a field inside an object if a policy needs it
([Agent identity](agent-identity.md#values-the-pdp-cannot-carry)).

## Logging and hooks

`DecisionEvent` and `EnrichmentEvent` carry subject attributes, tool arguments, and (for
enrichment) the inspected prompt or answer text. Anything you log from a hook is subject to the
same data-handling rules as the application's own logs. `LoggingHook` logs all of it at INFO,
and so does `AuditLogHook` (as JSON, on the `axiomatics_ai.audit` logger). Neither ever sees the
bearer token.

---

← Previous: [Hooks](hooks.md) · [Docs index](README.md) · Next: [Troubleshooting](troubleshooting.md) →

## Known limitations

These are known gaps in this release, each planned for a later one. The ones marked
**(can leak)** can let content the policy denies reach the model or the client; each says how
to avoid it.

- **OpenAI Agents** — hosted, shell, apply-patch and computer tools are decided by name, not per
  call with their arguments ([details](integrations/openai_agents.md#enforcement-points)).
  `agent.clone()` returns an agent without the authorization: authorize the clone.
- **Google ADK built-in tools (can leak)** — ADK's built-in Gemini tools (`google_search` and
  the like) run on Google's side and aren't function tools, so they are never sent to the PDP:
  they stay available whatever the policy says. Don't give them to an agent whose tools the
  policy must govern.
- **Google ADK** — a plugin's `after_tool_callback` that returns a value runs instead of the
  agent's callbacks, so post-invocation is skipped for that call; don't combine such a plugin
  with `AFTER_TOOL`.
- **Claude Agent SDK** — post-invocation rewrites a tool's output for MCP tools; the CLI expects
  its own output shape for built-in tools (`Read`, `Bash`, ...), so don't rely on
  post-invocation filtering there.
- **`@protected_tool` (can leak)** — a tool that returns a JSON string, a tuple, a pydantic
  model or a dataclass is passed through unfiltered (`ColumnMaskStrategy` included); return a
  list or a dict.
- **Input guard and conversation history (can leak)** — `BEFORE_MODEL` checks the latest user
  message only (LangChain, Strands, Google ADK). A blocked prompt stays in the conversation
  history (checkpointed thread, agent messages, ADK session), so a later, harmless prompt sends
  it to the model with the rest of the history; and with several user messages in one input,
  only the last is checked. Start a new thread after a block, and send one user message per
  turn. Content guards are probabilistic anyway ([above](#content-guards-are-probabilistic)).
- **Content next to the data under the default `fail_open` (can leak)** — post-invocation
  filters the data it can read; a block next to it that it can't read (plain text, an image, an
  embedded resource that restates the records) is reported and returned unfiltered under the
  default `on_unrecognized_output="fail_open"`. Set `"fail_closed"` on the `AFTER_TOOL`
  `Enforcer` to withhold it ([Post-invocation](post-invocation.md)).
- **Record filtering of nested records (can leak)** — `RecordFilterStrategy` treats a dict that
  holds a list of objects as a container: it filters the list and keeps the dict's own fields
  without asking the PDP. A single record with a nested list (`{"name": ..., "ssn": ...,
  "visits": [...]}`) therefore comes back whole. Return such records inside a list
  (`[{"name": ..., "visits": [...]}]`), where each is decided as a record, or use
  `ColumnMaskStrategy`.
- **Human approval: who resumes, and across processes (can leak)** — an answer is tied to the
  arguments the human saw and an OpenAI approval runs its tool once, but only within the process
  that paused the call; a checkpoint or `RunState` edited and resumed in another process is not
  checked against what was shown. Which caller resumes a paused run is not checked either (call
  ids aren't unique across callers): let only the user a run was paused for resume it, and keep
  resume state where callers can't change it.
- **FastMCP elicitation approves on the client's word (can leak)** — with no `approver`
  configured, FastMCP asks the MCP client by elicitation, so the client that made the call
  answers for it; a headless or agent client can approve its own calls. Configure an `approver`
  on the server where a human must decide.
- **A2A client guard** — `A2AClientAuthZ` authorizes JSON-RPC calls only; a client that uses the
  HTTP+JSON (REST) transport sends them unchecked. Configure the a2a-sdk client for JSON-RPC.
- **Gateway (preview)** — MCP resources and prompts are refused, not authorized; SSE responses
  are buffered; the A2A REST binding is refused. See
  [Scope](integrations/gateway.md#scope--not-yet-supported).
- **Retrieval** — a cardinality guard (`size(X) == n`) inside an `and` of a partial-evaluation
  result is dropped, not translated, whatever `n`; don't rely on one to narrow the documents.
  Outside an `and` it fails closed (no documents).
- **Configuration** — some misconfigurations are only logged, not rejected: an enforcement key
  given as a string instead of an `EnforcementPoint`, or a misspelt `Literal` value.
