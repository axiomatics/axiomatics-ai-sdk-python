# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **fastmcp / tool_decorator / a2a** — a request with more than one `Authorization` header is
  treated as anonymous (denied): authentication in front checks the first header, the SDK read
  the last, so a caller could pass with their own token and be decided as someone else.
- **human approval** — an answer counts only for the arguments the human saw: if they changed
  before the resume (an edited checkpoint or `RunState`), it is refused as not approved. On
  OpenAI an approval runs its call once; a `RunState` resumed again is refused. Both within the
  process that paused; which caller resumes a paused run is the application's to check.
- **fastmcp** — PDP errors from resources and prompts are masked too; an elicitation answer that
  isn't approve/reject is logged as a rejection.
- **pdp** — NotApplicable answers are cached again (an Indeterminate is not); numeric resource
  names in XACML batches are read again; NaN, infinity and lone surrogates are dropped like other
  values a PDP can't carry (with a warning) instead of failing the request.
- **packaging** — the `Development Status :: 3 - Alpha` classifier; human approval and the A2A
  client guard are marked preview in the docs.
- **claude_agent** — the hooks fail closed when authorization itself fails (a malformed PDP
  answer, a 4xx): the CLI treats a hook that raises as "no decision", so a denied tool ran under
  `allowed_tools`, unfiltered output reached the model and the prompt went unchecked. The tool is
  now denied, the output withheld (for MCP tools; the CLI keeps a built-in tool's own output)
  and the prompt blocked. The prompt guard runs off the event
  loop, and the per-session prompt cache is locked.
- **google_adk** — with tool visibility but no `BEFORE_TOOL`, a hidden tool leaves ADK's tool
  map again, so ADK refuses a call to it and ends the run (with `BEFORE_TOOL`, the per-call
  check refuses it and the run continues).
- **pdp** — a request holding a value JSON can't carry (NaN, infinity, a lone surrogate) raises
  `PdpRequestError` instead of counting as an outage that `fail_open` would permit. A malformed
  XACML MultiRequest result gives no decision instead of raising. An AuthZEN batch response
  that doesn't hold exactly one evaluation per name gives no decision for any of them (it is
  matched by position). Only a Permit or a Deny is cached, so a `fail_open` fallback for an
  Indeterminate or NotApplicable answer is decided again.
- **fastmcp / strands** — a PDP error raised inside a tool (FastMCP) or in post-invocation
  (Strands) reaches the client or the model as "Authorization is unavailable.", not with the
  PDP's URL and answer.
- **a2a** — `message/stream` works behind `A2AAuthZMiddleware` (the stream ended at once: the
  middleware reported a client disconnect after replaying the request body).
- **gateway** — the A2A forwarder keeps no cookies: one client serves every caller, so a cookie
  the target set for one caller went upstream on the next caller's requests.
- **fastmcp** — a request the PDP rejects (HTTP 3xx/4xx, misconfiguration) reaches the MCP
  client as "Authorization is unavailable."; the PDP's URL and answer stay in the server log
  (also for `@protected_tool` and the gateway's MCP routes).
- **google_adk** — tool visibility hides denied tools from the model only; a call to a hidden
  tool is refused at `BEFORE_TOOL` instead of ending the run with "Tool not found".
- **openai_agents** — a blanket `RunState.reject(always_reject=True)` in one caller's run no
  longer counts as a rejection of another caller's paused call.
- **docker** — `.dockerignore` keeps the ADS jars and licences (`examples/policies/`) and every
  `.env` out of the gateway image.
- **fastmcp** — a tool that returns JSON text (`-> str`) is no longer double-encoded after
  post-invocation (the client got a JSON string literal; also on the gateway's MCP routes,
  Strands and Google ADK). Non-text blocks next to the data (an embedded resource, an image)
  go through the same handling as text: reported, and withheld under `fail_closed`; they used
  to pass through unfiltered.
- **extras** — `strands` needs `strands-agents>=1.43.0`, the first release with the
  interventions API.
- **extras** — `fastmcp` is capped below 4 (fastmcp 4 brings mcp 2, which moved `McpError`, and
  the FastMCP middleware, `@protected_tool` and the gateway's MCP routes did not import) and
  `deepagents` below 0.8 (pre-1.0; the deep-agent factory builds its sub-agent specs).
- **extras** — `langchain-agents` no longer includes deepagents: use the new `deepagents`
  extra for `create_authorized_deep_agent`. `all` is every integration extra (no longer the
  example and dev dependencies, which are now the `examples` and `dev` dependency groups that
  `uv sync` installs). Importing an integration without its extra raises an `ImportError` that
  names the extra.
- **pdp** — a redirect (HTTP 3xx), or a success response that isn't a JSON decision, raises
  `PdpRequestError` instead of resolving by `fail_open`. Decisions are read case-insensitively,
  so another PDP's `"deny"` is a Deny, not an inconclusive answer that `fail_open` permits.
- **openai_agents** — with `BEFORE_TOOL` enforced, tools that don't go through `on_invoke_tool`
  (hosted, shell, apply-patch, computer) are decided by name each turn and removed unless the
  policy permits them; they used to run unchecked. The input guard runs before the model
  (`run_in_parallel=False`). A tool shared by several agents is checked with the calling
  agent's config.
- **google_adk** — a result the policy denies entirely is returned as `{"result": {}}`, never
  empty: ADK reads an empty callback result as "no change", so a later `after_tool_callback`
  returning `None` restored the unfiltered response.
- **gateway** — now marked preview. MCP routes hide and refuse the target's resources and
  prompts unless `Route(mcp_resources_and_prompts=True)`; a request path with a `.` or `..`
  segment gets HTTP 400 on A2A routes.
- **rag** — the constraint translator fails closed on anything it does not recognise (unknown
  functions such as `stringStartsWith` or regular expressions, the `all*` bag functions, a
  `size` guard outside an `and`, unknown node shapes). These used to become "no constraint"
  and return the whole corpus. `not` is now translated by negating the comparison.
- **rag** — every public search method on a wrapped store is filtered, including
  `similarity_search_by_vector`, `max_marginal_relevance_search_by_vector` and store-specific
  variants, which used to bypass the filter.
- **rag** — `to_authorized_vectorstore` raises `ValueError` when `config.retrieval` is unset,
  instead of returning a store that searches unfiltered.
- **rag** — on the in-memory store, a comparison on a missing metadata key never matches,
  `!=` included, as in XACML.
- **post-invocation** — every path unwraps the tool's data from the framework's result shape
  before the strategy runs: FastMCP `structured_content` is filtered together with the text,
  Strands `ToolResult` content blocks (`text` and `json`) are filtered, and a LangChain
  `ToolMessage` from an MCP tool has the records inside its content blocks filtered rather than
  the blocks themselves.
- **post-invocation** — `RecordFilterStrategy` filters every record list in a list or a dict,
  at any depth of the dict's values (`{"data": {"results": [...]}}`), in one batched request, and
  treats any other dict as a single record, replaced by `{}` when denied. A list mixing records
  with `null` or nested lists is filtered too. A text or image block next to the data, which can
  restate it, is reported as `skipped` and withheld under `fail_closed`. `ColumnMaskStrategy` masks
  fields at any depth. Custom strategies receive the unwrapped data (a list or a dict).
- **post-invocation** — output the strategy cannot inspect is logged and reported as a
  `skipped` decision on every path (ADK and Strands used to pass it silently).
- **a2a** — `A2AAuthZMiddleware` authorizes every JSON-RPC method (`tasks/get`,
  `tasks/cancel`, push-notification config, …), not only message sends, and rejects any other
  POST — batches, bodies without a method, the REST binding — with a JSON-RPC `-32600` error.
  Policies must now permit the task methods their agents use. Of non-POST requests only a
  `GET` / `HEAD` of the public agent card passes (plus `public_paths=`); anything else gets
  HTTP 403.
- **langchain_mcp** — `authorized_mcp_tools(context=None)` returns no tools (or every tool
  under `fail_open`) instead of every tool.
- **claude_agent** — tool invocation is enforced in a `PreToolUse` hook (`permissionDecision:
  "deny"`) instead of the `can_use_tool` permission callback, which the CLI skips for tools in
  `allowed_tools`, under `permission_mode="bypassPermissions"` and for settings allow rules. It
  also runs with one-shot `query(prompt=<str>)`. The policy decides: a permitted call returns
  `permissionDecision: "allow"`, so no `allowed_tools` or permission mode is needed, and the
  options' `can_use_tool` is not consulted for it. A deny from your own `PreToolUse` hook still
  blocks.
- **claude_agent** — an MCP tool's server is sent as its own resource attribute
  (`AttributeNames.mcp_server`, default `mcp_server`), so `mcp__trusted__search_docs`
  and `mcp__other__search_docs` are no longer identical to the policy. Enforcing
  `TOOL_VISIBILITY` or `AFTER_MODEL` on this path now logs a warning.
- **google_adk** — our callbacks run before the agent's existing ones: ADK stops at the first
  callback that returns a value, which could skip the answer check or result filtering.
- **content guards** — the output guard checks text the model sends alongside a tool call on
  LangChain, ADK and Strands; a block replaces the text and keeps the tool call. The classifier
  content is fenced in a random-nonce tag and marked as data.
- **human approval** — the approval request shows the tool call's own arguments, not the PDP
  resource attributes, and an approval is bound to both. All events of one tool call share a
  correlation id and a `request_id`, across a framework pause and its resume; a rejection ends
  with a final `deny` event (`not_approved: true`). A refusal where nobody could be asked (an
  MCP client without elicitation) is logged with `source: "none"`, like a missing approver.
- **human approval** — a paused tool call is remembered with the PDP decision that asked for
  approval, so the PDP isn't asked again: not when LangChain or Strands re-run the paused step
  on resume, and not when the answer is applied (also on ADK, OpenAI and FastMCP). The advice
  form asks the PDP twice in all, the obligation form once, with or without the decision cache,
  and `approval_required` is logged once per call.
- **hooks** — `detail["obligations"]` / `detail["advice"]` list each as `{"id": ..., "attributes":
  {...}}` (was: ids only), so an approval's `reason` or `ttl` reaches the hooks.
- **tool_decorator** — a sync tool runs with the caller's context (`asyncio.to_thread`), so a
  wrapped RAG store inside it sees the identity; positional arguments reach the policy.
- `AuthZConfig.validate()` warns when `TOOL_VISIBILITY` is enforced without `BEFORE_TOOL`, and
  when `on_unrecognized_output` is set outside `AFTER_TOOL`.
- **PDP client** — one pooled `httpx.Client` per adapter instead of a connection per decision.
  The DEBUG log names the URL, status and latency only, never a request or response body.
- **PDP client** — a single check with an empty subject never reaches the PDP and resolves by
  `fail_open`, like a batch check already did: a request with no token, or one that failed
  verification, is denied by default on every path.
- **cache** — a batch visibility result resolved by `fail_open` (PDP unreachable) is not cached,
  so an outage's fallback doesn't outlive it by the cache TTL (`authorized_tools`, `Authorizer`).
- **async paths** — the PDP calls of FastMCP, the A2A server middleware, the A2A client guard on
  an `httpx.AsyncClient`, OpenAI's tool wrapper and the async RAG searches run on a worker
  thread instead of blocking the event loop.
- **gateway** — `build_gateway` refuses identity from unverified tokens (the default provider)
  unless `GatewayConfig(allow_unverified_tokens=True)`. The A2A forwarder drops hop-by-hop
  headers, `Cookie` (with `Authorization`, unless the route opts in) and client-supplied
  `X-Forwarded-*` / `Forwarded`, sets its own `X-Forwarded-For/-Proto/-Host`, stops relaying
  `Content-Encoding` for a body httpx already decoded, keeps repeated `Set-Cookie` headers
  apart, and closes its HTTP client on shutdown. The agent-card rewrite also covers the legacy
  `/.well-known/agent.json`.

- **sub-agents** — sub-agents get the config of the agent you authorize: every deepagents
  sub-agent (including the general-purpose one; a compiled sub-agent must come from our
  factories, else `ValueError`), ADK `sub_agents` and `AgentTool` agents, OpenAI handoff targets
  and `as_tool()` agents, Strands `as_tool()` agents (with the caller's identity). They used to
  run their tools unauthorized. An agent already authorized keeps its config. The deepagents
  `task` tool no longer crashes post-invocation.
- **human approval** — a paused call is kept per caller, for an hour at most, and dropped by
  `invalidate_decisions()`. A remembered approval is reused only while the asking rule still has
  a `ttl`. The approval attribute (`approval_granted`) is dropped from a caller-supplied
  environment. On OpenAI, `approve(item, always_approve=True)` approves only the paused call,
  not later calls of the tool with other arguments.
- `ResourceNameAttributes` is renamed `AttributeNames`, and `AuthZConfig.resource_name_attributes`
  is `AuthZConfig.attribute_names`.

### Added
- The `jwt` extra (PyJWT, for decoding or verifying bearer tokens) and the `deepagents` extra.
- `Route(mcp_resources_and_prompts=...)` for gateway MCP routes.
- A "Known limitations" section in [Security](docs/security.md#known-limitations).
- `AttributeNames.acting_agent` (default `agent_name`): the subject attribute naming the agent or
  sub-agent in the app making the call, so one policy can treat sub-agents differently.
  `AuditLogHook` records it as `agent_name`.
- `A2AAuthZMiddleware(public_paths=...)`: more paths served to anyone with `GET` / `HEAD`.
- `PDPConnection(headers=..., auth=..., verify=..., cert=..., http_client=...)`: authentication
  (an API-key header, Basic, or an `httpx.Auth` for bearer tokens), TLS (a CA bundle, mTLS
  client certificate), or your own `httpx.Client`. `PdpClient.close()` and context-manager use.
- `PermittedNames`: the result of a batch check, a `set` with `answered` (False when some name
  was resolved by `fail_open`).
- `GatewayConfig(allow_unverified_tokens=..., public_url=...)`; `public_url` replaces the
  client's `Host` header when rewriting A2A agent cards.
- `Enforcer(on_unrecognized_output="fail_open" | "fail_closed")` for `AFTER_TOOL`:
  `fail_closed` withholds tool output the strategy cannot inspect.
- `ApprovalDecision(asked=False)` marks a refusal where no human could be asked.
- `authorized_tool_invocation(args=..., call_id=...)` and `needs_human_approval(args=..., call_id=...)`
  for adapters: the call's own arguments and the framework's tool-call id.

## [0.1.0]

Initial release — externalized ABAC for AI agents and tools. Decisions come from a Policy
Decision Point over XACML 3.0 JSON or OpenID AuthZEN; tested against Axiomatics ADS.

### Added
- **Agent identity on client paths** — `AgentIdentityContextProvider(token_from=...)` verifies a
  token the agent received (e.g. behind your own API), and `claims_from=...` maps claims an agent
  platform already verified (LangGraph Platform auth, AgentCore Runtime).
- **Core** — `AuthZConfig`, `PDPConnection` with configurable `PdpEndpoints` (ADS paths by
  default), `PdpClient` in `axiomatics_ai.core.pdp` (XACML JSON + AuthZEN, single and batch), decision cache, pluggable
  context providers (`DefaultClientContextProvider`, `DefaultServerContextProvider`), and
  post-invocation strategies (`RecordFilterStrategy`, `ColumnMaskStrategy`).
- **direct** — `Authorizer(config)` for applications without a framework adapter:
  `is_permitted`, `is_permitted_batch`, `partial_evaluation`, `partial_evaluation_text`.
- **langchain_agents** — `create_agent` middleware (`create_authorized_agent` /
  `LangChainAgentAuthZMiddleware`) with input/output content guards, tool visibility,
  tool-call authorization, and post-invocation. Includes deepagents (`create_authorized_deep_agent`).
- **langchain_mcp** — filter MCP tools before they reach a LangGraph graph.
- **rag** — authorization-filtered retrieval; a partial-evaluation reverse query (Axiomatics
  ADS CAQ) returns the permitted condition, pushed into the vector store as a metadata filter.
- **fastmcp** — server-side FastMCP middleware (visibility / invocation / post-invocation).
- **tool_decorator** — per-tool post-invocation via `@protected_tool`.
- **google_adk**, **strands**, **openai_agents**, **claude_agent** — client-side
  authorization for the respective agent frameworks (`authorized_adk_agent`,
  `authorized_strands_agent`, `authorized_openai_agent`, `authorized_claude_options`).
- **a2a** — Agent2Agent authorization at the wire layer: `A2AAuthZMiddleware` (server) and
  `A2AClientAuthZ` (client egress). Framework- and a2a-sdk-version-agnostic.
- **gateway** (experimental, not production grade; will move to its own package) — a
  standalone reverse proxy that authorizes A2A and MCP traffic in front of the servers you
  run; one gateway fronts many A2A agents and MCP servers. Configure it with
  `GatewayConfig(routes=[...], authz=AuthZConfig(...))` and import it from `axiomatics_ai.gateway`.
- **Content guards** — LLM-enricher input/output guards (`BEFORE_MODEL` / `AFTER_MODEL`) with
  ready-made presets (`prompt_injection_enricher`, `pii_enricher`, `toxicity_enricher`).
- **Intent-based access control** — `intent_enricher` at `BEFORE_TOOL` classifies the user's own
  recent messages (never tool output or assistant text) into environment attributes
  (`intent_actions`, `intent_<action>_recipients`, `intent_available`), so a policy can deny a
  tool call the user did not ask for, such as one planted by indirect prompt injection. Client
  paths only; on server paths intent is unavailable and fails closed.
- **Enricher caching and placement** — `EnricherSpec.cache_ttl_seconds` / `cache_maxsize` reuse
  a classification for identical content; `allowed_points` (set by the presets) lets
  `AuthZConfig.validate()` warn about an enricher placed at the wrong enforcement point.
- **Attribute contract** — the tool is sent as `tool_name` at tool visibility, invocation, and
  post-invocation, and the target agent as `agent_name` at A2A (both configurable on
  `AuthZConfig.attribute_names`: `AttributeNames(tool=..., agent=...)`). Tool arguments are sent as
  resource attributes of the same name; an argument named like the identifier is dropped
  with a warning, so the model cannot change which tool the policy sees.
- **Observability** — observe-only hooks (`AuthzHook`, `LoggingHook`) firing on every path.
  `DecisionEvent` carries the request's `environment` and a `timestamp`.
- **Verified agent identity** — `AgentIdentityContextProvider` with `JwtVerification` verifies
  bearer tokens on the server paths (RFC 9068 §4: signature via OIDC discovery or `jwks_url`,
  `iss`, `aud`, `exp`, `at+jwt` type, asymmetric algorithms only) and sends named attributes:
  `user_id` (only when a user is present), `agent_id` (`client_id`), `actor_id` (current `act`),
  `scopes`, `acr`, `amr`, `auth_time`, and `transaction_id` (`txn`). Extra claims are mapped
  explicitly (`claims=`, `Claim`); names are configurable (`AgentIdentityAttributes`);
  `preview()` shows the result. A token that fails verification yields an empty subject.
- **Rich Authorization Requests** — `RarMapping` maps exactly one matching RFC 9396
  `authorization_details` entry to `approved_*` attributes; separate grants are never merged.
- **Client-side agent identity** — `DefaultClientContextProvider` sends `agent_id` from the
  runtime context.
- **Value guard** — every PDP request drops objects, nested lists and mixed-type lists (one
  warning per attribute), so no tool argument, claim or record field can make a request invalid.
- **Revocation** — cached decisions expire no later than the verified token's `exp`;
  `AuthZConfig.invalidate_decisions(reason=..., **subject)` drops them across every cache built
  from the config and emits `on_invalidation` (`InvalidationEvent`).
- **Audit** — `AuditLogHook`: one JSON record per decision and per invalidation (agent, user,
  actor, resource, action, decision, environment, timestamp, correlation id), to the
  `axiomatics_ai.audit` logger or a callable sink, with an optional hash chain.
- **Gateway** — no token passthrough: the caller's `Authorization` header is stripped on MCP and
  A2A routes unless `Route(forward_authorization=True)`; `Route(audience=...)` verifies per
  upstream resource server.
- **Obligations and advice** — the PDP client reads XACML `Obligations` / `AssociatedAdvice`
  and ADS's AuthZEN `context.xacml` block into `Obligation` / `Advice` on a `PdpDecision`
  (`Decision` enum). `AuthZConfig(obligations_and_advice=ObligationsAndAdviceConfig(...))`:
  obligation `handlers` run on every decision (cache hits included); an obligation with no
  handler turns a Permit into a Deny (`on_unknown="deny"`, per XACML; `"ignore"` to permit);
  advice is advisory. Applied on every path, including `check_permission` and batch decisions.
- **Human approval of tool calls** — a policy asks for approval with `require-approval` as
  Deny + advice (after approval the PDP is asked again with `approval_granted = true`) or
  Permit + obligation (the approval fulfils it; one PDP call). The answer comes from the
  framework's own pause and resume — LangGraph `interrupt()`, ADK tool confirmation, a Strands
  interrupt, the OpenAI Agents approval flow (`needs_approval`), MCP elicitation on FastMCP — or
  from an `approver` callback (sync or async; the only way on the Claude Agent SDK).
  `ApprovalConfig(obligation_or_advice_ids=...)` sets which ids ask for approval; several on one
  decision make one request (all `reasons`, the smallest `ttl`), answered once. Approvals are
  bound to the call's arguments; a `ttl` makes one reusable, and `invalidate_decisions()` drops
  it. Decision events log `approval_required`, `approved` /
  `rejected` and the final decision (`evaluated_again`). A call that can't be approved is denied
  with "was not approved". Examples per framework on `domain_human_approval.yaml`.
- `DefaultServerContextProvider` still reads tokens unverified, and now logs a one-time warning
  saying so.
- Packaging: PEP 517 build (setuptools), `py.typed`, `requires-python >=3.11`, and per-path
  optional-dependency extras.
- Documentation under `docs/`: a sequential guide, one page per integration, architecture,
  running ADS, and agent identity and AIMS alignment; example policies under `examples/policies/`.

[Unreleased]: https://github.com/axiomatics/axiomatics-ai-sdk-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/axiomatics/axiomatics-ai-sdk-python/releases/tag/v0.1.0
