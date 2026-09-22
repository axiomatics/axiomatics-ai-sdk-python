# Performance

How many PDP round trips a turn costs, and what bounds them.

## Calls per agent turn

With all three tool points on, one model turn that calls one tool costs at most three PDP
requests, each a single HTTP round trip:

| Point | Requests | Notes |
|---|---|---|
| `TOOL_VISIBILITY` | 1 per model call | one batch request covering the whole tool list, however long |
| `BEFORE_TOOL` | 1 per tool call | with an `intent_enricher`, plus one LLM call per user turn (cached by message history) |
| `AFTER_TOOL` with `RecordFilterStrategy` | 1 per tool call | one batch request covering all records in the result |
| `AFTER_TOOL` with `ColumnMaskStrategy` | 1 per tool call | |
| `BEFORE_MODEL` / `AFTER_MODEL` | 1 per prompt / answer, plus one LLM call per enricher | the LLM call dominates |
| `A2A_INVOCATION` | 1 per message | |
| retrieval | 1 per search | a partial evaluation, then the store applies the filter itself |

Batching is what keeps this flat: ten tools or a hundred records are still one request. The
SDK never makes a per-tool or per-record round trip.

## The decision cache

Every decision is cached in memory for `cache_ttl_seconds` (default 300, `0` disables) with up
to `cache_maxsize` entries (default 1024), keyed on the full subject, action, resource, and
environment. In a conversation, the second and later model calls with the same tool list hit the
cache for visibility, and repeated calls of the same tool with the same arguments hit it for
invocation. A cache hit emits a `DecisionEvent` with `cached=True` and no transport event, so
hooks can measure the hit rate.

With `AgentIdentityContextProvider` an entry also expires at the token's `exp`, so short-lived
tokens mean shorter-lived entries; `config.invalidate_decisions()` drops entries early.

Two things defeat the cache: volatile environment attributes (timestamps, request ids) and
tool arguments that change every call. Keep the environment stable and accept that
argument-level decisions are usually fresh.

## Latency

A PDP call is one HTTP POST; against a local ADS it is typically a few milliseconds, dominated
by network. `PDPConnection(timeout=10.0)` bounds the worst case, and a timeout follows the
`fail_open` rule. The transport hooks (`on_pdp_response.latency_ms`) give you the distribution.

Content guards add an LLM call per enricher on the prompt and on the answer. Use a small, fast
model for classification; the enricher's job is a label, not an answer.

## Token verification

`AgentIdentityContextProvider` verifies each distinct token once. The verified claims are cached
per token (up to 1024 tokens, for at most 5 minutes and never past `exp`), so a session's later
tool calls cost a hash lookup; a rejected token is remembered for 30 seconds, so a bad token
doesn't cost a verification per call either. The issuer's discovery document is fetched once
per issuer (a failed discovery is retried after 30 seconds, not on every request), and
the signing keys are cached by PyJWT's JWKS client, so a token signed with a known key needs no network
call. The first request after start-up pays for discovery and the key fetch, bounded by
`JwtVerification(http_timeout=5.0)`.

## Sync calls in async frameworks

The PDP client uses synchronous `httpx` calls, over one pooled connection per adapter (no new
TCP or TLS handshake per decision). The async paths — LangChain's async hooks, FastMCP, the A2A
server middleware and the A2A client guard on an `httpx.AsyncClient`, OpenAI, Claude, the async
RAG searches — run each PDP call on a worker thread (`asyncio.to_thread`), so a slow PDP holds
that one request, not the event loop and every other session on it. Keep the timeout tight
anyway: the request still waits for the PDP.

**Google ADK and Strands** call the PDP (and any LLM enricher, and an approver) from their
synchronous callbacks, on the event loop: concurrent sessions on one loop wait for each other's
PDP calls, and an async approver that waits on that same loop never gets to run. Serve
concurrent users from separate workers, keep the PDP timeout short, and use a synchronous
approver on these paths.

## Payload size

Visibility batches carry one sub-request per tool, with only the tool name as resource. Record
filtering carries every record's fields as attributes; a result with thousands of wide records
becomes a large request. If that is your shape, filter at the source or move the check to
retrieval, where the PDP returns a condition once and the store does the work.

---

← Previous: [Troubleshooting](troubleshooting.md) · [Docs index](README.md)
