# rag — authorization-filtered vector-store retrieval

The rag path attaches to a LangChain vector store on the client side. Wrap the store once
with `to_authorized_vectorstore(store, config)`. Every search then asks the PDP for the
condition under which the current subject may read, and pushes the resulting metadata filter
**down into the store** at query time. The only enforcement point on this path is retrieval
(`RetrievalConfig`). Identity comes from an explicit `authz_context(user)` block or from the
ambient agent runtime.

Three properties follow from the push-down design:

- **Push-down, not post-filtering.** The store applies the filter during the search, so the
  top-K results are drawn from the authorized set. No relevant document is pushed out by one
  the user cannot see, and there are no per-document PDP round-trips.
- **Dialect-agnostic.** The PDP's Boolean Expression Tree is translated to the store's native
  filter language through LangChain's own query translators. Chroma, Pinecone, Qdrant,
  OpenSearch, Milvus, pgvector, Elasticsearch, Weaviate, MongoDB Atlas, and
  [many more](https://reference.langchain.com/python/integrations/overview) work without
  store-specific code.
- **Zero change to retrieval code.** `vector_store.similarity_search(query, k=4)` inside an
  agent tool, or `vector_store.as_retriever()` in a chain, are unchanged.

All decisions are driven by policy in the PDP. No policy logic lives in application code.

Retrieval filtering relies on **partial evaluation**: a reverse query that returns the
condition under which access is permitted. That is an Axiomatics ADS feature (CAQ) with no
standard equivalent, so this path works against Axiomatics ADS only. See
[PDP compatibility](../configuration.md#pdp-compatibility).

## Attach

Wrap the store once. Anywhere the store is searched is now authorization-filtered.

```python
from langchain.tools import tool
from axiomatics_ai import AuthZConfig, PDPConnection, RetrievalConfig, to_authorized_vectorstore

config = AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    retrieval=RetrievalConfig(),          # presence turns retrieval filtering on
)

vector_store = to_authorized_vectorstore(vector_store, config)

@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search internal documentation.

    Args:
        query: What to search for.
    """
    docs = vector_store.similarity_search(query, k=4)   # unchanged
    return "\n".join(d.page_content for d in docs)

# In an agent, identity flows from the context you already pass:
#   agent = create_agent(model=model, tools=[search_documentation])
#   agent.invoke({"messages": [...]}, context=user_ctx)
```

`to_authorized_vectorstore` mutates and returns the same store instance, so existing
references to the store are governed too.

In a static (non-agent) pipeline there is no agent runtime to read identity from. Bind it per
request with `authz_context`. It is backed by a contextvar, so it is task-local and safe when
one shared store serves many concurrent users.

```python
from axiomatics_ai import authz_context, to_authorized_vectorstore

vector_store = to_authorized_vectorstore(vector_store, config)   # wrap once, shared
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

with authz_context(current_user):        # per request
    docs = retriever.invoke(query)       # filtered for current_user
```

## Enforcement points

| Point | Framework hook used | Outcome |
|---|---|---|
| Retrieval (`RetrievalConfig`) | The store's search methods (`similarity_search` and siblings, which also covers `as_retriever()`) are wrapped on the instance | The PDP's condition becomes a native metadata filter merged into the search. If the subject may read nothing, the search short-circuits and returns no documents. |

Retrieval is a constraint-based enforcement, not a Permit/Deny decision. It is distinct from
the decision-based points (`BEFORE_MODEL`, `TOOL_VISIBILITY`, `BEFORE_TOOL`, `AFTER_TOOL`,
`AFTER_MODEL`, `A2A_INVOCATION`), none of which this path enforces. To also govern the tool
that performs the retrieval, attach one of the agent paths, for example
[langchain_agents](langchain_agents.md).

How a search is filtered:

```
context ─(ContextProvider)→ subject/action attributes
        ─→ PDP POST /authorization/constraints       (reverse query / partial evaluation)
        ─→ Boolean Expression Tree                    (condition on document metadata)
        ─(caq_to_ir)→ store-neutral filter IR
        ─(ir_to_vectorstore_filter)→ native filter → merged into the search call
```

If the PDP is unreachable, behaviour follows `PDPConnection.fail_open` (default: fail closed,
no documents).

## Identity

Identity is resolved at search time from, in order:

1. an explicit `with authz_context(user): ...` block — for static (non-agent) pipelines; then
2. the ambient LangGraph agent runtime — the object you pass to `agent.invoke(..., context=ctx)`,
   so agent tools need no extra wiring.

Whichever is found is handed to your `ContextProvider`, so the subject and action attributes
sent to the PDP are exactly what you configure. With the default `DefaultClientContextProvider`,
`context` should carry `username` / `user_id` / `tenant_id`. Supply a custom provider to send
whatever your policy needs (for example `department`, `level`). See [Identity](../identity.md).

The path fails closed in three cases:

- **Absent subject.** Neither source yields an identity, or the identity carries no usable
  subject attributes. There is no subject to authorize, so the search returns no documents
  rather than leaking the corpus.
- **Partial subject.** Unprovided subject attributes come back from the PDP as unbound free
  variables in the residual. Those cannot be translated to a document filter, so the SDK
  returns no documents.
- **PDP unreachable.** No condition can be obtained, so the search returns no documents.

- **A condition the SDK cannot translate.** Anything outside the supported subset below. The
  search returns no documents rather than dropping the condition.

All four honour `PDPConnection.fail_open`. Set it to opt into an unfiltered pass-through
instead.

### What the translator supports

| In the condition | Rendered as |
|---|---|
| `and`, `or` | the store's AND / OR |
| `not` | the negated comparison (`==` → `!=`, `<` → `>=`, …), De Morgan over `and`/`or` |
| `==`, `!=`, `<`, `<=`, `>`, `>=` against a constant, bare or via `anyOf` / `anyOfAny` | a metadata comparison; several constants under `==` become an OR of equals |
| `size(x) == n` inside an `and` | dropped (a single-valued cardinality guard) |

Anything else is untranslatable and fails closed: other functions (`stringStartsWith`, regular
expressions, `*-is-in`, …), the `all*` bag functions, a `size` guard outside an `and`, a
comparison between two document attributes, and unknown node shapes.

The translation assumes **single-valued** metadata — the `all*` functions only differ from
`any*` on multi-valued attributes, which is why they are refused.

A comparison on a metadata key the document does not have never matches, `!=` included, as in
XACML. The in-memory store implements this exactly. Other stores apply their own `!=` semantics
for a missing key, and some translators do not support `!=` at all (Qdrant's raises, so the
search fails rather than returns documents).

On a **FastMCP server** there is no agent runtime either. The server middleware binds the
request's `Authorization` header via `authz_context` (source 1) around each tool call. A store
wrapped with a `DefaultServerContextProvider`-configured `AuthZConfig` then filters inside the
tool with no extra wiring. See [fastmcp](fastmcp.md#rag-inside-a-tool-ambient-identity), and
its warning on why an absent subject is the SDK's responsibility, not the policy's. The
general pattern is in [Ambient identity for RAG inside a tool](../identity.md#ambient-identity-for-rag-inside-a-tool).

## Configuration notes

```python
AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080", fail_open=False),  # PDP unreachable → fail closed
    retrieval=RetrievalConfig(          # presence enables retrieval filtering (None → off)
        action_name="retrieval",        # action name sent on the reverse query
        attribute_map=None,             # optional {PDP attribute name: metadata key} overrides
    ),
    # context_provider=MyProvider(),    # supply the subject attributes your policy needs
)
```

- `retrieval` must be set. `to_authorized_vectorstore` raises `ValueError` without it, rather
  than return a store that looks authorized but searches unfiltered.
- Every public method with `search` in its name is filtered — `similarity_search`, the
  `*_by_vector` variants, `max_marginal_relevance_search*`, `search`, and store-specific ones
  such as Chroma's `similarity_search_by_image`.
- By default the document-metadata keys are the PDP resource attribute names with a `doc.` or
  `resource.` prefix stripped, so `resource.department` lines up with a `department` metadata
  key. Use `retrieval.attribute_map` when a name differs.
- The reverse query goes to `PdpEndpoints.partial_evaluation` regardless of
  `PDPConnection.protocol`. The display-only rendering below goes to
  `PdpEndpoints.partial_evaluation_text`.

Full field reference: [Configuration](../configuration.md#authzconfig-field-reference).

## What the PDP receives

One reverse query (`POST /authorization/constraints`) per search. This is a partial-evaluation
request (the Axiomatics ADS CAQ `providedAttributes` body), not XACML JSON or AuthZEN:

- **providedAttributes** — the subject attributes from `get_subject()`, the action
  (`{"name": retrieval.action_name}`), and the environment from `get_environment()`, flattened.
- The document-metadata attributes are left **unbound**. The PDP returns the condition over
  them under which the decision is Permit.

The returned Boolean Expression Tree is translated to the store's native metadata filter and
merged into the search. The full request shape and an example policy are in
[Enforcement points, 6. Retrieval](../enforcement-points.md#6-retrieval).

## Showing the constraint (display-only)

The same reverse query is also available in a human-readable form. Two entry points return it:

- `Authorizer.partial_evaluation_text(action=..., context=...)` — resolves the subject through
  your context provider, like `Authorizer.partial_evaluation`.
- `PdpClient.get_simplified_constraints(provided_attributes=...)` — takes the same
  `provided_attributes` as the Boolean Expression Tree reverse query.

Both call Axiomatics ADS's `/authorization/simplified-constraints` and return ADS's own
rendering of the condition as a string, for example
`resource.department in ("finance") AND resource.level <= 3`. A UI can show that rendering next
to the native vector-store filter it translates into.

It is **display-only** and must never drive enforcement. It returns `None` (or an empty string)
on failure rather than raising, so a missing rendering means "nothing to display", never
"permit everything".

## Supported stores

Any vector store LangChain can build a query translator for. `to_authorized_vectorstore`
resolves the translator via LangChain and falls back to importing the store's own
`query_constructors` translator directly, so store / `langchain-community` version skew does
not silently disable filtering. `InMemoryVectorStore` is supported via a metadata predicate.

## Examples

Runnable scripts in [`examples/rag/`](../../examples/rag/). Each assumes Axiomatics ADS is
already running with the domain loaded and reads `PDP_BASE_URL`, `LLM_MODEL`, and your
provider key from the environment / `.env`. See [Running Axiomatics ADS](../running-ads.md).

| Script | What it shows | Domain file |
|---|---|---|
| `agentic_rag.py` | Agentic RAG with `langchain.agents.create_agent` — retrieval as an agent tool on an `InMemoryVectorStore`. alice (engineering) and bob (finance) each retrieve only their permitted docs. | `examples/policies/domain_rag_retrieval.yaml` |
| `deepagents_rag.py` | The same, driven by `deepagents.create_deep_agent`. | `examples/policies/domain_rag_retrieval.yaml` |
| `static_rag.py` | Static pipeline: `as_retriever()` + `authz_context(user)` on a real Qdrant store in in-memory mode. No LLM required. | `examples/policies/domain_rag_retrieval.yaml` |

```bash
python examples/rag/agentic_rag.py
```

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
