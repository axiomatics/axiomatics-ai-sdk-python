# Architecture

Three deployment shapes, one mechanism. In each, the SDK is the Policy Enforcement Point: it
builds an authorization request from the caller's identity and the thing being attempted, asks
the PDP, and applies the answer. The PDP holds the policy; the SDK never does.

## Client side: agents, `langchain_mcp`, `rag`, `direct`

Identity is a dict or object the application passes at the call site. A
`DefaultClientContextProvider` (or your subclass) turns it into request attributes before each
PDP call.

```mermaid
sequenceDiagram
    participant C as Application / agent
    participant SDK as SDK (PEP)
    participant PDP as PDP
    participant MCP as Tool / MCP server
    Note over SDK: context provider maps the<br/>context dict → request attributes
    C->>SDK: list tools · context: username
    SDK->>PDP: batch request (one sub-request per tool)
    PDP-->>SDK: permitted tools ①
    SDK-->>C: filtered tool list
    C->>SDK: call tool "search_docs" · context: username
    SDK->>PDP: request (tool + arguments)
    PDP-->>SDK: Permit / Deny ②
    SDK->>MCP: tool call
    MCP-->>SDK: result
    SDK->>PDP: batch request (one sub-request per record)
    PDP-->>SDK: Permit / Deny per record ③
    SDK-->>C: filtered result
```

## Server side: `fastmcp`, `tool_decorator`, `a2a` server

Identity travels in HTTP headers with every request. A `DefaultServerContextProvider` reads
them; the application passes nothing. Headers can be `Authorization: Bearer <JWT>` or any
custom header.

```mermaid
sequenceDiagram
    participant C as Client
    participant S as MCP / A2A server
    participant PDP as PDP
    Note over S: SDK middleware reads headers →<br/>request attributes
    C->>S: list tools · Authorization: Bearer JWT
    S->>PDP: batch request
    PDP-->>S: permitted tools ①
    S-->>C: filtered tool list
    C->>S: call tool "get_records" · Authorization: Bearer JWT
    S->>PDP: request
    PDP-->>S: Permit / Deny ②
    S->>S: execute tool
    S->>PDP: batch request (records)
    PDP-->>S: Permit / Deny per record ③
    S-->>C: filtered result
```

## Gateway: reverse proxy in front of MCP and A2A backends

The out-of-process shape. Clients point only at the gateway; it authorizes every request on the
caller's JWT and forwards permitted traffic to backends that carry no authorization of their
own. For an A2A route it also rewrites the agent card's URL to itself so replies keep routing
through it.

```mermaid
sequenceDiagram
    participant C as Client
    participant G as Gateway (PEP)
    participant PDP as PDP
    participant B as Backend (MCP / A2A)
    Note over G: enforces on the caller's JWT.<br/>The backend has no authorization of its own
    C->>G: list tools · Authorization: Bearer JWT
    G->>B: list tools (proxy)
    B-->>G: all tools
    G->>PDP: batch request
    PDP-->>G: permitted tools ①
    G-->>C: filtered tool list
    C->>G: call tool "get_records" · Authorization: Bearer JWT
    G->>PDP: request
    PDP-->>G: Permit / Deny ②
    G->>B: tool call (if permitted)
    B-->>G: result
    G->>PDP: batch request (records)
    PDP-->>G: Permit / Deny per record ③
    G-->>C: filtered result
```

## Retrieval: a reverse query instead of a decision

RAG filtering asks a different question. Instead of "may alice read document X?" for every
candidate, the SDK asks once "under what condition may alice read?", receives a constraint on
the document attributes, and pushes it into the vector store as a native metadata filter. Only
permitted documents come back; retrieval code is unchanged. This needs a PDP that supports
partial evaluation; see [PDP compatibility](configuration.md#pdp-compatibility) and
[rag](integrations/rag.md).

## Where the pieces live

| Package | Role |
|---|---|
| `axiomatics_ai.core` | `AuthZConfig`, `PdpClient`, `Authorizer`, context providers, enforcement helpers, cache, hooks, enrichers, post-invocation strategies. Depends only on httpx, pydantic, cachetools. |
| `axiomatics_ai.<path>` | one adapter per framework, each a thin mapping from that framework's hooks onto the core helpers |
| `axiomatics_ai.gateway` | the reverse proxy: Starlette app composing the `fastmcp` and `a2a` adapters |

[Docs index](README.md)
