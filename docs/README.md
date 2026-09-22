# Documentation

Externalized ABAC for AI applications: the SDK is the Policy Enforcement Point (PEP), your
Policy Decision Point (PDP) holds the rules. Everything below assumes you have read the
[project README](../README.md) for the big picture.

## Guide

Read these in order; each page links to the next.

1. [Getting started](getting-started.md) — install, point at a PDP, first authorized agent
2. [Configuration](configuration.md) — `AuthZConfig`, `PDPConnection`, protocols, PDP compatibility
3. [Identity](identity.md) — context providers and the trust boundary
4. [Agent identity](agent-identity.md) — verified tokens, agent and user attributes, RAR, revocation, audit, and AIMS alignment
5. [Enforcement points](enforcement-points.md) — what the PDP receives at each point, and how to write policy for it
6. [Content guards](content-guards.md) — LLM-classified input and output guards
7. [Intent-based access control](intent-based-access.md) — check each tool call against what the user asked for
8. [Obligations and human approval](human-approval.md) — act on PDP obligations and advice; pause a tool call for a human's approval
9. [Post-invocation](post-invocation.md) — filtering and masking tool results
10. [Hooks](hooks.md) — observability: logging, audit trail, metrics
11. [Security](security.md) — fail-open, unverified tokens, caching, what the SDK does not do
12. [Troubleshooting](troubleshooting.md) — common errors and what they mean
13. [Performance](performance.md) — how many PDP calls a turn makes, and how caching bounds them

## Integrations

One page per way of attaching the SDK. Pick by the framework you use.

| Page | Attach point | Side |
|---|---|---|
| [langchain_agents](integrations/langchain_agents.md) | LangChain `create_agent` / deepagents middleware | client |
| [google_adk](integrations/google_adk.md) | Google ADK callbacks | client |
| [strands](integrations/strands.md) | Strands intervention handler | client |
| [openai_agents](integrations/openai_agents.md) | OpenAI Agents SDK tool seam + guardrails | client |
| [claude_agent](integrations/claude_agent.md) | Claude Agent SDK permission callback + hook | client |
| [langchain_mcp](integrations/langchain_mcp.md) | Filter MCP tools loaded with the LangChain adapters | client |
| [rag](integrations/rag.md) | Authorization-filtered vector-store retrieval | client |
| [fastmcp](integrations/fastmcp.md) | FastMCP server middleware | server |
| [tool_decorator](integrations/tool_decorator.md) | Per-tool `@protected_tool` on a FastMCP server | server |
| [a2a](integrations/a2a.md) | Agent2Agent wire-level middleware and client guard | both |
| [gateway](integrations/gateway.md) | Reverse proxy in front of MCP and A2A servers | out of process |
| [direct](integrations/direct.md) | `Authorizer` — ask the PDP from your own code | any |

## Reference

- [Architecture](architecture.md) — request-flow diagrams for the client, server, and gateway shapes
- [Running Axiomatics ADS](running-ads.md) — the PDP the SDK is tested with; domain files for the examples
- [Assets](../assets/README.md) — every diagram as its own file, plus exported pictures
- [Changelog](../CHANGELOG.md)
