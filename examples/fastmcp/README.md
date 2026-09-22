# fastmcp examples — server-side authorization on a FastMCP server

Each script assumes **ADS is already running** with the listed domain (see the
[Running ADS](../../docs/running-ads.md) for starting ADS) and reads `PDP_BASE_URL` (default
`http://localhost:8080`) plus `LLM_MODEL` + your provider key from the environment / `.env`.

| Script | Domain | What it shows |
|---|---|---|
| `authz_server.py` | `domain_mcp_tool_visibility.yaml` | Server-side enforcement via `FastMCPAuthZMiddleware` — alice sees all tools, bob only `search_docs`. |
| `authz_client.py` | `domain_mcp_tool_visibility.yaml` | A LangChain agent talking to the protected server above. |
| `to_authorized_server.py` | `domain_all_authz_enabled.yaml` | The same enforcement attached with the `to_authorized_fastmcp_server(server, config)` wrapper. |
| `rag_server.py` | `domain_rag_retrieval.yaml` | Authorization-filtered RAG inside a FastMCP tool. |
| `rag_client.py` | `domain_rag_retrieval.yaml` | A LangChain agent consuming the RAG server above. |
| `payments_server.py` | `domain_agent_identity.yaml` | A payments server that verifies the caller's OAuth access token (`AgentIdentityContextProvider`) and allows a payment only within the user's approved RAR grant (RFC 9396). |
| `approval_server.py` | `domain_human_approval.yaml` | A server whose large payments and cancellations need a human's approval, asked with an MCP **elicitation** to the client's user (port 8005). |
| `approval_client.py` | `domain_human_approval.yaml` | Calls that server's tools directly (no model) and answers each elicitation at the console, or with `--approve` / `--reject`. |
| `payments_agent.py` | `domain_agent_identity.yaml` | A LangChain agent (real model) paying invoices through that server. Its tokens come from a **local test issuer** (`examples/local_test_issuer.py`) standing in for your identity provider; verification and decisions are real. |

Run a server in one terminal and its client in another, e.g.:

```bash
python examples/fastmcp/authz_server.py     # terminal 1
python examples/fastmcp/authz_client.py      # terminal 2
```

See the [fastmcp module README](../../axiomatics_ai/fastmcp/README.md) for the API.
