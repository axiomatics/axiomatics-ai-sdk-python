# langchain_mcp examples — client-side MCP tool visibility

Filter MCP tools per user with `authorized_mcp_tools(tools, config, context=...)` before they
reach a LangGraph graph. Assumes **ADS is already running** with `domain_mcp_tool_visibility.yaml`
loaded (see [Running ADS](../../docs/running-ads.md)); reads `PDP_BASE_URL`, `LLM_MODEL`, and your
provider key from the environment / `.env`.

| Script | What it shows |
|---|---|
| `explicit_tool_filtering.py` | The recommended, call-site-explicit `authorized_mcp_tools(...)` on a loaded tool list. |
| `tool_filtering_langgraph.py` | Tool filtering wired into a custom LangGraph graph (single MCP server). |
| `multi_server_tool_filtering_langgraph.py` | The same across **multiple** MCP servers via `MultiServerMCPClient`. |

**Supporting MCP servers** (started by the examples above as needed): `mcp_server.py`
(single-server demo), `mcp_docs_server.py` + `mcp_account_server.py` (multi-server demo).

```bash
python examples/langchain_mcp/explicit_tool_filtering.py
```

See the [langchain_mcp module README](../../axiomatics_ai/langchain_mcp/README.md) for the API.
