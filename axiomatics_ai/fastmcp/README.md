# fastmcp

Server-side middleware for a FastMCP server (HTTP/SSE transports) that enforces `TOOL_VISIBILITY`, `BEFORE_TOOL`, and `AFTER_TOOL` on every `tools/list` and `tools/call`, and binds the caller's identity for authorization-filtered retrieval inside tools. A tool call the policy wants a human to approve is asked of the client's user with an MCP elicitation.

Documentation: [docs/integrations/fastmcp.md](../../docs/integrations/fastmcp.md) ·
Runnable examples: [examples/fastmcp/](../../examples/fastmcp/)
