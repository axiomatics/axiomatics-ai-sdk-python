# gateway

> **Experimental.** Not production grade yet; the gateway will move to its own package.

An out-of-process reverse proxy in front of the MCP servers and A2A agents you run, enforcing `TOOL_VISIBILITY`, `BEFORE_TOOL` and `AFTER_TOOL` on MCP routes and `A2A_INVOCATION` on A2A routes with no changes to those servers or their clients. The caller's token is verified with `AgentIdentityContextProvider` if configured, and is not forwarded to the targets unless a route sets `forward_authorization=True`.

Documentation: [docs/integrations/gateway.md](../../docs/integrations/gateway.md) ·
Runnable examples: [examples/gateway/](../../examples/gateway/)
