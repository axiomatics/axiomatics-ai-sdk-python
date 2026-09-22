# Gateway examples

> **Experimental.** The gateway is not production grade yet, and it will move to its own
> package. Import it from `axiomatics_ai.gateway`.

These show the **gateway** — a reverse proxy that authorizes A2A and MCP traffic in
front of the servers you run, so authorization applies with no changes to those servers or
their clients. One gateway fronts both an A2A agent and an MCP server here.

| File | What it is |
|---|---|
| `a2a_server.py` | A plain ADK billing agent (A2A) — no authz; the gateway enforces. |
| `mcp_server.py` | A plain FastMCP server (3 tools) — no authz; the gateway enforces. |
| `gateway.py` | The gateway: `/billing` → the A2A agent, `/tools` → the MCP server. |
| `a2a_client.py` | An ADK agent that calls `/billing` through the gateway (alice/bob). |
| `mcp_langchain.py` | A `create_agent` that loads its MCP tools through `/tools` (alice/bob). |

All backends are authorized against one ADS domain, [`examples/policies/domain_gateway.yaml`](../policies/domain_gateway.yaml):
**alice** may invoke `billing-agent` and sees all MCP tools; **bob** may not invoke billing
and sees only `search_docs`.

## Run order

Config in `.env`: `PDP_BASE_URL`, `LLM_MODEL`, `OPENROUTER_API_KEY`.

```bash
# 1. ADS with the combined domain (in examples/policies/, point deployment.yaml at domain_gateway.yaml)
cd examples/policies && java -jar access-decision-service-<version>.jar

# 2. the backends (they have no authorization themselves)
python examples/gateway/a2a_server.py       # :9211
python examples/gateway/mcp_server.py        # :9212

# 3. the gateway in front of both
python examples/gateway/gateway.py           # :9210

# 4. the clients — they talk ONLY to the gateway (:9210)
python examples/gateway/a2a_client.py        # alice reaches billing; bob is blocked
python examples/gateway/mcp_langchain.py     # alice loads 3 tools; bob loads only search_docs
```

Expected output:

```
[alice] → The company billing address is 1200 Market Street, Suite 400, San Francisco, CA 94103.
[bob]   → I was unable to retrieve the billing information.
[alice] tools visible via gateway: ['get_account_balance', 'search_docs', 'send_email']
[bob]   tools visible via gateway: ['search_docs']
```

## Notes

- **Clients point only at the gateway.** The A2A client is given `http://…:9210/billing`; the
  gateway rewrites the agent card's `url` to itself so every `message/send` routes back
  through it (it never learns the agent's real address). Required — a client pointed directly
  at a backend bypasses the gateway.
- **A2A client uses `AgentTool`, not `sub_agents`.** With `sub_agents` (transfer), a denied
  call doesn't return control to the orchestrator, so it can't relay the denial. `AgentTool`
  (call) gets the result/denial back and relays it. See the note in `a2a_client.py`.
- **MCP identity** rides in the JWT header on the transport connection (not `invoke(context=)`)
  — the gateway is the enforcement point, so the create_agent has no authz middleware.
- **MCP tools are identity-scoped at load time**, so this example builds the agent per user.
  A future example will show a long-lived shared agent with per-call invocation enforcement.
- **Identity here is a demo _unsigned_ JWT**, so `gateway.py` sets
  `GatewayConfig(allow_unverified_tokens=True)` — without it, `build_gateway` refuses to start.
  In production verify the token with `AgentIdentityContextProvider` and a `JwtVerification`,
  set as `GatewayConfig(authz=AuthZConfig(context_provider=...))` (see
  [docs/agent-identity.md](../../docs/agent-identity.md)), or keep the opt-out only behind an
  ingress that authenticates first. Set `GATEWAY_PUBLIC_URL` to the gateway's public address
  so A2A agent cards are rewritten to it rather than to the request's `Host` header.
