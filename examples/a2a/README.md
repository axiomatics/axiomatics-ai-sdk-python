# A2A authorization examples

These examples show Axiomatics authorization on the **A2A (Agent2Agent)** protocol, across
three agent frameworks and both enforcement directions:

- **Server** (`*_server.py`) — an agent you expose over A2A, protected by
  `A2AAuthZMiddleware`. Authoritative: it blocks unauthorized inbound `message/send`
  before the agent runs. Identity comes from the caller's `Authorization: Bearer <JWT>`.
- **Client** (`*_client.py`) — an agent that calls a remote agent, with `A2AClientAuthZ`
  on its httpx. Complementary egress control: it blocks a disallowed outbound call before
  it leaves the process.

The integration is **one line** and identical across frameworks — only the way you obtain
the ASGI app (server) or inject the httpx client (client) differs:

| Framework | Server | Client |
|---|---|---|
| Google ADK | `to_a2a(agent)` → `app.add_middleware(A2AAuthZMiddleware, ...)` | `RemoteA2aAgent(httpx_client=httpx.AsyncClient(auth=A2AClientAuthZ(...)))` |
| AWS Strands | `A2AServer(agent).to_starlette_app()` → `app.add_middleware(...)` | `A2AClientToolProvider(httpx_client_args={"auth": A2AClientAuthZ(...)})` |
| LangChain | `create_agent` in an a2a-sdk `AgentExecutor` → `add_middleware(...)` | a `@tool` calling the a2a client, httpx `auth=A2AClientAuthZ(...)` |

## Prerequisites

1. **Install** the example dependencies: `uv sync --extra all` (brings in a2a-sdk, ADK,
   Strands, and the LiteLLM providers).
2. **Start ADS** with the A2A domain. In `examples/policies/`, point `deployment.yaml` at
   `domain_a2a.yaml`, then run the jar:
   ```bash
   cd examples/policies && java -jar access-decision-service-<version>.jar
   ```
   Policy (`domain_a2a.yaml`): **alice** may send to `billing-agent`; **bob** may not
   (unless the message carries `skill=refund`).
3. **.env** at the repo root:
   ```
   PDP_BASE_URL=http://localhost:8080
   LLM_MODEL=openrouter/<your-model>
   OPENROUTER_API_KEY=<key>
   ```

## The zero-dependency example first

`self_wired_a2a.py` needs no LLM and no open ports — it runs a real a2a-sdk server + client
in-process against ADS and shows both enforcement directions:

```bash
python examples/a2a/self_wired_a2a.py
```

## The framework examples (two terminals)

Each framework is a **server** program and a **client** program. Start the server, then
run the client in another terminal. Ports: ADK `9101`, Strands `9102`, LangChain `9103`.

```bash
# terminal 1 — the protected agent server (ADK shown; same shape for strands/langchain)
python examples/a2a/adk_server.py                 # add --no-authz to disable the middleware

# terminal 2 — the calling agent
python examples/a2a/adk_client.py                 # runs as alice then bob
python examples/a2a/adk_client.py --user bob      # just bob
python examples/a2a/adk_client.py --no-authz      # disable client-side A2AClientAuthZ
```

The billing agent only knows one fact — Acme's billing address, **"1200 Market Street"**.
If that address appears in the client output, the remote agent was actually reached; if it
is missing, the call was blocked.

## Toggling both switches — the truth table

`--no-authz` on the **server** disables `A2AAuthZMiddleware`; `--no-authz` on the
**client** disables `A2AClientAuthZ`. Run bob under each combination:

| server authz | client authz | alice | bob | who blocked bob |
|:---:|:---:|:---:|:---:|---|
| on | on | reaches | blocked | client (before send) |
| on | off | reaches | blocked | server (`-32001`) |
| off | on | reaches | blocked | client (before send) |
| off | off | reaches | reaches | nobody — no enforcement |

When both are on, the **client** blocks bob first (egress), so the server never sees it —
realistic defense-in-depth. To watch the **server** enforce on its own, run the client
with `--no-authz`.

## Notes

- Identity is a demo Bearer JWT (`{"username": "alice"|"bob"}`); the default server
  provider reads its claims without verifying the signature. In production the token
  is verified by your auth layer upstream of the agent — or, if the agent is the edge,
  by overriding `get_subject` in a `DefaultServerContextProvider` subclass.
- `agent_name="billing-agent"` is passed explicitly as the ADS resource label — it is the
  policy-facing identity, decoupled from each framework's internal agent/card name (ADK,
  for instance, forbids hyphens in agent names).
- Skill-level gating (`skill=refund`) is easy to demonstrate from `self_wired_a2a.py`,
  which sets the message metadata directly; the framework clients here gate at the agent
  level (which is what a policy typically keys on).
