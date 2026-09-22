# Getting started

By the end of this page you have the SDK installed, a PDP reachable, and one agent whose tool
list and tool calls are decided by policy rather than by code.

## 1. Install

The package is not yet on PyPI. Install from the repository or a local clone, adding the
extra for the integration you use:

```bash
# from a local clone
uv add /path/to/axiomatics-ai-sdk-python --extra langchain-agents
pip install "/path/to/axiomatics-ai-sdk-python[langchain-agents]"

# from the repository
uv add git+https://github.com/axiomatics/axiomatics-ai-sdk-python.git --extra langchain-agents
```

Combine the extras for the integrations you use (`--extra a --extra b`, or `[a,b]`):

| Extra | Install when using |
|---|---|
| `langchain-agents` | LangChain `create_agent` middleware |
| `deepagents` | `create_authorized_deep_agent` |
| `langchain-mcp` | Filtering MCP tools loaded with the LangChain adapters |
| `rag` | Authorization-filtered retrieval |
| `fastmcp` | FastMCP server middleware |
| `tool-decorator` | `@protected_tool` |
| `a2a` | Agent2Agent middleware and client guard |
| `gateway` | The reverse-proxy gateway (preview) |
| `strands` | Strands agents |
| `claude-agent-sdk` | Claude Agent SDK |
| `llm-enrichment` | LLM content guards (pulls in `litellm`) |
| `jwt` | Decoding or verifying bearer tokens yourself (`DefaultServerContextProvider`, `AgentIdentityContextProvider`, `verify_access_token`); the server-side extras above include it |
| `all` | Every extra above |

Google ADK, OpenAI Agents, and the direct path (`Authorizer`) need no extra: the core package
(httpx, pydantic, cachetools) is enough, and you bring the framework. Importing an integration
without its extra raises an `ImportError` that names the extra.

To work on the SDK itself: clone, then `uv sync --extra all`. The `dev` and `examples` dependency
groups (test tools, and what the example scripts use) are installed by `uv sync` by default.

## 2. Point at a PDP

The SDK talks to any Policy Decision Point that speaks XACML 3.0 JSON or OpenID AuthZEN. It is
tested against [Axiomatics ADS](running-ads.md); the examples and integration tests ship with
ADS domain files. If you have another PDP, set its base URL and, if its paths differ from the
defaults, its [endpoints](configuration.md#endpoints).

Check connectivity before wiring anything:

```python
from axiomatics_ai import AuthZConfig, PDPConnection

config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"))

health = config.check_connection()          # GET <base_url><endpoints.health>; never raises
if not health:
    raise SystemExit(f"PDP not reachable: {health.error}")
print(f"PDP OK ({health.latency_ms:.0f} ms)  ·  {config.summary()}")
```

`config.summary()` is a one-line description of what the config enforces, useful in start-up
logs.

## 3. Your first authorized agent

The LangChain path is the shortest complete example. `create_authorized_agent` wraps
`create_agent` and injects the authorization middleware:

```python
from langchain.tools import tool
from langchain_litellm import ChatLiteLLM
from axiomatics_ai import AuthZConfig, PDPConnection, create_authorized_agent, standard_tool_enforcement

@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return "..."

@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return "sent"

agent = create_authorized_agent(
    model=ChatLiteLLM(model="openai/gpt-5.6-luna"),   # any LiteLLM model string
    tools=[search_docs, send_email],
    config=AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(),   # tool visibility + invocation + post-invocation
    ),
)

# The context dict is the user's identity; the PDP decides what alice may see and call.
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Find docs on deployment"}]},
    context={"username": "alice", "tenant_id": "acme"},
)
```

Nothing is enforced unless you say so: `enforcement` lists the points to enforce, and
`standard_tool_enforcement()` is the usual three tool points. The `context` dict is forwarded
to the PDP as subject attributes by the default [context provider](identity.md); replace it
with your own field names by subclassing the provider.

Not on LangChain? Every other framework has a one-call equivalent; see the
[integrations](README.md#integrations). No framework at all? Ask the PDP directly:

```python
from axiomatics_ai import Authorizer

authz = Authorizer(config)
if not authz.is_permitted(action="read_document", resource="doc-42", context={"username": "alice"}):
    raise PermissionError
```

See [direct](integrations/direct.md).

## 4. Write the policy

The SDK sends the same request shapes regardless of framework, so one policy works across all
paths. [Enforcement points](enforcement-points.md) lists exactly what the PDP receives at each
point with example requests and rules. Ready-made policies for every example live in
[`examples/policies/`](../examples/policies/) as ADS domain files.

## 5. Run the examples

Each `examples/<path>/` folder has a README with a table of scripts and the domain file each
one needs. All scripts read these variables from the environment or a `.env` file:

| Variable | Default | Meaning |
|---|---|---|
| `PDP_BASE_URL` | `http://localhost:8080` | Base URL of the PDP |
| `LLM_MODEL` | `openai/gpt-5.6-luna` | Any LiteLLM model string, e.g. `openai/gpt-5.6-luna`, `gemini/gemini-3.8-flash`, `openrouter/meta/muse-spark-1.3` |
| `LLM_API_BASE` | unset | Optional custom LLM API base URL |
| `AXIOMATICS_ENRICHER_MODEL` | unset | Default LLM for [content guards](content-guards.md) |

Plus the API key for whichever LLM provider `LLM_MODEL` names (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, ...). The SDK never requires a specific provider.

To run against ADS, follow [Running Axiomatics ADS](running-ads.md): each example names the
domain file it needs, and ADS serves one domain at a time.

## 6. Tests

```bash
uv run pytest tests/
```

Unit tests need no PDP. Integration tests (marked `ads_integration`) start ADS from the jar in
`examples/policies/` and are skipped automatically when the jar is absent.

---

[Docs index](README.md) · Next: [Configuration](configuration.md) →
