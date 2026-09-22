# simple-ui — Axiomatics Authorized Agent Example

A Streamlit app that demonstrates `LangChainAgentAuthZMiddleware` with two demo users and a policy that drives tool visibility, tool invocation, and post-invocation.

## Demo Users

| Username | Password | Policy |
|---|---|---|
| `alice` | `alice123` | Permitted all tools, all tool calls, all records |
| `bob` | `bob123` | Restricted by policy (see below) |

Bob's restrictions (defined in `examples/policies/domain_simple_ui.yaml`):

- **Tool listing** — can only see `search_docs` and `get_account_balance`
- **Tool invocation** — can call `search_docs` with any arguments; can call `get_account_balance` only with `account_id=ACC-001`
- **Post-invocation** — cannot see records where `doc_type=classified`

## Prerequisites

- Java (to run ADS)
- ADS jar, license file, and `deployment.yaml` — see [Axiomatics ADS docs](https://docs.axiomatics.com/access-decision-service/)
- An LLM API key (OpenAI, Anthropic, or OpenRouter)

## Setup

### 1. Install dependencies

From the repo root:

```bash
uv sync --extra all
```

### 2. Prepare ADS files

All ADS runtime files live in `examples/policies/`. Place these there:

- `access-decision-service-<version>.jar`
- `axiomatics_ADS.license`

The `deployment.yaml` and domain YAMLs are already in that directory.

### 3. Configure the domain

Edit `examples/policies/deployment.yaml` and set:

```yaml
domain:
  path: ./domain_simple_ui.yaml
```

### 4. Start ADS

ADS must be started from `examples/policies/` so it can find the jar, license, and domain files:

```bash
cd examples/policies
java -jar access-decision-service-<version>.jar
```

ADS will be available at `http://localhost:8080` by default.

### 5. Configure the app

Create a `.env` file (in the repo root or in `examples/langchain_agents/simple-ui/` — `load_dotenv()` searches upward from the working directory):

```env
PDP_BASE_URL=http://localhost:8080
INVOCATION_MODE=async
AGENT_RUNTIME=vanilla

# LLM credentials (read by ChatLiteLLM — set whichever you have)
OPENAI_API_KEY=...
LLM_MODEL=openai/gpt-5.6-luna

# or Anthropic
# ANTHROPIC_API_KEY=...
# LLM_MODEL=anthropic/claude-3-5-haiku-latest

# or OpenRouter
# OPENROUTER_API_KEY=...
# LLM_MODEL=openrouter/openai/gpt-5.6-luna
```

`LLM_MODEL` must be a model string recognized by [LiteLLM](https://docs.litellm.ai/docs/providers).

### 6. Start the app

From the repo root:

```bash
uv run streamlit run examples/langchain_agents/simple-ui/app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Configuration Options

| Env var | Values | Default | Notes |
|---|---|---|---|
| `PDP_BASE_URL` | URL | `http://localhost:8080` | ADS endpoint |
| `INVOCATION_MODE` | `sync`, `async` | `async` | async loads MCP tools via LangChain MCP adapter |
| `AGENT_RUNTIME` | `vanilla`, `deep-agents` | `vanilla` | `deep-agents` uses `deepagents.create_deep_agent` |

## MCP Tools

When `INVOCATION_MODE=async`, the app loads two additional tools from a local MCP server (`mcp_server.py`). The LangChain MCP adapter starts that server automatically — you do not need to launch it separately.

In sync mode, MCP tools are not included (the MCP adapter requires async).

## Policy File

`examples/policies/domain_simple_ui.yaml` contains a XACML `PolicySet` with three child policies:

| Policy | Action name | What it controls |
|---|---|---|
| `ToolListing` | `tool_visibility` | Which tools each user can see |
| `ToolExecution` | `tool_invocation` | Whether a specific tool call (with its arguments) is permitted |
| `RecordRead` | `post_invocation` | Which records from a tool response the user can see |

Edit that file to change authorization behavior without touching any application code.
