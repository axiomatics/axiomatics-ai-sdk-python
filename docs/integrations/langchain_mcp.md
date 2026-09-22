# langchain_mcp — filter MCP tools loaded with the LangChain adapters

This path attaches on the client side, between the LangChain MCP adapters and whatever consumes
the tool list: a LangGraph graph, `create_agent`, or a custom agent. It enforces one point,
`TOOL_VISIBILITY`: the loaded tool list is filtered through the PDP in one batched request, and
only the tools the subject may see are returned. Identity is the `context` dict or object you pass
at the call site. No Axiomatics-specific nodes or edges are required in the graph.

## Attach

Load tools with the MCP adapters as usual, then call `authorized_mcp_tools` on the result:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from axiomatics_ai import AuthZConfig, PDPConnection, standard_tool_enforcement, authorized_mcp_tools

config = AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False),
)
client = MultiServerMCPClient({"docs-server": {"url": "http://localhost:9000/mcp", "transport": "streamable_http"}})

# Load, then filter to the tools this user may see:
tools = authorized_mcp_tools(await client.get_tools(), config, context={"username": "alice"})
```

The single-session loader reads the same way:

```python
from langchain_mcp_adapters.tools import load_mcp_tools

tools = authorized_mcp_tools(await load_mcp_tools(session), config, context={"username": "alice"})
```

Signature:

```python
def authorized_mcp_tools(tools: list, config: AuthZConfig | None = None, *, context: Any) -> list
```

Loading stays with the MCP adapters; `authorized_mcp_tools` only filters the list it is given.
With no `TOOL_VISIBILITY` enforcer in the config, or an empty tool list, it returns the list
unchanged. A `None` context (no caller identity) returns no tools, and logs a warning — or every
tool under `PDPConnection(fail_open=True)`. Filtering is per subject, so build one agent or
graph per user from the filtered tools.

There is no "hold it yourself" alternative on this path. The function is a pure filter, so any
consumer of a tool list can take its output. For invocation and post-invocation enforcement on the
same tools, hand the filtered list to `create_authorized_agent` from the
[langchain_agents](langchain_agents.md) path.

## Enforcement points

| Point | Framework hook used | On Deny |
|---|---|---|
| `TOOL_VISIBILITY` | none; a call on the list returned by `client.get_tools()` / `load_mcp_tools(...)` | the tool is removed from the returned list |

`BEFORE_TOOL` and `AFTER_TOOL` are not available on this path. `authorized_mcp_tools` runs once,
before the agent is built, and never sees a tool call or its result. Any `BEFORE_TOOL` or
`AFTER_TOOL` enforcer in the config is ignored here. Use the [langchain_agents](langchain_agents.md)
path if you need those points.

## Identity

`context` is whatever dict or object you pass to `authorized_mcp_tools(..., context=...)`. It is
handed unchanged to the config's context provider, `DefaultClientContextProvider` unless you set
another. The default reads `username`, `user_id`, and `tenant_id` for the subject and `session_id`
for the environment.

Implement a custom provider when your `context` uses other field names, or when you need to
control the action name sent to the PDP. Set it on the `AuthZConfig` you pass in:

```python
from axiomatics_ai import AuthZConfig, DefaultClientContextProvider, standard_tool_enforcement, authorized_mcp_tools

class MyProvider(DefaultClientContextProvider):
    def get_subject(self, context):
        # context is the dict you pass to authorized_mcp_tools(..., context=...)
        return {"service_account": context.get("service_account", "")}

    def get_action(self, action_name, context):
        # action_name is the TOOL_VISIBILITY Enforcer's action_name (default: "tool_visibility")
        return {"name": action_name}

config = AuthZConfig(context_provider=MyProvider(), enforcement=standard_tool_enforcement())
tools = authorized_mcp_tools(await client.get_tools(), config, context={"service_account": "svc-1"})
```

See [Identity](../identity.md) for both base classes and the full method set.

## Configuration notes

Only the `TOOL_VISIBILITY` enforcer is read. The minimal explicit form:

```python
from axiomatics_ai import AuthZConfig, PDPConnection, EnforcementPoint, Enforcer

AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080", fail_open=False),  # deny if the PDP is unreachable
    enforcement={
        EnforcementPoint.TOOL_VISIBILITY: Enforcer(action_name="tool_visibility"),
    },
)
```

`standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False)` produces the
same thing. Passing `config=None` uses a default `AuthZConfig()`. The PDP client is stateless and
is built per call. See [Configuration](../configuration.md) for every field.

## What the PDP receives

One batch request per `authorized_mcp_tools` call, one sub-request per tool. Over XACML JSON this
is one MultiRequest under the Multiple Decision Profile; over AuthZEN it is one `evaluations`
request.

- **Subject** — `get_subject(context)`
- **Action** — `get_action(action_name, context)`, `action_name` being the visibility Enforcer's (default `tool_visibility`)
- **Resource** — one entry per tool, attribute id = `AuthZConfig.attribute_names.tool` (default `tool_name`), value = the bare tool name
- **Environment** — `get_environment(context)`

Tool names are the bare MCP tool name, the same whether they come from one server or from several
through `MultiServerMCPClient`. Tools that receive Deny, or that receive no decision when
`fail_open=False`, are removed from the returned list. Request shapes are in
[Enforcement points](../enforcement-points.md#1-tool-visibility).

## Building a LangGraph graph

```python
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

async def build_graph_for_user(user_context: dict):
    tools = authorized_mcp_tools(await load_mcp_tools(None, connection=connection), config, context=user_context)

    model_with_tools = model.bind_tools(tools)

    def model_node(state: MessagesState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", model_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue)
    graph.add_edge("tools", "model")
    return graph.compile()
```

Build one graph per user session. The tool list is fixed at build time.

## Multiple MCP servers

`authorized_mcp_tools` filters whatever list you hand it, so `MultiServerMCPClient` works the
same way:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "docs": docs_connection,
    "accounts": accounts_connection,
})

tools = authorized_mcp_tools(await client.get_tools(), config, context={"username": "alice"})
```

The combined list from every server goes to the PDP in the one batch request.

## Examples

Runnable scripts are in [`examples/langchain_mcp/`](../../examples/langchain_mcp/). All three
need a running PDP with `examples/policies/domain_mcp_tool_visibility.yaml` loaded, and read
`PDP_BASE_URL`, `LLM_MODEL`, and your provider key from the environment or `.env`. Each runs the
same question as `alice` and then `bob` and prints the tools each was permitted.

| Script | What it shows | Domain file |
|---|---|---|
| `explicit_tool_filtering.py` | `authorized_mcp_tools` on the list from `MultiServerMCPClient.get_tools()`, then `create_agent(model, tools)` with `ChatLiteLLM`; one agent per user | `domain_mcp_tool_visibility.yaml` |
| `tool_filtering_langgraph.py` | The same filter on `load_mcp_tools(...)` from a single MCP server, bound into a custom LangGraph graph; `context` carries `username` and `session_id`; alice sees all tools, bob only `search_docs` | `domain_mcp_tool_visibility.yaml` |
| `multi_server_tool_filtering_langgraph.py` | The same graph over the combined list from two MCP servers via `MultiServerMCPClient` | `domain_mcp_tool_visibility.yaml` |

Supporting MCP servers, started by the scripts as stdio subprocesses: `mcp_server.py`
(single-server demo), `mcp_docs_server.py` and `mcp_account_server.py` (multi-server demo).

```bash
python examples/langchain_mcp/explicit_tool_filtering.py
```

---

[Docs index](../README.md) · [All integrations](../README.md#integrations)
