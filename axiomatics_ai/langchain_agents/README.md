# langchain_agents

Attaches to `langchain.agents.create_agent` and `deepagents.create_deep_agent` as a middleware and enforces `BEFORE_MODEL`, `TOOL_VISIBILITY`, `BEFORE_TOOL`, `AFTER_TOOL`, and `AFTER_MODEL`. A tool call the policy wants a human to approve pauses the graph with LangGraph's `interrupt()`.

Documentation: [docs/integrations/langchain_agents.md](../../docs/integrations/langchain_agents.md) ·
Runnable examples: [examples/langchain_agents/](../../examples/langchain_agents/)
