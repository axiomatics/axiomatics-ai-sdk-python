# openai_agents

Attaches to an OpenAI Agents SDK `Agent` on the client side and enforces `TOOL_VISIBILITY`, `BEFORE_TOOL`, `AFTER_TOOL` through the `get_all_tools` seam, plus `BEFORE_MODEL` and `AFTER_MODEL` as SDK guardrails. A tool call the policy wants a human to approve goes through the SDK's own approval flow (`needs_approval`).

Documentation: [docs/integrations/openai_agents.md](../../docs/integrations/openai_agents.md) ·
Runnable examples: [examples/openai_agents/](../../examples/openai_agents/)
