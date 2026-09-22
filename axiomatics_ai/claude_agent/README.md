# claude_agent

Wraps a Claude Agent SDK `ClaudeAgentOptions` via `authorized_claude_options(...)` and enforces `BEFORE_MODEL` (input guard, hard block), `BEFORE_TOOL` (tool invocation via a `PreToolUse` hook, which `allowed_tools` and `permission_mode` cannot bypass; a permitted call is allowed, with no `allowed_tools` needed) and `AFTER_TOOL` (post-invocation via a `PostToolUse` hook); tool visibility and `AFTER_MODEL` are not available on this path. A tool call the policy wants a human to approve waits for the configured approver.

Documentation: [docs/integrations/claude_agent.md](../../docs/integrations/claude_agent.md) ·
Runnable examples: [examples/claude_agent/](../../examples/claude_agent/)
