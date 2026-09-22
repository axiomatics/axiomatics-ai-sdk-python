"""Client-side PDP authorization for the Claude Agent SDK."""
from axiomatics_ai._extras import requires_extra

with requires_extra("claude-agent-sdk", "axiomatics_ai.claude_agent"):
    from axiomatics_ai.claude_agent.authz import authorized_claude_options

__all__ = ["authorized_claude_options"]
