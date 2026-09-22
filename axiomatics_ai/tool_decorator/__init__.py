"""Per-tool post-invocation with the ``@protected_tool`` decorator.

Call ``configure_authz_decorator`` once at start-up, then decorate tools whose results a
strategy should filter or mask.
"""
from axiomatics_ai._extras import requires_extra
from axiomatics_ai.core.post_invocation import PostInvocationContext

with requires_extra("tool-decorator", "axiomatics_ai.tool_decorator"):
    from axiomatics_ai.tool_decorator.decorator import (
        configure_authz_decorator,
        protected_tool,
    )

__all__ = [
    "PostInvocationContext",
    "configure_authz_decorator",
    "protected_tool",
]
