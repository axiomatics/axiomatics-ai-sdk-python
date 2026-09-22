"""Client-side PDP authorization for Strands agents."""
from axiomatics_ai._extras import requires_extra

with requires_extra("strands", "axiomatics_ai.strands"):
    from axiomatics_ai.strands.authz import authorized_strands_agent

__all__ = ["authorized_strands_agent"]
