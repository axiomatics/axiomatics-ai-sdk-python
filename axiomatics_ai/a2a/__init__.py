"""A2A (Agent2Agent) authorization via an external PDP.

Enforcement happens at the HTTP/JSON-RPC wire layer, so it is framework-agnostic (works
with Google ADK, AWS Strands, LangGraph, and self-wired a2a-sdk) and a2a-sdk-version-
agnostic (0.3.x and 1.x). Two primitives:

- ``A2AAuthZMiddleware`` — server-side, the recommended enforcement point. A pure-ASGI
  middleware that blocks unauthorized inbound message sends before they reach the agent.
- ``A2AClientAuthZ`` — client-side, complementary egress control. An ``httpx.Auth`` that
  blocks unauthorized outbound message sends before they leave.

See the module docstrings (and README.md) for wiring examples per framework.
"""
from axiomatics_ai.a2a._enforce import A2AAuthorizationError
from axiomatics_ai.a2a.client import A2AClientAuthZ
from axiomatics_ai.a2a.server import A2AAuthZMiddleware

__all__ = [
    "A2AAuthZMiddleware",
    "A2AAuthorizationError",
    "A2AClientAuthZ",
]
