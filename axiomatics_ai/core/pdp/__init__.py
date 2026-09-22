"""pdp — talking to the Policy Decision Point.

- ``client``      — ``PdpClient``: transport, ``fail_open``, single and batch decisions, the
  partial-evaluation (reverse) query.
- ``xacml`` / ``authzen`` — each wire protocol's request builders and response parsers.
- ``decision``    — what a PDP returns: ``Decision``, ``Obligation``, ``Advice``, ``PdpDecision``.
- ``obligations`` — what the PEP does with obligations and advice, including human approval.
- ``attributes``  — the values a request can carry (the value guard).
"""
from axiomatics_ai.core.pdp.attributes import sendable_attributes, strip_none
from axiomatics_ai.core.pdp.client import (
    HealthCheckResult,
    PdpClient,
    PdpEndpoints,
    PdpRequestError,
    PermittedNames,
    flatten_to_provided_attributes,
)
from axiomatics_ai.core.pdp.decision import Advice, Decision, Obligation, PdpDecision
from axiomatics_ai.core.pdp.obligations import (
    ApprovalConfig,
    ApprovalDecision,
    ApprovalRequest,
    ObligationContext,
    ObligationsAndAdviceConfig,
    Outcome,
    Resolution,
)

__all__ = [
    "Advice",
    "ApprovalConfig",
    "ApprovalDecision",
    "ApprovalRequest",
    "Decision",
    "HealthCheckResult",
    "Obligation",
    "ObligationContext",
    "ObligationsAndAdviceConfig",
    "Outcome",
    "PdpClient",
    "PdpDecision",
    "PdpEndpoints",
    "PdpRequestError",
    "PermittedNames",
    "Resolution",
    "flatten_to_provided_attributes",
    "sendable_attributes",
    "strip_none",
]
