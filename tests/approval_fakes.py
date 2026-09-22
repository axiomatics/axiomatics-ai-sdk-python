"""
Shared fakes for the human-approval tests of each framework: a PDP stub that asks for approval
until the approval attribute is set, and a decision recorder.
"""
from __future__ import annotations

from axiomatics_ai import AuthzHook
from axiomatics_ai.core.pdp import Advice, Decision, PdpDecision

ASK = PdpDecision(Decision.DENY, advice=(Advice("require-approval", {"reason": "Payments over 100 need approval"}),))
PERMIT = PdpDecision(Decision.PERMIT)


class ApprovalPdp:
    """Stands in for ``enforcement.evaluate_tool_call``: asks for approval (``first``) until
    ``approval_granted`` is set, then permits. Records every request's environment."""

    def __init__(self, first=ASK):
        self.first = first
        self.environments: list[dict] = []

    def __call__(self, pdp, *, subject, action, resource, environment):
        self.environments.append(dict(environment))
        return PERMIT if environment.get("approval_granted") is True else self.first


class DecisionRecorder(AuthzHook):
    def __init__(self):
        self.events = []

    def on_decision(self, event):
        self.events.append(event)

    @property
    def decisions(self) -> list[str]:
        return [e.decision for e in self.events]
