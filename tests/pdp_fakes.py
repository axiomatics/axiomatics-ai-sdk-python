"""
Helpers for unit tests that stand in for the PDP.

The enforcement core asks the PDP for a decision (``PdpClient.evaluate`` / the
``enforcement.evaluate_tool_call`` seam) and resolves it (``PdpClient.resolve``: fail_open,
obligations, approval). Most fakes only know whether a call is permitted; these adapt that
bool to the decision-shaped API.
"""
from __future__ import annotations

import functools
from typing import Any

from axiomatics_ai.core.pdp import Decision, ObligationsAndAdviceConfig, PdpClient, PdpDecision


def permit_or_deny(permitted: Any) -> PdpDecision:
    return PdpDecision(Decision.PERMIT if permitted else Decision.DENY)


class FakePdpBase:
    """Base for fake PDPs that implement ``check_permission``. ``evaluate`` turns its answer
    into a decision; ``resolve`` is the real client's."""

    fail_open = False
    obligations = ObligationsAndAdviceConfig()

    def evaluate(self, *, subject, action, resource, environment) -> PdpDecision:
        return permit_or_deny(self.check_permission(
            subject=subject, action=action, resource=resource, environment=environment,
        ))

    resolve = PdpClient.resolve


def decision_seam(fake):
    """Adapt a fake that answers True/False into a fake ``evaluate_tool_call``."""
    @functools.wraps(fake)
    def evaluate_tool_call(pdp, **request):
        return permit_or_deny(fake(pdp, **request))
    return evaluate_tool_call


def classifier_input(messages: list[dict]) -> str:
    """The content an enricher asked its LLM to classify: the user message, without the
    ``<content-…>`` fence it arrives in."""
    return messages[-1]["content"].split("\n", 1)[1].rsplit("\n", 1)[0]


def patch_pdp_http(monkeypatch, method: str, fn) -> None:
    """Stub the PDP client's HTTP transport: every ``httpx.Client.<method>`` (``"post"`` or
    ``"get"``) call becomes ``fn(url, **kwargs)``. ``PdpClient`` sends through a pooled
    ``httpx.Client``, so this is the seam for a test that needs the real ``_authorize`` to run."""
    import httpx

    monkeypatch.setattr(httpx.Client, method, lambda self, url, **kwargs: fn(url, **kwargs))
