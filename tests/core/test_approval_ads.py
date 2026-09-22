"""
Human approval against a real ADS (examples/policies/domain_human_approval.yaml), over both
wire protocols: the PDP's advice reaches the SDK, the approver is asked, and the second
evaluation — with approval_granted — is the final decision.
"""
from __future__ import annotations

import pytest

from axiomatics_ai import AuthZConfig, AuthzHook, PDPConnection, standard_tool_enforcement
from axiomatics_ai.core.enforcement import authorized_tool_invocation
from axiomatics_ai.core.pdp import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig
from tests.conftest import ADS_URL

pytestmark = pytest.mark.ads_integration

ALICE = {"username": "alice"}
ACTION = {"name": "tool_invocation"}


class _Recorder(AuthzHook):
    def __init__(self):
        self.decisions = []
        self.pdp_calls = 0

    def on_decision(self, event):
        self.decisions.append(event.decision)

    def on_pdp_request(self, event):
        self.pdp_calls += 1


def _config(protocol, answer=None, recorder=None):
    requests = []

    def approver(request):
        requests.append(request)
        return answer

    config = AuthZConfig(
        connection=PDPConnection(base_url=ADS_URL, protocol=protocol),
        enforcement=standard_tool_enforcement(),
        obligations_and_advice=ObligationsAndAdviceConfig(
            approval=ApprovalConfig(approver=approver if answer is not None else None),
        ),
        hooks=[recorder] if recorder else [],
    )
    return config, requests


def _invoke(config, resource):
    return authorized_tool_invocation(
        pdp=config.build_pdp_client(), cache=config.build_cache(), config=config, surface="test",
        subject=ALICE, action=ACTION, resource=resource, environment={}, user_messages=[],
    ).permitted


def _pay(config, amount):
    return _invoke(config, {"tool_name": "pay_invoice", "amount": amount})


@pytest.mark.usefixtures("ads_server")
@pytest.mark.parametrize("protocol", ["xacml", "authzen"])
class TestHumanApprovalAgainstAds:
    domain_yaml = "domain_human_approval.yaml"

    def test_small_payment_needs_no_approval(self, protocol):
        config, requests = _config(protocol, ApprovalDecision(True))
        assert _pay(config, 40.0) is True
        assert requests == []

    def test_large_payment_is_approved_with_a_second_pdp_call(self, protocol):
        # Deny + advice: only the PDP can turn the Deny into a Permit, so it is asked again.
        recorder = _Recorder()
        config, requests = _config(protocol, ApprovalDecision(True, approver="manager"), recorder)
        assert _pay(config, 500.0) is True
        assert requests[0].reason == "Payments over 100 need a human's approval"
        assert recorder.decisions == ["approval_required", "approved", "permit"]
        assert recorder.pdp_calls == 2

    def test_cancellation_is_approved_with_one_pdp_call(self, protocol):
        # Permit + obligation: the approval fulfils the obligation, so the Permit stands.
        recorder = _Recorder()
        config, requests = _config(protocol, ApprovalDecision(True, approver="manager"), recorder)
        assert _invoke(config, {"tool_name": "cancel_invoice"}) is True
        assert requests[0].reason == "Cancelling an invoice needs a human's approval"
        assert recorder.decisions == ["approval_required", "approved", "permit"]
        assert recorder.pdp_calls == 1

    def test_cancellation_is_rejected(self, protocol):
        config, _ = _config(protocol, ApprovalDecision(False))
        assert _invoke(config, {"tool_name": "cancel_invoice"}) is False

    def test_large_payment_is_rejected(self, protocol):
        config, _ = _config(protocol, ApprovalDecision(False))
        assert _pay(config, 500.0) is False

    def test_large_payment_without_approver_is_denied(self, protocol):
        config, _ = _config(protocol)
        assert _pay(config, 500.0) is False
