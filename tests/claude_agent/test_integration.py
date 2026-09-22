"""
Integration tests for the claude_agent path against REAL ADS (via the ads_server fixture).

Drive the enforcement engine against a live ADS PDP + a real XACML domain, deterministically,
with no LLM and no Claude Agent SDK runtime (the full path is exercised by the example). The
engine takes plain values, so no SDK is needed here.

  - Invocation  → domain_tool_invocation.yaml    (alice any; bob get_account_balance only ACC-001)
  - Post-invoc. → domain_all_authz_enabled.yaml  (bob's 'classified' records are filtered out)

There is no visibility class: tool visibility is not supported on this path.
"""
from __future__ import annotations

import json

import pytest

from axiomatics_ai import AuthZConfig
from axiomatics_ai.claude_agent.authz import _ClaudeAuthZ
from axiomatics_ai import PDPConnection, standard_tool_enforcement

ADS_URL = "http://localhost:8080"


def _engine(identity):
    return _ClaudeAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()), identity)


@pytest.mark.ads_integration
class TestClaudeInvocationAgainstRealAds:
    domain_yaml = "domain_tool_invocation.yaml"

    def test_invocation_denial_is_arg_level(self, ads_server):
        alice = _engine({"username": "alice"})
        bob = _engine({"username": "bob"})

        assert alice.invocation_denial("get_account_balance", {"account_id": "ACC-002"}) is None
        assert bob.invocation_denial("get_account_balance", {"account_id": "ACC-001"}) is None   # arg-level permit
        assert bob.invocation_denial("get_account_balance", {"account_id": "ACC-002"}) is not None
        assert bob.invocation_denial("send_email", {"to": "x@y.z"}) is not None


@pytest.mark.ads_integration
class TestClaudePostInvocationAgainstRealAds:
    domain_yaml = "domain_all_authz_enabled.yaml"

    def test_filter_result_per_subject(self, ads_server):
        records = [{"id": "d1", "doc_type": "public"}, {"id": "d2", "doc_type": "classified"}]
        payload = {"content": [{"type": "text", "text": json.dumps(records)}], "structuredContent": {"result": records}}

        alice_out = _engine({"username": "alice"}).filter_result("search_docs", {}, payload)
        assert [r["id"] for r in alice_out["structuredContent"]["result"]] == ["d1", "d2"]

        bob_out = _engine({"username": "bob"}).filter_result("search_docs", {}, payload)
        assert [r["id"] for r in bob_out["structuredContent"]["result"]] == ["d1"]
