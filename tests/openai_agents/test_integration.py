"""
Integration tests for the openai_agents path against REAL ADS (via the ads_server fixture).

These drive the enforcement engine against a live ADS PDP + a real XACML domain, so they
verify the actual ADS request/decision wiring — deterministically, with NO LLM and no Agents
runner (that full agentic path is exercised by the runnable example). The engine imports no
`agents`, so no SDK is needed here.

  - Visibility  → domain_gateway.yaml            (alice sees all tools; bob sees only search_docs)
  - Invocation  → domain_tool_invocation.yaml    (alice any; bob get_account_balance only for ACC-001)
  - Post-invoc. → domain_all_authz_enabled.yaml  (bob's 'classified' records are filtered out)
"""
from __future__ import annotations

import json

import pytest

from axiomatics_ai import AuthZConfig
from axiomatics_ai.openai_agents.authz import _OpenAIAuthZ
from axiomatics_ai import PDPConnection, standard_tool_enforcement

ADS_URL = "http://localhost:8080"
MCP_TOOLS = ["search_docs", "get_account_balance", "send_email"]


@pytest.mark.ads_integration
class TestOpenAIVisibilityAgainstRealAds:
    domain_yaml = "domain_gateway.yaml"

    def test_permitted_tools_per_subject(self, ads_server):
        az = _OpenAIAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))
        assert az.permitted_tools({"username": "alice"}, list(MCP_TOOLS)) == set(MCP_TOOLS)
        assert az.permitted_tools({"username": "bob"}, list(MCP_TOOLS)) == {"search_docs"}


@pytest.mark.ads_integration
class TestOpenAIInvocationAgainstRealAds:
    domain_yaml = "domain_tool_invocation.yaml"

    def test_invocation_denial_per_policy(self, ads_server):
        az = _OpenAIAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))
        alice = {"username": "alice"}
        bob = {"username": "bob"}

        assert az.invocation_denial(alice, "get_account_balance", {"account_id": "ACC-002"}, []) is None
        assert az.invocation_denial(bob, "get_account_balance", {"account_id": "ACC-001"}, []) is None
        assert az.invocation_denial(bob, "get_account_balance", {"account_id": "ACC-002"}, []) is not None
        assert az.invocation_denial(bob, "send_email", {"to": "x@y.z"}, []) is not None


@pytest.mark.ads_integration
class TestOpenAIPostInvocationAgainstRealAds:
    domain_yaml = "domain_all_authz_enabled.yaml"

    def test_filter_result_per_subject(self, ads_server):
        az = _OpenAIAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))
        records = [{"id": "d1", "doc_type": "public"}, {"id": "d2", "doc_type": "classified"}]
        # default MCP on_invoke_tool shape: {'type':'text','text':'<json list>'}
        payload = {"type": "text", "text": json.dumps(records)}

        alice_out = az.filter_result({"username": "alice"}, "search_docs", {}, payload)
        assert [r["id"] for r in json.loads(alice_out["text"])] == ["d1", "d2"]

        bob_out = az.filter_result({"username": "bob"}, "search_docs", {}, payload)
        assert [r["id"] for r in json.loads(bob_out["text"])] == ["d1"]
