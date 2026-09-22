"""
Integration tests for the google_adk path against REAL ADS (via the ads_server fixture).

These drive the enforcement callbacks against a live ADS PDP + a real XACML domain, so they
verify the actual ADS request/decision wiring — deterministically, with NO LLM and no ADK
agent loop (that full agentic path is exercised by the runnable example). The callbacks are
core-only (they duck-type ADK objects), so ADK-shaped fakes are sufficient here.

  - Visibility → domain_gateway.yaml   (alice sees all tools; bob sees only search_docs)
  - Invocation → domain_tool_invocation.yaml (alice any; bob get_account_balance only for ACC-001)
"""
from __future__ import annotations

import pytest

from axiomatics_ai import AuthZConfig
from axiomatics_ai.google_adk.authz import _AdkAuthZ

from .conftest import FakeContext, FakeLlmRequest, FakeTool
from axiomatics_ai import PDPConnection, standard_tool_enforcement

ADS_URL = "http://localhost:8080"
MCP_TOOLS = ["search_docs", "get_account_balance", "send_email"]


@pytest.mark.ads_integration
class TestAdkVisibilityAgainstRealAds:
    domain_yaml = "domain_gateway.yaml"

    def test_before_model_filters_tools_per_subject(self, ads_server):
        az = _AdkAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))

        alice_req = FakeLlmRequest(list(MCP_TOOLS))
        az.before_model(FakeContext({"username": "alice"}), alice_req)
        alice_decls = {fd.name for t in alice_req.config.tools for fd in t.function_declarations}
        assert alice_decls == set(MCP_TOOLS)                        # alice sees all

        bob_req = FakeLlmRequest(list(MCP_TOOLS))
        az.before_model(FakeContext({"username": "bob"}), bob_req)
        # bob sees only search_docs: the model-facing declarations are pruned
        bob_decls = {fd.name for t in bob_req.config.tools for fd in t.function_declarations}
        assert bob_decls == {"search_docs"}


@pytest.mark.ads_integration
class TestAdkInvocationAgainstRealAds:
    domain_yaml = "domain_tool_invocation.yaml"

    def test_before_tool_blocks_and_allows_per_policy(self, ads_server):
        az = _AdkAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))
        alice = FakeContext({"username": "alice"})
        bob = FakeContext({"username": "bob"})

        # alice may call any tool
        assert az.before_tool(FakeTool("get_account_balance"), {"account_id": "ACC-002"}, alice) is None
        # bob may only call get_account_balance for ACC-001 (arg-level policy)
        assert az.before_tool(FakeTool("get_account_balance"), {"account_id": "ACC-001"}, bob) is None
        denied = az.before_tool(FakeTool("get_account_balance"), {"account_id": "ACC-002"}, bob)
        assert isinstance(denied, dict) and "Access denied" in denied["error"]
        # bob may not send email at all
        assert isinstance(az.before_tool(FakeTool("send_email"), {"to": "x@y.z"}, bob), dict)
