"""
Integration tests for the strands path against REAL ADS (via the ads_server fixture).

These drive the enforcement engine against a live ADS PDP + a real XACML domain, so they
verify the actual ADS request/decision wiring — deterministically, with NO LLM and no Strands
agent loop (that full agentic path is exercised by the runnable example). The engine
duck-types Strands events, so Strands-shaped fakes are sufficient here.

  - Visibility  → domain_gateway.yaml            (alice sees all tools; bob sees only search_docs)
  - Invocation  → domain_tool_invocation.yaml    (alice any; bob get_account_balance only for ACC-001)
  - Post-invoc. → domain_all_authz_enabled.yaml  (bob's 'classified' records are filtered out)
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("strands")

from axiomatics_ai import AuthZConfig
from axiomatics_ai.strands.authz import _StrandsAuthZ

from .conftest import FakeAgent, FakeModelEvent, FakeToolEvent
from axiomatics_ai import PDPConnection, standard_tool_enforcement

ADS_URL = "http://localhost:8080"
MCP_TOOLS = ["search_docs", "get_account_balance", "send_email"]


def _mcp_result(records):
    return {
        "content": [{"text": json.dumps(records)}],
        "structuredContent": {"result": records},
        "status": "success",
        "toolUseId": "t1",
    }


@pytest.mark.ads_integration
class TestStrandsVisibilityAgainstRealAds:
    domain_yaml = "domain_gateway.yaml"

    def test_before_model_filters_tools_per_subject(self, ads_server):
        az = _StrandsAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))

        alice_agent = FakeAgent(tools=list(MCP_TOOLS))
        az.before_model(FakeModelEvent(alice_agent, {"username": "alice"}))
        alice_seen = az.filter_tools_config(alice_agent.tool_registry.get_all_tools_config())
        assert set(alice_seen) == set(MCP_TOOLS)                    # alice sees all

        bob_agent = FakeAgent(tools=list(MCP_TOOLS))
        az.before_model(FakeModelEvent(bob_agent, {"username": "bob"}))
        bob_seen = az.filter_tools_config(bob_agent.tool_registry.get_all_tools_config())
        assert set(bob_seen) == {"search_docs"}                    # bob sees only search_docs


@pytest.mark.ads_integration
class TestStrandsInvocationAgainstRealAds:
    domain_yaml = "domain_tool_invocation.yaml"

    def test_before_tool_blocks_and_allows_per_policy(self, ads_server):
        az = _StrandsAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))
        alice = {"username": "alice"}
        bob = {"username": "bob"}

        # alice may call any tool
        assert az.before_tool(FakeToolEvent("get_account_balance", {"account_id": "ACC-002"}, invocation_state=alice)) is None
        # bob may only call get_account_balance for ACC-001 (arg-level policy)
        assert az.before_tool(FakeToolEvent("get_account_balance", {"account_id": "ACC-001"}, invocation_state=bob)) is None
        denied = az.before_tool(FakeToolEvent("get_account_balance", {"account_id": "ACC-002"}, invocation_state=bob))
        assert isinstance(denied, str) and "Access denied" in denied
        # bob may not send email at all
        assert isinstance(az.before_tool(FakeToolEvent("send_email", {"to": "x@y.z"}, invocation_state=bob)), str)


@pytest.mark.ads_integration
class TestStrandsPostInvocationAgainstRealAds:
    domain_yaml = "domain_all_authz_enabled.yaml"

    def test_after_tool_filters_records_per_subject(self, ads_server):
        az = _StrandsAuthZ(AuthZConfig(connection=PDPConnection(base_url=ADS_URL), enforcement=standard_tool_enforcement()))
        records = [{"id": "d1", "doc_type": "public"}, {"id": "d2", "doc_type": "classified"}]

        alice_out = az.after_tool(FakeToolEvent("search_docs", invocation_state={"username": "alice"}, result=_mcp_result(records)))
        assert [r["id"] for r in alice_out["structuredContent"]["result"]] == ["d1", "d2"]   # alice sees both

        bob_out = az.after_tool(FakeToolEvent("search_docs", invocation_state={"username": "bob"}, result=_mcp_result(records)))
        assert [r["id"] for r in bob_out["structuredContent"]["result"]] == ["d1"]           # bob: classified filtered
