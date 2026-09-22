"""
Integration tests for the MCP example ADS domain YAML files.

Each test class starts ADS with a specific domain YAML and calls get_permitted_tools
directly against the real ADS instance to verify that the policy produces the
expected decisions for each user.

ADS lifecycle fixtures (ads_server, ads) come from tests/conftest.py.

These tests require:
  - Java in PATH
  - ADS jar at examples/policies/access-decision-service-*.jar
  - ADS license at examples/policies/axiomatics_ADS.license
  - Port 8080 available

Run them explicitly:
  uv run pytest tests/ -m ads_integration -v

They are skipped by default unless the marker is selected.
"""
from __future__ import annotations

import pytest

from axiomatics_ai.core.authz_helpers import get_permitted_tools

pytestmark = pytest.mark.ads_integration


# ── MCP tool filtering ─────────────────────────────────────────────────────────

@pytest.mark.usefixtures("ads_server")
class TestMcpToolVisibilityPolicy:
    """Verifies domain_mcp_tool_visibility.yaml — the policy used by examples/langchain_mcp/tool_filtering.py.

    Tools exposed by the MCP server: search_docs, get_account_balance, send_email.
      alice   → all three tools
      bob     → search_docs only
      charlie → no tools (no matching rule, fail_open=False)
    """

    domain_yaml = "domain_mcp_tool_visibility.yaml"
    all_tools = ["search_docs", "get_account_balance", "send_email"]

    def test_alice_can_see_all_tools(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "alice"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == set(self.all_tools)

    def test_bob_can_only_see_search_docs(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "bob"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == {"search_docs"}

    def test_unknown_user_sees_no_tools(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "charlie"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == set()
