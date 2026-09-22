"""
Integration tests for the agents example ADS domain YAML files.

Each test class starts ADS with a specific domain YAML, calls the SDK's
authorization helpers (get_permitted_tools / PdpClient.check_permission / get_permitted_records)
against the real ADS instance, and verifies the policy produces the expected
decisions.

ADS lifecycle fixtures (ads_server, ads) are defined in tests/conftest.py and
shared with tests/mcp/test_example_ads_policies.py.

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

from axiomatics_ai.core.authz_helpers import get_permitted_records, get_permitted_tools

pytestmark = pytest.mark.ads_integration


# ── Tool visibility only ────────────────────────────────────────────────────────

@pytest.mark.usefixtures("ads_server")
class TestToolVisibilityOnlyPolicy:
    """Verifies domain_tool_visibility_only.yaml — alice sees all, bob sees search_docs only."""

    domain_yaml = "domain_tool_visibility_only.yaml"
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


# ── Tool invocation ────────────────────────────────────────────────────

@pytest.mark.usefixtures("ads_server")
class TestToolCallAuthorizationPolicy:
    """Verifies domain_tool_invocation.yaml."""

    domain_yaml = "domain_tool_invocation.yaml"
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

    def test_bob_can_see_search_docs_and_get_account_balance(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "bob"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == {"search_docs", "get_account_balance"}

    def test_alice_can_execute_any_tool(self, ads):
        for tool_name in ["search_docs", "get_account_balance", "send_email"]:
            assert ads.check_permission(
                subject={"username": "alice"},
                action={"name": "tool_invocation"},
                resource={"tool_name": tool_name},
                environment={},
            ), f"alice should be permitted to execute {tool_name}"

    def test_bob_can_execute_search_docs_with_any_query(self, ads):
        assert ads.check_permission(
            subject={"username": "bob"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "search_docs", "query": "anything"},
            environment={},
        )

    def test_bob_can_execute_get_account_balance_for_acc_001(self, ads):
        assert ads.check_permission(
            subject={"username": "bob"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "get_account_balance", "account_id": "ACC-001"},
            environment={},
        )

    def test_bob_cannot_execute_get_account_balance_for_other_accounts(self, ads):
        for account_id in ["ACC-002", "ACC-999", "ACC-000"]:
            assert not ads.check_permission(
                subject={"username": "bob"},
                action={"name": "tool_invocation"},
                resource={"tool_name": "get_account_balance", "account_id": account_id},
                environment={},
            ), f"bob should be denied get_account_balance for {account_id}"

    def test_bob_cannot_execute_send_email(self, ads):
        assert not ads.check_permission(
            subject={"username": "bob"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "send_email", "to": "someone@example.com", "body": "hello"},
            environment={},
        )


# ── Post-invocation ───────────────────────────────────────────────────────────

@pytest.mark.usefixtures("ads_server")
class TestPostInvocationPolicy:
    """Verifies domain_post_invocation.yaml."""

    domain_yaml = "domain_post_invocation.yaml"
    records = [
        {"id": "r1", "name": "Alice Johnson", "doc_type": "profile"},
        {"id": "r2", "name": "Bob Smith",     "doc_type": "classified"},
        {"id": "r3", "name": "Carol Davis",   "doc_type": "profile"},
    ]

    def test_alice_can_see_all_tools(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "alice"},
            tool_names=["list_customer_records"],
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == {"list_customer_records"}

    def test_alice_sees_all_records(self, ads):
        result = get_permitted_records(
            ads,
            records=self.records,
            subject={"username": "alice"},
            action={"name": "post_invocation"},
            environment={},
        )
        assert len(result) == 3

    def test_bob_sees_only_non_classified_records(self, ads):
        result = get_permitted_records(
            ads,
            records=self.records,
            subject={"username": "bob"},
            action={"name": "post_invocation"},
            environment={},
        )
        assert len(result) == 2
        assert all(r["doc_type"] != "classified" for r in result)

    def test_unknown_user_sees_no_records(self, ads):
        result = get_permitted_records(
            ads,
            records=self.records,
            subject={"username": "charlie"},
            action={"name": "post_invocation"},
            environment={},
        )
        assert result == []


# ── All authz enabled ──────────────────────────────────────────────────────────

@pytest.mark.usefixtures("ads_server")
class TestAllAuthzEnabledPolicy:
    """Verifies domain_all_authz_enabled.yaml — all three features combined."""

    domain_yaml = "domain_all_authz_enabled.yaml"
    all_tools = ["search_docs", "get_account_balance", "send_email"]
    records = [
        {"id": "r1", "title": "Getting Started", "doc_type": "public"},
        {"id": "r2", "title": "Internal Spec",   "doc_type": "classified"},
    ]

    def test_alice_sees_all_tools(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "alice"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == set(self.all_tools)

    def test_bob_sees_search_docs_and_get_account_balance(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "bob"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == {"search_docs", "get_account_balance"}

    def test_alice_can_execute_any_tool(self, ads):
        assert ads.check_permission(
            subject={"username": "alice"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "send_email", "to": "x@example.com", "body": "hi"},
            environment={},
        )

    def test_bob_can_execute_get_account_balance_only_for_acc_001(self, ads):
        assert ads.check_permission(
            subject={"username": "bob"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "get_account_balance", "account_id": "ACC-001"},
            environment={},
        )
        assert not ads.check_permission(
            subject={"username": "bob"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "get_account_balance", "account_id": "ACC-002"},
            environment={},
        )

    def test_alice_sees_all_records(self, ads):
        result = get_permitted_records(
            ads,
            records=self.records,
            subject={"username": "alice"},
            action={"name": "post_invocation"},
            environment={},
        )
        assert len(result) == 2

    def test_bob_cannot_see_classified_records(self, ads):
        result = get_permitted_records(
            ads,
            records=self.records,
            subject={"username": "bob"},
            action={"name": "post_invocation"},
            environment={},
        )
        assert len(result) == 1
        assert result[0]["doc_type"] == "public"


# ── Simple-UI policy ───────────────────────────────────────────────────────────

@pytest.mark.usefixtures("ads_server")
class TestSimpleUiPolicy:
    """Verifies domain_simple_ui.yaml — the full policy used by the simple-ui demo app."""

    domain_yaml = "domain_simple_ui.yaml"
    all_tools = ["search_docs", "create_ticket", "send_email", "get_account_balance"]
    records = [
        {"id": "r1", "title": "Public Guide",    "doc_type": "profile"},
        {"id": "r2", "title": "Classified Brief", "doc_type": "classified"},
        {"id": "r3", "title": "Release Notes",    "doc_type": "profile"},
    ]

    def test_alice_sees_all_tools(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "alice"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == set(self.all_tools)

    def test_bob_sees_search_docs_and_get_account_balance(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "bob"},
            tool_names=self.all_tools,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == {"search_docs", "get_account_balance"}

    def test_alice_can_execute_any_tool(self, ads):
        assert ads.check_permission(
            subject={"username": "alice"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "send_email", "to": "x@example.com", "body": "hi"},
            environment={},
        )

    def test_bob_can_execute_get_account_balance_only_for_acc_001(self, ads):
        assert ads.check_permission(
            subject={"username": "bob"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "get_account_balance", "account_id": "ACC-001"},
            environment={},
        )
        assert not ads.check_permission(
            subject={"username": "bob"},
            action={"name": "tool_invocation"},
            resource={"tool_name": "get_account_balance", "account_id": "ACC-999"},
            environment={},
        )

    def test_alice_sees_all_records(self, ads):
        result = get_permitted_records(
            ads,
            records=self.records,
            subject={"username": "alice"},
            action={"name": "post_invocation"},
            environment={},
        )
        assert len(result) == 3

    def test_bob_cannot_see_classified_records(self, ads):
        result = get_permitted_records(
            ads,
            records=self.records,
            subject={"username": "bob"},
            action={"name": "post_invocation"},
            environment={},
        )
        assert len(result) == 2
        assert all(r["doc_type"] != "classified" for r in result)
