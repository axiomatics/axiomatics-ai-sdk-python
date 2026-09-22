"""
Integration test for a MULTI-VALUED (list-of-strings) subject attribute against a
real ADS instance.

Uses ``domain_list_attribute.yaml``, whose policy keys tool visibility off a
``roles`` bag. Bob is permitted via "auditor" — the SECOND element of his roles
list — so a passing test proves the SDK serialises and sends every element of the
list, not just the first.

Skipped automatically when the ADS jar is absent (see tests/conftest.py).
"""
from __future__ import annotations

import pytest

from axiomatics_ai.core.authz_helpers import get_permitted_tools

pytestmark = pytest.mark.ads_integration  # starts the real ADS (ads_server)

ALL_TOOLS = ["search_docs", "get_account_balance", "send_email"]


@pytest.mark.usefixtures("ads_server")
class TestListAttributePolicy:
    domain_yaml = "domain_list_attribute.yaml"

    # Usernames are deliberately neutral (no username-based rule exists in this
    # domain). Outcomes therefore depend ONLY on the roles bag — so these tests
    # would fail, not pass-by-coincidence, if a different domain were loaded.

    def test_admin_role_first_element_sees_all_tools(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "user-1", "roles": ["admin", "ops"]},
            tool_names=ALL_TOOLS,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == set(ALL_TOOLS)

    def test_auditor_role_second_element_sees_only_search_docs(self, ads):
        # "auditor" is the second element — matching it confirms the full list is sent.
        permitted = get_permitted_tools(
            ads,
            subject={"username": "user-2", "roles": ["viewer", "auditor"]},
            tool_names=ALL_TOOLS,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == {"search_docs"}

    def test_no_matching_role_sees_no_tools(self, ads):
        permitted = get_permitted_tools(
            ads,
            subject={"username": "user-3", "roles": ["guest", "viewer"]},
            tool_names=ALL_TOOLS,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == set()

    def test_single_role_as_plain_string_still_works(self, ads):
        # A scalar (non-list) string is the degenerate one-element bag.
        permitted = get_permitted_tools(
            ads,
            subject={"username": "user-4", "roles": "admin"},
            tool_names=ALL_TOOLS,
            action={"name": "tool_visibility"},
            environment={},
        )
        assert permitted == set(ALL_TOOLS)
