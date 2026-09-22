"""
Integration tests for the agents example CAQ reverse-evaluation behavior.

These tests use the real ADS/CAQ service and verify that the example domain
attribute catalogs support reduced reverse-evaluation responses.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.conftest import ADS_URL

pytestmark = pytest.mark.ads_integration


def _caq_constraints(*, provided_attributes: list[dict[str, Any]]) -> dict[str, Any]:
    response = httpx.post(
        f"{ADS_URL}/authorization/constraints",
        json={
            "providedAttributes": provided_attributes,
            "pdpDecisionSet": ["PERMIT"],
            "excludeIndeterminate": True,
            "contextType": "NONE",
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def _equal_constants_for_attribute(node: Any, attribute_name: str) -> set[str]:
    if isinstance(node, dict):
        apply = node.get("apply")
        if isinstance(apply, dict) and apply.get("function") == "anyOf":
            arguments = apply.get("arguments", [])
            constants = {
                arg["constant"]["value"]
                for arg in arguments
                if isinstance(arg, dict) and isinstance(arg.get("constant"), dict)
            }
            has_attribute = any(
                isinstance(arg, dict)
                and arg.get("attributeDesignator", {}).get("attributeName") == attribute_name
                for arg in arguments
            )
            if has_attribute:
                return constants
        values: set[str] = set()
        for value in node.values():
            values.update(_equal_constants_for_attribute(value, attribute_name))
        return values
    if isinstance(node, list):
        values: set[str] = set()
        for item in node:
            values.update(_equal_constants_for_attribute(item, attribute_name))
        return values
    return set()


def _provided(attribute: str, *values: str) -> dict[str, Any]:
    result: dict[str, Any] = {"attribute": attribute}
    if values:
        result["values"] = list(values)
    return result


@pytest.mark.usefixtures("ads_server")
class TestToolVisibilityOnlyCaqPolicy:
    domain_yaml = "domain_tool_visibility_only.yaml"

    def test_bob_visible_tool_constraint(self):
        body = _caq_constraints(
            provided_attributes=[
                _provided("username", "bob"),
                _provided("name", "tool_visibility"),
            ]
        )

        assert _equal_constants_for_attribute(body, "tool_name") == {"search_docs"}


@pytest.mark.usefixtures("ads_server")
class TestToolCallAuthorizationCaqPolicy:
    domain_yaml = "domain_tool_invocation.yaml"

    def test_bob_visible_tool_constraints(self):
        body = _caq_constraints(
            provided_attributes=[
                _provided("username", "bob"),
                _provided("name", "tool_visibility"),
            ]
        )

        assert _equal_constants_for_attribute(body, "tool_name") == {
            "search_docs",
            "get_account_balance",
        }

    def test_bob_get_account_balance_requires_acc_001(self):
        body = _caq_constraints(
            provided_attributes=[
                _provided("username", "bob"),
                _provided("name", "tool_invocation"),
                _provided("tool_name", "get_account_balance"),
            ]
        )

        assert _equal_constants_for_attribute(body, "account_id") == {"ACC-001"}


@pytest.mark.usefixtures("ads_server")
class TestAllAuthzEnabledCaqPolicy:
    domain_yaml = "domain_all_authz_enabled.yaml"

    def test_bob_visible_tool_constraints(self):
        body = _caq_constraints(
            provided_attributes=[
                _provided("username", "bob"),
                _provided("name", "tool_visibility"),
            ]
        )

        assert _equal_constants_for_attribute(body, "tool_name") == {
            "search_docs",
            "get_account_balance",
        }

    def test_bob_get_account_balance_requires_acc_001(self):
        body = _caq_constraints(
            provided_attributes=[
                _provided("username", "bob"),
                _provided("name", "tool_invocation"),
                _provided("tool_name", "get_account_balance"),
            ]
        )

        assert _equal_constants_for_attribute(body, "account_id") == {"ACC-001"}
