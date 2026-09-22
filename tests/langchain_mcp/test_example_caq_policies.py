"""
Integration tests for the MCP example CAQ reverse-evaluation behavior.
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
class TestMcpToolVisibilityCaqPolicy:
    domain_yaml = "domain_mcp_tool_visibility.yaml"

    def test_bob_visible_tool_constraint(self):
        body = _caq_constraints(
            provided_attributes=[
                _provided("username", "bob"),
                _provided("name", "tool_visibility"),
            ]
        )

        assert _equal_constants_for_attribute(body, "tool_name") == {"search_docs"}
