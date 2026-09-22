"""
End-to-end test for the documented multi-server MCP example:
a custom provider overriding the action AttributeId (``action_id``) plus a config
overriding the tool-visibility resource AttributeId (``tool_id``), filtered by a
REAL ADS via ``authorized_mcp_tools``.

No live MCP servers — the tool objects are stand-ins — so the test exercises the real
request-building against a real ADS running ``domain_mcp_custom_attrs.yaml``.

Skipped automatically when the ADS jar is absent (see tests/conftest.py).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from axiomatics_ai import AuthZConfig, AttributeNames, standard_tool_enforcement
from axiomatics_ai.core.config import DefaultClientContextProvider
from axiomatics_ai.langchain_mcp import authorized_mcp_tools

pytestmark = pytest.mark.ads_integration  # starts the real ADS (ads_server)


def _tool(name: str):
    t = MagicMock()
    t.name = name
    return t


ALL_TOOLS = [_tool("search_docs"), _tool("get_account_balance"), _tool("send_email")]


class MyProvider(DefaultClientContextProvider):
    def get_action(self, action_name, context):
        return {"action_id": action_name}  # action AttributeId → "action_id"


@pytest.mark.usefixtures("ads_server")
class TestMcpCustomAttrsEndToEnd:
    domain_yaml = "domain_mcp_custom_attrs.yaml"

    def _permitted_names(self, username):
        config = AuthZConfig(
            context_provider=MyProvider(),
            attribute_names=AttributeNames(tool="tool_id"),
            enforcement=standard_tool_enforcement(),
        )
        tools = authorized_mcp_tools(ALL_TOOLS, config, context={"username": username})
        return {t.name for t in tools}

    def test_alice_sees_all_tools(self):
        assert self._permitted_names("alice") == {"search_docs", "get_account_balance", "send_email"}

    def test_bob_sees_only_search_docs(self):
        assert self._permitted_names("bob") == {"search_docs"}

    def test_unknown_user_sees_no_tools(self):
        assert self._permitted_names("charlie") == set()
