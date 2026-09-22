"""
Tests for the simple-ui example app, verifying that LangChainAgentAuthZMiddleware
correctly filters tools before presenting them to the LLM.

These tests monkeypatch ``get_permitted_tools`` in ``authz_helpers`` so no real ADS
instance is required. Captured calls are inspected to assert that:
  - The correct subject was used for the ADS request.
  - Alice receives all available tools.
  - Bob receives only the subset allowed by the policy stub.
  - MCP tools (async mode) are included in the available set.
  - MCP tools are absent in sync mode.
"""
import asyncio
import sys
from pathlib import Path

import pytest
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "langchain_agents" / "simple-ui"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import agent as example_agent
import axiomatics_ai.core.enforcement as enforcement_module
from data_models import AppConfig, AppContext


class StubChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@tool
def lookup_customer_profile(customer_id: str) -> str:
    """Look up a customer's profile summary."""
    return f"Customer {customer_id}"


@tool
def get_account_balance(account_id: str) -> str:
    """Retrieve a mock account balance."""
    return f"Account {account_id}"


def invoke_agent(agent, invocation_mode: str, app_context: AppContext):
    payload = {"messages": [{"role": "user", "content": "What can you do?"}]}
    if invocation_mode == "sync":
        return agent.invoke(payload, context=app_context)
    return asyncio.run(agent.ainvoke(payload, context=app_context))


@pytest.mark.parametrize("agent_runtime", ["vanilla", "deep-agents"])
@pytest.mark.parametrize("invocation_mode", ["sync", "async"])
@pytest.mark.parametrize("username", ["alice", "bob"])
def test_middleware_tool_filter_matrix(monkeypatch, agent_runtime, invocation_mode, username):
    case_label = f"runtime={agent_runtime}, invocation={invocation_mode}, user={username}"

    # Capture calls so we can assert on what the middleware sent to ADS.
    filter_calls: list[dict] = []

    def fake_get_permitted_tools(pdp_client, *, subject, tool_names, action, environment, tool_name_attribute="tool_name", fail_open=False):
        username_ = subject.get("username", "")
        if username_ == "alice":
            permitted = set(tool_names)
        elif username_ == "bob":
            permitted = {"search_docs", "get_account_balance"} & set(tool_names)
        else:
            permitted = set()
        filter_calls.append({
            "subject": subject,
            "tool_names": set(tool_names),
            "permitted": permitted,
        })
        return permitted

    monkeypatch.setattr(enforcement_module, "get_permitted_tools", fake_get_permitted_tools)
    monkeypatch.setattr(
        example_agent,
        "_build_model",
        lambda: StubChatModel(responses=[AIMessage(content="ok")]),
    )
    monkeypatch.setattr(
        example_agent,
        "_load_mcp_tools",
        lambda: (lookup_customer_profile, get_account_balance),
    )

    config = AppConfig(
        app_name="test",
        app_url="http://localhost:8501",
        ads_base_url="http://localhost:8080",
        invocation_mode=invocation_mode,
        agent_runtime=agent_runtime,
    )
    app_context = AppContext(
        user_id=f"user-{username}",
        username=username,
        tenant_id="tenant-acme",
        agent_name="support-agent",
        session_id="session-test",
    )
    agent = example_agent.build_agent(config, app_context)
    result = invoke_agent(agent, invocation_mode, app_context)

    assert result["messages"], f"{case_label}: expected at least one message"
    assert filter_calls, f"{case_label}: get_permitted_tools was never called"

    call = filter_calls[0]
    available_tool_names = call["tool_names"]
    allowed_tool_names = call["permitted"]

    assert call["subject"].get("username") == username, (
        f"{case_label}: expected subject username='{username}', got '{call['subject']}'"
    )
    assert {"search_docs", "create_ticket", "send_email"}.issubset(available_tool_names), (
        f"{case_label}: base tools missing from available set: {sorted(available_tool_names)}"
    )

    if invocation_mode == "async":
        assert {"lookup_customer_profile", "get_account_balance"}.issubset(available_tool_names), (
            f"{case_label}: MCP tools expected in async mode, got: {sorted(available_tool_names)}"
        )
    else:
        assert "lookup_customer_profile" not in available_tool_names, (
            f"{case_label}: sync mode should not include lookup_customer_profile"
        )
        assert "get_account_balance" not in available_tool_names, (
            f"{case_label}: sync mode should not include get_account_balance"
        )

    if username == "alice":
        assert allowed_tool_names == available_tool_names, (
            f"{case_label}: alice should be allowed all available tools, "
            f"available={sorted(available_tool_names)}, allowed={sorted(allowed_tool_names)}"
        )
    else:
        expected_allowed = {"search_docs"}
        if invocation_mode == "async":
            expected_allowed.add("get_account_balance")
        assert allowed_tool_names == expected_allowed, (
            f"{case_label}: bob's allowed tools should match policy stub, "
            f"expected={sorted(expected_allowed)}, got={sorted(allowed_tool_names)}"
        )
