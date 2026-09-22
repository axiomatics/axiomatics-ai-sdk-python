"""
Tests for LangChainAgentAuthZMiddleware enforcement outcomes.

Each test creates a real agent with real tools and a fake LLM, stubs ADS at the
authz_helpers level, and asserts on observable enforcement outcomes:
  - which tool names the model actually received after filtering
  - whether a denied tool call returns an access-denied ToolMessage
  - which records survive in the ToolMessage content after record filtering

A ToolCapturingMiddleware is placed after LangChainAgentAuthZMiddleware in the chain
so it sees the already-filtered tool list — this is how we verify what the model
actually received, not just what ADS was asked about.

No real ADS instance is required.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage as LangChainToolMessage

import axiomatics_ai.core.authz_helpers as authz_helpers_module
import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import LangChainAgentAuthZMiddleware, AuthZConfig
from axiomatics_ai.core.config import DefaultClientContextProvider
from axiomatics_ai import PDPConnection, standard_tool_enforcement
from tests.pdp_fakes import decision_seam


# ── Helpers ────────────────────────────────────────────────────────────────────

class StubChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class ToolCapturingMiddleware(AgentMiddleware):
    """
    Placed after LangChainAgentAuthZMiddleware in the middleware chain.
    Captures the tool names present in the ModelRequest at the point they
    reach the model — i.e., after authz filtering has already been applied.
    """

    def __init__(self):
        self.received_tool_names: set[str] = set()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self.received_tool_names = {t.name for t in request.tools}
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self.received_tool_names = {t.name for t in request.tools}
        return await handler(request)


def _invoke(agent, mode: str, context: dict) -> dict:
    payload = {"messages": [{"role": "user", "content": "help"}]}
    if mode == "async":
        return asyncio.run(agent.ainvoke(payload, context=context))
    return agent.invoke(payload, context=context)


def _tool_messages(result: dict) -> list[LangChainToolMessage]:
    return [m for m in result["messages"] if isinstance(m, LangChainToolMessage)]


def _tool_call_ai_message(tool_name: str, args: dict, call_id: str = "call_001") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# ── Tools used across tests ────────────────────────────────────────────────────

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return "search results"


@tool
def get_account_balance(account_id: str) -> str:
    """Get account balance."""
    return "Balance: $1,000"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return "Email sent"


@tool
def list_customer_records(query: str) -> str:
    """Return a JSON list of customer records."""
    return json.dumps([
        {"id": "r1", "name": "Alice Johnson", "doc_type": "profile"},
        {"id": "r2", "name": "Bob Smith",     "doc_type": "classified"},
        {"id": "r3", "name": "Carol Davis",   "doc_type": "profile"},
    ])


BASE_TOOLS = [search_docs, get_account_balance, send_email]
CONTEXT = {"username": "alice", "session_id": "test-session"}


# ── Tool visibility ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_visibility_removes_tools_denied_by_policy(monkeypatch, mode):
    """Denied tools are absent from the ModelRequest the model receives."""
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, **kw: {"search_docs", "get_account_balance"},
    )
    spy = ToolCapturingMiddleware()
    middleware = [LangChainAgentAuthZMiddleware(AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False))), spy]
    agent = create_agent(
        model=StubChatModel(responses=[AIMessage(content="ok")]),
        tools=BASE_TOOLS,
        middleware=middleware,
    )

    _invoke(agent, mode, CONTEXT)

    assert spy.received_tool_names == {"search_docs", "get_account_balance"}, (
        f"send_email should have been filtered out; model received: {spy.received_tool_names}"
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_visibility_disabled_sends_all_tools_to_model(monkeypatch, mode):
    """With no TOOL_VISIBILITY enforcer all tools reach the model and ADS is never called."""
    ads_calls: list = []
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda *a, **kw: ads_calls.append(True) or set(),
    )
    spy = ToolCapturingMiddleware()
    middleware = [LangChainAgentAuthZMiddleware(AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=False, invocation=False, post_invocation=False))), spy]
    agent = create_agent(
        model=StubChatModel(responses=[AIMessage(content="ok")]),
        tools=BASE_TOOLS,
        middleware=middleware,
    )

    _invoke(agent, mode, CONTEXT)

    assert not ads_calls, "get_permitted_tools should not be called when tool filtering is disabled"
    assert spy.received_tool_names == {t.name for t in BASE_TOOLS}


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_visibility_fail_open_permits_all_tools_when_ads_is_unreachable(monkeypatch, mode):
    """With fail_open=True, all tools are permitted when ADS returns None (unreachable)."""
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, fail_open=False, **kw:
            set(tool_names) if fail_open else set(),
    )
    spy = ToolCapturingMiddleware()
    middleware = [LangChainAgentAuthZMiddleware(AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080", fail_open=True), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False))), spy]
    agent = create_agent(
        model=StubChatModel(responses=[AIMessage(content="ok")]),
        tools=BASE_TOOLS,
        middleware=middleware,
    )

    _invoke(agent, mode, CONTEXT)

    assert spy.received_tool_names == {t.name for t in BASE_TOOLS}


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_visibility_fail_closed_denies_all_tools_when_ads_is_unreachable(monkeypatch, mode):
    """With fail_open=False (default), no tools are permitted when ADS returns None."""
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, fail_open=False, **kw:
            set(tool_names) if fail_open else set(),
    )
    spy = ToolCapturingMiddleware()
    middleware = [LangChainAgentAuthZMiddleware(AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080", fail_open=False), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False))), spy]
    agent = create_agent(
        model=StubChatModel(responses=[AIMessage(content="ok")]),
        tools=BASE_TOOLS,
        middleware=middleware,
    )

    _invoke(agent, mode, CONTEXT)

    assert spy.received_tool_names == set(), (
        f"All tools should be denied when ADS unreachable and fail_open=False; "
        f"got: {spy.received_tool_names}"
    )


# ── Tool invocation ────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_invocation_denied_returns_access_denied_tool_message(monkeypatch, mode):
    """When ADS denies the tool call, the ToolMessage content says access denied."""
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call",
                        decision_seam(lambda *a, **kw: False))

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=True, post_invocation=False))
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("get_account_balance", {"account_id": "ACC-999"}),
            AIMessage(content="I cannot do that."),
        ]),
        tools=BASE_TOOLS,
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    result = _invoke(agent, mode, CONTEXT)

    tool_msgs = _tool_messages(result)
    assert tool_msgs, "Expected a ToolMessage in the result"
    assert "Access denied" in tool_msgs[0].content, (
        f"Expected access-denied content; got: {tool_msgs[0].content!r}"
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_invocation_permitted_executes_tool_and_returns_real_output(monkeypatch, mode):
    """When ADS permits the call, the tool executes and returns its actual output."""
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call",
                        decision_seam(lambda *a, **kw: True))

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=True, post_invocation=False))
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("get_account_balance", {"account_id": "ACC-001"}),
            AIMessage(content="Balance retrieved."),
        ]),
        tools=BASE_TOOLS,
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    result = _invoke(agent, mode, CONTEXT)

    tool_msgs = _tool_messages(result)
    assert tool_msgs, "Expected a ToolMessage in the result"
    assert "Access denied" not in tool_msgs[0].content
    assert "Balance" in tool_msgs[0].content


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_invocation_disabled_always_executes_tool_without_asking_ads(monkeypatch, mode):
    """With no BEFORE_TOOL enforcer the tool always executes and ADS is never asked."""
    ads_calls: list = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call",
                        decision_seam(lambda *a, **kw: ads_calls.append(True) or False))

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False))
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("get_account_balance", {"account_id": "ACC-001"}),
            AIMessage(content="Done."),
        ]),
        tools=BASE_TOOLS,
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    result = _invoke(agent, mode, CONTEXT)

    assert not ads_calls, "evaluate_tool_call should not be called when authz is disabled"
    tool_msgs = _tool_messages(result)
    assert tool_msgs and "Access denied" not in tool_msgs[0].content


# ── Post-invocation ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["sync", "async"])
def test_post_invocation_removes_records_denied_by_policy(monkeypatch, mode):
    """Records denied by ADS are absent from the ToolMessage returned to the model."""
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(
        authz_helpers_module, "get_permitted_records",
        lambda _ads, *, records, **kw: [r for r in records if r.get("doc_type") != "classified"],
    )

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=True))
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("list_customer_records", {"query": "all"}),
            AIMessage(content="Here are the results."),
        ]),
        tools=[list_customer_records],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    result = _invoke(agent, mode, CONTEXT)

    tool_msgs = _tool_messages(result)
    assert tool_msgs, "Expected a ToolMessage with record results"
    returned = json.loads(tool_msgs[0].content)
    assert len(returned) == 2, f"Expected 2 records after filtering; got {len(returned)}"
    assert all(r["doc_type"] != "classified" for r in returned), (
        f"Classified record should have been removed; got: {returned}"
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_post_invocation_disabled_returns_all_records_without_asking_ads(monkeypatch, mode):
    """With no AFTER_TOOL enforcer all records are returned and ADS is never called."""
    ads_calls: list = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(authz_helpers_module, "get_permitted_records",
                        lambda *a, **kw: ads_calls.append(True) or [])

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False))
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("list_customer_records", {"query": "all"}),
            AIMessage(content="Done."),
        ]),
        tools=[list_customer_records],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    result = _invoke(agent, mode, CONTEXT)

    assert not ads_calls, "get_permitted_records should not be called when record filtering is disabled"
    tool_msgs = _tool_messages(result)
    returned = json.loads(tool_msgs[0].content)
    assert len(returned) == 3


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_post_invocation_non_json_tool_result_is_passed_through_unchanged(monkeypatch, mode):
    """Tools returning plain strings (not a JSON array) bypass record filtering entirely."""
    ads_calls: list = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(authz_helpers_module, "get_permitted_records",
                        lambda *a, **kw: ads_calls.append(True) or [])

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=True))
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("get_account_balance", {"account_id": "ACC-001"}),
            AIMessage(content="Done."),
        ]),
        tools=BASE_TOOLS,
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    result = _invoke(agent, mode, CONTEXT)

    assert not ads_calls, "get_permitted_records should not be called for non-JSON tool results"
    tool_msgs = _tool_messages(result)
    assert "Balance" in tool_msgs[0].content


@tool
def list_records_as_mcp_blocks(query: str) -> list:
    """Return records the way a langchain-mcp-adapters tool does: as text content blocks."""
    return [{"type": "text", "text": json.dumps([
        {"id": "r1", "doc_type": "profile"},
        {"id": "r2", "doc_type": "classified"},
    ]), "id": "lc_block_1"}]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_post_invocation_filters_records_inside_mcp_content_blocks(monkeypatch, mode):
    """An MCP tool's ToolMessage content is a list of blocks. The blocks are not records:
    the records inside the text block are filtered and the block is kept."""
    seen: list = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))

    def permitted(_ads, *, records, **kw):
        seen.extend(records)
        return [r for r in records if r.get("doc_type") != "classified"]

    monkeypatch.setattr(authz_helpers_module, "get_permitted_records", permitted)

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=True))
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("list_records_as_mcp_blocks", {"query": "all"}),
            AIMessage(content="Done."),
        ]),
        tools=[list_records_as_mcp_blocks],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    result = _invoke(agent, mode, CONTEXT)

    assert [r["id"] for r in seen] == ["r1", "r2"]       # the records, never the blocks
    content = _tool_messages(result)[0].content
    assert isinstance(content, list) and content[0]["type"] == "text"
    assert json.loads(content[0]["text"]) == [{"id": "r1", "doc_type": "profile"}]


# ── Custom context provider ────────────────────────────────────────────────────
#
# The motivation: the default provider reads well-known context keys (username,
# user_id, tenant_id, session_id). A custom provider is needed when the runtime
# context uses different field names or when you want to inject additional
# attributes (e.g. region, classification) into any XACML category.
#
# The subject dict can have any key — the full dict is forwarded to get_permitted_tools
# as the ``subject`` parameter, so {"service_account": "svc-billing"} is passed as-is.


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_custom_provider_maps_nonstandard_subject_fields(monkeypatch, mode):
    """
    Custom provider reads non-standard context fields and maps them to XACML
    subject attributes. The mapped subject dict must reach get_permitted_tools.

    Use case: the runtime context uses ``external_id`` (not ``username`` or
    ``user_id``), so the default provider would produce an empty subject.
    """
    captured_subjects: list[dict] = []

    class MyProvider(DefaultClientContextProvider):
        def get_subject(self, context):
            # Use external_id as the primary identifier; also forward department.
            return {
                "external_id": context.get("external_id", ""),
                "department": context.get("department", ""),
            }

    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, subject, tool_names, **kw:
            captured_subjects.append(subject) or set(tool_names),
    )
    spy = ToolCapturingMiddleware()
    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False), context_provider=MyProvider())
    agent = create_agent(
        model=StubChatModel(responses=[AIMessage(content="ok")]),
        tools=BASE_TOOLS,
        middleware=[LangChainAgentAuthZMiddleware(config), spy],
    )

    _invoke(agent, mode, {"external_id": "ext-user-99", "department": "finance"})

    assert captured_subjects, "get_permitted_tools was never called"
    assert captured_subjects[0] == {"external_id": "ext-user-99", "department": "finance"}, (
        f"Expected subject dict from custom provider; got {captured_subjects[0]!r}"
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_custom_provider_injects_extra_resource_attributes_from_context(monkeypatch, mode):
    """
    Custom provider adds extra resource attributes drawn from the runtime context.

    Use case: a ``classification`` field in the context should be forwarded as
    a resource attribute so the ADS policy can restrict calls by data classification.
    """
    captured_resources: list[dict] = []

    class MyProvider(DefaultClientContextProvider):
        def get_resource(self, resource_name, context, *, args=None, resource_name_attribute):
            resource = super().get_resource(
                resource_name, context, args=args, resource_name_attribute=resource_name_attribute,
            )
            classification = context.get("classification") if isinstance(context, dict) else None
            if classification:
                resource["classification"] = classification
            return resource

    def capturing_permit(*a, resource, **kw):
        captured_resources.append(resource)
        return True

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(capturing_permit))

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=True, post_invocation=False), context_provider=MyProvider())
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("get_account_balance", {"account_id": "ACC-001"}),
            AIMessage(content="Done."),
        ]),
        tools=BASE_TOOLS,
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    _invoke(agent, mode, {"username": "alice", "classification": "confidential"})

    assert captured_resources, "evaluate_tool_call was never called"
    resource = captured_resources[0]
    assert resource["tool_name"] == "get_account_balance"
    assert resource["account_id"] == "ACC-001"
    assert resource["classification"] == "confidential"


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_custom_provider_injects_environment_attributes_from_context(monkeypatch, mode):
    """
    Custom provider reads additional fields from context into the XACML environment.

    Use case: a ``region`` field in the context should be forwarded as an
    environment attribute so region-based access policies can be enforced.
    """
    captured_envs: list[dict] = []

    class MyProvider(DefaultClientContextProvider):
        def get_environment(self, context):
            env = super().get_environment(context)
            region = context.get("region") if isinstance(context, dict) else None
            if region:
                env["region"] = region
            return env

    def capturing_permit(*a, environment, **kw):
        captured_envs.append(environment)
        return True

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(capturing_permit))

    config = AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080"), enforcement=standard_tool_enforcement(visibility=True, invocation=True, post_invocation=False), context_provider=MyProvider())
    agent = create_agent(
        model=StubChatModel(responses=[
            _tool_call_ai_message("search_docs", {"query": "hello"}),
            AIMessage(content="Done."),
        ]),
        tools=BASE_TOOLS,
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    _invoke(agent, mode, {"username": "alice", "session_id": "s-1", "region": "eu-west-1"})

    assert captured_envs, "evaluate_tool_call was never called"
    assert captured_envs[0].get("region") == "eu-west-1"
    assert captured_envs[0].get("session_id") == "s-1"
