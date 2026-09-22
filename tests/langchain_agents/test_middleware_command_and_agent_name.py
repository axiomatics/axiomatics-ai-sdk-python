"""
Tests for LangChainAgentAuthZMiddleware post-invocation on ``Command`` results (a tool that
updates the graph state, like deepagents' ``task``), tools whose output is left unchecked,
and the ``agent_name`` it adds to the subject.

The PDP is stubbed at ``get_permitted_records`` / ``get_permitted_tools`` — no real ADS.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

import axiomatics_ai.core.authz_helpers as authz_helpers_module
import axiomatics_ai.core.enforcement as enforcement_module
from axiomatics_ai import AuthZConfig, LangChainAgentAuthZMiddleware, PDPConnection, standard_tool_enforcement

from .test_middleware_enforcement import CONTEXT, StubChatModel, _invoke, search_docs

RECORDS = [{"id": "r1", "doc_type": "profile"}, {"id": "r2", "doc_type": "classified"}]
KEPT = [{"id": "r1", "doc_type": "profile"}]


def _post_only_config() -> AuthZConfig:
    return AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(visibility=False, invocation=False, post_invocation=True),
    )


def _request(tool_name: str = "task") -> SimpleNamespace:
    return SimpleNamespace(
        tool_call={"name": tool_name, "args": {}, "id": "call_1", "type": "tool_call"},
        runtime=SimpleNamespace(context=CONTEXT),
    )


@pytest.fixture
def filtering(monkeypatch):
    """Stub the PDP: classified records are denied; returns the list of batches asked."""
    batches: list = []

    def permitted(_ads, *, records, **kw):
        batches.append(records)
        return [r for r in records if r.get("doc_type") != "classified"]

    monkeypatch.setattr(authz_helpers_module, "get_permitted_records", permitted)
    return batches


def _records_message(call_id: str = "call_1") -> ToolMessage:
    return ToolMessage(content=json.dumps(RECORDS), tool_call_id=call_id)


# ── Command results ───────────────────────────────────────────────────────────


def test_command_tool_messages_are_post_processed_and_other_update_keys_kept(filtering):
    mw = LangChainAgentAuthZMiddleware(_post_only_config())
    other = AIMessage(content="not a tool message")
    command = Command(update={"messages": [_records_message(), other], "files": {"a.txt": "x"}})

    out = mw.wrap_tool_call(_request(), lambda req: command)

    assert isinstance(out, Command) and out is not command
    assert out.update["files"] == {"a.txt": "x"}
    tool_msg, kept_other = out.update["messages"]
    assert json.loads(tool_msg.content) == KEPT and tool_msg.tool_call_id == "call_1"
    assert kept_other is other
    assert len(filtering) == 1
    # The original Command is not mutated.
    assert json.loads(command.update["messages"][0].content) == RECORDS


def test_command_without_messages_is_returned_unchanged(filtering):
    mw = LangChainAgentAuthZMiddleware(_post_only_config())
    command = Command(update={"files": {"a.txt": "x"}})
    assert mw.wrap_tool_call(_request(), lambda req: command) is command
    goto = Command(goto="somewhere")
    assert mw.wrap_tool_call(_request(), lambda req: goto) is goto
    assert filtering == []


def test_unchecked_output_tools_are_not_post_processed(filtering):
    mw = LangChainAgentAuthZMiddleware(_post_only_config())
    mw._unchecked_output_tools = frozenset({"task"})
    msg = _records_message()
    command = Command(update={"messages": [msg]})
    assert mw.wrap_tool_call(_request("task"), lambda req: msg) is msg
    assert mw.wrap_tool_call(_request("task"), lambda req: command) is command
    assert filtering == []
    # Another tool is still checked.
    out = mw.wrap_tool_call(_request("list_docs"), lambda req: _records_message())
    assert json.loads(out.content) == KEPT


def test_awrap_tool_call_post_processes_on_a_worker_thread(filtering):
    mw = LangChainAgentAuthZMiddleware(_post_only_config())

    async def handler(req):
        return _records_message()

    async def handler_command(req):
        return Command(update={"messages": [_records_message()]})

    out = asyncio.run(mw.awrap_tool_call(_request("list_docs"), handler))
    assert json.loads(out.content) == KEPT
    out = asyncio.run(mw.awrap_tool_call(_request("task"), handler_command))
    assert isinstance(out, Command) and json.loads(out.update["messages"][0].content) == KEPT
    assert len(filtering) == 2


# ── agent_name in the subject ─────────────────────────────────────────────────


def _visibility_subjects(monkeypatch, agent_name, context) -> list[dict]:
    subjects: list = []

    def permitted_tools(_ads, *, subject, tool_names, **kw):
        subjects.append(subject)
        return set(tool_names)

    monkeypatch.setattr(enforcement_module, "get_permitted_tools", permitted_tools)
    config = AuthZConfig(
        connection=PDPConnection(base_url="http://localhost:8080"),
        enforcement=standard_tool_enforcement(visibility=True, invocation=False, post_invocation=False),
    )
    agent = create_agent(
        model=StubChatModel(responses=[AIMessage(content="Hello.")]),
        tools=[search_docs],
        middleware=[LangChainAgentAuthZMiddleware(config, agent_name=agent_name)],
    )
    _invoke(agent, "sync", context)
    return subjects


def test_agent_name_is_added_to_the_subject(monkeypatch):
    (subject,) = _visibility_subjects(monkeypatch, "researcher", CONTEXT)
    assert subject["agent_name"] == "researcher"
    assert subject["username"] == "alice"


def test_no_agent_name_leaves_the_subject_alone(monkeypatch):
    (subject,) = _visibility_subjects(monkeypatch, None, CONTEXT)
    assert "agent_name" not in subject


def test_agent_name_is_not_added_to_an_empty_subject():
    mw = LangChainAgentAuthZMiddleware(_post_only_config(), agent_name="researcher")
    assert mw._resolve({}).subject == {}
    assert mw._resolve(CONTEXT).subject["agent_name"] == "researcher"
