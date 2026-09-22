"""
Unit tests for the default context providers' resource dict: the identifier is sent as
``resource_name_attribute``, invocation arguments are merged in, and an argument that shares
the identifier's name is dropped (with a warning) so the model can't choose the resource.
"""
from __future__ import annotations

import logging

import pytest

from axiomatics_ai.core.config import DefaultClientContextProvider, DefaultServerContextProvider

_LOGGER = "axiomatics_ai.core.config"


@pytest.fixture(params=["client", "server"])
def provider(request):
    return DefaultClientContextProvider() if request.param == "client" else DefaultServerContextProvider()


def test_identifier_sent_as_resource_name_attribute(provider):
    assert provider.get_resource(
        "search_docs", {}, args={"query": "q"}, resource_name_attribute="tool_name",
    ) == {"tool_name": "search_docs", "query": "q"}


def test_none_attribute_sends_no_identifier(provider):
    assert provider.get_resource("before_model", {}, resource_name_attribute=None) == {}
    assert provider.get_resource(
        "before_model", {}, args={"x": 1}, resource_name_attribute=None,
    ) == {"x": 1}


def test_colliding_arg_is_dropped_with_warning(provider, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resource = provider.get_resource(
            "search_docs", {}, args={"tool_name": "delete_everything", "query": "q"},
            resource_name_attribute="tool_name",
        )
    assert resource == {"tool_name": "search_docs", "query": "q"}   # identifier wins
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "tool_name" in warnings[0].getMessage() and "search_docs" in warnings[0].getMessage()


def test_no_warning_without_collision(provider, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        provider.get_resource("search_docs", {}, args={"name": "x"}, resource_name_attribute="tool_name")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# ── Subject: agent_id from context; unverified-token warning ─────────────────────

def test_client_subject_includes_agent_id():
    subject = DefaultClientContextProvider().get_subject(
        {"username": "alice", "agent_id": "spiffe://example.org/agent/a", "other": "x"},
    )
    assert subject == {"username": "alice", "agent_id": "spiffe://example.org/agent/a"}


def test_client_subject_omits_missing_agent_id():
    assert "agent_id" not in DefaultClientContextProvider().get_subject({"username": "alice"})


def test_server_provider_warns_once_about_unverified_tokens(monkeypatch, caplog):
    import axiomatics_ai.core.config as config_module

    monkeypatch.setattr(config_module, "_warned_unverified_tokens", False)
    provider = DefaultServerContextProvider()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        provider.get_subject({})
        provider.get_subject({})
        DefaultServerContextProvider().get_subject({})
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "WITHOUT verification" in warnings[0] and "AgentIdentityContextProvider" in warnings[0]
