"""
The acting agent's name in the subject: ``resolve_authz_context(..., agent_name=,
agent_attribute=)`` and ``AttributeNames.acting_agent``.
"""
from __future__ import annotations

from axiomatics_ai import AttributeNames, AuthZConfig
from axiomatics_ai.core.authz_helpers import resolve_authz_context


class _Provider:
    def __init__(self, subject, environment=None):
        self.subject, self.environment = subject, environment or {}

    def get_subject(self, context):
        return dict(self.subject)

    def get_environment(self, context):
        return dict(self.environment)


def _resolve(subject, **kwargs):
    return resolve_authz_context(_Provider(subject), None, **kwargs)


def test_added_to_a_non_empty_subject():
    resolved = _resolve({"user_id": "alice"}, agent_name="billing", agent_attribute="agent_name")
    assert resolved.subject == {"agent_name": "billing", "user_id": "alice"}


def test_the_providers_own_value_wins():
    resolved = _resolve({"user_id": "alice", "agent_name": "from-provider"},
                        agent_name="billing", agent_attribute="agent_name")
    assert resolved.subject["agent_name"] == "from-provider"


def test_not_added_to_an_empty_subject():
    assert _resolve({}, agent_name="billing", agent_attribute="agent_name").subject == {}


def test_a_subject_of_only_none_values_counts_as_empty():
    assert _resolve({"user_id": None}, agent_name="billing", agent_attribute="agent_name").subject == {}


def test_not_added_without_an_attribute():
    assert _resolve({"user_id": "alice"}, agent_name="billing", agent_attribute=None).subject == {"user_id": "alice"}


def test_not_added_without_a_name():
    assert _resolve({"user_id": "alice"}, agent_name=None, agent_attribute="agent_name").subject == {"user_id": "alice"}


def test_without_the_keywords_nothing_changes():
    resolved = resolve_authz_context(_Provider({"user_id": "alice"}, {"channel": "chat", "x": None}), None)
    assert resolved.subject == {"user_id": "alice"}
    assert resolved.environment == {"channel": "chat"}


def test_a_custom_attribute():
    resolved = _resolve({"user_id": "alice"}, agent_name="billing", agent_attribute="acting_agent")
    assert resolved.subject == {"acting_agent": "billing", "user_id": "alice"}


def test_attribute_names_defaults():
    names = AttributeNames()
    assert names.acting_agent == "agent_name"
    assert (names.tool, names.agent, names.mcp_server) == ("tool_name", "agent_name", "mcp_server")
    assert AuthZConfig().attribute_names == names


def test_acting_agent_can_be_turned_off():
    assert AttributeNames(acting_agent=None).acting_agent is None
