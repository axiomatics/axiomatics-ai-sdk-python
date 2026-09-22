"""
Tests for typed / multi-valued XACML attribute serialization and the
provider-controlled action attribute name.

Focus is the tool-visibility (batch MultiRequest) path, with single-request
coverage for the shared serialization helper. Rather than stubbing the return
value, these tests capture the request handed to ``_authorize`` and assert on the
XACML that the SDK emits — no real ADS instance is required.
"""
from __future__ import annotations


import logging

import pytest

import axiomatics_ai.core.pdp.attributes as attributes_module
from axiomatics_ai.core.pdp import PdpClient, authzen, flatten_to_provided_attributes, sendable_attributes, xacml
from axiomatics_ai.core.pdp.attributes import is_sendable
from axiomatics_ai.core.pdp.xacml import _ACTION_CAT, _ENV_CAT, _RESOURCE_CAT, _SUBJECT_CAT

_STRING = "http://www.w3.org/2001/XMLSchema#string"
_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"
_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"


def _capture(monkeypatch):
    """Return (client, captured) where captured['request'] is the JSON sent to ADS.

    ``_authorize`` is stubbed to record the request and return ``None``. The request
    is built before that call, so triggering any check populates ``captured['request']``
    as a side effect — that dict, not the check's return value, is what the tests assert on.
    """
    client = PdpClient(base_url="http://ads-stub")
    captured: dict = {}

    def fake_authorize(self, request, *, endpoint="/authorize"):
        captured["request"] = request
        return

    monkeypatch.setattr(PdpClient, "_authorize", fake_authorize)
    return client, captured


def _batch_section_attrs(request: dict, section: str) -> dict:
    """AttributeId → attribute dict for the first entry of a batch section."""
    entries = request["Request"][section]
    return {a["AttributeId"]: a for a in entries[0]["Attribute"]}


def _category_attrs(request: dict, category_id: str) -> dict:
    """AttributeId → attribute dict for a single-request category."""
    for category in request["Request"]["Category"]:
        if category["CategoryId"] == category_id:
            return {a["AttributeId"]: a for a in category["Attribute"]}
    return {}


def _run_batch(client, *, subject, action, resource_names=("search_docs",)):
    """Trigger a batch check so the stubbed _authorize captures the built request."""
    client.check_permission_batch(
        subject=subject,
        action=action,
        resource_names=list(resource_names),
        environment={},
        resource_attr_id="tool_name",
    )


# ── Tool visibility (batch) — typed & multi-valued subject attributes ──────────

class TestBatchTypedAttributes:
    def test_string_attribute(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"username": "alice"}, action={"name": "tool_visibility"})
        attr = _batch_section_attrs(captured["request"], "AccessSubject")["username"]
        assert attr["DataType"] == _STRING
        assert attr["Value"] == ["alice"]

    def test_list_of_strings_is_multivalued(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"roles": ["admin", "ops"]}, action={"name": "tool_visibility"})
        attr = _batch_section_attrs(captured["request"], "AccessSubject")["roles"]
        assert attr["DataType"] == _STRING
        assert attr["Value"] == ["admin", "ops"]

    def test_int_attribute(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"clearance_level": 5}, action={"name": "tool_visibility"})
        attr = _batch_section_attrs(captured["request"], "AccessSubject")["clearance_level"]
        assert attr["DataType"] == _INTEGER
        assert attr["Value"] == [5]
        assert isinstance(attr["Value"][0], int)

    def test_float_attribute(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"risk_score": 1.5}, action={"name": "tool_visibility"})
        attr = _batch_section_attrs(captured["request"], "AccessSubject")["risk_score"]
        assert attr["DataType"] == _DOUBLE
        assert attr["Value"] == [1.5]

    def test_bool_attribute(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"is_admin": True}, action={"name": "tool_visibility"})
        attr = _batch_section_attrs(captured["request"], "AccessSubject")["is_admin"]
        assert attr["DataType"] == _BOOLEAN
        assert attr["Value"] == [True]

    def test_bool_is_not_serialized_as_integer(self, monkeypatch):
        """Regression: bool is a subclass of int, so it must be checked first."""
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"is_admin": False, "level": 0}, action={"name": "tool_visibility"})
        attrs = _batch_section_attrs(captured["request"], "AccessSubject")
        assert attrs["is_admin"]["DataType"] == _BOOLEAN
        assert attrs["level"]["DataType"] == _INTEGER

    def test_mixed_types_in_one_subject(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(
            client,
            subject={"username": "alice", "roles": ["a", "b"], "level": 3, "score": 0.5, "active": True},
            action={"name": "tool_visibility"},
        )
        attrs = _batch_section_attrs(captured["request"], "AccessSubject")
        assert attrs["username"]["DataType"] == _STRING and attrs["username"]["Value"] == ["alice"]
        assert attrs["roles"]["DataType"] == _STRING and attrs["roles"]["Value"] == ["a", "b"]
        assert attrs["level"]["DataType"] == _INTEGER and attrs["level"]["Value"] == [3]
        assert attrs["score"]["DataType"] == _DOUBLE and attrs["score"]["Value"] == [0.5]
        assert attrs["active"]["DataType"] == _BOOLEAN and attrs["active"]["Value"] == [True]


# ── Tool visibility (batch) — provider-controlled action attribute name ────────

class TestBatchActionAttributeName:
    def test_default_action_attr_id_is_name(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"username": "alice"}, action={"name": "tool_visibility"})
        attrs = _batch_section_attrs(captured["request"], "Action")
        assert "name" in attrs
        assert attrs["name"]["Value"] == ["tool_visibility"]

    def test_custom_action_attr_id_from_action_dict(self, monkeypatch):
        """The action dict's key becomes the AttributeId — no extra parameter or config."""
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"username": "alice"}, action={"action_id": "tool_visibility"})
        attrs = _batch_section_attrs(captured["request"], "Action")
        assert set(attrs) == {"action_id"}
        assert attrs["action_id"]["Value"] == ["tool_visibility"]

    def test_action_dict_supports_extra_and_typed_attributes(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(
            client,
            subject={"username": "alice"},
            action={"name": "browse_tools", "risk_tier": 2},
        )
        attrs = _batch_section_attrs(captured["request"], "Action")
        assert attrs["name"]["Value"] == ["browse_tools"]
        assert attrs["risk_tier"]["DataType"] == _INTEGER and attrs["risk_tier"]["Value"] == [2]


# ── Tool visibility (batch) — resource entry still correct ─────────────────────

class TestBatchResourceEntry:
    def test_resource_name_is_included_in_result(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        _run_batch(client, subject={"username": "alice"}, action={"name": "tool_visibility"},
                   resource_names=["search_docs"])
        resource_attr = _batch_section_attrs(captured["request"], "Resource")["tool_name"]
        assert resource_attr["IncludeInResult"] is True
        assert resource_attr["Value"] == ["search_docs"]


# ── Configurable resource attribute key (tool visibility & record filtering) ───

class TestConfigurableResourceAttrId:
    def test_tool_visibility_tool_name_attribute(self, monkeypatch):
        from axiomatics_ai.core.authz_helpers import get_permitted_tools

        client, captured = _capture(monkeypatch)
        get_permitted_tools(
            client,
            subject={"username": "alice"},
            action={"name": "tool_visibility"},
            tool_names=["search_docs"],
            environment={},
            tool_name_attribute="tool_id",
        )
        attrs = _batch_section_attrs(captured["request"], "Resource")
        assert set(attrs) == {"tool_id"}
        assert attrs["tool_id"]["IncludeInResult"] is True
        assert attrs["tool_id"]["Value"] == ["search_docs"]

    def test_record_filter_key_attribute(self, monkeypatch):
        from axiomatics_ai.core.authz_helpers import get_permitted_records

        client, captured = _capture(monkeypatch)
        get_permitted_records(
            client,
            records=[{"doc_type": "public"}],
            subject={"username": "alice"},
            action={"name": "post_invocation"},
            environment={},
            key_attribute="row_key",
        )
        attrs = _batch_section_attrs(captured["request"], "Resource")
        # The correlation key is renamed; record fields still ride alongside it.
        assert "row_key" in attrs and "record_hash" not in attrs
        assert attrs["row_key"]["IncludeInResult"] is True
        assert attrs["doc_type"]["Value"] == ["public"]

    def test_record_filter_strategy_key_attribute(self, monkeypatch):
        from axiomatics_ai.core.post_invocation import PostInvocationContext
        from axiomatics_ai.core.presets import RecordFilterStrategy

        client, captured = _capture(monkeypatch)
        ctx = PostInvocationContext(
            pdp_client=client, subject={"username": "alice"}, environment={},
            tool_name="list_docs", args={}, resource={"tool_name": "list_docs"},
            action={"name": "post_invocation"}, fail_open=False,
        )
        RecordFilterStrategy(key_attribute="row").apply([{"doc_type": "public"}], ctx)
        attrs = _batch_section_attrs(captured["request"], "Resource")
        assert "row" in attrs and "record_hash" not in attrs
        assert attrs["row"]["IncludeInResult"] is True
        # Only the correlation key is IncludeInResult; record fields ride alongside it.
        assert [k for k, a in attrs.items() if a.get("IncludeInResult")] == ["row"]
        assert attrs["doc_type"]["Value"] == ["public"]

    def test_record_filter_strategy_default_key_attribute(self, monkeypatch):
        from axiomatics_ai.core.post_invocation import PostInvocationContext
        from axiomatics_ai.core.presets import RecordFilterStrategy

        client, captured = _capture(monkeypatch)
        ctx = PostInvocationContext(
            pdp_client=client, subject={"username": "alice"}, environment={},
            tool_name="list_docs", args={}, resource={"tool_name": "list_docs"},
            action={"name": "post_invocation"}, fail_open=False,
        )
        RecordFilterStrategy().apply([{"doc_type": "public"}], ctx)
        assert "record_hash" in _batch_section_attrs(captured["request"], "Resource")

    def test_defaults_unchanged(self, monkeypatch):
        from axiomatics_ai.core.authz_helpers import get_permitted_tools

        client, captured = _capture(monkeypatch)
        get_permitted_tools(
            client,
            subject={"username": "alice"},
            action={"name": "tool_visibility"},
            tool_names=["search_docs"],
            environment={},
        )
        assert "tool_name" in _batch_section_attrs(captured["request"], "Resource")


# ── Record filtering (post-invocation) — provider-controlled action key ────────

class TestRecordFilteringActionAttribute:
    def test_custom_action_attr_id_reaches_record_filter_request(self, monkeypatch):
        """get_permitted_records forwards the action dict, so its key controls the AttributeId."""
        from axiomatics_ai.core.authz_helpers import get_permitted_records

        client, captured = _capture(monkeypatch)
        get_permitted_records(
            client,
            records=[{"id": "r1"}, {"id": "r2"}],
            subject={"username": "alice"},
            action={"operation": "post_invocation"},
            environment={},
        )
        attrs = _batch_section_attrs(captured["request"], "Action")
        assert set(attrs) == {"operation"}
        assert attrs["operation"]["Value"] == ["post_invocation"]

    def test_record_fields_are_typed(self, monkeypatch):
        """Record fields forwarded as extra resource attributes are typed, not stringified."""
        from axiomatics_ai.core.authz_helpers import get_permitted_records

        client, captured = _capture(monkeypatch)
        get_permitted_records(
            client,
            records=[{"amount": 100, "public": True, "tags": ["a", "b"]}],
            subject={"username": "alice"},
            action={"name": "post_invocation"},
            environment={},
        )
        attrs = _batch_section_attrs(captured["request"], "Resource")
        assert attrs["amount"]["DataType"] == _INTEGER and attrs["amount"]["Value"] == [100]
        assert attrs["public"]["DataType"] == _BOOLEAN and attrs["public"]["Value"] == [True]
        assert attrs["tags"]["DataType"] == _STRING and attrs["tags"]["Value"] == ["a", "b"]


# ── Single request — shared serialization helper ───────────────────────────────

class TestSingleRequestTypedAttributes:
    def _check(self, client, **kwargs):
        client.check_permission(
            # An empty subject never reaches the PDP, so every request carries one.
            subject=kwargs.get("subject", {"username": "alice"}),
            action=kwargs.get("action", {}),
            resource=kwargs.get("resource", {}),
            environment=kwargs.get("environment", {}),
        )

    def test_typed_subject_attributes(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        self._check(client, subject={"level": 5, "active": True, "name": "alice"},
                    action={"name": "tool_invocation"})
        attrs = _category_attrs(captured["request"], _SUBJECT_CAT)
        assert attrs["level"]["DataType"] == _INTEGER and attrs["level"]["Value"] == [5]
        assert attrs["active"]["DataType"] == _BOOLEAN and attrs["active"]["Value"] == [True]
        assert attrs["name"]["DataType"] == _STRING and attrs["name"]["Value"] == ["alice"]

    def test_list_of_strings_in_resource(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        self._check(client, action={"name": "read"}, resource={"tags": ["finance", "pii"]})
        attr = _category_attrs(captured["request"], _RESOURCE_CAT)["tags"]
        assert attr["DataType"] == _STRING
        assert attr["Value"] == ["finance", "pii"]

    def test_custom_action_attribute_name_via_dict(self, monkeypatch):
        """Single requests take the action key straight from the provider's dict."""
        client, captured = _capture(monkeypatch)
        self._check(client, action={"operation": "read"})
        attrs = _category_attrs(captured["request"], _ACTION_CAT)
        assert set(attrs) == {"operation"}
        assert attrs["operation"]["Value"] == ["read"]

    def test_float_in_environment(self, monkeypatch):
        client, captured = _capture(monkeypatch)
        self._check(client, action={"name": "read"}, environment={"request_score": 0.25})
        attr = _category_attrs(captured["request"], _ENV_CAT)["request_score"]
        assert attr["DataType"] == _DOUBLE and attr["Value"] == [0.25]


# ── Unsendable values (objects, nested / mixed lists) are dropped ─────────────

_UNSENDABLE = {
    "obj": {"a": 1},
    "nested": [[1], [2]],
    "mixed": [1, "a"],
    "bool_and_int": [True, 1],
    "int_and_float": [1, 2.5],
    "list_with_none": ["a", None],
    "list_with_obj": [{"a": 1}],
}
_SENDABLE = {
    "s": "x", "i": 3, "f": 0.5, "b": False,
    "empty": [], "strs": ["a", "b"], "ints": [1, 2], "floats": [0.5, 1.5], "bools": [True, False],
}
_PDP_LOGGER = "axiomatics_ai.core.pdp.attributes"


@pytest.fixture
def fresh_warnings(monkeypatch):
    """Reset the once-per-attribute warning registry so each test sees its own warnings."""
    monkeypatch.setattr(attributes_module, "_warned_unsendable", set())


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == _PDP_LOGGER]


class TestSendableAttributes:
    @pytest.mark.parametrize("name", sorted(_SENDABLE))
    def test_sendable(self, name):
        assert is_sendable(_SENDABLE[name])

    @pytest.mark.parametrize("name", sorted(_UNSENDABLE))
    def test_unsendable(self, name):
        assert not is_sendable(_UNSENDABLE[name])

    def test_none_and_bad_values_dropped(self, fresh_warnings):
        assert sendable_attributes({**_SENDABLE, **_UNSENDABLE, "none": None}) == _SENDABLE
        assert sendable_attributes(None) == {}

    def test_warns_once_per_attribute_and_not_for_none(self, fresh_warnings, caplog):
        with caplog.at_level(logging.WARNING, logger=_PDP_LOGGER):
            sendable_attributes({"obj": {"a": 1}, "none": None})
            sendable_attributes({"obj": {"b": 2}, "mixed": [1, "a"]})
        warnings = _warnings(caplog)
        assert len(warnings) == 2
        assert "'obj'" in warnings[0] and "dict" in warnings[0]
        assert "'mixed'" in warnings[1]

    def test_xacml_category(self, fresh_warnings):
        category = xacml._category(_SUBJECT_CAT, {"ok": "x", "empty": [], **_UNSENDABLE})
        assert [a["AttributeId"] for a in category["Attribute"]] == ["ok", "empty"]

    def test_xacml_category_entry(self, fresh_warnings):
        entry = xacml._category_entry("AccessSubject", {"ok": 1, "obj": {"a": 1}, "none": None})
        assert [a["AttributeId"] for a in entry["Attribute"]] == ["ok"]

    def test_xacml_resource_entry_extras(self, fresh_warnings):
        entry = xacml._resource_entry("r1", "record_hash", {"owner": "alice", "meta": {"a": 1}, "mixed": [1, "a"]})
        assert [a["AttributeId"] for a in entry["Attribute"]] == ["record_hash", "owner"]

    def test_authzen_properties(self, fresh_warnings):
        assert authzen._properties({"ok": [1, 2], "none": None, **_UNSENDABLE}) == {"ok": [1, 2]}

    def test_flatten_to_provided_attributes(self, fresh_warnings):
        assert flatten_to_provided_attributes({"a": "x", "obj": {"a": 1}}, None, {"b": [], "c": [1, "a"]}) == [
            {"attribute": "a", "values": ["x"]},
            {"attribute": "b", "values": []},
        ]

    def test_single_request_end_to_end(self, monkeypatch, fresh_warnings):
        client, captured = _capture(monkeypatch)
        client.check_permission(
            subject={"user": "alice", "claims": {"roles": ["x"]}},
            action={"name": "read"},
            resource={"tool_name": "t", "args": [[1, 2]]},
            environment={"mix": [True, 1], "ok": 1.5},
        )
        request = captured["request"]
        assert set(_category_attrs(request, _SUBJECT_CAT)) == {"user"}
        assert set(_category_attrs(request, _RESOURCE_CAT)) == {"tool_name"}
        assert set(_category_attrs(request, _ENV_CAT)) == {"ok"}

    def test_batch_end_to_end(self, monkeypatch, fresh_warnings):
        client, captured = _capture(monkeypatch)
        client.check_permission_batch(
            subject={"user": "alice", "obj": {"a": 1}}, action={"name": "vis"},
            resource_names=["a"], environment={"mix": [1, "x"]}, resource_attr_id="tool_name",
            resource_extra_attrs=[{"meta": {"k": "v"}, "tag": "t"}],
        )
        request = captured["request"]
        assert set(_batch_section_attrs(request, "AccessSubject")) == {"user"}
        assert set(_batch_section_attrs(request, "Resource")) == {"tool_name", "tag"}
        assert _batch_section_attrs(request, "Environment") == {}

    def test_authzen_end_to_end(self, monkeypatch, fresh_warnings):
        client = PdpClient(base_url="http://ads-stub", protocol="authzen")
        captured: dict = {}

        def fake_authorize(self, request, *, endpoint="/authorize"):
            captured["request"] = request

        monkeypatch.setattr(PdpClient, "_authorize", fake_authorize)
        client.check_permission(
            subject={"user": "alice", "obj": {"a": 1}}, action={"name": "read"},
            resource={"tool_name": "t", "mix": [1, "a"]}, environment={"ok": True, "nested": [[1]]},
        )
        text = repr(captured["request"])
        assert "alice" in text and "'ok': True" in text
        for dropped in ("'obj'", "'mix'", "'nested'"):
            assert dropped not in text
