"""
Tests for RecordFilterStrategy, ColumnMaskStrategy, and PostInvocationContext.

All ADS calls are stubbed at the pdp_client level — no real ADS instance required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from axiomatics_ai import standard_tool_enforcement
from axiomatics_ai.core.post_invocation import PostInvocationContext
from axiomatics_ai.core.presets import ColumnMaskStrategy, RecordFilterStrategy
from tests.pdp_fakes import decision_seam


# ── Helpers ────────────────────────────────────────────────────────────────────


def _pdp_client(decision: dict | None = None, *, permitted: bool = True) -> MagicMock:
    """Return a mock PdpClient."""
    client = MagicMock()
    client.authorize.return_value = decision
    client.check_permission.return_value = permitted
    return client


def _context(
    pdp_client=None,
    action_name: str = "post_invocation",
    fail_open: bool = False,
    tool_name: str = "my_tool",
    **kwargs,
) -> PostInvocationContext:
    return PostInvocationContext(
        pdp_client=pdp_client or _pdp_client(),
        subject={"username": "alice"},
        environment={},
        tool_name=tool_name,
        args={},
        resource={"tool_name": tool_name},
        action={"name": action_name},
        fail_open=fail_open,
    )


# ── RecordFilterStrategy ───────────────────────────────────────────────────────


class TestRecordFilterStrategy:
    def test_filters_permitted_records(self, monkeypatch):
        records = [
            {"id": "r1", "value": "a"},
            {"id": "r2", "value": "b"},
            {"id": "r3", "value": "c"},
        ]
        strategy = RecordFilterStrategy()
        client = _pdp_client()

        # Stub get_permitted_records to return only first two records
        import axiomatics_ai.core.authz_helpers as helpers_module
        monkeypatch.setattr(
            helpers_module,
            "get_permitted_records",
            lambda _client, *, records, **kw: records[:2],
        )

        ctx = _context(pdp_client=client)
        result = strategy.apply(records, ctx)
        assert result == records[:2]

    def test_non_container_returned_unchanged(self):
        strategy = RecordFilterStrategy()
        ctx = _context()
        assert strategy.apply("plain string", ctx) == "plain string"
        assert strategy.apply(42, ctx) == 42

    def test_empty_list_passed_through(self, monkeypatch):
        strategy = RecordFilterStrategy()
        import axiomatics_ai.core.authz_helpers as helpers_module
        monkeypatch.setattr(
            helpers_module,
            "get_permitted_records",
            lambda _client, *, records, **kw: records,
        )
        ctx = _context()
        result = strategy.apply([], ctx)
        assert result == []



# ── ColumnMaskStrategy ─────────────────────────────────────────────────────────


    def test_container_dict_filters_every_record_list(self, monkeypatch):
        import axiomatics_ai.core.authz_helpers as helpers_module
        monkeypatch.setattr(
            helpers_module, "get_permitted_records",
            lambda _client, *, records, **kw: [r for r in records if r.get("ok")],
        )
        result = RecordFilterStrategy().apply(
            {"a": [{"ok": True}, {"ok": False}], "b": [{"ok": False}], "count": 3}, _context(),
        )
        assert result == {"a": [{"ok": True}], "b": [], "count": 3}

    def test_single_record_is_kept_or_replaced(self, monkeypatch):
        import axiomatics_ai.core.authz_helpers as helpers_module
        monkeypatch.setattr(
            helpers_module, "get_permitted_records",
            lambda _client, *, records, **kw: [r for r in records if r.get("ok")],
        )
        assert RecordFilterStrategy().apply({"ok": True, "ssn": "1"}, _context()) == {"ok": True, "ssn": "1"}
        assert RecordFilterStrategy().apply({"ok": False, "ssn": "1"}, _context()) == {}


class TestColumnMaskStrategy:
    def test_permit_returns_full_result(self):
        records = [{"name": "Alice", "ssn": "123-45-6789"}]
        strategy = ColumnMaskStrategy("ssn")
        ctx = _context(pdp_client=_pdp_client(permitted=True))
        result = strategy.apply(records, ctx)
        assert result == records

    def test_deny_masks_fields_in_list_of_dicts(self):
        records = [
            {"name": "Alice", "ssn": "111", "dob": "1990-01-01"},
            {"name": "Bob", "ssn": "222", "dob": "1985-05-05"},
        ]
        strategy = ColumnMaskStrategy("ssn", "dob")
        ctx = _context(pdp_client=_pdp_client(permitted=False))
        result = strategy.apply(records, ctx)
        assert result == [
            {"name": "Alice", "ssn": "***", "dob": "***"},
            {"name": "Bob", "ssn": "***", "dob": "***"},
        ]

    def test_deny_masks_fields_in_dict(self):
        record = {"name": "Alice", "ssn": "111", "department": "Eng"}
        strategy = ColumnMaskStrategy("ssn")
        ctx = _context(pdp_client=_pdp_client(permitted=False))
        result = strategy.apply(record, ctx)
        assert result == {"name": "Alice", "ssn": "***", "department": "Eng"}

    def test_deny_masks_nested_fields(self):
        strategy = ColumnMaskStrategy("ssn")
        ctx = _context(pdp_client=_pdp_client(permitted=False))
        result = strategy.apply(
            {"patient": {"name": "Bob", "ssn": "1"}, "visits": [{"ssn": "2", "date": "d"}]}, ctx,
        )
        assert result == {"patient": {"name": "Bob", "ssn": "***"},
                          "visits": [{"ssn": "***", "date": "d"}]}

    def test_custom_mask_string(self):
        record = {"name": "Alice", "email": "alice@example.com"}
        strategy = ColumnMaskStrategy("email", mask="[REDACTED]")
        ctx = _context(pdp_client=_pdp_client(permitted=False))
        result = strategy.apply(record, ctx)
        assert result["email"] == "[REDACTED]"

    def test_unsupported_type_returned_unchanged(self):
        strategy = ColumnMaskStrategy("field")
        ctx = _context(pdp_client=_pdp_client(permitted=False))
        result = strategy.apply("plain string", ctx)
        assert result == "plain string"

    def test_fail_open_permits_on_pdp_unavailable(self):
        record = {"name": "Alice", "ssn": "111"}
        strategy = ColumnMaskStrategy("ssn")
        client = _pdp_client(permitted=True)  # fail_open=True → check_permission returns True
        ctx = _context(pdp_client=client, fail_open=True)
        result = strategy.apply(record, ctx)
        assert result == record  # Permit path — full result



# ── Integration: custom strategy via AuthZConfig ───────────────────────────────


class TestCustomStrategy:
    def test_custom_strategy_called_by_agents_middleware(self, monkeypatch):
        """The AFTER_TOOL Enforcer's strategy is used instead of the default."""
        applied = []

        class MyStrategy:
            def apply(self, result, context):
                applied.append((result, context.tool_name))
                return result[:1]

        from axiomatics_ai import AuthZConfig, LangChainAgentAuthZMiddleware
        from langchain.agents.middleware.types import ToolMessage

        config = AuthZConfig(enforcement=standard_tool_enforcement(strategy=MyStrategy()))
        mw = LangChainAgentAuthZMiddleware(config=config)

        import axiomatics_ai.core.authz_helpers as helpers_module
        monkeypatch.setattr(helpers_module, "get_permitted_tools", lambda *a, **kw: set())
        monkeypatch.setattr(helpers_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))

        records = [{"id": "r1"}, {"id": "r2"}]
        tool_msg = ToolMessage(content=json.dumps(records), tool_call_id="tc1")

        fake_request = MagicMock()
        fake_request.tool_call = {"name": "search", "args": {}, "id": "tc1"}
        fake_request.runtime.context = {"username": "alice", "session_id": "s1"}

        result = mw._apply_post_invocation(tool_msg, fake_request)
        assert applied == [(records, "search")]
        assert json.loads(result.content) == records[:1]
