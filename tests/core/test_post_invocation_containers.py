"""
Tests for RecordFilterStrategy on nested containers, and for post_process_tool_output on
content-block lists where a block next to the records holds no data the strategy can read.

The PDP is a fake ``check_permission_batch`` (no network): it permits a record whose
``kind`` is ``"public"`` and counts the batched calls, so "all records in one request" is
pinned and the real ``get_permitted_records`` (hashing, matching back) runs.
"""
from __future__ import annotations

import json
import logging

import pytest

from axiomatics_ai import AuthZConfig, Enforcer, RecordFilterStrategy
from axiomatics_ai.core.enforcement import EnforcementAction, post_process_tool_output
from axiomatics_ai.core.hooks import AuthzHook
from axiomatics_ai.core.post_invocation import PostInvocationContext

PUB_A = {"id": "a", "kind": "public"}
PUB_B = {"id": "b", "kind": "public"}
SECRET = {"id": "s", "kind": "classified", "ssn": "123"}


class BatchPdp:
    """A fake PDP client: permits public records, records each batched call."""

    def __init__(self):
        self.batches: list[list[dict]] = []

    def check_permission_batch(self, *, resource_names, resource_extra_attrs, **kw):
        self.batches.append(list(resource_extra_attrs))
        return {
            name for name, rec in zip(resource_names, resource_extra_attrs, strict=True)
            if rec.get("kind") == "public"
        }


def _context(pdp: BatchPdp, tool_name: str = "list_docs") -> PostInvocationContext:
    return PostInvocationContext(
        pdp_client=pdp, subject={"username": "alice"}, environment={}, tool_name=tool_name,
        args={}, resource={"tool_name": tool_name}, action={"name": "post_invocation"},
        fail_open=False,
    )


def _apply(result):
    pdp = BatchPdp()
    out = RecordFilterStrategy().apply(result, _context(pdp))
    return out, pdp.batches


# ── RecordFilterStrategy: records at any depth, one batch ─────────────────────────


class TestRecordFilterContainers:
    def test_nested_dict_container_filters_the_inner_list_and_keeps_other_keys(self):
        out, batches = _apply({"data": {"results": [PUB_A, SECRET, PUB_B], "page": 1}, "count": 3})
        assert out == {"data": {"results": [PUB_A, PUB_B], "page": 1}, "count": 3}
        assert len(batches) == 1 and batches[0] == [PUB_A, SECRET, PUB_B]

    def test_several_lists_at_different_depths_go_in_one_batch(self):
        out, batches = _apply({
            "top": [SECRET, PUB_A],
            "meta": {"inner": {"deep": [PUB_B, SECRET]}, "label": "x"},
            "count": 4,
        })
        assert out == {
            "top": [PUB_A],
            "meta": {"inner": {"deep": [PUB_B]}, "label": "x"},
            "count": 4,
        }
        assert len(batches) == 1 and len(batches[0]) == 4

    def test_list_of_lists(self):
        out, batches = _apply([[PUB_A, SECRET], [SECRET], [PUB_B]])
        assert out == [[PUB_A], [], [PUB_B]]
        assert len(batches) == 1 and len(batches[0]) == 4

    def test_mixed_list_keeps_scalars_and_none_and_searches_nested_lists(self):
        out, batches = _apply([SECRET, None, "s", 7, [PUB_A, SECRET]])
        assert out == [None, "s", 7, [PUB_A]]
        assert len(batches) == 1 and batches[0] == [SECRET, PUB_A, SECRET]

    def test_a_records_own_fields_are_never_searched(self):
        # The record in the list is decided whole; its nested list is not a container.
        rec = {"id": "s", "kind": "classified", "children": [PUB_A]}
        out, batches = _apply([rec, PUB_B])
        assert out == [PUB_B]
        assert batches == [[rec, PUB_B]]

    def test_single_record_with_a_list_of_scalars_is_one_record(self):
        rec = {"id": "s", "kind": "classified", "tags": ["x", "y"]}
        out, batches = _apply(rec)
        assert out == {}
        assert batches == [[rec]]
        kept = {"id": "a", "kind": "public", "tags": ["x"]}
        out, _ = _apply(kept)
        assert out == kept

    def test_denied_single_record_becomes_empty_dict(self):
        out, batches = _apply(SECRET)
        assert out == {} and len(batches) == 1

    @pytest.mark.parametrize("payload", [
        pytest.param([], id="empty-list"),
        pytest.param([[], []], id="list-of-empty-lists"),
        pytest.param([None, "s", 1], id="scalars-only"),
    ])
    def test_containers_without_records_are_unchanged_and_never_reach_the_pdp(self, payload):
        out, batches = _apply(payload)
        assert out == payload and batches == []

    def test_empty_inner_list_in_a_container_is_kept(self):
        out, batches = _apply({"a": [], "b": [SECRET, PUB_A]})
        assert out == {"a": [], "b": [PUB_A]} and len(batches) == 1


# ── post_process_tool_output: blocks next to the data ─────────────────────────────


class _Recorder(AuthzHook):
    def __init__(self):
        self.events = []

    def on_decision(self, event):
        self.events.append(event)


SUMMARY = "Summary: alice ssn 123"


def _mcp_blocks():
    return [
        {"type": "text", "text": json.dumps([PUB_A, SECRET])},
        {"type": "text", "text": SUMMARY},
    ]


def _process(payload, on_unrecognized_output, **kw):
    hook = _Recorder()
    config = AuthZConfig(hooks=[hook])
    enforcer = Enforcer(
        action_name="post_invocation", strategy=RecordFilterStrategy(),
        on_unrecognized_output=on_unrecognized_output,
    )
    pdp = BatchPdp()
    result = post_process_tool_output(
        payload, enforcer=enforcer, post_ctx=_context(pdp), config=config, surface="test", **kw,
    )
    skips = [e for e in hook.events
             if e.enforcement_point == "post_invocation" and "withheld" in (e.detail or {})]
    return result, skips, pdp


class TestUnreadBlocks:
    def test_fail_closed_withholds_the_summary_block_and_filters_the_records(self, caplog):
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            result, skips, pdp = _process(_mcp_blocks(), "fail_closed")
        assert result.action is EnforcementAction.TRANSFORM
        records_block, summary_block = result.content
        assert json.loads(records_block["text"]) == [PUB_A]
        assert summary_block["type"] == "text"
        assert "withheld" in summary_block["text"] and "ssn" not in summary_block["text"]
        assert len(pdp.batches) == 1
        (skip,) = skips
        assert skip.detail["withheld"] is True and skip.decision == "deny"
        assert "content block" in skip.detail["reason"]
        assert any("withholding them" in r.getMessage() for r in caplog.records)

    def test_fail_open_keeps_the_summary_block_but_records_the_skip(self, caplog):
        with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
            result, skips, _ = _process(_mcp_blocks(), "fail_open")
        records_block, summary_block = result.content
        assert json.loads(records_block["text"]) == [PUB_A]
        assert summary_block == {"type": "text", "text": SUMMARY}
        (skip,) = skips
        assert skip.detail["withheld"] is False and skip.decision == "skipped"
        assert any("returning them unfiltered" in r.getMessage() for r in caplog.records)

    def test_image_block_next_to_data_is_withheld_too(self):
        image = {"type": "image", "data": "...", "mimeType": "image/png"}
        result, skips, _ = _process([image, _mcp_blocks()[0]], "fail_closed")
        assert "withheld" in result.content[0]["text"]
        assert len(skips) == 1 and "1 content block" in skips[0].detail["reason"]

    def test_strands_blocks_are_withheld_with_the_strands_renderer(self):
        payload = {
            "toolUseId": "t1", "status": "success",
            "content": [{"text": json.dumps([PUB_A, SECRET])}, {"text": SUMMARY}],
        }
        result, skips, _ = _process(payload, "fail_closed", render_text_block=lambda t: {"text": t})
        out = result.content
        assert out["toolUseId"] == "t1"
        assert json.loads(out["content"][0]["text"]) == [PUB_A]
        assert set(out["content"][1]) == {"text"} and "withheld" in out["content"][1]["text"]
        assert len(skips) == 1

    def test_no_block_event_when_every_block_held_data(self):
        payload = [{"type": "text", "text": json.dumps([PUB_A, SECRET])}]
        result, skips, _ = _process(payload, "fail_closed")
        assert json.loads(result.content[0]["text"]) == [PUB_A]
        assert skips == []


def test_row_lists_are_withheld_under_fail_closed():
    """A list of rows (lists of scalars) holds nothing a strategy can check, so it is withheld."""
    from axiomatics_ai import AuthZConfig, Enforcer, RecordFilterStrategy
    from axiomatics_ai.core.enforcement import post_process_tool_output
    from axiomatics_ai.core.post_invocation import PostInvocationContext

    enforcer = Enforcer(
        action_name="post_invocation", strategy=RecordFilterStrategy(), on_unrecognized_output="fail_closed",
    )
    post_ctx = PostInvocationContext(
        pdp_client=None, subject={"user_id": "alice"}, environment={}, tool_name="query",
        args={}, resource={"tool_name": "query"}, action={"name": "post_invocation"}, fail_open=False,
    )
    outcome = post_process_tool_output(
        [["alice", "123-45-6789"]], enforcer=enforcer, post_ctx=post_ctx, config=AuthZConfig(), surface="t",
    )
    assert outcome.is_block
