"""Unit tests for unwrapping a tool result for post-invocation filtering, and rebuilding it.

``transform_tool_output`` is the single shape-dispatcher every enforcement surface uses, so a
gap here silently returns records policy meant to withhold. Each shape is pinned twice: what
the strategy is handed (the data, never a wrapper), and how the result is rebuilt around what
the strategy returned.
"""
from __future__ import annotations

import json

import pytest

from axiomatics_ai.core.records_helpers import is_content_blocks, transform_tool_output

RECORDS = [{"id": "a", "kind": "public"}, {"id": "b", "kind": "classified"}]
KEPT = [{"id": "a", "kind": "public"}]


class Recorder:
    """A strategy stand-in: records what it was handed and keeps only public records."""

    def __init__(self):
        self.seen = []

    def __call__(self, value):
        self.seen.append(value)
        if isinstance(value, list):
            return [r for r in value if r.get("kind") == "public"]
        return {**value, "masked": True}


def run(payload, **kw):
    fn = Recorder()
    out, found = transform_tool_output(payload, fn, **kw)
    return out, found, fn.seen


# ── shapes that carry data ─────────────────────────────────────────────────────────

def test_bare_record_list():
    out, found, seen = run(RECORDS)
    assert found and seen == [RECORDS] and out == KEPT


def test_json_string_is_parsed_and_re_serialized():
    out, found, _ = run(json.dumps(RECORDS))
    # Re-serialized, because the caller handed us a string and expects one back.
    assert found and json.loads(out) == KEPT


def test_a_container_dict_is_handed_to_the_strategy_whole():
    # Which keys hold records is the strategy's call (RecordFilterStrategy filters each list).
    out, found, seen = run({"results": RECORDS, "total": 2})
    assert found and seen == [{"results": RECORDS, "total": 2}]
    assert out["masked"] is True


def test_a_single_dict_is_handed_to_the_strategy():
    out, found, seen = run({"name": "Bob", "ssn": "123"})
    assert found and seen == [{"name": "Bob", "ssn": "123"}]


def test_result_wrapper_is_unwrapped():
    # ADK and FastMCP wrap a non-object output as {"result": ...}.
    out, found, seen = run({"result": RECORDS})
    assert found and seen == [RECORDS] and out == {"result": KEPT}


def test_mcp_result_rewrites_structured_content_and_text_together():
    payload = {
        "structuredContent": {"result": RECORDS},
        "content": [{"type": "text", "text": json.dumps(RECORDS)}],
        "isError": False,
    }
    out, found, seen = run(payload)
    assert found and seen == [RECORDS]
    # Both renderings must agree, or a client reading structuredContent sees removed data.
    assert out["structuredContent"] == {"result": KEPT}
    assert json.loads(out["content"][0]["text"]) == KEPT  # the wrapped value, not the wrapper
    assert out["isError"] is False


def test_mcp_result_of_a_tool_that_returns_json_text_is_not_encoded_twice():
    """A ``-> str`` tool returning JSON: FastMCP wraps the string as ``{"result": "<json>"}``."""
    text = json.dumps(RECORDS)
    payload = {"structuredContent": {"result": text}, "content": [{"type": "text", "text": text}]}
    out, found, _ = run(payload)
    assert found
    assert json.loads(out["content"][0]["text"]) == KEPT   # a list, not a JSON string literal
    assert json.loads(out["structuredContent"]["result"]) == KEPT


def test_mcp_result_with_an_object_as_structured_content():
    payload = {"structuredContent": {"patients": RECORDS}, "content": [{"type": "text", "text": "x"}]}
    out, found, seen = run(payload)
    assert found and seen == [{"patients": RECORDS}]
    assert json.loads(out["content"][0]["text"]) == out["structuredContent"]


def test_custom_text_block_renderer_is_used():
    # Strands uses {"text": ...} where MCP uses {"type": "text", "text": ...}.
    payload = {"structuredContent": {"result": RECORDS}, "content": []}
    out, _, _ = run(payload, render_text_block=lambda t: {"text": t})
    assert out["content"] == [{"text": json.dumps(KEPT)}]


def test_strands_tool_result_text_block():
    payload = {"toolUseId": "t1", "status": "success", "content": [{"text": json.dumps(RECORDS)}]}
    out, found, seen = run(payload)
    assert found and seen == [RECORDS]
    assert out["toolUseId"] == "t1" and json.loads(out["content"][0]["text"]) == KEPT


def test_strands_tool_result_json_block():
    payload = {"toolUseId": "t1", "status": "success", "content": [{"json": RECORDS}]}
    out, found, _ = run(payload)
    assert found and out["content"] == [{"json": KEPT}]


def test_bare_block_list_is_unwrapped_not_treated_as_records():
    # A LangChain ToolMessage's content from an MCP tool: the blocks are not records.
    blocks = [{"type": "text", "text": json.dumps(RECORDS), "id": "lc_1"}]
    out, found, seen = run(blocks)
    assert found and seen == [RECORDS]
    assert out[0]["id"] == "lc_1" and json.loads(out[0]["text"]) == KEPT


def test_lone_block_is_unwrapped():
    # OpenAI Agents' MCP output shape.
    out, found, seen = run({"type": "text", "text": json.dumps(RECORDS)})
    assert found and seen == [RECORDS] and json.loads(out["text"]) == KEPT


def test_non_data_blocks_are_kept_beside_filtered_ones():
    image = {"type": "image", "data": "...", "mimeType": "image/png"}
    out, found, _ = run([image, {"type": "text", "text": json.dumps(RECORDS)}])
    assert found and out[0] is image and json.loads(out[1]["text"]) == KEPT


def test_unchanged_output_returns_the_original_object():
    out, found = transform_tool_output(RECORDS, lambda v: v)
    assert found and out is RECORDS


def test_empty_list_is_recognized_without_calling_the_strategy():
    out, found, seen = run([])
    assert found and out == [] and seen == []


# ── shapes that carry none ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    pytest.param("hello", id="non-json-string"),
    pytest.param("42", id="json-scalar"),
    pytest.param(None, id="none"),
    pytest.param(["a", "b"], id="list-of-non-dicts"),
    pytest.param({"result": "Email sent"}, id="wrapped-scalar"),
    pytest.param([{"type": "text", "text": "sunny, 21C"}], id="plain-text-blocks"),
    pytest.param({"structuredContent": None, "content": []}, id="empty-mcp-result"),
    pytest.param({"toolUseId": "t", "status": "success", "content": []}, id="empty-strands-result"),
])
def test_payloads_without_data_are_not_found(payload):
    # found=False is the signal callers turn into a skipped (or withheld) decision.
    out, found, seen = run(payload)
    assert not found and out is payload and seen == []


def test_is_content_blocks():
    assert is_content_blocks([{"type": "text", "text": "x"}])
    assert is_content_blocks([{"json": {}}])
    assert not is_content_blocks(RECORDS)
    assert not is_content_blocks([])


# ── lists holding any object or list are data ──────────────────────────────────────

@pytest.mark.parametrize("payload", [
    pytest.param([{"id": "a"}, None, "s"], id="dict-beside-scalars"),
    pytest.param([[{"id": "a"}], [{"id": "b"}]], id="list-of-lists"),
])
def test_a_list_holding_a_dict_is_handed_to_the_strategy(payload):
    seen = []
    out, found = transform_tool_output(payload, lambda v: seen.append(v) or "filtered")
    assert found and seen == [payload] and out == "filtered"


def test_a_json_string_of_a_mixed_list_is_handed_to_the_strategy():
    payload = [{"id": "a"}, None]
    seen = []
    out, found = transform_tool_output(json.dumps(payload), lambda v: seen.append(v) or [])
    assert found and seen == [payload] and json.loads(out) == []


@pytest.mark.parametrize("payload", [
    ["a", "b"],
    [1, None, 2.5],
    [None, [1, 2]],
    [["alice", "123-45-6789"], ["bob", "987-65-4321"]],   # SQL-style rows: nothing to check
])
def test_a_list_holding_no_dict_is_unrecognized(payload):
    out, found, seen = run(payload)
    assert not found and out is payload and seen == []


# ── on_unread_block: blocks next to the data ───────────────────────────────────────

def test_on_unread_block_replaces_every_block_next_to_data():
    image = {"type": "image", "data": "...", "mimeType": "image/png"}
    summary = {"type": "text", "text": "Summary: Bob is classified"}
    unread = []
    out, found = transform_tool_output(
        [summary, {"type": "text", "text": json.dumps(RECORDS)}, image],
        Recorder(),
        on_unread_block=lambda b: unread.append(b) or {"type": "text", "text": "[withheld]"},
    )
    assert found and unread == [summary, image]
    assert out[0] == {"type": "text", "text": "[withheld]"}
    assert json.loads(out[1]["text"]) == KEPT
    assert out[2] == {"type": "text", "text": "[withheld]"}


def test_on_unread_block_applies_inside_a_strands_result():
    payload = {"toolUseId": "t1", "status": "success",
               "content": [{"text": json.dumps(RECORDS)}, {"text": "note"}]}
    out, found = transform_tool_output(payload, Recorder(), on_unread_block=lambda b: {"text": "x"})
    assert found and out["content"][1] == {"text": "x"} and out["toolUseId"] == "t1"


def test_on_unread_block_not_called_when_no_block_held_data():
    blocks = [{"type": "text", "text": "sunny"}, {"type": "image", "data": "..."}]
    called = []
    out, found = transform_tool_output(blocks, Recorder(), on_unread_block=called.append)
    assert not found and out is blocks and called == []


def test_on_unread_block_not_called_when_every_block_held_data():
    called = []
    out, found = transform_tool_output(
        [{"type": "text", "text": json.dumps(RECORDS)}], Recorder(), on_unread_block=called.append,
    )
    assert found and called == [] and json.loads(out[0]["text"]) == KEPT


def test_without_on_unread_block_other_blocks_are_kept_as_they_are():
    summary = {"type": "text", "text": "Summary: Bob is classified"}
    out, found, _ = run([{"type": "text", "text": json.dumps(RECORDS)}, summary])
    assert found and out[1] is summary


def test_unread_blocks_returned_unchanged_keep_the_original_list():
    # The data block didn't change and the callback kept the block: "nothing changed" via `is`.
    blocks = [{"type": "text", "text": json.dumps(KEPT)}, {"type": "text", "text": "hi"}]
    out, found = transform_tool_output(blocks, lambda v: v, on_unread_block=lambda b: b)
    assert found and out is blocks
