"""Shared unwrapping of a tool result, for post-invocation filtering.

Every surface hands the post-invocation strategy a tool result, but frameworks wrap the tool's
actual output differently:

- a JSON string of it (FastMCP text blocks, the Claude Agent SDK, OpenAI);
- the MCP ``CallToolResult`` shape — ``structuredContent`` holds the data while ``content`` is
  a text rendering of the same data;
- content blocks — ``{"content": [{"text": ...}]}`` (Strands' ``ToolResult``, an MCP result
  without ``structuredContent``), with a ``{"json": ...}`` block on Strands;
- ``{"result": <value>}`` — how ADK and FastMCP wrap an output that is not an object.

``transform_tool_output`` strips these transport wrappers, hands the underlying value (a list
or a dict) to a callback, and rebuilds the result in kind, so the model only sees what the
strategy returned. What the value means — a record list, a container of lists, a single
record — is the strategy's call (see ``core.presets``); this module only unwraps and re-wraps.

The one per-framework difference is how a text content block is shaped when a result is
re-rendered from ``structuredContent`` — MCP uses ``{"type": "text", "text": ...}`` while some
frameworks use ``{"text": ...}`` — so callers pass ``render_text_block`` to control it.
"""
from __future__ import annotations

import json
from typing import Any
from collections.abc import Callable

# Returned by the walk when the output holds no list or object the strategy could inspect
# (plain text, a scalar, an image block).
_UNRECOGNIZED = object()


def mcp_text_block(text: str) -> dict[str, Any]:
    """Build an MCP text content block.

    Args:
        text (str): The block's text.

    Returns:
        dict[str, Any]: ``{"type": "text", "text": text}``.
    """
    return {"type": "text", "text": text}


def transform_tool_output(
    payload: Any,
    fn: Callable[[Any], Any],
    *,
    render_text_block: Callable[[str], Any] = mcp_text_block,
    on_unread_block: Callable[[Any], Any] | None = None,
) -> tuple[Any, bool]:
    """Apply ``fn`` to the data inside a tool result and rebuild the result around it.

    Args:
        payload (Any): The framework's tool result.
        fn (Callable[[Any], Any]): Called with the unwrapped data (a list or a dict); returns
            what replaces it.
        render_text_block (Callable[[str], Any]): Builds a text block when a result is
            re-rendered from ``structuredContent``.
        on_unread_block (Callable[[Any], Any], optional): Called with each content block that
            holds no data ``fn`` could read (plain text, an image) when another block of the same
            result did; returns what replaces it. Such a block can restate the data the
            strategy just filtered. Kept as it is by default.

    Returns:
        tuple[Any, bool]: The rebuilt result, and whether data was found. When nothing was
        found, or nothing changed, the result is ``payload`` itself, so a caller can tell
        "nothing changed" with ``is``.
    """
    out = _transform(payload, fn, render_text_block, on_unread_block)
    if out is _UNRECOGNIZED:
        return payload, False
    return out, True


def _transform(value: Any, fn: Callable[[Any], Any], render: Callable[[str], Any], unread: Any) -> Any:
    if isinstance(value, str):
        return _transform_json_string(value, fn, render, unread)
    if isinstance(value, list):
        if not value:
            return value  # no records: nothing to filter, nothing withheld
        if is_content_blocks(value):
            # A bare block list (a LangChain ToolMessage's content from an MCP tool): the data
            # is inside the blocks, never the blocks themselves.
            return _transform_block_list(value, fn, render, unread)
        # A list holding an object, here or in a nested list, is data (records, possibly next to
        # nulls): the strategy decides what it means. Scalars, or lists of them (SQL-style rows),
        # hold nothing a strategy can check.
        return fn(value) if _holds_object(value) else _UNRECOGNIZED
    if isinstance(value, dict):
        return _transform_dict(value, fn, render, unread)
    return _UNRECOGNIZED


def _holds_object(items: list) -> bool:
    return any(isinstance(x, dict) or (isinstance(x, list) and _holds_object(x)) for x in items)


def _transform_json_string(text: str, fn: Callable[[Any], Any], render: Callable[[str], Any], unread: Any):
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return _UNRECOGNIZED
    if not isinstance(parsed, (list, dict)):
        return _UNRECOGNIZED
    out = _transform(parsed, fn, render, unread)
    if out is _UNRECOGNIZED:
        return out
    return text if out is parsed else json.dumps(out)


# Keys that mark a dict as a framework's result envelope (MCP ``CallToolResult``, Strands
# ``ToolResult``) rather than a domain record.
_ENVELOPE_KEYS = frozenset({"structuredContent", "isError", "toolUseId"})

# MCP content block types, and the single keys of a Strands ``ToolResultContent`` block.
_MCP_BLOCK_TYPES = frozenset({"text", "image", "audio", "resource", "resource_link"})
_STRANDS_BLOCK_KEYS = frozenset({"text", "json", "image", "document", "video"})


def _transform_dict(payload: dict, fn: Callable[[Any], Any], render: Callable[[str], Any], unread: Any):
    if payload.get("structuredContent") is not None:
        return _transform_mcp_result(payload, fn, render, unread)
    if is_content_blocks(payload.get("content")):
        blocks = _transform_block_list(payload["content"], fn, render, unread)
        if blocks is _UNRECOGNIZED:
            return blocks
        return payload if blocks is payload["content"] else {**payload, "content": blocks}
    if _ENVELOPE_KEYS & payload.keys():
        return _UNRECOGNIZED  # a result envelope that carries no data blocks
    if _is_content_block(payload):
        return _transform_block(payload, fn, render, unread)  # a lone block (OpenAI's MCP output)
    if set(payload) == {"result"}:
        inner = _transform(payload["result"], fn, render, unread)
        if inner is _UNRECOGNIZED:
            return inner
        return payload if inner is payload["result"] else {"result": inner}
    return fn(payload)


def _transform_mcp_result(payload: dict, fn: Callable[[Any], Any], render: Callable[[str], Any], unread: Any):
    """Transform an MCP ``CallToolResult``: ``structuredContent`` and the text that renders it.

    Both are rewritten from the transformed data, so they stay in sync.
    """
    structured = payload["structuredContent"]
    out = _transform(structured, fn, render, unread)
    if out is _UNRECOGNIZED or out is structured:
        return out if out is _UNRECOGNIZED else payload
    # A ``{"result": ...}`` wrapper's text block renders the wrapped value, not the wrapper. A
    # string is already the text (a tool that returns JSON text): encoding it again would make
    # it a JSON string literal.
    rendered = out["result"] if isinstance(out, dict) and set(out) == {"result"} else out
    text = rendered if isinstance(rendered, str) else json.dumps(rendered)
    return {**payload, "structuredContent": out, "content": [render(text)]}


def _is_content_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") in _MCP_BLOCK_TYPES:
        return True
    return len(block) == 1 and next(iter(block)) in _STRANDS_BLOCK_KEYS


def is_content_blocks(value: Any) -> bool:
    """Whether ``value`` is a non-empty list of MCP or Strands content blocks (not records).

    Args:
        value (Any): The value to test.

    Returns:
        bool: True for a list of content blocks.
    """
    return isinstance(value, list) and bool(value) and all(_is_content_block(b) for b in value)


def _transform_block_list(blocks: list, fn: Callable[[Any], Any], render: Callable[[str], Any], unread: Any):
    """Transform every text or json block that holds data.

    The other blocks (images, plain text) go to ``unread`` when some block held data, since they
    can restate it; with no ``unread`` they are kept as they are.
    """
    outs = [_transform_block(block, fn, render, unread) for block in blocks]
    if all(out is _UNRECOGNIZED for out in outs):
        return _UNRECOGNIZED
    new_blocks = [
        (block if unread is None else unread(block)) if out is _UNRECOGNIZED else out
        for block, out in zip(blocks, outs, strict=True)
    ]
    if all(n is b for n, b in zip(new_blocks, blocks, strict=True)):
        return blocks
    return new_blocks


def _transform_block(block: dict, fn: Callable[[Any], Any], render: Callable[[str], Any], unread: Any):
    """One content block: its ``json`` value, or its ``text`` when that holds JSON data."""
    if "json" in block:
        key = "json"
    elif isinstance(block.get("text"), str):
        key = "text"
    else:
        return _UNRECOGNIZED
    out = _transform(block[key], fn, render, unread)
    if out is _UNRECOGNIZED:
        return out
    return block if out is block[key] else {**block, key: out}
