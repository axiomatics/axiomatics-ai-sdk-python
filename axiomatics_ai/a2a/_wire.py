"""A2A JSON-RPC wire parsing, shared by the client and server guards.

Everything here works on the raw JSON-RPC request body, never on ``a2a-sdk`` Python types,
which makes enforcement immune to the protobuf-vs-pydantic split between a2a-sdk 0.3.x and
1.x. A2A is JSON-RPC 2.0 over HTTP POST::

    {"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {...}}

Method names differ by a2a-sdk line (both are recognised): 0.3.x uses ``message/send`` /
``message/stream``; 1.x uses ``SendMessage`` / ``SendStreamingMessage``.
"""
from __future__ import annotations

import json
from typing import Any

# RPC methods that invoke the remote agent (i.e. should be authorized). Union of the
# 0.3.x slash-names and the 1.x ProtoJSON names so one check covers both versions.
INVOCATION_METHODS: frozenset[str] = frozenset(
    {
        "message/send",           # a2a-sdk 0.3.x
        "message/stream",         # a2a-sdk 0.3.x (streaming)
        "SendMessage",            # a2a-sdk 1.x
        "SendStreamingMessage",   # a2a-sdk 1.x (streaming)
    }
)


def parse_jsonrpc(body: bytes | str | None) -> dict[str, Any] | None:
    """Parse a JSON-RPC request body.

    Args:
        body (bytes | str | None): The raw request body.

    Returns:
        dict[str, Any] | None: The request object, or None (never raises) for an empty body,
        invalid JSON, a batch array, or a payload without ``method``. The server middleware
        rejects those; the client guard lets them through, since it only gates the message
        sends it originates.
    """
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or "method" not in parsed:
        return None
    return parsed


def is_invocation_method(method: str | None) -> bool:
    """Whether ``method`` sends a message to an agent (either a2a-sdk version).

    Args:
        method (str | None): The JSON-RPC method.

    Returns:
        bool: True for ``message/send``, ``message/stream``, ``SendMessage`` and
        ``SendStreamingMessage``.
    """
    return method in INVOCATION_METHODS


def extract_skill_hint(params: dict[str, Any] | None) -> str | None:
    """Find a skill id in the message, if the caller or framework supplied one.

    The base A2A protocol routes a message to an agent and lets the agent pick the skill, so a
    skill selector is not guaranteed. Some frameworks put one in ``message.metadata``
    (``skillId`` / ``skill_id`` / ``skill``) or in the send ``configuration``. It is chosen by
    the caller.

    Args:
        params (dict[str, Any] | None): The request's ``params``.

    Returns:
        str | None: The skill id, or None.
    """
    if not isinstance(params, dict):
        return None
    message = _get_message(params)
    for container in (message.get("metadata"), params.get("metadata"), params.get("configuration")):
        if isinstance(container, dict):
            for key in ("skillId", "skill_id", "skill"):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def extract_message_text(params: dict[str, Any] | None) -> str | None:
    """Join the text parts of the message.

    Handles both wire shapes: 0.3.x ``params.message.parts[].text`` and the 1.x ProtoJSON
    ``params.request.msg.content[].text``. Off the enforcement path by default (privacy and
    cache-key churn); callers opt in with ``include_message_text``.

    Args:
        params (dict[str, Any] | None): The request's ``params``.

    Returns:
        str | None: The text, or None when there is no text part.
    """
    if not isinstance(params, dict):
        return None
    message = _get_message(params)
    parts = message.get("parts") or message.get("content") or []
    if not isinstance(parts, list):
        return None
    texts = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(texts) if texts else None


def _get_message(params: dict[str, Any]) -> dict[str, Any]:
    """Locate the Message object across a2a-sdk versions.

    0.3.x: ``params["message"]``. 1.x ProtoJSON: ``params["request"]["msg"]`` (also
    accepts ``["request"]["message"]`` defensively). Returns ``{}`` when not found.
    """
    message = params.get("message")
    if isinstance(message, dict):
        return message
    request = params.get("request")
    if isinstance(request, dict):
        for key in ("msg", "message"):
            candidate = request.get(key)
            if isinstance(candidate, dict):
                return candidate
    return {}
