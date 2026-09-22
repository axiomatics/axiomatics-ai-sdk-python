"""Unit tests for the version-proof JSON-RPC wire parsing (no a2a-sdk, no ADS)."""
from __future__ import annotations

import pytest

from axiomatics_ai.a2a import _wire

from .conftest import jsonrpc, params_v03, params_v1


@pytest.mark.parametrize(
    "method,expected",
    [
        ("message/send", True),          # 0.3.x
        ("message/stream", True),        # 0.3.x streaming
        ("SendMessage", True),           # 1.x
        ("SendStreamingMessage", True),  # 1.x streaming
        ("tasks/get", False),
        ("GetTask", False),
        (None, False),
    ],
)
def test_is_invocation_method(method, expected):
    assert _wire.is_invocation_method(method) is expected


def test_parse_jsonrpc_rejects_non_rpc():
    assert _wire.parse_jsonrpc(b"") is None
    assert _wire.parse_jsonrpc(b"not json") is None
    assert _wire.parse_jsonrpc(b'{"no":"method"}') is None
    assert _wire.parse_jsonrpc(b'[{"method":"x"}]') is None  # batch array unsupported


def test_parse_jsonrpc_accepts_rpc():
    rpc = _wire.parse_jsonrpc(jsonrpc("message/send", params_v03()))
    assert rpc is not None and rpc["method"] == "message/send"


@pytest.mark.parametrize("params_fn", [params_v03, params_v1])
def test_extract_message_text_both_versions(params_fn):
    assert _wire.extract_message_text(params_fn("hi there")) == "hi there"


@pytest.mark.parametrize("params_fn", [params_v03, params_v1])
def test_extract_skill_hint_both_versions(params_fn):
    assert _wire.extract_skill_hint(params_fn(skill="refund")) == "refund"
    assert _wire.extract_skill_hint(params_fn()) is None  # absent → None
