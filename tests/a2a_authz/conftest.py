"""
Shared helpers for A2A *unit* tests — a fake ADS client, no live ADS required.

This is the unit tier (fast, mocked ADS). The integration tier (Phase 5) reuses the real
``ads`` / ``ads_server`` fixtures from ``tests/conftest.py`` behind the ``ads_integration``
marker, against ``examples/policies/domain_a2a.yaml`` and real per-framework A2A servers/clients.
"""
from __future__ import annotations

import json
from typing import Any
from tests.pdp_fakes import FakePdpBase


class FakePdp(FakePdpBase):
    """Stand-in for ``PdpClient`` that returns a fixed decision and records calls.

    Only ``check_permission`` is exercised by the A2A invocation path (via the core's
    ``evaluate_tool_call``); it records each call's XACML categories so tests can
    assert what reached ADS.
    """

    def __init__(self, permit: bool = True, *, fail_open: bool = False) -> None:
        self.permit = permit
        self.fail_open = fail_open
        self.calls: list[dict[str, Any]] = []

    def check_permission(self, *, subject, action, resource, environment, fail_open=False) -> bool:
        self.calls.append(
            {"subject": subject, "action": action, "resource": resource, "environment": environment}
        )
        return self.permit


def jsonrpc(method: str, params: dict[str, Any] | None = None, *, id: str = "1") -> bytes:
    """Encode a JSON-RPC request body."""
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        body["params"] = params
    return json.dumps(body).encode()


# Message-send params in both a2a-sdk wire shapes, so tests prove version-agnosticism.
def params_v03(text: str = "hello", *, skill: str | None = None) -> dict[str, Any]:
    """a2a-sdk 0.3.x shape: params.message.parts[].text."""
    message: dict[str, Any] = {"role": "user", "parts": [{"kind": "text", "text": text}]}
    if skill is not None:
        message["metadata"] = {"skillId": skill}
    return {"message": message}


def params_v1(text: str = "hello", *, skill: str | None = None) -> dict[str, Any]:
    """a2a-sdk 1.x ProtoJSON shape: params.request.msg.content[].text."""
    msg: dict[str, Any] = {"role": "user", "content": [{"text": text}]}
    if skill is not None:
        msg["metadata"] = {"skill_id": skill}
    return {"request": {"msg": msg}}
