"""Shared helpers for gateway unit tests — fake ADS, JWT, and an ASGI driver (no live ADS)."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import jwt
from tests.pdp_fakes import FakePdpBase


class FakePdp(FakePdpBase):
    """Stand-in for PdpClient. With ``allowed_tools`` set, only those resource/tool names
    are permitted (drives MCP visibility + invocation); otherwise a fixed ``permit`` bool."""

    def __init__(self, permit: bool = True, *, allowed_tools: set[str] | None = None, fail_open: bool = False) -> None:
        self.permit = permit
        self.allowed_tools = allowed_tools
        self.fail_open = fail_open
        self.calls: list[dict[str, Any]] = []

    def check_permission(self, *, subject, action, resource, environment, fail_open=False) -> bool:
        self.calls.append({"subject": subject, "action": action, "resource": resource})
        if self.allowed_tools is not None:
            return resource.get("tool_name") in self.allowed_tools
        return self.permit

    def check_permission_batch(self, *, subject, action, resource_names, environment,
                               resource_attr_id, resource_extra_attrs=None, fail_open=False) -> set[str]:
        if self.allowed_tools is not None:
            return {n for n in resource_names if n in self.allowed_tools}
        return set(resource_names) if self.permit else set()


def bearer(username: str) -> tuple[bytes, bytes]:
    """An Authorization header (name, value) pair as ASGI raw headers expect."""
    token = jwt.encode({"username": username}, "demo-secret", algorithm="HS256")
    return (b"authorization", f"Bearer {token}".encode())


def drive_asgi(app, *, method: str = "POST", path: str = "/", body: bytes = b"",
               headers: list[tuple[bytes, bytes]] | None = None, query: bytes = b"") -> tuple[int, bytes]:
    """Drive an ASGI app once; return (status, body). Buffers the response."""
    scope = {
        "type": "http", "method": method, "path": path,
        "query_string": query, "headers": headers or [],
    }

    async def run() -> tuple[int, bytes]:
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        chunks = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return status, chunks

    return asyncio.run(run())


def message_send(text: str = "hi") -> bytes:
    # messageId is required by a real a2a-sdk server (the integration target validates it).
    return json.dumps({
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "m1", "parts": [{"kind": "text", "text": text}]}},
    }).encode()
