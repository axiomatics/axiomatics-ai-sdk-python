"""
Fakes for the claude_agent unit tests. The engine takes plain values (tool name / args /
result); the wiring returns Claude Agent SDK types. `FakePdp` mirrors the other paths' tests.
"""
from __future__ import annotations

from typing import Any
from tests.pdp_fakes import FakePdpBase


class FakePdp(FakePdpBase):
    """Stand-in for PdpClient. With ``allowed_tools`` set, only those names are permitted
    (drives invocation); otherwise a fixed ``permit`` bool (drives post records)."""

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
