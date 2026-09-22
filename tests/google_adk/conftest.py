"""
Fakes for the google_adk unit tests. No ADK is imported — authorized_adk_agent duck-types the
objects ADK passes to its callbacks, so we stand in tiny fakes for State / LlmRequest / tools.
"""
from __future__ import annotations

from typing import Any
from tests.pdp_fakes import FakePdpBase


class FakePdp(FakePdpBase):
    """Stand-in for PdpClient. With ``allowed_tools`` set, only those names are permitted
    (drives visibility + invocation); otherwise a fixed ``permit`` bool (drives post records)."""

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


# ── tiny ADK-shaped fakes (duck-typed by authz.py) ──────────────────────────────

class FakeState:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class FakeContext:
    """Stands in for CallbackContext / ToolContext (only `.state` is used)."""
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = FakeState(state)


class _FD:
    def __init__(self, name: str) -> None:
        self.name = name


class _DeclTool:
    """A genai-style Tool holding function_declarations."""
    def __init__(self, names: list[str]) -> None:
        self.function_declarations = [_FD(n) for n in names]


class _Config:
    def __init__(self, names: list[str]) -> None:
        self.tools = [_DeclTool(names)]


class FakeLlmRequest:
    def __init__(self, tool_names: list[str] | None = None, *, contents: list[Any] | None = None) -> None:
        tool_names = tool_names or []
        self.tools_dict = {n: object() for n in tool_names}
        self.config = _Config(tool_names)
        self.contents = contents or []          # genai-style conversation turns


class FakePart:
    """A genai-style Part: exactly one of text / function_call / function_response is set."""
    def __init__(self, *, text: str | None = None, function_call: Any = None, function_response: Any = None) -> None:
        self.text = text
        self.function_call = function_call
        self.function_response = function_response


class FakeContent:
    """A genai-style Content (a conversation turn)."""
    def __init__(self, role: str, parts: list[FakePart]) -> None:
        self.role = role
        self.parts = parts


class FakeLlmResponse:
    def __init__(self, parts: list[FakePart], *, partial: bool = False) -> None:
        self.content = FakeContent("model", parts)
        self.partial = partial


class FakeTool:
    """Stands in for BaseTool (only `.name` is used)."""
    def __init__(self, name: str) -> None:
        self.name = name


class FakeAgent:
    """Stands in for an ADK Agent — just carries the callback fields authorized_adk_agent sets."""
    before_model_callback = None
    after_model_callback = None
    before_tool_callback = None
    after_tool_callback = None
