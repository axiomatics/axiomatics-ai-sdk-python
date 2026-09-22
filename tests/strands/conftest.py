"""
Fakes for the strands unit tests. The enforcement engine duck-types the event objects
Strands hands its intervention handlers, so we stand in tiny fakes for the agent / state /
tool registry / events. ``FakePdp`` mirrors the one in the google_adk tests.
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


# ── tiny Strands-shaped fakes (duck-typed by the engine) ─────────────────────────

class FakeState:
    """Stands in for AgentState — ``get()`` with no key returns the whole dict."""
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def get(self, key: str | None = None) -> Any:
        return dict(self._data) if key is None else self._data.get(key)


class FakeRegistry:
    """Stands in for ToolRegistry — the engine reads ``registry``/``dynamic_tools`` for names
    and ``get_all_tools_config()`` for the specs the model sees (which the wrapper filters)."""
    def __init__(self, names: list[str]) -> None:
        self.registry = {n: object() for n in names}
        self.dynamic_tools: dict[str, Any] = {}

    def get_all_tools_config(self) -> dict[str, Any]:
        return {n: {"name": n} for n in self.registry}


class FakeAgent:
    def __init__(self, tools: list[str] | tuple[str, ...] = (), state: dict[str, Any] | None = None,
                 messages: list[dict[str, Any]] | None = None) -> None:
        self.tool_registry = FakeRegistry(list(tools))
        self.state = FakeState(state) if state is not None else None
        self.messages = messages or []      # Strands conversation (role + content blocks)


class FakeModelEvent:
    """Stands in for BeforeModelCallEvent — agent + invocation_state; ``agent.messages`` carries
    the conversation the BEFORE_MODEL guard reads."""
    def __init__(self, agent: FakeAgent, invocation_state: dict[str, Any] | None = None) -> None:
        self.agent = agent
        self.invocation_state = invocation_state or {}


class _FakeStopResponse:
    def __init__(self, message: dict[str, Any]) -> None:
        self.message = message


class FakeAfterModelEvent:
    """Stands in for AfterModelCallEvent — ``stop_response.message`` is the model's answer that
    the AFTER_MODEL guard reads and (on block) rewrites."""
    def __init__(self, message: dict[str, Any], *, agent: FakeAgent | None = None,
                 invocation_state: dict[str, Any] | None = None) -> None:
        self.stop_response = _FakeStopResponse(message)
        self.agent = agent or FakeAgent()
        self.invocation_state = invocation_state or {}


class FakeToolEvent:
    """Stands in for Before/AfterToolCallEvent (tool_use + result + identity carriers)."""
    def __init__(self, name: str, tool_input: dict[str, Any] | None = None, *,
                 agent: FakeAgent | None = None, invocation_state: dict[str, Any] | None = None,
                 result: Any = None) -> None:
        self.tool_use = {"name": name, "input": tool_input or {}}
        self.agent = agent or FakeAgent()
        self.invocation_state = invocation_state or {}
        self.result = result
