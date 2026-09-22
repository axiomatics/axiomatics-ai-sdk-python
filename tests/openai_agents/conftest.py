"""
Fakes for the openai_agents unit tests. The adapter imports no `agents` and duck-types the
objects the SDK passes around, so tiny fakes for the agent / tools / contexts suffice.
`FakePdp` mirrors the one in the other MCP-client paths' tests.
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


# ── tiny OpenAI-Agents-shaped fakes (duck-typed by the adapter) ──────────────────

class FakeTool:
    """Stands in for FunctionTool — carries a name and an async on_invoke_tool."""
    def __init__(self, name: str, result: Any = None) -> None:
        self.name = name
        self._result = result

        async def on_invoke_tool(tool_context: Any, input_json: Any):
            return self._result

        self.on_invoke_tool = on_invoke_tool


class FakeToolContext:
    """Stands in for ToolContext (`.context`, and `.turn_input` for intent classification)."""
    def __init__(self, context: dict[str, Any], turn_input: Any = None) -> None:
        self.context = context
        self.turn_input = turn_input if turn_input is not None else []


class FakeRunContext:
    """Stands in for RunContextWrapper (only `.context` is read)."""
    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context


class FakeAgent:
    """Stands in for an OpenAI Agents SDK Agent — exposes an async get_all_tools."""
    def __init__(self, tools: list[FakeTool]) -> None:
        self._tools = list(tools)

        async def get_all_tools(run_context: Any):
            return list(self._tools)

        self.get_all_tools = get_all_tools
