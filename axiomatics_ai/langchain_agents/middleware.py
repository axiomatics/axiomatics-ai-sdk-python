"""``LangChainAgentAuthZMiddleware``: LangChain agent middleware enforcing a PDP's decisions.

The content guards and tool visibility hook the model call; tool invocation (with human
approval through LangGraph's ``interrupt()``) and post-invocation hook the tool call.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.agents.middleware.types import ToolMessage
from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt

from axiomatics_ai.core.authz_helpers import resolve_authz_context
from axiomatics_ai.core.config import (
    AuthZConfig,
    ContextProvider,
    DefaultClientContextProvider,
    EnforcementPoint,
)
from axiomatics_ai.core.enforcement import (
    EnforcementResult,
    ToolCallDecision,
    authorized_tool_invocation,
    authorized_tools,
    denial_message,
    guard_content,
    post_process_tool_output,
)
from axiomatics_ai.core.pdp import ApprovalDecision, ApprovalRequest
from axiomatics_ai.core.post_invocation import PostInvocationContext
from axiomatics_ai.core.records_helpers import is_content_blocks

logger = logging.getLogger(__name__)


def _ask_through_interrupt(request: ApprovalRequest) -> ApprovalDecision:
    """Pause the graph for a human's approval with LangGraph's ``interrupt()``.

    It needs the agent to have a checkpointer. The first time, this raises ``GraphInterrupt``
    and the run returns with ``__interrupt__`` carrying ``request.to_payload()``; the app
    resumes with ``Command(resume={"approved": True, ...})``, the node runs again, and here the
    answer is returned.
    """
    return ApprovalDecision.from_answer(interrupt(request.to_payload()))


def _user_prompt(request: ModelRequest) -> str | None:
    """Get the user's prompt text for a BEFORE_MODEL guard, or None if this is not a fresh user turn.

    Only fires when the last message is a ``HumanMessage`` (so continuation turns after a tool
    result are skipped), and returns only its text: reasoning and other content blocks are
    excluded by ``.text``.
    """
    messages = getattr(request, "messages", None) or []
    if not messages or getattr(messages[-1], "type", None) != "human":
        return None
    text = (messages[-1].text or "").strip()
    return text or None


def _user_messages(request: ToolCallRequest) -> list[str]:
    """Get the user's own messages in the agent state, oldest first, for intent classification.

    Only ``HumanMessage`` s count; tool results and assistant turns are skipped, so injected
    tool output can't shape the intent.
    """
    state = getattr(request, "state", None) or {}
    messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
    return [m.text for m in messages or [] if getattr(m, "type", None) == "human" and m.text]


def _answer_text(response: ModelResponse) -> str | None:
    """Get the model's text for an AFTER_MODEL guard, or None if it wrote none.

    Text sent alongside a tool call counts: it reaches the user too. Reasoning/thinking and
    tool-call args are excluded by ``.text``.
    """
    messages = getattr(response, "result", None) or []
    text = "".join((m.text or "") for m in messages if getattr(m, "type", None) == "ai").strip()
    return text or None


def _refusal_response(reason: str) -> ModelResponse:
    return ModelResponse(result=[AIMessage(content=reason)])


def _withheld_answer(response: ModelResponse, reason: str) -> ModelResponse:
    """Replace a response's text with ``reason``, keeping its tool calls.

    Only the text is withheld; each call is still authorized at BEFORE_TOOL.
    """
    messages = getattr(response, "result", None) or []
    tool_calls = [
        call for m in messages if getattr(m, "type", None) == "ai"
        for call in (getattr(m, "tool_calls", None) or [])
    ]
    return ModelResponse(result=[AIMessage(content=reason, tool_calls=tool_calls)])


class LangChainAgentAuthZMiddleware(AgentMiddleware):
    """LangChain agent middleware enforcing authorization through a PDP.

    - ``wrap_model_call`` / ``awrap_model_call``: ``BEFORE_MODEL`` guards the user prompt (a
      block returns a refusal and the model is never called); ``TOOL_VISIBILITY`` filters the
      tool list on every model turn (one batched request); ``AFTER_MODEL`` guards the model's
      text, including text sent alongside a tool call (a block replaces the text and keeps the
      tool call).
    - ``wrap_tool_call`` / ``awrap_tool_call``: ``BEFORE_TOOL`` checks each call with its
      arguments (and asks for human approval when the policy requires it, pausing with
      LangGraph's ``interrupt()``); ``AFTER_TOOL`` runs the post-invocation strategy on the
      result.

    Streamed tokens reach the user before the answer is checked. Decisions are cached for the
    configured TTL. Identity comes from the runtime context (``agent.invoke(..., context=...)``)
    through the config's context provider (``DefaultClientContextProvider`` by default).

    Args:
        config (AuthZConfig, optional): What to enforce and where the PDP is; ``AuthZConfig()``
            (nothing enforced) by default.
        agent_name (str, optional): The agent this middleware runs in, sent as the subject's
            ``config.attribute_names.acting_agent``. The factories pass the agent's ``name``
            and each deepagents sub-agent's.
    """

    _SURFACE = "langchain_agents"

    def __init__(self, config: AuthZConfig | None = None, *, agent_name: str | None = None):
        self.config = config or AuthZConfig()
        self.agent_name = agent_name
        # Tools whose output isn't post-processed: deepagents' ``task`` when every sub-agent
        # is authorized (set by create_authorized_deep_agent).
        self._unchecked_output_tools: frozenset[str] = frozenset()
        self._provider: ContextProvider = self.config.context_provider or DefaultClientContextProvider()
        self._ads = self.config.build_pdp_client()
        self._cache = self.config.build_cache()

    # ── wrap_model_call: tool visibility ──────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # BEFORE_MODEL: guard the user prompt before it reaches the model.
        """Guard the prompt, filter the tool list, call the model, and guard its text.

        Args:
            request (ModelRequest): The model call.
            handler (Callable[[ModelRequest], ModelResponse]): Calls the model.

        Returns:
            ModelResponse: The model's response, a refusal (prompt blocked), or the response with
            its text withheld (answer blocked).
        """
        if self.config.enforces(EnforcementPoint.BEFORE_MODEL):
            prompt = _user_prompt(request)
            if prompt is not None:
                result = self._guard_model_io(EnforcementPoint.BEFORE_MODEL, prompt, request.runtime.context)
                if result.is_block:
                    return _refusal_response(result.reason)

        # TOOL_VISIBILITY: filter the tool list for this model turn.
        if self.config.enforces(EnforcementPoint.TOOL_VISIBILITY):
            permitted = self._get_permitted_tools(request)
            request = request.override(tools=[t for t in request.tools if t.name in permitted])
        response = handler(request)

        # AFTER_MODEL: guard the model's text, including text sent alongside a tool call.
        if self.config.enforces(EnforcementPoint.AFTER_MODEL):
            answer = _answer_text(response)
            if answer is not None:
                result = self._guard_model_io(EnforcementPoint.AFTER_MODEL, answer, request.runtime.context)
                if result.is_block:
                    return _withheld_answer(response, result.reason)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async ``wrap_model_call``; the PDP calls run on a worker thread.

        Args:
            request (ModelRequest): The model call.
            handler (Callable[[ModelRequest], Awaitable[ModelResponse]]): Calls the model.

        Returns:
            ModelResponse: As for ``wrap_model_call``.
        """
        if self.config.enforces(EnforcementPoint.BEFORE_MODEL):
            prompt = _user_prompt(request)
            if prompt is not None:
                result = await asyncio.to_thread(
                    self._guard_model_io, EnforcementPoint.BEFORE_MODEL, prompt, request.runtime.context
                )
                if result.is_block:
                    return _refusal_response(result.reason)

        if self.config.enforces(EnforcementPoint.TOOL_VISIBILITY):
            permitted = await asyncio.to_thread(self._get_permitted_tools, request)
            request = request.override(tools=[t for t in request.tools if t.name in permitted])
        response = await handler(request)

        if self.config.enforces(EnforcementPoint.AFTER_MODEL):
            answer = _answer_text(response)
            if answer is not None:
                result = await asyncio.to_thread(
                    self._guard_model_io, EnforcementPoint.AFTER_MODEL, answer, request.runtime.context
                )
                if result.is_block:
                    return _withheld_answer(response, result.reason)
        return response

    # ── Private: model-I/O content guards (BEFORE_MODEL / AFTER_MODEL) ─────────

    def _guard_model_io(self, point: EnforcementPoint, content: str, run_context: Any) -> EnforcementResult:
        """Run the shared ``guard_content`` for this middleware.

        Lets the four call sites (sync/async × before/after) pass just point, content and context.
        """
        return guard_content(
            config=self.config,
            pdp=self._ads,
            cache=self._cache,
            provider=self._provider,
            surface=self._SURFACE,
            point=point,
            content=content,
            run_context=run_context,
            agent_name=self.agent_name,
        )

    def _resolve(self, run_context: Any):
        return resolve_authz_context(
            self._provider, run_context,
            agent_name=self.agent_name, agent_attribute=self.config.attribute_names.acting_agent,
        )

    # ── wrap_tool_call: tool invocation + post-invocation ─────────────────────

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Authorize a tool call, run it, and post-process its result.

        Args:
            request (ToolCallRequest): The tool call.
            handler (Callable[[ToolCallRequest], ToolMessage]): Runs the tool.

        Returns:
            ToolMessage: The (filtered) result, or a message saying the call was denied or not
            approved.

        Raises:
            GraphInterrupt: The call needs a human's approval; the run pauses until the app
                resumes it with the answer.
        """
        if self.config.enforces(EnforcementPoint.BEFORE_TOOL):
            decision = self._check_call(request)
            if decision.pending is not None:
                decision = self._check_call(request, _ask_through_interrupt(decision.pending))
            if not decision.permitted:
                return self._denied_message(request, decision)
        result = handler(request)
        return self._apply_post_invocation(result, request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Async ``wrap_tool_call``; the PDP calls run on a worker thread.

        Args:
            request (ToolCallRequest): The tool call.
            handler (Callable[[ToolCallRequest], Awaitable[ToolMessage]]): Runs the tool.

        Returns:
            ToolMessage: As for ``wrap_tool_call``.

        Raises:
            GraphInterrupt: The call needs a human's approval.
        """
        if self.config.enforces(EnforcementPoint.BEFORE_TOOL):
            decision = await asyncio.to_thread(self._check_call, request)
            if decision.pending is not None:
                # interrupt() runs here, on the graph's own task, not in the worker thread.
                answer = _ask_through_interrupt(decision.pending)
                decision = await asyncio.to_thread(self._check_call, request, answer)
            if not decision.permitted:
                return self._denied_message(request, decision)
        result = await handler(request)
        return await asyncio.to_thread(self._apply_post_invocation, result, request)

    # ── Private: tool visibility ───────────────────────────────────────────────

    def _get_permitted_tools(self, request: ModelRequest) -> set[str]:
        ctx = self._resolve(request.runtime.context)
        enforcer = self.config.enforcer(EnforcementPoint.TOOL_VISIBILITY)
        action = self._provider.get_action(enforcer.action_name, request.runtime.context)
        return authorized_tools(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._SURFACE,
            subject=ctx.subject,
            action=action,
            tool_names=[t.name for t in request.tools],
            environment=ctx.environment,
        )

    # ── Private: tool invocation ───────────────────────────────────────────────

    def _check_call(self, request: ToolCallRequest, approval: ApprovalDecision | None = None) -> ToolCallDecision:
        ctx = self._resolve(request.runtime.context)
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {})
        enforcer = self.config.enforcer(EnforcementPoint.BEFORE_TOOL)
        action = self._provider.get_action(enforcer.action_name, request.runtime.context)
        resource = self._provider.get_resource(
            tool_name, request.runtime.context, args=args,
            resource_name_attribute=self.config.attribute_names.tool,
        )
        return authorized_tool_invocation(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._SURFACE,
            subject=ctx.subject,
            action=action,
            resource=resource,
            environment=ctx.environment,
            user_messages=_user_messages(request),
            approval=approval,
            can_interrupt=True,
            args=args,
            call_id=request.tool_call.get("id"),
        )

    def _denied_message(self, request: ToolCallRequest, decision: ToolCallDecision) -> ToolMessage:
        return ToolMessage(
            content=denial_message(request.tool_call.get("name", "unknown"), decision),
            tool_call_id=request.tool_call.get("id", ""),
        )

    # ── Private: post-invocation ───────────────────────────────────────────────

    def _apply_post_invocation(self, result: ToolMessage | Command, request: ToolCallRequest) -> ToolMessage | Command:
        enforcer = self.config.enforcer(EnforcementPoint.AFTER_TOOL)
        if enforcer is None or enforcer.strategy is None:
            return result
        if request.tool_call.get("name") in self._unchecked_output_tools:
            return result
        if isinstance(result, Command):
            return self._post_process_command(result, request, enforcer)
        if not isinstance(result, ToolMessage):
            return result
        return self._post_process_message(result, request, enforcer)

    def _post_process_command(self, result: Command, request: ToolCallRequest, enforcer: Any) -> Command:
        """Post-process the ToolMessages a state-updating tool (deepagents' ``task``) returns."""
        update = result.update
        messages = update.get("messages") if isinstance(update, dict) else None
        if not isinstance(messages, list):
            return result
        filtered = [
            self._post_process_message(m, request, enforcer) if isinstance(m, ToolMessage) else m
            for m in messages
        ]
        return dataclasses.replace(result, update={**update, "messages": filtered})

    def _post_process_message(self, result: ToolMessage, request: ToolCallRequest, enforcer: Any) -> ToolMessage:
        ctx = self._resolve(request.runtime.context)
        tool_name = request.tool_call.get("name", "")
        action_name = enforcer.action_overrides.get(tool_name, enforcer.action_name)
        action = self._provider.get_action(action_name, request.runtime.context)

        post_ctx = PostInvocationContext(
            pdp_client=self._ads,
            subject=ctx.subject,
            environment=ctx.environment,
            tool_name=tool_name,
            args=request.tool_call.get("args", {}),
            resource=self._provider.get_resource(
                tool_name, request.runtime.context, args=request.tool_call.get("args", {}),
                resource_name_attribute=self.config.attribute_names.tool,
            ),
            action=action,
            fail_open=self.config.connection.fail_open,
        )

        outcome = post_process_tool_output(
            result.content, enforcer=enforcer, post_ctx=post_ctx, config=self.config,
            surface=self._SURFACE,
        )
        if outcome.is_block:
            return ToolMessage(
                content=outcome.reason, tool_call_id=result.tool_call_id, status="error",
            )
        if outcome.content is None:
            return result
        new_content = outcome.content
        # Content blocks (an MCP tool's output) stay blocks; filtered records are serialized.
        if not isinstance(new_content, str) and not is_content_blocks(new_content):
            new_content = json.dumps(new_content)
        return ToolMessage(content=new_content, tool_call_id=result.tool_call_id)
