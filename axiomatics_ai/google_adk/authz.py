"""Client-side PDP authorization for Google ADK agents (e.g. agents that consume MCP tools).

``authorized_adk_agent(agent, config)`` attaches ADK callbacks — it does NOT wrap or subclass the
agent or its toolsets:

- ``before_model_callback`` → **BEFORE_MODEL content guard** and **tool visibility**. The
  content guard classifies the user prompt into PDP attributes (LLM enrichers) and, on a
  block, returns a refusal ``LlmResponse`` so the model is never called. Visibility then makes
  one batched PDP ``get_permitted_tools`` call (cached) and removes denied tools from the
  ``LlmRequest``.
- ``after_model_callback`` → **AFTER_MODEL content guard**: classifies the model's final
  answer; a block returns a replacement ``LlmResponse`` carrying the refusal.
- ``before_tool_callback`` → **tool invocation**: ``authorized_tool_invocation``; a denied call
  is blocked by returning an error dict (the tool never runs). A call the policy wants a human
  to approve pauses with ADK tool confirmation (``tool_context.request_confirmation``) unless
  an ``approver`` is configured.
- ``after_tool_callback`` → **post-invocation**: the configured strategy transforms/filters
  the tool result.

The model-I/O guards inspect real user prompts and the model's text — function-response
continuation turns are skipped, while text sent alongside a function call is checked (its
block keeps the call). Streamed partials reach the user before the complete response is
checked (see ``_adk_user_prompt`` / ``_adk_answer_text``).

Our callbacks run before any already configured on the agent: ADK stops at the first callback
that returns a value, so one placed after a user callback could be skipped.

Identity is the PDP subject read from ADK **session state** (``context.state``), which the
app can populate either once per user (``create_session(state=...)``) or per invocation
(``run_async(state_delta=...)``). The state dict is handed to the (reused)
``DefaultClientContextProvider``; pass a custom ``context_provider`` to change the mapping.
"""
from __future__ import annotations

import logging
from typing import Any

from axiomatics_ai.core.authz_helpers import resolve_authz_context
from axiomatics_ai.core.config import AuthZConfig, DefaultClientContextProvider, EnforcementPoint
from axiomatics_ai.core.enforcement import (
    authorized_tool_invocation,
    authorized_tools,
    denial_message,
    guard_content,
    post_process_tool_output,
)
from axiomatics_ai.core.pdp import ApprovalDecision
from axiomatics_ai.core.post_invocation import PostInvocationContext

logger = logging.getLogger(__name__)


def _state_context(context: Any) -> dict[str, Any]:
    """Extract the runtime context (a plain dict) from an ADK callback/tool context.

    ADK's ``State`` is dict-like but not a real dict, so we coerce via ``to_dict()`` and
    feed the result to the context provider (see module docstring).
    """
    state = getattr(context, "state", None)
    if state is None:
        return {}
    if hasattr(state, "to_dict"):
        return state.to_dict()
    if isinstance(state, dict):
        return dict(state)
    return {}


class _AdkAuthZ:
    """Holds the shared PDP client/cache and provides the three ADK callbacks."""

    _SURFACE = "google_adk"

    def __init__(self, config: AuthZConfig) -> None:
        self.config = config
        self._provider = self.config.context_provider or DefaultClientContextProvider()
        self._ads = config.build_pdp_client()
        self._cache = config.build_cache()

    def _guard_model_io(self, point: EnforcementPoint, content: str, ctx: dict, agent_name: str | None):
        """Instance-bound adapter over the shared ``guard_content`` (BEFORE_MODEL/AFTER_MODEL)."""
        return guard_content(
            config=self.config, pdp=self._ads, cache=self._cache, provider=self._provider,
            surface=self._SURFACE, point=point, content=content, run_context=ctx,
            agent_name=agent_name,
        )

    def _resolve(self, ctx: dict, adk_context: Any):
        """The subject and environment, with the name of the agent the callback runs for."""
        return resolve_authz_context(
            self._provider, ctx, agent_name=getattr(adk_context, "agent_name", None),
            agent_attribute=self.config.attribute_names.acting_agent,
        )

    # ── BEFORE_MODEL content guard + tool visibility (before_model_callback) ────
    def before_model(self, callback_context: Any, llm_request: Any):
        """ADK ``before_model_callback``: guard the prompt, then prune denied tools from the request.

        Returns a refusal ``LlmResponse`` when the prompt is blocked (the model is not called),
        else None.
        """
        ctx = _state_context(callback_context)

        # BEFORE_MODEL content guard — a block short-circuits the model with a refusal.
        if self.config.enforces(EnforcementPoint.BEFORE_MODEL):
            prompt = _adk_user_prompt(llm_request)
            if prompt is not None:
                result = self._guard_model_io(
                    EnforcementPoint.BEFORE_MODEL, prompt, ctx, getattr(callback_context, "agent_name", None),
                )
                if result.is_block:
                    return _adk_refusal(result.reason)

        # TOOL_VISIBILITY — remove denied tools from the request (model still runs).
        if self.config.enforces(EnforcementPoint.TOOL_VISIBILITY):
            tool_names = list(getattr(llm_request, "tools_dict", {}) or {})
            if tool_names:
                resolved = self._resolve(ctx, callback_context)
                action = self._provider.get_action(
                    self.config.enforcer(EnforcementPoint.TOOL_VISIBILITY).action_name, ctx
                )
                permitted = authorized_tools(
                    pdp=self._ads,
                    cache=self._cache,
                    config=self.config,
                    surface=self._SURFACE,
                    subject=resolved.subject,
                    action=action,
                    tool_names=tool_names,
                    environment=resolved.environment,
                )
                denied = set(tool_names) - set(permitted)
                if denied:
                    _prune_tools(llm_request, permitted, refused_on_call=self.config.enforces(EnforcementPoint.BEFORE_TOOL))
                    logger.debug("ADK tool visibility [%s]: hiding %s", resolved.subject, sorted(denied))
        return None  # request mutated in place; do not short-circuit the model

    # ── AFTER_MODEL content guard (after_model_callback) ───────────────────────
    def after_model(self, callback_context: Any, llm_response: Any):
        """ADK ``after_model_callback``: guard the model's text.

        Returns a replacement ``LlmResponse`` (text withheld, function calls kept) when blocked,
        else None.
        """
        if not self.config.enforces(EnforcementPoint.AFTER_MODEL):
            return None
        answer = _adk_answer_text(llm_response)
        if answer is None:
            return None
        result = self._guard_model_io(
            EnforcementPoint.AFTER_MODEL, answer, _state_context(callback_context),
            getattr(callback_context, "agent_name", None),
        )
        if result.is_block:
            return _adk_refusal(result.reason, llm_response)  # replaces the model's answer
        return None

    # ── invocation (before_tool_callback) ──────────────────────────────────────
    def before_tool(self, tool: Any, args: dict[str, Any], tool_context: Any):
        """ADK ``before_tool_callback``: authorize the call, pausing for tool confirmation if asked.

        Returns an error dict (the tool doesn't run) for a denied or pending call, else None.
        """
        ctx = _state_context(tool_context)
        tool_name = getattr(tool, "name", "")
        resolved = self._resolve(ctx, tool_context)
        action = self._provider.get_action(
            self.config.enforcer(EnforcementPoint.BEFORE_TOOL).action_name, ctx
        )
        resource = self._provider.get_resource(
            tool_name, ctx, args=args or {},
            resource_name_attribute=self.config.attribute_names.tool,
        )

        decision = authorized_tool_invocation(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._SURFACE,
            subject=resolved.subject,
            action=action,
            resource=resource,
            environment=resolved.environment,
            user_messages=_adk_user_messages(tool_context),
            approval=_adk_confirmation(tool_context),
            can_interrupt=_can_request_confirmation(tool_context),
            args=args or {},
            call_id=getattr(tool_context, "function_call_id", None),
        )
        if decision.pending is not None:
            # ADK pauses the run with an adk_request_confirmation function call; the app answers
            # with a FunctionResponse ({"confirmed": bool, "payload": {...}}), and ADK runs this
            # callback again with tool_context.tool_confirmation set.
            tool_context.request_confirmation(
                hint=decision.pending.reason or f"Approve the call to '{tool_name}'?",
                payload=decision.pending.to_payload(),
            )
            return {"error": f"'{tool_name}' is waiting for a human's approval."}
        if not decision.permitted:
            logger.debug("ADK tool call [%s] by [%s]: Deny", tool_name, resolved.subject)
            return {"error": denial_message(tool_name, decision)}
        return None  # permitted → let the tool run

    # ── post-invocation (after_tool_callback) ──────────────────────────────────
    def after_tool(self, tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any):
        """ADK ``after_tool_callback``: run the post-invocation strategy on the tool's response.

        Returns the rewritten response, an error dict for withheld output, or None (unchanged).
        """
        post_enforcer = self.config.enforcer(EnforcementPoint.AFTER_TOOL)
        if post_enforcer is None or post_enforcer.strategy is None:
            return None
        if _is_agent_tool(tool):
            # The wrapped agent is authorized too (authorized_adk_agent walks into it): its own
            # tool results were post-processed, and its answer is text no strategy reads.
            return None

        ctx = _state_context(tool_context)
        resolved = self._resolve(ctx, tool_context)
        tool_name = getattr(tool, "name", "")
        action_name = post_enforcer.action_overrides.get(tool_name, post_enforcer.action_name)
        action = self._provider.get_action(action_name, ctx)
        post_ctx = PostInvocationContext(
            pdp_client=self._ads,
            subject=resolved.subject,
            environment=resolved.environment,
            tool_name=tool_name,
            args=args or {},
            resource=self._provider.get_resource(
                tool_name, ctx, args=args or {},
                resource_name_attribute=self.config.attribute_names.tool,
            ),
            action=action,
            fail_open=self.config.connection.fail_open,
        )
        outcome = post_process_tool_output(
            tool_response, enforcer=post_enforcer, post_ctx=post_ctx, config=self.config,
            surface=self._SURFACE,
        )
        if outcome.is_block:
            return {"error": outcome.reason}
        if outcome.content is not None and not outcome.content:
            # Everything was denied ({} or []). ADK reads a falsy callback result as "no change"
            # and moves on to the next callback, whose None would restore the original
            # response: wrap it so it is never empty.
            return {"result": outcome.content}
        return outcome.content  # None (unchanged) lets ADK keep the original response


def authorized_adk_agent(agent: Any, config: AuthZConfig | None = None):
    """Attach authorization to an ADK ``Agent`` and every agent below it, and return it.

    Adds only the callbacks the config enforces: the BEFORE_MODEL guard and tool visibility
    (``before_model_callback``), the AFTER_MODEL guard (``after_model_callback``), tool
    invocation (``before_tool_callback``, pausing with ADK tool confirmation when the policy
    asks for approval) and post-invocation (``after_tool_callback``). Callbacks already on the
    agent are kept and run after ours. Identity comes from the session state through the
    config's context provider (``DefaultClientContextProvider`` by default).

    The agents below it get the same config: its ``sub_agents`` (also through workflow agents
    such as ``SequentialAgent``) and the agents its ``AgentTool`` s wrap. Each request names the
    agent making the call (its ``name``) in the subject's ``config.attribute_names.acting_agent``,
    so a policy can treat sub-agents differently. An agent already authorized is left as it is:
    to give a sub-agent another config, authorize it first with that config.

    Args:
        agent (Agent): The ADK agent.
        config (AuthZConfig, optional): What to enforce and where the PDP is.

    Returns:
        Agent: ``agent``, with the callbacks attached.
    """
    config = config or AuthZConfig()
    az = _AdkAuthZ(config)
    for each in _agent_tree(agent):
        _attach(each, az)
    return agent


def _agent_tree(root: Any) -> list[Any]:
    """The agent and every agent below it (sub-agents, agents an AgentTool wraps), once each."""
    seen: dict[int, Any] = {}
    stack = [root]
    while stack:
        agent = stack.pop()
        if id(agent) in seen:
            continue
        seen[id(agent)] = agent
        below = list(getattr(agent, "sub_agents", None) or [])
        below += [t.agent for t in getattr(agent, "tools", None) or [] if _is_agent_tool(t)]
        stack.extend(reversed(below))
    return list(seen.values())


def _is_agent_tool(tool: Any) -> bool:
    return type(tool).__name__ == "AgentTool" and getattr(tool, "agent", None) is not None


def _attach(agent: Any, az: _AdkAuthZ) -> None:
    """Attach the callbacks to one agent that calls a model (an ``LlmAgent``), unless already there."""
    fields = getattr(type(agent), "model_fields", None)
    if fields is not None and "before_tool_callback" not in fields:
        return  # a workflow agent (Sequential, Parallel, Loop) calls no model or tool itself
    if _is_authorized(agent):
        return
    config = az.config
    if config.enforces(EnforcementPoint.BEFORE_MODEL) or config.enforces(EnforcementPoint.TOOL_VISIBILITY):
        _prepend_callback(agent, "before_model_callback", az.before_model)
    if config.enforces(EnforcementPoint.AFTER_MODEL):
        _prepend_callback(agent, "after_model_callback", az.after_model)
    if config.enforces(EnforcementPoint.BEFORE_TOOL):
        _prepend_callback(agent, "before_tool_callback", az.before_tool)
    if config.enforces(EnforcementPoint.AFTER_TOOL):
        _prepend_callback(agent, "after_tool_callback", az.after_tool)


def _is_authorized(agent: Any) -> bool:
    """Whether one of our callbacks is already on the agent."""
    for field in ("before_model_callback", "after_model_callback", "before_tool_callback", "after_tool_callback"):
        existing = getattr(agent, field, None)
        callbacks = existing if isinstance(existing, list) else [existing]
        if any(isinstance(getattr(cb, "__self__", None), _AdkAuthZ) for cb in callbacks):
            return True
    return False


def _prepend_callback(agent: Any, field: str, callback) -> None:
    """Put our callback first in an ADK agent callback field, keeping the existing ones after it.

    The field accepts a callable or a list. First, because ADK stops at the first callback that
    returns a value: behind a user callback that does, ours would never run.
    """
    existing = getattr(agent, field, None)
    if existing is None:
        new = callback
    elif isinstance(existing, list):
        new = [callback, *existing]
    else:
        new = [callback, existing]
    setattr(agent, field, new)


# ── helpers ──────────────────────────────────────────────────────────────────


def _can_request_confirmation(tool_context: Any) -> bool:
    """Whether this tool call can pause for ADK tool confirmation.

    The context needs ``request_confirmation`` and the function call id.
    """
    return callable(getattr(tool_context, "request_confirmation", None)) and bool(
        getattr(tool_context, "function_call_id", None)
    )


def _adk_confirmation(tool_context: Any) -> ApprovalDecision | None:
    """Get the human's answer when ADK re-runs a confirmed tool call, or None on a first run.

    ``approver`` / ``note`` are read from the answer's payload, if set.
    """
    confirmation = getattr(tool_context, "tool_confirmation", None)
    if confirmation is None:
        return None
    payload = getattr(confirmation, "payload", None)
    payload = payload if isinstance(payload, dict) else {}
    return ApprovalDecision.from_answer({**payload, "approved": getattr(confirmation, "confirmed", False) is True})


def _parts_text(parts: Any) -> str:
    return "".join(getattr(p, "text", "") or "" for p in (parts or []))


def _adk_user_messages(tool_context: Any) -> list[str]:
    """Get the user's own messages in the session, oldest first, for intent classification.

    Only events authored by the user count, minus function-response events, so tool output
    can't shape the intent. The current turn's ``user_content`` is appended if the session
    doesn't hold it yet.
    """
    events = getattr(getattr(tool_context, "session", None), "events", None) or []
    messages = []
    for event in events:
        if getattr(event, "author", None) != "user":
            continue
        parts = getattr(getattr(event, "content", None), "parts", None) or []
        if any(getattr(p, "function_response", None) for p in parts):
            continue
        text = _parts_text(parts).strip()
        if text:
            messages.append(text)
    current = _parts_text(getattr(getattr(tool_context, "user_content", None), "parts", None)).strip()
    if current and (not messages or messages[-1] != current):
        messages.append(current)
    return messages


def _adk_user_prompt(llm_request: Any) -> str | None:
    """Get the latest user turn's text from an ``LlmRequest``, or None if it isn't a fresh prompt.

    A function-response continuation after a tool ran is not a fresh prompt.
    """
    contents = getattr(llm_request, "contents", None) or []
    if not contents or getattr(contents[-1], "role", None) != "user":
        return None
    parts = getattr(contents[-1], "parts", None) or []
    if any(getattr(p, "function_response", None) for p in parts):
        return None  # tool-result continuation, not a new user prompt
    text = _parts_text(parts).strip()
    return text or None


def _adk_answer_text(llm_response: Any) -> str | None:
    """Get the model's text from a complete ``LlmResponse``, or None.

    None for a streaming partial or a response with no text. Text sent alongside a function
    call counts: it reaches the user too. (Partials stream to the user before the complete
    response is checked.)
    """
    if getattr(llm_response, "partial", False):
        return None
    parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
    text = _parts_text(parts).strip()
    return text or None


def _adk_refusal(reason: str, llm_response: Any = None):
    """Build an ADK ``LlmResponse`` carrying the refusal text.

    Returned from ``before_model`` to block the model, or from ``after_model`` to replace its
    answer. The function calls of ``llm_response`` are kept: only its text is withheld. ADK is
    imported lazily so importing this module never requires google-adk.
    """
    from google.adk.models import LlmResponse
    from google.genai import types

    parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
    calls = [p for p in parts if getattr(p, "function_call", None)]
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=reason), *calls]))


def _prune_tools(llm_request: Any, permitted: set[str], *, refused_on_call: bool) -> None:
    """Remove denied tools from the function declarations the model sees.

    With ``before_tool`` enforcing (``refused_on_call``), ADK's name → tool map is kept: a call
    to a hidden tool then reaches ``before_tool``, which refuses it, where a missing entry would
    make ADK raise and end the run. Without it, nothing else would stop that call, so the
    denied tools leave the map too and ADK refuses them.
    """
    _prune_declarations(getattr(llm_request, "config", None), permitted)
    tools_dict = getattr(llm_request, "tools_dict", None)
    if not refused_on_call and isinstance(tools_dict, dict):
        for name in [n for n in tools_dict if n not in permitted]:
            del tools_dict[name]


def _prune_declarations(config: Any, permitted: set[str]) -> None:
    """Drop denied function declarations from the generate-content config.

    Then drop any function tool left with no declarations (a non-function tool is kept).
    """
    tools = getattr(config, "tools", None)
    if not tools:
        return
    for tool in tools:
        decls = getattr(tool, "function_declarations", None)
        if decls:
            tool.function_declarations = [fd for fd in decls if getattr(fd, "name", None) in permitted]
    config.tools = [t for t in tools if getattr(t, "function_declarations", None) or not _is_function_tool(t)]


def _is_function_tool(tool: Any) -> bool:
    return getattr(tool, "function_declarations", None) is not None
