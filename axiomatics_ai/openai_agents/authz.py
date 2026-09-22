"""Client-side PDP authorization for OpenAI Agents SDK agents (e.g. agents that consume MCP tools).

``authorized_openai_agent(agent, config)`` attaches PDP enforcement to an already-constructed
``Agent`` and returns it — it does NOT wrap or subclass the agent. It imports **no** ``agents``
(it duck-types the objects the SDK passes around), so the SDK adds no dependency; you bring
your own agent.

The SDK exposes a single seam that the Runner calls every turn with the run's identity
context: ``agent.get_all_tools(run_context)`` returns *all* tools — native ``@function_tool``
tools and MCP tools (converted to ``FunctionTool``) alike. All three enforcement points hang
off wrapping that one method:

- **tool visibility** — one batched PDP ``get_permitted_tools`` call (cached), then the
  returned tool list is filtered so the model never sees denied tools.
- **tool invocation** — each surviving tool's ``on_invoke_tool`` is wrapped to check
  ``authorized_tool_invocation`` first; a denied call returns a denial string (the tool never runs).
  A call the policy wants a human to approve goes through the SDK's own approval flow: the
  tool's ``needs_approval`` also asks the PDP (combined with any setting of its own), unless
  an ``approver`` is configured.
- **post-invocation** — the same wrapper filters the tool's result records via the configured
  ``AFTER_TOOL`` Enforcer's ``strategy`` before they reach the model.

The **BEFORE_MODEL / AFTER_MODEL content guards** are wired as SDK **guardrails** (an input
guardrail on the user prompt, an output guardrail on the final answer). Unlike the in-band
refusal on LangChain/ADK/Strands, an OpenAI guardrail can only *tripwire*: on an PDP block it
sets ``tripwire_triggered=True`` and the Runner raises ``InputGuardrailTripwireTriggered`` /
``OutputGuardrailTripwireTriggered`` — the caller catches that and reads the refusal from the
result's ``output_info["reason"]``. (This is OpenAI's native content-guard mechanism; it halts
the run rather than substituting a refusal message.)

Lifecycle hooks (``RunHooks``/``AgentHooks``) are intentionally NOT used — they are purely
observational (every method returns ``None``); they cannot block a call or rewrite a result.

Identity is the PDP subject read from the object passed to ``Runner.run(agent, input,
context=...)`` (reached as ``run_context.context`` and, inside the tool call, as
``tool_context.context``). Populate it per invocation, or reuse the same object across runs to
bind once. It is handed to the reused ``DefaultClientContextProvider`` (override with
``context_provider=``).
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import threading
from collections import OrderedDict
from typing import Any

from axiomatics_ai.core.authz_helpers import resolve_authz_context
from axiomatics_ai.core.config import AuthZConfig, DefaultClientContextProvider, EnforcementPoint
from axiomatics_ai.core.enforcement import (
    ToolCallDecision,
    authorized_tool_invocation,
    authorized_tools,
    denial_message,
    guard_content,
    needs_human_approval,
    post_process_tool_output,
)
from axiomatics_ai.core.pdp import ApprovalDecision, ApprovalRequest
from axiomatics_ai.core.pdp.obligations import args_hash
from axiomatics_ai.core.post_invocation import PostInvocationContext

logger = logging.getLogger(__name__)

_WRAPPED_FLAG = "_ads_authz_wrapped"


def _as_dict(context: Any) -> dict[str, Any]:
    """Coerce the run context into a plain dict for the context provider.

    The run context is whatever the app passed to ``Runner.run(context=...)``.
    """
    if context is None:
        return {}
    if isinstance(context, dict):
        return context
    if hasattr(context, "to_dict"):
        # Third-party context objects: to_dict() may raise anything, and a failure here just
        # means we fall through to vars() below — never a reason to fail the request.
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            data = context.to_dict()
            if isinstance(data, dict):
                return data
    try:
        return dict(vars(context))
    except TypeError:
        return {}


def _parse_args(input_json: Any) -> dict[str, Any]:
    if isinstance(input_json, dict):
        return input_json
    if isinstance(input_json, str) and input_json:
        try:
            parsed = json.loads(input_json)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


class _OpenAIAuthZ:
    """The PDP client, cache and enforcement logic behind the OpenAI adapter.

    Its methods take duck-typed SDK objects, so they can be unit-tested with plain fakes.
    """

    _SURFACE = "openai_agents"

    def __init__(self, config: AuthZConfig, agent_name: str | None = None) -> None:
        self.config = config
        self.agent_name = agent_name   # sent as the subject's attribute_names.acting_agent
        self._provider = self.config.context_provider or DefaultClientContextProvider()
        self._ads = config.build_pdp_client()
        self._cache = config.build_cache()
        # Calls paused for human approval in this process: call_id → (tool name, context,
        # arguments). The SDK applies a rejection on resume without calling our code, so the
        # next turn's get_all_tools logs it from here (see log_rejections). A run resumed in
        # another process isn't in here, so its rejection isn't logged by us.
        self._pending: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        # Calls whose human approval was used: a saved RunState resumed again must not run
        # the tool again on the same approval (bounded; in this process only).
        self._used_approvals: OrderedDict[str, None] = OrderedDict()

    def use_approval(self, call_id: str | None, ctx: dict[str, Any], args: dict[str, Any]) -> bool:
        """Record that a call ran on its approval; False if that approval was already used.

        Keyed by the call id with its caller and arguments: a provider may number call ids per
        run, so the same id can come back for another call.
        """
        if not call_id:
            return True
        used = f"{call_id}:{args_hash(ctx)}:{args_hash(args)}"
        with self._pending_lock:
            if used in self._used_approvals:
                return False
            self._used_approvals[used] = None
            while len(self._used_approvals) > 4096:
                self._used_approvals.popitem(last=False)
            return True

    def remember_pending(self, call_id: str, tool_name: str, ctx: dict[str, Any], args: dict[str, Any]) -> None:
        """Remember a call paused for approval, so its rejection can be logged later."""
        with self._pending_lock:
            self._pending[call_id] = (tool_name, ctx, args)

    def forget_pending(self, call_id: str | None) -> bool:
        """Forget a paused call (answered, or never paused); return whether it was paused here."""
        with self._pending_lock:
            return self._pending.pop(call_id or "", None) is not None

    def log_rejections(self, run_context: Any) -> None:
        """Log a rejection for each call paused here that the app rejected (``RunState.reject``)."""
        with self._pending_lock:
            pending = list(self._pending.items())
        run_ctx = _as_dict(getattr(run_context, "context", None))
        for call_id, (tool_name, ctx, args) in pending:
            status = _sdk_approval(run_context, tool_name, call_id)
            if status is None or status.approved:
                continue   # not answered yet (or in another run), or approved: on_invoke_tool forgets it
            if ctx != run_ctx and not _rejected_by_call_id(run_context, tool_name, call_id):
                continue   # a blanket reject in another caller's run says nothing about this call
            self.forget_pending(call_id)
            self.invocation_decision(
                ctx, tool_name, args, _openai_user_messages(run_context),
                approval=ApprovalDecision(approved=False, note="rejected with RunState.reject"),
                call_id=call_id,
            )

    # ── model-I/O content guard (BEFORE_MODEL / AFTER_MODEL) ────────────────────
    def guard_model_io(self, ctx: dict[str, Any], point: EnforcementPoint, content: str):
        """Run the point's enrichers on ``content`` and let PDP decide (allow / block)."""
        return guard_content(
            config=self.config, pdp=self._ads, cache=self._cache, provider=self._provider,
            surface=self._SURFACE, point=point, content=content, run_context=ctx,
            agent_name=self.agent_name,
        )

    def _resolve(self, ctx: dict[str, Any]):
        return resolve_authz_context(
            self._provider, ctx,
            agent_name=self.agent_name, agent_attribute=self.config.attribute_names.acting_agent,
        )

    # ── visibility ──────────────────────────────────────────────────────────────
    def permitted_tools(
        self, ctx: dict[str, Any], tool_names: list[str],
        point: EnforcementPoint = EnforcementPoint.TOOL_VISIBILITY,
    ) -> set[str]:
        """Decide which of ``tool_names`` the caller may use at ``point`` (one batched request).

        At ``TOOL_VISIBILITY``: which tools the caller may see. At ``BEFORE_TOOL``: which of the
        tools that can't be checked call by call (hosted, shell, computer) the caller may use.
        """
        if not tool_names:
            return set()
        resolved = self._resolve(ctx)
        action = self._provider.get_action(self.config.enforcer(point).action_name, ctx)
        return set(authorized_tools(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._SURFACE,
            subject=resolved.subject,
            action=action,
            tool_names=tool_names,
            environment=resolved.environment,
            enforcement_point="tool_visibility" if point is EnforcementPoint.TOOL_VISIBILITY else "tool_invocation",
        ))

    # ── invocation ──────────────────────────────────────────────────────────────
    def invocation_decision(
        self, ctx: dict[str, Any], tool_name: str, args: dict[str, Any], user_messages: list[str],
        *, approval: ApprovalDecision | None = None, call_id: str | None = None,
    ) -> ToolCallDecision:
        """The BEFORE_TOOL decision for one call (see ``authorized_tool_invocation``)."""
        resolved = self._resolve(ctx)
        action = self._provider.get_action(
            self.config.enforcer(EnforcementPoint.BEFORE_TOOL).action_name, ctx
        )
        resource = self._provider.get_resource(
            tool_name, ctx, args=args,
            resource_name_attribute=self.config.attribute_names.tool,
        )
        return authorized_tool_invocation(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._SURFACE,
            subject=resolved.subject,
            action=action,
            resource=resource,
            environment=resolved.environment,
            user_messages=user_messages,
            approval=approval,
            args=args,
            call_id=call_id,
        )

    def pending_approval(
        self, ctx: dict[str, Any], tool_name: str, args: dict[str, Any], user_messages: list[str],
        call_id: str | None = None,
    ) -> ApprovalRequest | None:
        """Get the approval request when the call must pause for a human, else None.

        The pre-check behind ``needs_approval`` (see ``needs_human_approval``); it logs only
        ``approval_required``, so an ordinary call is logged once, by ``on_invoke_tool``.
        """
        resolved = self._resolve(ctx)
        action = self._provider.get_action(
            self.config.enforcer(EnforcementPoint.BEFORE_TOOL).action_name, ctx
        )
        resource = self._provider.get_resource(
            tool_name, ctx, args=args,
            resource_name_attribute=self.config.attribute_names.tool,
        )
        return needs_human_approval(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._SURFACE,
            subject=resolved.subject,
            action=action,
            resource=resource,
            environment=resolved.environment,
            user_messages=user_messages,
            args=args,
            call_id=call_id,
        )

    def invocation_denial(
        self, ctx: dict[str, Any], tool_name: str, args: dict[str, Any], user_messages: list[str],
        *, approval: ApprovalDecision | None = None, call_id: str | None = None,
    ) -> str | None:
        """Get a denial message if the call is not permitted, else None.

        ``approval`` is the human's answer when the SDK's own approval flow ran first
        (``needs_approval``).
        """
        decision = self.invocation_decision(
            ctx, tool_name, args, user_messages, approval=approval, call_id=call_id,
        )
        if not decision.permitted:
            logger.debug("OpenAI tool call [%s]: Deny", tool_name)
            return denial_message(tool_name, decision)
        return None

    # ── post-invocation ─────────────────────────────────────────────────────────
    def filter_result(self, ctx: dict[str, Any], tool_name: str, args: dict[str, Any], result: Any) -> Any:
        """Run the post-invocation strategy on a tool result.

        Returns the filtered result, the result unchanged when not filterable, or a withheld
        message under ``on_unrecognized_output="fail_closed"``.
        """
        post_enforcer = self.config.enforcer(EnforcementPoint.AFTER_TOOL)
        if post_enforcer is None or post_enforcer.strategy is None:
            return result
        resolved = self._resolve(ctx)
        action_name = post_enforcer.action_overrides.get(tool_name, post_enforcer.action_name)
        action = self._provider.get_action(action_name, ctx)
        post_ctx = PostInvocationContext(
            pdp_client=self._ads,
            subject=resolved.subject,
            environment=resolved.environment,
            tool_name=tool_name,
            args=args,
            resource=self._provider.get_resource(
                tool_name, ctx, args=args,
                resource_name_attribute=self.config.attribute_names.tool,
            ),
            action=action,
            fail_open=self.config.connection.fail_open,
        )
        outcome = post_process_tool_output(
            result, enforcer=post_enforcer, post_ctx=post_ctx, config=self.config,
            surface=self._SURFACE,
        )
        if outcome.is_block:
            return outcome.reason
        return result if outcome.content is None else outcome.content


def _input_text(run_input: Any) -> str:
    """Get the user-prompt text from a run input (a ``str`` or a list of input items).

    Concatenates the text of user-role items; ignores tool and other items.
    """
    if isinstance(run_input, str):
        return run_input.strip()
    if isinstance(run_input, list):
        parts = [text for item in run_input for text in _user_item_texts(item)]
        return " ".join(parts).strip()
    return ""


def _openai_user_messages(tool_context: Any) -> list[str]:
    """Get the user's own messages in this run's input, oldest first, for intent.

    ``turn_input`` is a string (one user message) or a list of items; only ``user``-role items
    count, so tool output can't shape the intent.
    """
    turn_input = getattr(tool_context, "turn_input", None)
    if isinstance(turn_input, str):
        return [turn_input] if turn_input.strip() else []
    messages = [" ".join(_user_item_texts(item)).strip() for item in turn_input or []]
    return [m for m in messages if m]


def _user_item_texts(item: Any) -> list[str]:
    """Get the text parts of one run-input item, or nothing if it isn't a user message.

    ``content`` is either a plain string or a list of content blocks with a ``text`` key.
    """
    if not isinstance(item, dict) or item.get("role") != "user":
        return []
    content = item.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")]
    return []


def _output_text(agent_output: Any) -> str:
    """Get the final-answer text from an agent's output.

    A plain string, or a structured object rendered via ``str``.
    """
    if agent_output is None:
        return ""
    return (agent_output if isinstance(agent_output, str) else str(agent_output)).strip()


def _wrap_on_invoke(tool: Any, engine: _OpenAIAuthZ, config: AuthZConfig) -> None:
    """Wrap a tool's ``on_invoke_tool`` with invocation + post-invocation enforcement (once).

    A tool object shared by several agents is wrapped once; each call is checked by the engine
    of the agent making it (``tool_context.agent``), so with that agent's config and name.
    ``needs_approval`` gets no agent, so its pre-check uses the first agent's engine.
    """
    original = getattr(tool, "on_invoke_tool", None)
    if original is None or getattr(tool, _WRAPPED_FLAG, False):
        return
    tool_name = getattr(tool, "name", "")
    first_engine = engine

    async def guarded(tool_context: Any, input_json: Any):
        engine = getattr(getattr(tool_context, "agent", None), _ENGINE_ATTR, None) or first_engine
        config = engine.config
        ctx = _as_dict(getattr(tool_context, "context", None))
        args = _parse_args(input_json)
        if config.enforces(EnforcementPoint.BEFORE_TOOL):
            call_id = getattr(tool_context, "tool_call_id", None)
            approval = _call_approval(tool_context, tool_name, call_id, engine, first_engine, ctx, args)
            # The PDP calls block, so they run on a worker thread, off the event loop.
            denial = await asyncio.to_thread(
                engine.invocation_denial,
                ctx, tool_name, args, _openai_user_messages(tool_context),
                approval=approval, call_id=call_id,
            )
            if denial is not None:
                return denial
        result = await original(tool_context, input_json)
        # An agent used as a tool is authorized too (authorized_openai_agent walks into it): its
        # tool results were post-processed there, and its answer is text no strategy reads.
        if config.enforces(EnforcementPoint.AFTER_TOOL) and not getattr(tool, "_is_agent_tool", False):
            return await asyncio.to_thread(engine.filter_result, ctx, tool_name, args, result)
        return result

    tool.on_invoke_tool = guarded
    if config.enforces(EnforcementPoint.BEFORE_TOOL) and hasattr(tool, "needs_approval"):
        tool.needs_approval = _pdp_needs_approval(tool.needs_approval, tool_name, engine)
    # Frozen/slotted tool objects reject the marker; the only cost is that a re-wrap can't be
    # detected, which is harmless (guarded is idempotent).
    with contextlib.suppress(AttributeError, TypeError):  # pragma: no cover - defensive
        setattr(tool, _WRAPPED_FLAG, True)


def _call_approval(
    tool_context: Any, tool_name: str, call_id: str | None, engine: _OpenAIAuthZ, first_engine: _OpenAIAuthZ,
    ctx: dict[str, Any], args: dict[str, Any],
) -> ApprovalDecision | None:
    """The human's answer for this call, if the SDK holds one that still counts.

    Answered (approved), or never paused: a blanket ``always_approve`` counts only for the call
    paused here (see ``_sdk_approval``; the pre-check paused it on ``first_engine``), and an
    approval runs its call once.
    """
    paused_here = engine.forget_pending(call_id)
    paused_here = first_engine.forget_pending(call_id) or paused_here
    approval = _sdk_approval(tool_context, tool_name, call_id, paused_here=paused_here)
    if approval is not None and approval.approved and not first_engine.use_approval(call_id, ctx, args):
        return ApprovalDecision(approved=False, note="this approval was already used")
    return approval


def _sdk_approval(
    context: Any, tool_name: str, call_id: str | None, *, paused_here: bool = True,
) -> ApprovalDecision | None:
    """Get the human's answer from the SDK's approval flow, or None if this call had none.

    The answer is set by ``RunState.approve`` / ``reject``. A blanket approval
    (``approve(item, always_approve=True)``) approves every later call of the tool, whatever its
    arguments, which is not what the human saw: it counts only for the call this process paused
    (``paused_here``). An approval of the exact call id always counts, so a ``RunState`` resumed
    in another process still works.
    """
    is_tool_approved = getattr(context, "is_tool_approved", None)
    if not callable(is_tool_approved) or not call_id:
        return None
    status = is_tool_approved(tool_name, call_id)
    if status is True and not paused_here and not _approved_by_call_id(context, tool_name, call_id):
        return None
    return None if status is None else ApprovalDecision(approved=status is True)


def _approved_by_call_id(context: Any, tool_name: str, call_id: str) -> bool:
    """Whether the SDK holds an approval of this exact call, not a blanket one for the tool."""
    record = (getattr(context, "_approvals", None) or {}).get(tool_name)
    approved = getattr(record, "approved", None)
    return isinstance(approved, list) and call_id in approved


def _rejected_by_call_id(context: Any, tool_name: str, call_id: str) -> bool:
    """Whether the SDK holds a rejection of this exact call, not a blanket one for the tool."""
    record = (getattr(context, "_approvals", None) or {}).get(tool_name)
    rejected = getattr(record, "rejected", None)
    return isinstance(rejected, list) and call_id in rejected


def _pdp_needs_approval(original: Any, tool_name: str, engine: _OpenAIAuthZ):
    """Build a ``needs_approval`` that also asks the PDP.

    True when the tool's own setting says so, or when the policy wants a human to approve and
    no approver answered: the SDK then pauses the run with a ``ToolApprovalItem`` in
    ``result.interruptions``. After the app's ``approve`` / ``reject``, the SDK calls this again:
    a rejection is logged here (the tool never runs), an approval is checked in
    ``on_invoke_tool``.
    """

    async def needs_approval(run_context: Any, params: dict[str, Any], call_id: str) -> bool:
        if original is True or (callable(original) and await _maybe_await(original(run_context, params, call_id))):
            return True
        if _sdk_approval(run_context, tool_name, call_id) is not None:
            # Already answered: the SDK applies it (on_invoke_tool checks an approval). The SDK
            # won't pause a call a blanket approval covers, so if the policy wants approval for
            # it, on_invoke_tool refuses it as not approved.
            return True
        ctx = _as_dict(getattr(run_context, "context", None))
        request = await asyncio.to_thread(
            engine.pending_approval, ctx, tool_name, params, _openai_user_messages(run_context), call_id,
        )
        if request is not None:
            engine.remember_pending(call_id, tool_name, ctx, params)
        return request is not None

    return needs_approval


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _tools_visible_to(ctx: dict, tools: list, engine: _OpenAIAuthZ) -> list:
    """The subset of ``tools`` the caller may see, per the TOOL_VISIBILITY decision."""
    names = [name for name in (getattr(t, "name", None) for t in tools) if name]
    permitted = engine.permitted_tools(ctx, names)
    denied = set(names) - permitted
    if denied:
        logger.debug("OpenAI tool visibility [%s]: hiding %s", ctx, sorted(denied))
    return [t for t in tools if getattr(t, "name", None) in permitted]


def _without_denied_unchecked_tools(ctx: dict, tools: list, engine: _OpenAIAuthZ) -> list:
    """Drop the tools the call-by-call check can't reach, unless the policy permits them by name.

    Hosted tools (web search, file search, code interpreter, image generation, hosted MCP) run
    on OpenAI's servers, and shell, apply-patch and computer tools run through the SDK's own
    executors: none goes through ``on_invoke_tool``. Their names are decided at BEFORE_TOOL, in
    one batched request, without arguments; a tool the policy doesn't permit is removed.
    """
    unchecked = [t.name for t in tools if not hasattr(t, "on_invoke_tool") and getattr(t, "name", None)]
    if not unchecked:
        return tools
    denied = set(unchecked) - engine.permitted_tools(ctx, unchecked, EnforcementPoint.BEFORE_TOOL)
    if denied:
        logger.debug("OpenAI tools without a per-call check [%s]: removing %s", ctx, sorted(denied))
    return [t for t in tools if getattr(t, "name", None) not in denied]


def _wrap_tools_for_invocation(tools: list, engine: _OpenAIAuthZ, config: AuthZConfig) -> None:
    """Wrap every tool's ``on_invoke_tool`` for BEFORE_TOOL / AFTER_TOOL enforcement."""
    for tool in tools:
        if hasattr(tool, "on_invoke_tool"):
            _wrap_on_invoke(tool, engine, config)


def authorized_openai_agent(agent: Any, config: AuthZConfig | None = None):
    """Attach authorization to an OpenAI Agents SDK ``Agent`` and return it (the same instance).

    Wraps ``agent.get_all_tools`` (which the Runner calls every turn with the run's context) to
    filter the tool list (visibility) and wrap each tool's ``on_invoke_tool`` (invocation and
    post-invocation; a call the policy wants approved pauses through the SDK's own
    ``needs_approval`` flow). The content guards are attached as input and output guardrails: a
    block raises ``InputGuardrailTripwireTriggered`` / ``OutputGuardrailTripwireTriggered``.
    Only the points the config enforces are wired. Identity comes from
    ``Runner.run(..., context=...)`` through the config's context provider
    (``DefaultClientContextProvider`` by default).

    The agents below it get the same config: its handoff targets and the agents it uses as
    tools (``agent.as_tool()``). Each request names the agent making the call (its ``name``) in
    the subject's ``config.attribute_names.acting_agent``, so a policy can treat them
    differently. An agent already authorized is left as it is: to give one another config,
    authorize it first with that config. Clones (``agent.clone()``) need their own call.

    Args:
        agent (Agent): The OpenAI Agents SDK agent.
        config (AuthZConfig, optional): What to enforce and where the PDP is.

    Returns:
        Agent: ``agent``, with enforcement attached.
    """
    config = config or AuthZConfig()
    for each in _agent_tree(agent):
        if not getattr(each, _AUTHORIZED_FLAG, False):
            _attach(each, config)
    return agent


_AUTHORIZED_FLAG = "_axiomatics_authorized"
_ENGINE_ATTR = "_axiomatics_engine"


def _agent_tree(root: Any) -> list[Any]:
    """The agent and every agent below it (handoff targets, agents used as tools), once each."""
    seen: dict[int, Any] = {}
    stack = [root]
    while stack:
        agent = stack.pop()
        if agent is None or id(agent) in seen:
            continue
        seen[id(agent)] = agent
        below = [_handoff_agent(h) for h in getattr(agent, "handoffs", None) or []]
        below += [
            t._agent_instance for t in getattr(agent, "tools", None) or []
            if getattr(t, "_is_agent_tool", False)
        ]
        stack.extend(reversed(below))
    return list(seen.values())


def _handoff_agent(handoff: Any) -> Any:
    """The agent a handoff leads to: an ``Agent``, or a ``Handoff`` made by ``handoff(agent)``.

    A ``Handoff`` built by hand (its own ``on_invoke_handoff``) names no agent; it gets a
    warning, since the agent it returns is not authorized here.
    """
    if hasattr(handoff, "get_all_tools"):
        return handoff
    ref = getattr(handoff, "_agent_ref", None)
    target = ref() if callable(ref) else None
    if target is None:
        logger.warning(
            "Handoff '%s' names no agent: authorize the agent it hands off to yourself.",
            getattr(handoff, "agent_name", handoff),
        )
    return target


def _attach(agent: Any, config: AuthZConfig) -> None:
    """Attach enforcement to one agent."""
    engine = _OpenAIAuthZ(config, getattr(agent, "name", None))
    original_get_all_tools = agent.get_all_tools

    async def get_all_tools(run_context: Any):
        tools = await original_get_all_tools(run_context)
        ctx = _as_dict(getattr(run_context, "context", None))
        if config.enforces(EnforcementPoint.BEFORE_TOOL):
            await asyncio.to_thread(engine.log_rejections, run_context)
        if config.enforces(EnforcementPoint.TOOL_VISIBILITY):
            tools = await asyncio.to_thread(_tools_visible_to, ctx, tools, engine)
        if config.enforces(EnforcementPoint.BEFORE_TOOL):
            tools = await asyncio.to_thread(_without_denied_unchecked_tools, ctx, tools, engine)
        if config.enforces(EnforcementPoint.BEFORE_TOOL) or config.enforces(EnforcementPoint.AFTER_TOOL):
            _wrap_tools_for_invocation(tools, engine, config)
        return tools

    agent.get_all_tools = get_all_tools
    _attach_model_io_guardrails(agent, engine, config)
    with contextlib.suppress(AttributeError, TypeError):  # pragma: no cover - defensive
        setattr(agent, _AUTHORIZED_FLAG, True)
        setattr(agent, _ENGINE_ATTR, engine)   # a shared tool's wrapper finds the calling agent's


def _attach_model_io_guardrails(agent: Any, engine: _OpenAIAuthZ, config: AuthZConfig) -> None:
    """Wire BEFORE_MODEL / AFTER_MODEL content guards as OpenAI Agents guardrails.

    Unlike the in-band refusal on the other paths, an OpenAI guardrail can only *tripwire*: on
    a block it sets ``tripwire_triggered=True`` and the Runner raises
    ``Input/OutputGuardrailTripwireTriggered`` (the refusal is carried in ``output_info``). The
    caller catches that and surfaces the refusal. ``agents`` is imported lazily so this module
    stays import-free until guardrails are actually wired.
    """
    if not (config.enforces(EnforcementPoint.BEFORE_MODEL) or config.enforces(EnforcementPoint.AFTER_MODEL)):
        return
    from agents import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail

    _allow = GuardrailFunctionOutput(output_info={"reason": None}, tripwire_triggered=False)

    def _verdict(ctx: Any, point: EnforcementPoint, content: str):
        result = engine.guard_model_io(_as_dict(getattr(ctx, "context", None)), point, content)
        return GuardrailFunctionOutput(output_info={"reason": result.reason}, tripwire_triggered=result.is_block)

    if config.enforces(EnforcementPoint.BEFORE_MODEL):
        async def before_model_guard(ctx: Any, agent_: Any, run_input: Any):
            content = _input_text(run_input)
            if not content:
                return _allow  # no user text to classify → nothing to guard
            return await asyncio.to_thread(_verdict, ctx, EnforcementPoint.BEFORE_MODEL, content)

        agent.input_guardrails = [
            *(getattr(agent, "input_guardrails", None) or []),
            # Not in parallel: the guard decides before the model runs (and calls tools), not
            # while it does.
            InputGuardrail(
                guardrail_function=before_model_guard, name="axiomatics-before-model", run_in_parallel=False,
            ),
        ]

    if config.enforces(EnforcementPoint.AFTER_MODEL):
        async def after_model_guard(ctx: Any, agent_: Any, agent_output: Any):
            content = _output_text(agent_output)
            if not content:
                return _allow  # no answer text to classify → nothing to guard
            return await asyncio.to_thread(_verdict, ctx, EnforcementPoint.AFTER_MODEL, content)

        agent.output_guardrails = [
            *(getattr(agent, "output_guardrails", None) or []),
            OutputGuardrail(guardrail_function=after_model_guard, name="axiomatics-after-model"),
        ]
