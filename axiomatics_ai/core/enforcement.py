"""Shared enforcement helpers: cache → PDP → decision-event, in one place.

Every surface (agents, FastMCP, A2A, ADK, OpenAI, Claude, MCP) runs the same three-step
dance at each enforcement point: check the decision cache, fall back to an PDP call, cache
the result — and now also emit an ``on_decision`` hook event (with the right ``cached`` flag)
under a correlation scope. These helpers hold that dance so each engine is a one-line call
instead of a re-implementation, and so the hook taxonomy stays identical across surfaces.

- ``authorized_tools`` — batch visibility (``get_permitted_tools``).
- ``authorized_call``  — single invocation Permit/Deny (``evaluate_tool_call``).
- ``authorized_tool_invocation`` — BEFORE_TOOL decision, adding the user's classified intent
  and asking for human approval when the policy requires it.
- ``apply_post_invocation`` — run a post-invocation strategy and emit its outcome.
- ``post_process_tool_output`` — the same, on a framework's tool result: unwrap the data,
  run the strategy, rebuild the result (or withhold output the strategy cannot inspect).

``cache`` may be ``None`` for surfaces that don't cache (e.g. the MCP visibility patch).
The cache holds the PDP's decision with its obligations and advice, so obligation handlers
run on every call, cache hit or not. The dispatcher is built from ``config`` per call
(cheap; a no-op when no hooks are set).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from axiomatics_ai.core.authz_helpers import (
    evaluate_tool_call,
    get_permitted_tools,
    resolve_authz_context,
)
from axiomatics_ai.core.config import EnforcementPoint, block_message_for
from axiomatics_ai.core.pdp.client import _elapsed_ms
from axiomatics_ai.core.pdp.decision import Decision, Obligation
from axiomatics_ai.core.pdp.obligations import (
    ApprovalDecision,
    ApprovalRequest,
    Outcome,
    Resolution,
    args_hash,
    reasons,
    ttl_seconds,
)
from axiomatics_ai.core.post_invocation import PostInvocationContext, PostInvocationStrategy
from axiomatics_ai.core.records_helpers import mcp_text_block, transform_tool_output

logger = logging.getLogger(__name__)

# Only a policy's answer is cached (fail_open is applied again on every read): an Indeterminate
# (often a transient PDP or PIP error) is decided again next time, not for the cache's TTL.
_DEFINITE = (Decision.PERMIT, Decision.DENY, Decision.NOT_APPLICABLE)


class EnforcementAction(str, Enum):
    """What an enforcement point does with the content flowing through it.

    Attributes:
        ALLOW: Pass the content through unchanged.
        TRANSFORM: Replace the content with a transformed version.
        BLOCK: Stop the content, with a reason for the agent or user.
    """
    ALLOW = "allow"
    TRANSFORM = "transform"
    BLOCK = "block"


@dataclass(frozen=True)
class EnforcementResult:
    """The outcome at an enforcement point.

    Every point is a transformation of what flows through it; ``BLOCK`` is the degenerate
    transform (deny). Build one with ``allow()``, ``transform(content)`` or ``block(reason)``.

    Attributes:
        action (EnforcementAction): Allow, transform, or block.
        content (Any): The replacement content, for ``TRANSFORM``.
        reason (str | None): The agent- or user-facing reason, for ``BLOCK``.
    """

    action: EnforcementAction
    content: Any = None
    reason: str | None = None

    @classmethod
    def allow(cls) -> EnforcementResult:
        """Build an ``ALLOW`` result: the content passes through unchanged.

        Returns:
            EnforcementResult: The result.
        """
        return cls(EnforcementAction.ALLOW)

    @classmethod
    def transform(cls, content: Any) -> EnforcementResult:
        """Build a ``TRANSFORM`` result.

        Args:
            content (Any): What replaces the original content.

        Returns:
            EnforcementResult: The result.
        """
        return cls(EnforcementAction.TRANSFORM, content=content)

    @classmethod
    def block(cls, reason: str) -> EnforcementResult:
        """Build a ``BLOCK`` result.

        Args:
            reason (str): The agent- or user-facing reason.

        Returns:
            EnforcementResult: The result.
        """
        return cls(EnforcementAction.BLOCK, reason=reason)

    @property
    def is_block(self) -> bool:
        """Whether this result blocks the content.

        Returns:
            bool: True for ``BLOCK``.
        """
        return self.action is EnforcementAction.BLOCK


def authorized_tools(
    *,
    pdp: Any,
    cache: Any | None,
    config: Any,
    surface: str,
    subject: dict,
    action: dict,
    tool_names: list[str],
    environment: dict,
    enforcement_point: str = "tool_visibility",
) -> set[str]:
    """Ask which tools the subject may see, in one batched request, and emit the decision.

    Each tool name is sent as ``config.attribute_names.tool``, the same attribute tool
    invocation and post-invocation use. The result is cached unless it was resolved by
    ``fail_open`` (PDP unreachable), so an outage's fallback does not outlive it.

    Args:
        pdp (PdpClient): The PDP client.
        cache (AuthzDecisionCacheWithTTL | None): The decision cache, or None for no caching.
        config (AuthZConfig): The config (fail_open, hooks, attribute names).
        surface (str): The adapter's label for decision events, e.g. ``"langchain_agents"``.
        subject (dict): Subject attributes.
        action (dict): Action attributes, e.g. ``{"name": "tool_visibility"}``.
        tool_names (list[str]): The tools to decide.
        environment (dict): Environment attributes.
        enforcement_point (str): The point the decision event names; ``"tool_invocation"``
            when tools that can't be checked call by call are decided by name.

    Returns:
        set[str]: The permitted tool names.
    """
    dispatcher = config.build_hook_dispatcher()
    resource = {"tool_names": sorted(tool_names)}
    with dispatcher.correlation():
        cached = cache.get(subject, action, resource, environment) if cache is not None else None
        if cached is not None:
            _emit_visibility(
                dispatcher, surface, subject, action, resource, cached, tool_names,
                cached=True, environment=environment, enforcement_point=enforcement_point,
            )
            return cached
        permitted = get_permitted_tools(
            pdp,
            subject=subject,
            action=action,
            tool_names=tool_names,
            environment=environment,
            tool_name_attribute=config.attribute_names.tool,
            fail_open=config.connection.fail_open,
        )
        # A result resolved by fail_open (PDP unreachable) is not cached: it would outlive the
        # outage by the cache TTL — every tool shown, or none, after the PDP is back.
        if cache is not None and getattr(permitted, "answered", True):
            cache.set(subject, action, resource, environment, permitted)
        _emit_visibility(
            dispatcher, surface, subject, action, resource, permitted, tool_names,
            cached=False, environment=environment, enforcement_point=enforcement_point,
        )
        return permitted


def authorized_call(
    *,
    pdp: Any,
    cache: Any | None,
    config: Any,
    surface: str,
    enforcement_point: str = "tool_invocation",
    subject: dict,
    action: dict,
    resource: dict,
    environment: dict,
) -> bool:
    """Ask whether one call is permitted, with caching, and emit the decision.

    A decision that asks for human approval is a Deny here: only a tool call can wait for
    approval (see ``authorized_tool_invocation``).

    Args:
        pdp (PdpClient): The PDP client.
        cache (AuthzDecisionCacheWithTTL | None): The decision cache, or None for no caching.
        config (AuthZConfig): The config (fail_open, hooks, attribute names).
        surface (str): The adapter's label for decision events, e.g. ``"langchain_agents"``.
        enforcement_point (str): The point label for the decision event.
        subject (dict): Subject attributes.
        action (dict): Action attributes.
        resource (dict): Resource attributes.
        environment (dict): Environment attributes.

    Returns:
        bool: Whether the call is permitted.
    """
    dispatcher = config.build_hook_dispatcher()
    with dispatcher.correlation():
        return _decide_call(
            pdp=pdp, cache=cache, config=config, surface=surface,
            enforcement_point=enforcement_point, subject=subject, action=action,
            resource=resource, environment=environment,
        ).permitted


def _decide_call(
    *, pdp, cache, config, surface, enforcement_point,
    subject, action, resource, environment, approval_possible=False, event_detail=None,
    log_approval_required=True, approval_detail=None,
) -> Resolution:
    """Run cache → PDP → resolve → decision event for one single request.

    The cache holds the PDP's decision (not the outcome), so obligation handlers run on every
    call; a decision the PDP never gave (unreachable) is not cached. ``log_approval_required``
    False skips the event for an "approval required" outcome that was already logged (the
    engine is calling back with the human's answer); ``approval_detail`` is added to that event
    only. Assumes an active correlation scope (its caller opens one): the correlation id is a
    context variable, so the event joins it from this function's own dispatcher.
    """
    dispatcher = config.build_hook_dispatcher()
    decision = cache.get(subject, action, resource, environment) if cache is not None else None
    cached = decision is not None
    if not cached:
        decision = evaluate_tool_call(
            pdp, subject=subject, action=action, resource=resource, environment=environment,
        )
        if cache is not None and decision.decision in _DEFINITE:   # an inconclusive fallback must not outlive it
            cache.set(subject, action, resource, environment, decision)
    resolution = pdp.resolve(
        decision, subject=subject, action=action, resource=resource, environment=environment,
        fail_open=config.connection.fail_open, approval_possible=approval_possible,
    )
    approval_required = resolution.outcome is Outcome.APPROVAL_REQUIRED
    if log_approval_required or not approval_required:
        _emit_invocation(
            dispatcher, surface, enforcement_point, subject, action, resource,
            resolution.outcome.value, cached=cached, environment=environment,
            detail={
                **resolution.detail, **(event_detail or {}),
                **((approval_detail or {}) if approval_required else {}),
            },
        )
    return resolution


def authorized_tool_invocation(
    *,
    pdp: Any,
    cache: Any | None,
    config: Any,
    surface: str,
    subject: dict,
    action: dict,
    resource: dict,
    environment: dict,
    user_messages: list[str],
    approval: ApprovalDecision | None = None,
    can_interrupt: bool = False,
    args: dict | None = None,
    call_id: str | None = None,
) -> ToolCallDecision:
    """Decide a tool call (BEFORE_TOOL), with intent checks and human approval.

    **Intent.** When the BEFORE_TOOL Enforcer has enrichers, each classifies its last
    ``spec.num_user_messages`` user messages (cached per history, so a turn costs one LLM
    call) and the attributes join the request's environment, so the policy can compare the
    call with what the user asked for. No user messages (a server-side path) means intent is
    unavailable, which ``on_unavailable="fail_closed"`` turns into a deny.

    **Approval.** When the policy asks for approval (the ``require-approval`` obligation or
    advice), the answer comes from, in order: ``approval`` (the human's answer when the engine
    resumes), a still-valid remembered approval, the configured approver. With none of these,
    an engine that ``can_interrupt`` gets the request back as ``pending`` (it asks through its
    framework and calls again with ``approval=``); any other engine gets a deny. On approval
    the call is decided again with the approval attribute set, and that decision is final.

    **Pauses.** With a ``call_id``, a paused call is remembered with the decision that asked
    for approval, so the PDP is not asked that again: a replay of the paused step before the
    answer (LangGraph, Strands) gets the same pending request, and the answer is applied to
    the stored decision. All of the call's events share one correlation id.

    Args:
        pdp (PdpClient): The PDP client.
        cache (AuthzDecisionCacheWithTTL | None): The decision cache, or None for no caching.
        config (AuthZConfig): The config (fail_open, hooks, attribute names).
        surface (str): The adapter's label for decision events, e.g. ``"langchain_agents"``.
        subject (dict): Subject attributes.
        action (dict): Action attributes, e.g. ``{"name": "tool_invocation"}``.
        resource (dict): Resource attributes: the tool name and the forwarded arguments.
        environment (dict): Environment attributes.
        user_messages (list[str]): The user's own messages, oldest first; never tool output or
            assistant text.
        approval (ApprovalDecision, optional): The human's answer, when the engine resumes.
        can_interrupt (bool): Whether the engine can pause the call to ask a human.
        args (dict, optional): The tool call's own arguments, shown to the approver; without
            them the approver sees the resource's attributes.
        call_id (str, optional): The framework's id for this tool call.

    Returns:
        ToolCallDecision: Whether the call may run, or the pending approval request.
    """
    dispatcher = config.build_hook_dispatcher()
    with dispatcher.correlation(_call_correlation_id(surface, call_id)):
        environment = _without_approval_attribute(environment, pdp)
        environment, unavailable = _with_intent(config, environment, user_messages, dispatcher)
        if unavailable:
            _emit_invocation(
                dispatcher, surface, "tool_invocation", subject, action, resource, "deny",
                cached=False, environment=environment,
            )
            return ToolCallDecision(permitted=False)
        call = {
            "pdp": pdp, "cache": cache, "config": config, "surface": surface,
            "enforcement_point": "tool_invocation", "subject": subject, "action": action,
            "resource": resource,
        }
        binding = _approval_binding(resource, args)
        paused_key = _paused_key(subject, surface, call_id, binding)
        paused = config._state.paused.get(paused_key) if paused_key else None
        if approval is not None and paused_key and config._state.paused.paused_otherwise(
            paused_key, subject, ignore=config.attribute_names.acting_agent,
        ):
            # The answer was given to this call as it was paused; its arguments have changed
            # since (an edited checkpoint or RunState).
            config._state.paused.pop(paused_key)
            _emit_invocation(
                dispatcher, surface, "tool_invocation", subject, action, resource, "deny",
                cached=False, environment=environment,
                detail={"not_approved": True, "reason": "the answer was for the call as it was paused"},
            )
            return ToolCallDecision(permitted=False, not_approved=True)
        if paused is not None and approval is None:
            return ToolCallDecision(permitted=False, pending=paused[0])   # a replay, still waiting
        if paused is None:
            first = _first_decision(
                call, environment, args=args, binding=binding,
                request_id=_request_id(surface, call_id, binding),
                log_approval_required=approval is None,
            )
            if isinstance(first, ToolCallDecision):
                return first
            paused = first
        request, pdp_decision = paused
        answer = _get_approval(request, dispatcher, call, given=approval, can_interrupt=can_interrupt)
        if answer is None:
            if paused_key:
                config._state.paused.put(paused_key, request, pdp_decision)
            return ToolCallDecision(permitted=False, pending=request)
        if paused_key:
            config._state.paused.pop(paused_key)
        return _decide_after_answer(dispatcher, call, request, pdp_decision, answer, environment)


def _first_decision(
    call: dict, environment: dict, *,
    args: dict | None, binding: str, request_id: str, log_approval_required: bool,
) -> ToolCallDecision | tuple[ApprovalRequest, Any]:
    """Ask the PDP about the call: the final decision, or the approval request and its decision.

    Returns the final ``ToolCallDecision`` when no approval is needed, else the approval
    request and the PDP decision that asked for it.
    """
    resolution = _decide_call(
        **call, environment=environment, approval_possible=True,
        log_approval_required=log_approval_required, approval_detail={"request_id": request_id},
    )
    if resolution.outcome is not Outcome.APPROVAL_REQUIRED:
        return ToolCallDecision(permitted=resolution.permitted)
    request = _approval_request(
        resolution, call["subject"], call["resource"], call["config"],
        args=args, binding=binding, request_id=request_id,
    )
    return request, resolution.decision


def _decide_after_answer(
    dispatcher, call: dict, request: ApprovalRequest, pdp_decision: Any,
    answer: ApprovalDecision, environment: dict,
) -> ToolCallDecision:
    """Decide the call once the approval was answered.

    A deny on rejection; on approval, the decision resolved again (obligation form) or the PDP
    asked again with the approval attribute (advice form).
    """
    if not answer.approved:
        # The call's final decision: without it an audit trail ends at "rejected".
        _emit_invocation(
            dispatcher, call["surface"], "tool_invocation", call["subject"], call["action"],
            call["resource"], "deny", cached=False, environment=environment,
            detail={"not_approved": True, "request_id": request.request_id},
        )
        return ToolCallDecision(permitted=False, not_approved=True)
    if all(isinstance(asked, Obligation) for asked in request.asked_by):
        final = _fulfil_approval(dispatcher, call, pdp_decision, environment)
    else:
        approved_env = {**environment, call["pdp"].obligations.approval.attribute: True}
        final = _decide_call(
            **call, environment=approved_env, event_detail={"evaluated_again": True},
        )
    if final.permitted:
        call["config"]._state.approvals.remember(request, answer)
    return ToolCallDecision(permitted=final.permitted)


def _with_intent(config: Any, environment: dict, user_messages: list[str], dispatcher) -> tuple[dict, bool]:
    """Add the BEFORE_TOOL enrichers' intent attributes to the environment.

    Returns the new environment and whether intent was unavailable under
    ``on_unavailable="fail_closed"`` (the call is then denied).
    """
    enforcer = config.enforcer(EnforcementPoint.BEFORE_TOOL)
    if enforcer is None or not enforcer.enrichers:
        return environment, False
    attributes, available = _classify_user_messages(enforcer.enrichers, user_messages, dispatcher)
    return _merge_intent(environment, attributes), not available and enforcer.on_unavailable == "fail_closed"


def needs_human_approval(
    *,
    pdp: Any,
    cache: Any | None,
    config: Any,
    surface: str,
    subject: dict,
    action: dict,
    resource: dict,
    environment: dict,
    user_messages: list[str],
    args: dict | None = None,
    call_id: str | None = None,
) -> ApprovalRequest | None:
    """Pre-check a tool call for a framework that decides before the call whether to pause it.

    For the OpenAI Agents SDK's ``needs_approval``. A pre-check with no side effects on the
    outcome: it logs only ``approval_required`` and runs no obligation handlers; the call's own
    check (``authorized_tool_invocation``) logs and enforces the rest. With the same
    ``call_id``, both checks share a correlation id and a ``request_id``, and the call's own
    check reuses this decision instead of asking the PDP again.

    Args:
        pdp (PdpClient): The PDP client.
        cache (AuthzDecisionCacheWithTTL | None): The decision cache, or None for no caching.
        config (AuthZConfig): The config (fail_open, hooks, attribute names).
        surface (str): The adapter's label for decision events, e.g. ``"langchain_agents"``.
        subject (dict): Subject attributes.
        action (dict): Action attributes.
        resource (dict): Resource attributes.
        environment (dict): Environment attributes.
        user_messages (list[str]): The user's own messages, oldest first.
        args (dict, optional): The tool call's own arguments.
        call_id (str, optional): The framework's id for this tool call.

    Returns:
        ApprovalRequest | None: The request when the policy asks for approval and only a human
        can answer (no approver configured, no remembered approval), else None.
    """
    if pdp.obligations.approval.approver is not None:
        return None   # the approver answers inside the call's own check
    dispatcher = config.build_hook_dispatcher()
    with dispatcher.correlation(_call_correlation_id(surface, call_id)):
        environment = _without_approval_attribute(environment, pdp)
        environment, unavailable = _with_intent(config, environment, user_messages, dispatcher)
        if unavailable:
            return None   # the call's own check denies it
        decision = cache.get(subject, action, resource, environment) if cache is not None else None
        cached = decision is not None
        if not cached:
            decision = evaluate_tool_call(pdp, subject=subject, action=action, resource=resource, environment=environment)
            if cache is not None and decision.decision in _DEFINITE:
                cache.set(subject, action, resource, environment, decision)
        resolution = pdp.resolve(
            decision, subject=subject, action=action, resource=resource, environment=environment,
            fail_open=config.connection.fail_open, approval_possible=True, run_handlers=False,
        )
        if resolution.outcome is not Outcome.APPROVAL_REQUIRED:
            return None
        binding = _approval_binding(resource, args)
        request = _approval_request(
            resolution, subject, resource, config, args=args, binding=binding,
            request_id=_request_id(surface, call_id, binding),
        )
        if _recall_approval(config, request) is not None:
            return None   # remembered: the call's own check reuses it
        paused_key = _paused_key(subject, surface, call_id, binding)
        if paused_key:
            # The call's own check, after the answer, applies it to this decision.
            config._state.paused.put(paused_key, request, resolution.decision)
        _emit_invocation(
            dispatcher, surface, "tool_invocation", subject, action, resource,
            resolution.outcome.value, cached=cached, environment=environment,
            detail={**resolution.detail, "request_id": request.request_id},
        )
        return request


def _fulfil_approval(dispatcher, call: dict, pdp_decision: Any, environment: dict) -> Resolution:
    """Resolve a Permit with an approval obligation again, now that it was approved.

    The approval fulfils the obligation, so the PDP's Permit stands: resolved again without a
    PDP call (its other obligations still have to be met) and logged as the final decision.
    (Deny + approval advice instead asks the PDP again, since only the PDP can turn a Deny into
    a Permit.)
    """
    final = call["pdp"].resolve(
        pdp_decision, subject=call["subject"], action=call["action"], resource=call["resource"],
        environment=environment, fail_open=call["config"].connection.fail_open, approval_given=True,
    )
    _emit_invocation(
        dispatcher, call["surface"], call["enforcement_point"], call["subject"], call["action"],
        call["resource"], final.outcome.value, cached=False, environment=environment,
        detail={**final.detail, "evaluated_again": False},
    )
    return final


@dataclass(frozen=True)
class ToolCallDecision:
    """The BEFORE_TOOL outcome.

    Attributes:
        permitted (bool): Whether the call may run.
        pending (ApprovalRequest | None): The approval request when the policy asks for human
            approval and nothing has answered yet (only for an engine that ``can_interrupt``):
            the engine asks through its framework, then calls again with ``approval=``.
        not_approved (bool): True when the call was blocked because a human (or the lack of an
            approver) did not approve it, rather than by the policy itself.
    """

    permitted: bool
    pending: ApprovalRequest | None = None
    not_approved: bool = False


def denial_message(tool_name: str, decision: ToolCallDecision) -> str:
    """What an engine tells the model about a blocked tool call.

    Args:
        tool_name (str): The blocked tool.
        decision (ToolCallDecision): The decision that blocked it.

    Returns:
        str: "not approved" for a call a human did not approve, else "access denied".
    """
    if decision.not_approved:
        return f"'{tool_name}' was not approved, so it did not run."
    return f"Access denied: not authorized to call '{tool_name}'."


def _paused_key(subject: dict, surface: str, call_id: str | None, binding: str) -> tuple | None:
    """Key a paused call by caller, surface, call id and argument binding; None without a call id.

    The caller is part of it so another caller reusing a call id (per-turn numbered ids exist
    in practice) never gets someone else's pending request or decision.
    """
    return (args_hash(subject), surface, call_id, binding) if call_id else None


def _recall_approval(config: Any, request: ApprovalRequest) -> ApprovalDecision | None:
    """Get a still-valid remembered approval, but only when the current rule allows reuse.

    An approval remembered under one rule's ``ttl`` is not reused for a request whose asking
    rule is single-use (no ``ttl``), so no approval outlives what the current rule allows.
    """
    if request.ttl_seconds is None:
        return None
    return config._state.approvals.recall(request)


# Whether a provider-supplied approval attribute was already warned about (once).
_warned_approval_attribute = False


def _without_approval_attribute(environment: dict, pdp: Any) -> dict:
    """Drop the approval attribute from the provider's environment.

    Only the SDK sets it, after a human approved; a provider (or anything a caller can put in
    the context) setting it would skip the approval.
    """
    global _warned_approval_attribute
    attribute = pdp.obligations.approval.attribute
    if attribute not in environment:
        return environment
    if not _warned_approval_attribute:
        _warned_approval_attribute = True
        logger.warning(
            "The context provider set the approval attribute %r; it is reserved for the SDK "
            "and was dropped.", attribute,
        )
    return {k: v for k, v in environment.items() if k != attribute}


def _call_correlation_id(surface: str, call_id: str | None) -> str | None:
    """Derive a correlation id from the framework's tool-call id.

    The events logged when an interrupted call is resumed then join those logged before it
    paused. None (a fresh id per scope) without a call id.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"axiomatics:{surface}:{call_id}").hex if call_id else None


def _approval_binding(resource: dict, args: dict | None) -> str:
    """Hash what an approval is bound to: the PDP resource and, when known, the call's arguments.

    An argument the context provider does not forward still needs its own approval when it
    changes.
    """
    return args_hash(resource if args is None else {"resource": resource, "args": args})


def _request_id(surface: str, call_id: str | None, binding: str) -> str:
    """Make the approval request id: stable per tool call and arguments, else random.

    Stable so the request logged before an interrupt and the answer logged on resume match.
    """
    if call_id:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"axiomatics:{surface}:{call_id}:{binding}").hex
    return uuid.uuid4().hex


def _approval_request(
    resolution: Resolution, subject: dict, resource: dict, config: Any, *,
    args: dict | None, binding: str, request_id: str,
) -> ApprovalRequest:
    """Build what the approver is shown.

    The tool, its arguments (the call's own when known, else the call's other resource
    attributes), the policy's reasons, and how long the approval may be reused. Several
    approval obligations or advice on one decision make one request.
    """
    tool_attribute = config.attribute_names.tool
    asked_by = resolution.approval
    return ApprovalRequest(
        request_id=request_id,
        tool_name=str(resource.get(tool_attribute, "")),
        args=dict(args) if args is not None else {k: v for k, v in resource.items() if k != tool_attribute},
        args_hash=binding,
        reasons=reasons(asked_by),
        ttl_seconds=ttl_seconds(asked_by),
        subject=dict(subject),
        asked_by=asked_by,
    )


def _get_approval(
    request: ApprovalRequest, dispatcher, call: dict, *, given: ApprovalDecision | None, can_interrupt: bool,
) -> ApprovalDecision | None:
    """Get the answer to ``request`` and emit an approval decision event.

    The answer ``given`` on resume, a remembered approval, or the approver's. None (no event)
    when nothing answered and the engine will ask through its framework's interrupt.
    """
    answer, source = given, ("interrupt" if given is None or given.asked else "none")
    if answer is None:
        answer, source = _recall_approval(call["config"], request), "remembered"
    approver = call["pdp"].obligations.approval.approver
    if answer is None and approver is not None:
        answer, source = _ask_approver(approver, request), "approver"
    if answer is None and can_interrupt:
        return None
    if answer is None:
        logger.warning(
            "Tool call %r needs human approval, but no approver is configured "
            "(ObligationsAndAdviceConfig.approval.approver) and this integration can't ask "
            "through its framework; it is denied.", request.tool_name,
        )
        answer = ApprovalDecision(approved=False, note="no approver configured", asked=False)
        source = "none"
    dispatcher.decision(
        surface=call["surface"],
        enforcement_point="tool_invocation",
        subject=call["subject"],
        action=call["action"],
        resource=call["resource"],
        decision="approved" if answer.approved else "rejected",
        detail={
            "request_id": request.request_id,
            "approver": answer.approver,
            "note": answer.note,
            "source": source,
        },
    )
    return answer


def _ask_approver(approver: Any, request: ApprovalRequest) -> ApprovalDecision:
    """Call the approver, sync or async; any failure counts as a rejection.

    A failure is the approver raising, or returning something other than an
    ``ApprovalDecision``.
    """
    try:
        answer = approver(request)
        if inspect.isawaitable(answer):
            answer = _run_awaitable(answer)
    except Exception:
        # The approver is user code; any failure means the call was not approved.
        logger.exception("Approver failed for tool call %r; it is denied.", request.tool_name)
        return ApprovalDecision(approved=False, note="approver failed")
    if not isinstance(answer, ApprovalDecision):
        logger.error("Approver returned %r, not an ApprovalDecision; the call is denied.", answer)
        return ApprovalDecision(approved=False, note="approver returned no decision")
    return answer


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _run_awaitable(awaitable: Any) -> Any:
    """Wait for an async approver's answer from sync code.

    With no event loop in this thread (the async engines call in via ``asyncio.to_thread``) it
    runs here; inside a running loop (a sync framework hook) it runs on a helper thread, since
    the loop can't be re-entered.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(awaitable))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _await(awaitable)).result()


# Each user message is cut to this many characters before classification, so one pasted
# document can't blow up the classifier's cost or context.
_MAX_USER_MESSAGE_CHARS = 4000


def _format_user_messages(messages: list[str]) -> str:
    """The classifier input: earlier requests for context, then the current request."""
    trimmed = [m[:_MAX_USER_MESSAGE_CHARS] for m in messages]
    *earlier, current = trimmed
    parts = []
    if earlier:
        parts.append("Earlier requests (oldest first):\n" + "\n".join(f"{i}. {m}" for i, m in enumerate(earlier, 1)))
    parts.append(f"Current request:\n{current}")
    return "\n\n".join(parts)


def _classify_user_messages(enrichers: list, user_messages: list[str], dispatcher) -> tuple[dict, bool]:
    """Run each BEFORE_TOOL enricher on its window of the user's messages.

    Returns ``(merged_attributes, available)`` like ``run_enrichers``.
    """
    messages = [m for m in (m.strip() for m in user_messages) if m]
    if not messages:
        for enricher in enrichers:
            dispatcher.enrichment(
                point="before_tool", enricher=enricher.spec.name, content="",
                attributes={enricher.spec.status_attribute: False}, available=False,
                latency_ms=0.0, error="no user messages available on this path",
            )
        return {e.spec.status_attribute: False for e in enrichers}, False
    merged: dict[str, Any] = {}
    available = True
    for enricher in enrichers:
        content = _format_user_messages(messages[-enricher.spec.num_user_messages:])
        attributes, ok = run_enrichers(
            enrichers=[enricher], content=content, dispatcher=dispatcher, point="before_tool",
        )
        merged.update(attributes)
        available = available and ok
    return merged, available


# Attribute names already warned about in _merge_intent, so a clash is logged once, not on
# every tool call.
_warned_intent_clashes: set[str] = set()


def _merge_intent(environment: dict, attributes: dict) -> dict:
    """Add the enrichers' attributes to the environment; they win on a name clash.

    The provider's value is dropped with a warning (once per name), so nothing else can pose as
    the intent.
    """
    for key in (attributes.keys() & environment.keys()) - _warned_intent_clashes:
        _warned_intent_clashes.add(key)
        logger.warning(
            "Environment attribute %r from the context provider has the same name as an intent "
            "attribute and is replaced by it. Rename one of them.", key,
        )
    return {**environment, **attributes}


def run_enrichers(*, enrichers: list, content: str, dispatcher, point: str) -> tuple[dict, bool]:
    """Run each LLM attribute enricher (a PIP) on ``content`` and merge what they produce.

    Emits an ``on_enrichment`` event per enricher. The enrichers never decide; their
    attributes feed the PDP decision.

    Args:
        enrichers (list[LLMAttributeEnricher]): The enrichers to run.
        content (str): The text to classify.
        dispatcher (HookDispatcher): Receives the ``on_enrichment`` events.
        point (str): The enforcement point label for the events.

    Returns:
        tuple[dict, bool]: The union of every enricher's attributes (each
        ``{name}_available`` sentinel included), and whether **every** enricher classified the
        content (one failure means "not fully checked").
    """
    merged: dict[str, Any] = {}
    available = True
    for enricher in enrichers:
        start = time.perf_counter()
        outcome = enricher.enrich(content)
        latency_ms = _elapsed_ms(start)
        merged.update(outcome.attributes)
        available = available and outcome.available
        dispatcher.enrichment(
            point=point,
            enricher=enricher.spec.name,
            content=content,
            attributes=outcome.attributes,
            available=outcome.available,
            latency_ms=latency_ms,
            error=outcome.error,
        )
    return merged, available


def enrich_and_authorize(
    *,
    pdp: Any,
    cache: Any | None,
    config: Any,
    surface: str,
    enforcement_point: str,
    subject: dict,
    action: dict,
    resource: dict,
    environment: dict,
    enrichers: list,
    content: str,
    on_unavailable: str = "fail_closed",
    block_reason: str,
) -> EnforcementResult:
    """Decide a content guard: classify ``content``, then let the PDP decide on the attributes.

    The enrichers' attributes, and the availability sentinel (always, so a policy can
    condition on it), are merged into the request's resource. With no enrichers this is a
    plain PDP decision.

    Args:
        pdp (PdpClient): The PDP client.
        cache (AuthzDecisionCacheWithTTL | None): The decision cache, or None for no caching.
        config (AuthZConfig): The config (fail_open, hooks, attribute names).
        surface (str): The adapter's label for decision events, e.g. ``"langchain_agents"``.
        enforcement_point (str): The point label for the decision event.
        subject (dict): Subject attributes.
        action (dict): Action attributes.
        resource (dict): Resource attributes, before the enrichers' attributes are added.
        environment (dict): Environment attributes.
        enrichers (list[LLMAttributeEnricher]): The point's enrichers.
        content (str): The text to classify.
        on_unavailable (str): ``"fail_closed"`` blocks without asking the PDP when an enricher
            could not classify; ``"fail_open"`` asks the PDP with the sentinel, so the policy
            decides.
        block_reason (str): The refusal text for a block.

    Returns:
        EnforcementResult: ``ALLOW``, or ``BLOCK`` with ``block_reason``.
    """
    dispatcher = config.build_hook_dispatcher()
    with dispatcher.correlation():
        attributes, available = run_enrichers(
            enrichers=enrichers, content=content, dispatcher=dispatcher, point=enforcement_point,
        )
        resource = {**resource, **attributes}
        if not available and on_unavailable == "fail_closed":
            _emit_invocation(
                dispatcher, surface, enforcement_point, subject, action, resource, "deny",
                cached=False, environment=environment,
            )
            return EnforcementResult.block(block_reason)
        permitted = _decide_call(
            pdp=pdp, cache=cache, config=config, surface=surface,
            enforcement_point=enforcement_point, subject=subject, action=action,
            resource=resource, environment=environment,
        ).permitted
        return EnforcementResult.allow() if permitted else EnforcementResult.block(block_reason)


def guard_content(
    *,
    config: Any,
    pdp: Any,
    cache: Any | None,
    provider: Any,
    surface: str,
    point: EnforcementPoint,
    content: str,
    run_context: Any,
    agent_name: str | None = None,
) -> EnforcementResult:
    """Content-guard one model-I/O text at ``point`` (BEFORE_MODEL or AFTER_MODEL).

    The glue every client adapter shares: resolve identity from ``run_context`` via
    ``provider``, build the action and resource for ``point``, then run the point's enrichers
    and let the PDP decide (``enrich_and_authorize``). The adapter turns a block into its own
    framework's refusal. Assumes ``config.enforces(point)``.

    Args:
        config (AuthZConfig): The config holding the point's Enforcer.
        pdp (PdpClient): The PDP client.
        cache (AuthzDecisionCacheWithTTL | None): The decision cache, or None.
        provider (ContextProvider): Turns ``run_context`` into attributes.
        surface (str): The adapter's label for decision events.
        point (EnforcementPoint): ``BEFORE_MODEL`` or ``AFTER_MODEL``.
        content (str): The prompt or answer text.
        run_context (Any): The framework's runtime context (the caller's identity).
        agent_name (str, optional): The agent whose model this is, sent as
            ``config.attribute_names.acting_agent``.

    Returns:
        EnforcementResult: ``ALLOW``, or ``BLOCK`` with the refusal text.
    """
    resolved = resolve_authz_context(
        provider, run_context, agent_name=agent_name, agent_attribute=config.attribute_names.acting_agent,
    )
    enforcer = config.enforcer(point)
    action = provider.get_action(enforcer.action_name, run_context)
    # The action already names the point, so the resource carries no identifier of its own —
    # only whatever the provider adds plus the enrichers' attributes.
    resource = provider.get_resource(point.value, run_context, resource_name_attribute=None)
    return enrich_and_authorize(
        pdp=pdp,
        cache=cache,
        config=config,
        surface=surface,
        enforcement_point=point.value,
        subject=resolved.subject,
        action=action,
        resource=resource,
        environment=resolved.environment,
        enrichers=enforcer.enrichers,
        content=content,
        on_unavailable=enforcer.on_unavailable,
        block_reason=block_message_for(enforcer, point),
    )


def emit_post_invocation_skip(
    *,
    post_ctx: PostInvocationContext,
    config: Any,
    surface: str,
    reason: str,
    withheld: bool = False,
) -> None:
    """Record that post-invocation was configured but the strategy never ran.

    Without this event, "the strategy ran and permitted everything" and "the strategy never
    ran" look the same in the hook stream: both end in an unmodified result.

    Args:
        post_ctx (PostInvocationContext): The call's context.
        config (AuthZConfig): The config (its hooks receive the event).
        surface (str): The adapter's label for the event.
        reason (str): Why the strategy did not run.
        withheld (bool): True when the output was blocked instead
            (``on_unrecognized_output="fail_closed"``); the decision is then ``"deny"``.
    """
    dispatcher = config.build_hook_dispatcher()
    with dispatcher.correlation():
        dispatcher.decision(
            surface=surface,
            enforcement_point="post_invocation",
            subject=post_ctx.subject,
            action=post_ctx.action,
            resource=post_ctx.resource,
            decision="deny" if withheld else "skipped",
            detail={"reason": reason, "withheld": withheld},
            environment=post_ctx.environment,
        )


def apply_post_invocation(
    *,
    strategy: PostInvocationStrategy,
    records: Any,
    post_ctx: PostInvocationContext,
    config: Any,
    surface: str,
) -> Any:
    """Run a post-invocation strategy and emit its decision event.

    Args:
        strategy (PostInvocationStrategy): The strategy to run.
        records (Any): The data handed to the strategy.
        post_ctx (PostInvocationContext): The call's context.
        config (AuthZConfig): The config (its hooks receive the event).
        surface (str): The adapter's label for the event.

    Returns:
        Any: What the strategy returned. The event says ``"passthrough"`` when it returned the
        input object itself, ``"filtered"`` otherwise.
    """
    dispatcher = config.build_hook_dispatcher()
    with dispatcher.correlation():
        processed = strategy.apply(records, post_ctx)
        detail = {}
        if isinstance(records, list) and isinstance(processed, list):
            detail = {"before": len(records), "after": len(processed)}
        dispatcher.decision(
            surface=surface,
            enforcement_point="post_invocation",
            subject=post_ctx.subject,
            action=post_ctx.action,
            resource=post_ctx.resource,
            decision="passthrough" if processed is records else "filtered",
            detail=detail,
            environment=post_ctx.environment,
        )
        return processed


# What replaces a content block the strategy can't inspect, next to data it did, under fail_closed.
_WITHHELD_BLOCK_TEXT = "[withheld: the authorization policy could not check this part of the output]"


def post_process_tool_output(
    payload: Any,
    *,
    enforcer: Any,
    post_ctx: PostInvocationContext,
    config: Any,
    surface: str,
    render_text_block: Any = mcp_text_block,
) -> EnforcementResult:
    """Run the ``AFTER_TOOL`` Enforcer's strategy on a framework's tool result.

    The data is unwrapped from the framework's result shape (``transform_tool_output``),
    handed to the strategy, and the result is rebuilt around what the strategy returns. Every
    output the strategy cannot inspect is logged and emitted as a decision event: the whole
    result, or a content block (plain text, an image) next to the data it read, which can
    restate that data. Under ``"fail_closed"`` such a block is replaced by a "withheld" text
    block; the whole result is blocked.

    Args:
        payload (Any): The framework's tool result.
        enforcer (Enforcer): The AFTER_TOOL Enforcer (its strategy and
            ``on_unrecognized_output``).
        post_ctx (PostInvocationContext): The call's context.
        config (AuthZConfig): The config (its hooks receive the events).
        surface (str): The adapter's label for the events.
        render_text_block (Callable[[str], Any]): Builds a text content block when a result is
            re-rendered from ``structuredContent``, or a block is withheld.

    Returns:
        EnforcementResult: ``ALLOW`` when the result is unchanged, ``TRANSFORM`` with the
        rebuilt result, or ``BLOCK`` with a message to return in place of output the strategy
        cannot inspect, when ``on_unrecognized_output`` is ``"fail_closed"``.
    """
    withheld = enforcer.on_unrecognized_output == "fail_closed"
    unread_blocks: list[Any] = []

    def on_unread_block(block: Any) -> Any:
        # A text or image block next to the data the strategy read: it can restate that data.
        unread_blocks.append(block)
        return render_text_block(_WITHHELD_BLOCK_TEXT) if withheld else block

    new_payload, found = transform_tool_output(
        payload,
        lambda value: apply_post_invocation(
            strategy=enforcer.strategy, records=value, post_ctx=post_ctx,
            config=config, surface=surface,
        ),
        render_text_block=render_text_block,
        on_unread_block=on_unread_block,
    )
    if unread_blocks:
        reason = f"{len(unread_blocks)} content block(s) next to the data hold nothing the strategy can inspect"
        logger.warning(
            "Post-invocation [%s]: tool '%s': %s; %s.", surface, post_ctx.tool_name, reason,
            "withholding them" if withheld else "returning them unfiltered",
        )
        emit_post_invocation_skip(
            post_ctx=post_ctx, config=config, surface=surface, reason=reason, withheld=withheld,
        )
    if found:
        if new_payload is payload:
            return EnforcementResult.allow()
        logger.debug("Post-invocation [%s] on [%s]: strategy applied", surface, post_ctx.tool_name)
        return EnforcementResult.transform(new_payload)

    reason = "output holds no list or object the strategy can inspect"
    logger.warning(
        "Post-invocation [%s]: tool '%s' %s; %s.", surface, post_ctx.tool_name, reason,
        "withholding it" if withheld else "returning it unfiltered",
    )
    emit_post_invocation_skip(
        post_ctx=post_ctx, config=config, surface=surface, reason=reason, withheld=withheld,
    )
    if withheld:
        return EnforcementResult.block(
            f"The output of '{post_ctx.tool_name}' was withheld: the authorization policy "
            f"could not check it."
        )
    return EnforcementResult.allow()


def _emit_visibility(
    dispatcher, surface, subject, action, resource, permitted, tool_names, *,
    cached, environment, enforcement_point="tool_visibility",
):
    dispatcher.decision(
        surface=surface,
        enforcement_point=enforcement_point,
        subject=subject,
        action=action,
        resource=resource,
        decision="partial",
        cached=cached,
        detail={"permitted": len(permitted), "total": len(tool_names)},
        environment=environment,
    )


def _emit_invocation(
    dispatcher, surface, enforcement_point, subject, action, resource, decision, *,
    cached, environment, detail=None,
):
    dispatcher.decision(
        surface=surface,
        enforcement_point=enforcement_point,
        subject=subject,
        action=action,
        resource=resource,
        decision=decision,
        cached=cached,
        detail=detail or {},
        environment=environment,
    )
