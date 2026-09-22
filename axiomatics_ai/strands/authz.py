"""Client-side PDP authorization for Strands agents (e.g. agents that consume MCP tools).

``authorized_strands_agent(agent, config)`` attaches a single Strands *intervention handler* (the
newer, higher-level control API) to an already-constructed ``Agent`` and returns it — it does
NOT wrap or subclass the agent or its toolsets. The enforcement points map onto Strands as
follows:

- ``before_model_call`` → **BEFORE_MODEL content guard** and **tool visibility**. The content
  guard classifies the user prompt (LLM enrichers) and, on a block, returns ``Deny(reason=...)``
  — Strands cancels the model call and surfaces the reason as the assistant message. Visibility
  then makes one batched PDP ``get_permitted_tools`` call (cached) and filters the tool specs
  the model receives via a non-destructive registry wrapper (source-agnostic: MCP + native
  ``@tool``).
- ``after_model_call`` → **AFTER_MODEL content guard**: classifies the model's final answer;
  a block returns ``Transform`` that rewrites the answer message with the refusal (there is no
  ``Deny`` at this point).
- ``before_tool_call`` → **tool invocation**: ``authorized_tool_invocation``; a denied call is
  blocked by returning ``Deny(reason=...)`` (the tool never runs). A call the policy wants a
  human to approve pauses with a Strands interrupt (``axiomatics-approval``) unless an
  ``approver`` is configured.
- ``after_tool_call`` → **post-invocation**: the configured ``AFTER_TOOL`` Enforcer ``strategy``
  (built-in or custom) filters/transforms the tool result's records via ``Transform``.

The model-I/O guards only inspect real user prompts and final answers — tool-result
continuation turns and intermediate tool-call turns are skipped (see ``_strands_user_prompt``
/ ``_strands_final_answer``).

Identity (the PDP subject) is read by merging the agent's persistent state
(``Agent(state={"username": ...})`` — bound once per user) with the per-invocation
``invocation_state`` (``agent(prompt, username=...)``); the latter takes precedence. The
merged dict is handed to the reused ``DefaultClientContextProvider`` (override with
``context_provider=``).

Note: a single ``Agent`` instance sharing one bound identity is assumed per turn; the
engine tracks the current turn's permitted set on itself, so concurrent invocations of the
same instance with *different* subjects are not supported (consistent with the other paths).
"""
from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from typing import Any

from strands.interventions import Deny, InterventionHandler, Proceed, Transform
from strands.interventions.registry import InterventionRegistry

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

# The name of the Strands interrupt a tool call raises while it waits for human approval; its
# ``reason`` is the approval request payload.
_APPROVAL_INTERRUPT = "axiomatics-approval"

# The caller's context while an agent used as a tool runs. Strands starts that agent without
# the caller's invocation_state, so its own handler reads the caller's identity from here.
_caller_context: ContextVar[dict[str, Any] | None] = ContextVar("axiomatics_strands_caller", default=None)


def _strands_text_block(text: str) -> dict[str, Any]:
    """A Strands ``ToolResultContent`` text block (no ``type`` key, unlike raw MCP)."""
    return {"text": text}


def _registry_tool_names(agent: Any) -> list[str]:
    """List every tool name registered on the agent, from the raw registry.

    Not the filtered view the visibility wrapper exposes to the model.
    """
    reg = getattr(agent, "tool_registry", None)
    if reg is None:
        return []
    names = list(getattr(reg, "registry", {}) or {})
    for name in getattr(reg, "dynamic_tools", {}) or {}:
        if name not in names:
            names.append(name)
    return names


def _blocks_text(content: Any) -> str:
    return "".join(b.get("text", "") for b in (content or []) if isinstance(b, dict) and "text" in b)


def _strands_user_messages(event: Any) -> list[str]:
    """Get the user's own messages in the agent's history, oldest first, for intent.

    Strands stores tool results as ``user``-role messages too, so only their text blocks count
    and ``toolResult`` messages are skipped: tool output can't shape the intent.
    """
    messages = getattr(getattr(event, "agent", None), "messages", None) or []
    texts = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content") or []
        if any(isinstance(b, dict) and "toolResult" in b for b in content):
            continue
        text = _blocks_text(content).strip()
        if text:
            texts.append(text)
    return texts


def _strands_user_prompt(event: Any) -> str | None:
    """Get the latest user prompt from a BeforeModelCallEvent, or None.

    None when the last message is not a fresh user turn (e.g. a tool-result continuation).
    """
    messages = getattr(getattr(event, "agent", None), "messages", None) or []
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    content = last.get("content") or []
    if any(isinstance(b, dict) and "toolResult" in b for b in content):
        return None  # tool-result continuation, not a new user prompt
    text = _blocks_text(content).strip()
    return text or None


def _strands_final_answer(event: Any) -> str | None:
    """Get the model's text from an AfterModelCallEvent, or None if it wrote none.

    Read from ``stop_response.message``. Text sent alongside a tool call counts: it reaches
    the user too.
    """
    message = getattr(getattr(event, "stop_response", None), "message", None)
    if not isinstance(message, dict):
        return None
    text = _blocks_text(message.get("content") or []).strip()
    return text or None


def _replace_answer_text(event: Any, text: str) -> None:
    """Rewrite the model's message in place: its text becomes one refusal block.

    The AFTER_MODEL block path, applied through a Strands ``Transform``. ``toolUse`` blocks are
    kept, since the turn's stop reason still says a tool runs next; each call is authorized at
    BEFORE_TOOL.
    """
    message = getattr(getattr(event, "stop_response", None), "message", None)
    if isinstance(message, dict):
        tool_uses = [b for b in message.get("content") or [] if isinstance(b, dict) and "toolUse" in b]
        message["content"] = [{"text": text}, *tool_uses]


class _StrandsAuthZ:
    """The PDP client, cache and enforcement logic behind the Strands handler.

    Its methods take duck-typed Strands event objects (``event.agent``,
    ``event.invocation_state``, ``event.tool_use``, ``event.result``), so they can be unit-tested
    with plain fakes.
    """

    _SURFACE = "strands"

    def __init__(self, config: AuthZConfig) -> None:
        self.config = config
        self._provider = self.config.context_provider or DefaultClientContextProvider()
        self._ads = config.build_pdp_client()
        self._cache = config.build_cache()
        # Permitted-tool set for the current turn (None = no visibility filtering yet /
        # everything permitted). Set by before_model, consumed by filter_tools_config.
        self._permitted: set[str] | None = None
        # toolUseId → the token that set _caller_context for a running agent tool.
        self._caller_tokens: dict[str, Any] = {}

    # ── context / identity ──────────────────────────────────────────────────────
    def _context(self, event: Any) -> dict[str, Any]:
        # The caller's context, when this agent runs as another's tool; its own state and
        # invocation state take precedence.
        ctx: dict[str, Any] = dict(_caller_context.get() or {})
        agent = getattr(event, "agent", None)
        state = getattr(agent, "state", None)
        if state is not None and hasattr(state, "get"):
            try:
                data = state.get()  # AgentState.get() with no key returns the whole dict
            except TypeError:
                data = None
            if isinstance(data, dict):
                ctx.update(data)
        inv = getattr(event, "invocation_state", None)
        if isinstance(inv, dict):
            ctx.update(inv)  # per-invocation overrides bound-once state
        return ctx

    # ── model-I/O content guard (BEFORE_MODEL / AFTER_MODEL) ────────────────────
    def guard_model_io(self, point: EnforcementPoint, content: str, event: Any):
        """Run the shared ``guard_content`` for this engine.

        Identity comes from the same merged state/invocation context as the tool points.
        """
        return guard_content(
            config=self.config, pdp=self._ads, cache=self._cache, provider=self._provider,
            surface=self._SURFACE, point=point, content=content, run_context=self._context(event),
            agent_name=_agent_name(event),
        )

    def _resolve(self, ctx: dict[str, Any], event: Any):
        return resolve_authz_context(
            self._provider, ctx,
            agent_name=_agent_name(event), agent_attribute=self.config.attribute_names.acting_agent,
        )

    # ── visibility (before_model_call) ──────────────────────────────────────────
    def before_model(self, event: Any) -> None:
        """Decide tool visibility for this turn; ``filter_tools_config`` applies it."""
        agent = getattr(event, "agent", None)
        tool_names = _registry_tool_names(agent)
        if not tool_names:
            self._permitted = None
            return

        ctx = self._context(event)
        resolved = self._resolve(ctx, event)
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

        self._permitted = set(permitted)
        denied = set(tool_names) - self._permitted
        if denied:
            logger.debug("Strands tool visibility [%s]: hiding %s", resolved.subject, sorted(denied))

    def filter_tools_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Filter a tool-spec config dict to the permitted set.

        Used by the registry wrapper installed on the agent. A no-op until ``before_model`` has
        run.
        """
        if self._permitted is None:
            return config
        return {name: spec for name, spec in config.items() if name in self._permitted}

    # ── invocation (before_tool_call) ───────────────────────────────────────────
    def before_tool(self, event: Any) -> str | None:
        """Authorize a tool call, pausing with a Strands interrupt if the policy asks for approval.

        Returns a denial reason if the call is not permitted, else None.
        """
        tool_use = getattr(event, "tool_use", None) or {}
        tool_name = tool_use.get("name", "")
        args = tool_use.get("input", {}) or {}
        ctx = self._context(event)
        resolved = self._resolve(ctx, event)
        action = self._provider.get_action(
            self.config.enforcer(EnforcementPoint.BEFORE_TOOL).action_name, ctx
        )
        resource = self._provider.get_resource(
            tool_name, ctx, args=args,
            resource_name_attribute=self.config.attribute_names.tool,
        )

        def decide(approval: ApprovalDecision | None = None):
            return authorized_tool_invocation(
                pdp=self._ads,
                cache=self._cache,
                config=self.config,
                surface=self._SURFACE,
                subject=resolved.subject,
                action=action,
                resource=resource,
                environment=resolved.environment,
                user_messages=_strands_user_messages(event),
                approval=approval,
                can_interrupt=callable(getattr(event, "interrupt", None)),
                args=args,
                call_id=tool_use.get("toolUseId"),
            )

        decision = decide()
        if decision.pending is not None:
            # Strands interrupt: the first call raises InterruptException (the run stops with
            # stop_reason "interrupt"); the app resumes with an interruptResponse, this hook
            # runs again, and interrupt() returns the human's answer.
            answer = event.interrupt(_APPROVAL_INTERRUPT, reason=decision.pending.to_payload())
            decision = decide(ApprovalDecision.from_answer(answer))
        if not decision.permitted:
            logger.debug("Strands tool call [%s] by [%s]: Deny", tool_name, resolved.subject)
            return denial_message(tool_name, decision)
        if _is_agent_tool(getattr(event, "selected_tool", None)):
            # Runs in the same task as the tool, so the agent it starts reads this context;
            # reset when the call ends (release_caller_context).
            self._caller_tokens[tool_use.get("toolUseId") or ""] = _caller_context.set(ctx)
        return None

    def release_caller_context(self, event: Any) -> None:
        """Reset the caller's context set for an agent tool, once the call has ended."""
        tool_use = getattr(event, "tool_use", None) or {}
        token = self._caller_tokens.pop(tool_use.get("toolUseId") or "", None)
        if token is not None:
            with contextlib.suppress(ValueError):   # set in another context: nothing to reset here
                _caller_context.reset(token)

    # ── post-invocation (after_tool_call) ───────────────────────────────────────
    def after_tool(self, event: Any) -> Any | None:
        """Run the post-invocation strategy on a tool result.

        Returns the rewritten ``ToolResult`` (or an error result for withheld output), or None to
        leave it unchanged. The records are read from the result's content blocks (``{"text":
        <json>}`` or ``{"json": ...}``), or its ``structuredContent`` for an MCP tool.
        """
        result = getattr(event, "result", None)
        post_enforcer = self.config.enforcer(EnforcementPoint.AFTER_TOOL)
        if post_enforcer is None or post_enforcer.strategy is None:
            return None
        if _is_agent_tool(getattr(event, "selected_tool", None)):
            # The wrapped agent is authorized too (authorized_strands_agent walks into it): its
            # tool results were post-processed there, and its answer is text no strategy reads.
            return None

        ctx = self._context(event)
        resolved = self._resolve(ctx, event)
        tool_use = getattr(event, "tool_use", None) or {}
        tool_name = tool_use.get("name", "")
        args = tool_use.get("input", {}) or {}
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
            surface=self._SURFACE, render_text_block=_strands_text_block,
        )
        if outcome.is_block:
            return {**(result or {}), "status": "error", "content": [{"text": outcome.reason}]}
        return outcome.content


def _agent_name(event: Any) -> str | None:
    return getattr(getattr(event, "agent", None), "name", None)


def _is_agent_tool(tool: Any) -> bool:
    """Whether a tool runs an agent (``agent.as_tool()``)."""
    return type(tool).__name__ == "_AgentAsTool" and getattr(tool, "agent", None) is not None


class _AuthzInterventionHandler(InterventionHandler):
    """Thin Strands intervention handler around the engine.

    Translates the engine's plain return values into Strands intervention actions
    (``Proceed`` / ``Deny`` / ``Transform``).
    """

    name = "axiomatics-pdp-authz"

    def __init__(self, engine: _StrandsAuthZ, config: AuthZConfig) -> None:
        self._engine = engine
        self._config = config

    def before_model_call(self, event: Any, **kwargs: Any):
        # BEFORE_MODEL content guard — a block cancels the model and surfaces the reason.
        """Guard the prompt (``Deny`` on a block) and decide tool visibility; else ``Proceed``."""
        if self._config.enforces(EnforcementPoint.BEFORE_MODEL):
            prompt = _strands_user_prompt(event)
            if prompt is not None:
                result = self._engine.guard_model_io(EnforcementPoint.BEFORE_MODEL, prompt, event)
                if result.is_block:
                    return Deny(reason=result.reason)
        if self._config.enforces(EnforcementPoint.TOOL_VISIBILITY):
            self._engine.before_model(event)
        return Proceed()

    def after_model_call(self, event: Any, **kwargs: Any):
        # AFTER_MODEL content guard — a block rewrites the answer message with the refusal.
        """Guard the model's text: ``Transform`` it to the refusal on a block; else ``Proceed``."""
        if self._config.enforces(EnforcementPoint.AFTER_MODEL):
            answer = _strands_final_answer(event)
            if answer is not None:
                result = self._engine.guard_model_io(EnforcementPoint.AFTER_MODEL, answer, event)
                if result.is_block:
                    refusal = result.reason

                    def apply(ev: Any) -> None:
                        _replace_answer_text(ev, refusal)

                    return Transform(apply=apply)
        return Proceed()

    def before_tool_call(self, event: Any, **kwargs: Any):
        """Authorize the call: ``Deny`` with the reason, or ``Proceed``."""
        if self._config.enforces(EnforcementPoint.BEFORE_TOOL):
            reason = self._engine.before_tool(event)
            if reason:
                return Deny(reason=reason)
        return Proceed()

    def after_tool_call(self, event: Any, **kwargs: Any):
        """Post-process the result: ``Transform`` it when rewritten, else ``Proceed``."""
        self._engine.release_caller_context(event)
        if self._config.enforces(EnforcementPoint.AFTER_TOOL):
            try:
                new_result = self._engine.after_tool(event)
            except Exception:   # Strands would hand the error text (the PDP's URL and answer) to the model
                logger.exception("Strands post-invocation failed")
                result = getattr(event, "result", None) or {}
                new_result = {**result, "status": "error", "content": [{"text": "Authorization is unavailable."}]}
            if new_result is not None:
                def apply(ev: Any) -> None:
                    ev.result = new_result
                return Transform(apply=apply)
        return Proceed()


def _install_visibility_filter(agent: Any, engine: _StrandsAuthZ) -> None:
    """Filter the tool specs the model sees, non-destructively.

    Wraps the registry's ``get_all_tools_config`` (which ``get_all_tool_specs`` reads before
    each model call).
    """
    reg = getattr(agent, "tool_registry", None)
    if reg is None or getattr(reg, "_ads_visibility_wrapped", False):
        return
    original = reg.get_all_tools_config

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return engine.filter_tools_config(original(*args, **kwargs))

    reg.get_all_tools_config = wrapped
    reg._ads_visibility_wrapped = True


def authorized_strands_agent(agent: Any, config: AuthZConfig | None = None):
    """Attach authorization to a Strands ``Agent`` and the agents it uses as tools, and return it.

    Registers one intervention handler on the agent's existing hook registry (handlers already
    there are kept); each of its methods acts only on the points the config enforces: the
    content guards (``before_model_call`` / ``after_model_call``), tool visibility
    (``before_model_call`` plus a registry filter), invocation (``before_tool_call``, pausing
    with a Strands interrupt when the policy asks for approval) and post-invocation
    (``after_tool_call``). Identity comes from the agent state and invocation state through the
    config's context provider (``DefaultClientContextProvider`` by default).

    The agents it uses as tools (``agent.as_tool()``) get the same config, and the caller's
    identity. Each request names the agent making the call (its ``name``) in the subject's
    ``config.attribute_names.acting_agent``, so a policy can treat them differently. An agent
    already authorized is left as it is: to give one another config, authorize it first with
    that config.

    Args:
        agent (Agent): The Strands agent.
        config (AuthZConfig, optional): What to enforce and where the PDP is.

    Returns:
        Agent: ``agent``, with the handler registered.
    """
    config = config or AuthZConfig()
    for each in _agent_tree(agent):
        if not getattr(each, _AUTHORIZED_FLAG, False):
            _attach(each, config)
    return agent


_AUTHORIZED_FLAG = "_axiomatics_authorized"


def _agent_tree(root: Any) -> list[Any]:
    """The agent and every agent below it (agents used as tools), once each."""
    seen: dict[int, Any] = {}
    stack = [root]
    while stack:
        agent = stack.pop()
        if id(agent) in seen:
            continue
        seen[id(agent)] = agent
        registry = getattr(agent, "tool_registry", None)
        tools = getattr(registry, "registry", None) or {}
        stack.extend(reversed([t.agent for t in tools.values() if _is_agent_tool(t)]))
    return list(seen.values())


def _attach(agent: Any, config: AuthZConfig) -> None:
    """Register the handler on one agent."""
    engine = _StrandsAuthZ(config)
    if config.enforces(EnforcementPoint.TOOL_VISIBILITY):
        _install_visibility_filter(agent, engine)
    handler = _AuthzInterventionHandler(engine, config)
    InterventionRegistry([handler], agent.hooks)
    setattr(agent, _AUTHORIZED_FLAG, True)
