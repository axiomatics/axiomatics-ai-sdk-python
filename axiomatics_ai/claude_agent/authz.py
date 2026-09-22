"""Client-side PDP authorization for the Claude Agent SDK (e.g. agents that consume MCP tools).

Unlike the other MCP-client paths, the Claude Agent SDK has **no ``Agent`` object** — you pass
a ``ClaudeAgentOptions`` to ``query()`` / ``ClaudeSDKClient`` and the SDK drives the Claude Code
runtime. So authorization is configured on the *options*:

``authorized_claude_options(options, config, *, identity)`` returns a copy of ``options`` with PDP
enforcement wired in:

- **BEFORE_MODEL content guard (input hard-block)** → a ``UserPromptSubmit`` hook that runs the
  BEFORE_MODEL enrichers on the user prompt and, on an PDP deny, returns ``decision: "block"``
  so the CLI never sends the turn to the model (the refusal is surfaced as the block reason).
  Note this is a **hard block only** — the Claude Agent SDK's ``UserPromptSubmit`` exposes no
  prompt-rewrite / model-authored refusal, unlike LangChain/ADK/Strands.
- **tool invocation** → a ``PreToolUse`` hook that runs PDP ``authorized_tool_invocation``
  (with the tool's args as resource attributes) and returns ``permissionDecision: "deny"`` for
  a disallowed call, so the tool never runs. This is the security boundary and is arg-level
  (e.g. permit ``get_account_balance`` only for a specific account). A hook, not the
  ``can_use_tool`` permission callback: the CLI skips that callback for tools in
  ``allowed_tools``, under ``permission_mode="bypassPermissions"`` and for allow rules in
  settings files, while ``PreToolUse`` runs for every call, in every mode. The policy
  decides: a permitted call is allowed (``permissionDecision: "allow"``), with no
  ``allowed_tools`` or permission mode needed. A call the
  policy wants a human to approve waits for the configured ``approver``; there is no
  framework interrupt to pause on, so without an approver the call is denied.
- **post-invocation** → a ``PostToolUse`` hook that filters the tool result's records via the
  configured ``AFTER_TOOL`` Enforcer ``strategy`` and returns ``updatedToolOutput``.

**AFTER_MODEL is not supported on this path.** The Claude Agent SDK exposes no hook that sees
(or can redact) the assistant's final answer: it is delivered as an ``AssistantMessage`` in the
receive stream (observe-only), and none of the hook events — ``UserPromptSubmit``, ``Pre/PostToolUse``,
``Stop`` (which only sees ``stop_hook_active`` and whose block forces *more* generation) — can
intercept it. An ``AFTER_MODEL`` Enforcer is ignored. (If an app funnels its answer through a
designated tool, guard that tool at ``AFTER_TOOL`` instead.)

**Tool visibility is not supported on this path.** The SDK exposes no per-turn tool-list filter
and no runtime identity at tool-list assembly, so policies for Claude Agent SDK agents must be
written at the *invocation* point, not visibility. (A ``TOOL_VISIBILITY`` Enforcer is ignored
here, with a warning.) This is unaffected by MCP *tool search*: the ``PreToolUse`` hook fires
at call time and the ``PostToolUse`` hook at result time, whether or not tool definitions were
deferred.

**Tool names.** Claude namespaces MCP tools as ``mcp__<server>__<tool>``. The PDP gets the bare
tool name (so one policy is portable across paths) and the server name as a separate resource
attribute (``AuthZConfig.attribute_names.mcp_server``), so a policy can tell
``mcp__trusted__search_docs`` from ``mcp__other__search_docs``.

**Identity** is captured by closure: the callbacks are built bound to the ``identity`` passed to
``authorized_claude_options`` (the SDK's hook/permission callbacks carry no app-supplied identity). So
build one ``ClaudeAgentOptions`` per user (bind-once-per-user); the same options object cannot
serve different subjects.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import threading
from typing import Any

from cachetools import LRUCache
from claude_agent_sdk import HookMatcher

from axiomatics_ai.core.authz_helpers import resolve_authz_context
from axiomatics_ai.core.config import AuthZConfig, DefaultClientContextProvider, EnforcementPoint
from axiomatics_ai.core.enforcement import (
    authorized_tool_invocation,
    denial_message,
    guard_content,
    post_process_tool_output,
)
from axiomatics_ai.core.post_invocation import PostInvocationContext

logger = logging.getLogger(__name__)

# Upper bound on sessions (and pending tool calls) whose prompts are kept for intent
# classification, so a long-running process doesn't grow without limit; the least recently
# used are dropped first.
_MAX_TRACKED_SESSIONS = 1024

# Enforcement points the Claude Agent SDK has no hook for (see the module docstring).
_UNSUPPORTED_POINTS = (EnforcementPoint.TOOL_VISIBILITY, EnforcementPoint.AFTER_MODEL)


def _split_tool_name(name: str) -> tuple[str, str | None]:
    """A Claude Agent SDK tool name as ``(bare tool name, MCP server or None)``.

    Claude namespaces MCP tools as ``mcp__<server>__<tool>`` (e.g. ``mcp__acme__search_docs``),
    whereas the other framework paths present bare names (``search_docs``). The bare name keeps
    a single PDP policy portable across all paths; the server is sent as its own attribute.
    Non-MCP tool names are returned unchanged, with no server.
    """
    if name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:]), parts[1]
    return name, None


def _as_dict(identity: Any) -> dict[str, Any]:
    if identity is None:
        return {}
    if isinstance(identity, dict):
        return identity
    if hasattr(identity, "to_dict"):
        # Third-party identity objects: to_dict() may raise anything, and a failure here just
        # means we fall through to vars() below — never a reason to fail the request.
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            data = identity.to_dict()
            if isinstance(data, dict):
                return data
    try:
        return dict(vars(identity))
    except TypeError:
        return {}


class _ClaudeAuthZ:
    """The PDP client, cache and enforcement logic behind the Claude Agent SDK hooks.

    Its methods take plain values (tool name, args, result), so they can be unit-tested with
    fakes and no Claude Agent SDK import.
    """

    _SURFACE = "claude_agent"

    def __init__(self, config: AuthZConfig, identity: dict[str, Any]) -> None:
        self.config = config
        self._ctx = identity
        self._provider = self.config.context_provider or DefaultClientContextProvider()
        self._ads = config.build_pdp_client()
        self._cache = config.build_cache()
        # The PreToolUse hook sees a tool call's session but not its conversation. So for
        # intent classification at BEFORE_TOOL the UserPromptSubmit hook keeps each session's
        # recent prompts. Keyed by session, one options object can serve several conversations
        # without mixing their intent.
        before_tool = config.enforcer(EnforcementPoint.BEFORE_TOOL)
        enrichers = before_tool.enrichers if before_tool else []
        self._prompt_window = max((e.spec.num_user_messages for e in enrichers), default=0)
        self._session_prompts: LRUCache = LRUCache(maxsize=_MAX_TRACKED_SESSIONS)
        # Written on the event loop, read from worker threads; an LRU read reorders it.
        self._session_prompts_lock = threading.Lock()

    @property
    def captures_prompts(self) -> bool:
        """Whether BEFORE_TOOL has enrichers that need the user's prompts."""
        return self._prompt_window > 0

    def remember_prompt(self, session_id: str, prompt: str) -> None:
        """Keep an accepted user prompt for its session (only the most recent ones needed)."""
        if self.captures_prompts:
            with self._session_prompts_lock:
                recent = self._session_prompts.get(session_id, [])
                self._session_prompts[session_id] = [*recent, prompt][-self._prompt_window:]

    def _user_messages_for(self, session_id: str | None) -> list[str]:
        """Get the kept prompts of a session.

        None if the session is unknown, which makes intent unavailable (fail closed).
        """
        if not session_id:
            return []
        with self._session_prompts_lock:
            return list(self._session_prompts.get(session_id, []))

    def _resource(self, tool_name: str, server: str | None, tool_input: dict[str, Any]) -> dict:
        """The PDP resource for a tool call: the provider's, plus the MCP server when known."""
        resource = self._provider.get_resource(
            tool_name, self._ctx, args=tool_input,
            resource_name_attribute=self.config.attribute_names.tool,
        )
        if server is None:
            return resource
        return {**resource, self.config.attribute_names.mcp_server: server}

    # ── BEFORE_MODEL content guard (input hard-block) ─────────────────────────────
    def prompt_denial(self, prompt: str) -> str | None:
        """Get a refusal message if the BEFORE_MODEL guard blocks the prompt, else None.

        This path can only block (the Claude Agent SDK has no in-band refusal substitution); the
        message is surfaced as the ``UserPromptSubmit`` block reason.
        """
        result = guard_content(
            config=self.config, pdp=self._ads, cache=self._cache, provider=self._provider,
            surface=self._SURFACE, point=EnforcementPoint.BEFORE_MODEL, content=prompt,
            run_context=self._ctx,
        )
        return result.reason if result.is_block else None

    # ── invocation ──────────────────────────────────────────────────────────────
    def invocation_denial(
        self, tool_name: str, tool_input: dict[str, Any], *,
        server: str | None = None, session_id: str | None = None, tool_use_id: str | None = None,
    ) -> str | None:
        """Get a denial message if the call is not permitted, else None.

        ``session_id`` ties the call to its session's prompts for intent classification;
        ``tool_use_id`` groups the call's decision events.
        """
        ctx = self._ctx
        resolved = resolve_authz_context(self._provider, ctx)
        action = self._provider.get_action(
            self.config.enforcer(EnforcementPoint.BEFORE_TOOL).action_name, ctx
        )
        # No framework interrupt here: human approval comes only from the configured approver,
        # which the hook simply waits for (in process); without one the call is denied.
        decision = authorized_tool_invocation(
            pdp=self._ads,
            cache=self._cache,
            config=self.config,
            surface=self._SURFACE,
            subject=resolved.subject,
            action=action,
            resource=self._resource(tool_name, server, tool_input or {}),
            environment=resolved.environment,
            user_messages=self._user_messages_for(session_id),
            args=tool_input or {},
            call_id=tool_use_id,
        )
        if not decision.permitted:
            logger.debug("Claude tool call [%s] by [%s]: Deny", tool_name, resolved.subject)
            return denial_message(tool_name, decision)
        return None

    # ── post-invocation ─────────────────────────────────────────────────────────
    def filter_result(
        self, tool_name: str, tool_input: dict[str, Any], tool_response: Any, server: str | None = None,
    ) -> Any:
        """Run the post-invocation strategy on a tool response.

        Returns the filtered result, a withheld message under
        ``on_unrecognized_output="fail_closed"``, or None for no change.
        """
        post_enforcer = self.config.enforcer(EnforcementPoint.AFTER_TOOL)
        if post_enforcer is None or post_enforcer.strategy is None:
            return None
        ctx = self._ctx
        resolved = resolve_authz_context(self._provider, ctx)
        action_name = post_enforcer.action_overrides.get(tool_name, post_enforcer.action_name)
        action = self._provider.get_action(action_name, ctx)
        post_ctx = PostInvocationContext(
            pdp_client=self._ads,
            subject=resolved.subject,
            environment=resolved.environment,
            tool_name=tool_name,
            args=tool_input or {},
            resource=self._resource(tool_name, server, tool_input or {}),
            action=action,
            fail_open=self.config.connection.fail_open,
        )
        outcome = post_process_tool_output(
            tool_response, enforcer=post_enforcer, post_ctx=post_ctx, config=self.config,
            surface=self._SURFACE,
        )
        return outcome.reason if outcome.is_block else outcome.content


def _build_user_prompt_submit_hook(engine: _ClaudeAuthZ):
    """Build the ``UserPromptSubmit`` hook: the BEFORE_MODEL guard, and prompts kept for intent.

    A hard block: on a deny the CLI never sends the turn to the model and the reason is
    surfaced. The guard runs only if BEFORE_MODEL is enforced (the hook may be installed just to
    keep prompts), and a blocked prompt is not kept, since it never reaches the model.
    """
    guards = engine.config.enforces(EnforcementPoint.BEFORE_MODEL)

    async def user_prompt_submit(input_data: dict[str, Any], tool_use_id: Any, context: Any):
        prompt = (input_data.get("prompt") or "").strip()
        try:   # off the event loop: the guard calls the PDP and any LLM enricher
            denial = await asyncio.to_thread(engine.prompt_denial, prompt) if prompt and guards else None
        except Exception:   # the CLI treats a failing hook as no decision: fail closed instead
            logger.exception("UserPromptSubmit: authorization failed")
            denial = _UNAVAILABLE
        if denial is not None:
            return {"decision": "block", "reason": denial}
        if prompt:
            engine.remember_prompt(input_data.get("session_id", ""), prompt)
        return {}

    return user_prompt_submit


# What a hook answers when authorization itself fails (the details go to the log).
_UNAVAILABLE = "Authorization is unavailable."


def _build_pre_tool_use_hook(engine: _ClaudeAuthZ):
    """Build the ``PreToolUse`` hook that carries BEFORE_TOOL enforcement.

    It runs for every tool call, whatever the permission mode or allow rules. The policy
    decides: a permitted call gets ``permissionDecision: "allow"`` (so the CLI runs it without
    needing ``allowed_tools`` or a permission mode), a denied one ``"deny"``. For an extra gate
    of your own, add another ``PreToolUse`` hook: a deny from any hook blocks the call. The PDP
    call runs on a worker thread, so a slow PDP (or approver) doesn't stall the event loop.
    """
    async def pre_tool_use(input_data: dict[str, Any], tool_use_id: Any, context: Any):
        tool_name, server = _split_tool_name(input_data.get("tool_name", ""))
        try:
            denial = await asyncio.to_thread(
                engine.invocation_denial,
                tool_name,
                input_data.get("tool_input") or {},
                server=server,
                session_id=input_data.get("session_id"),
                tool_use_id=input_data.get("tool_use_id") or tool_use_id,
            )
        except Exception:   # the CLI treats a failing hook as no decision: fail closed instead
            logger.exception("PreToolUse [%s]: authorization failed", tool_name)
            denial = _UNAVAILABLE
        decision = {"permissionDecision": "allow"} if denial is None else {
            "permissionDecision": "deny", "permissionDecisionReason": denial,
        }
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", **decision}}

    return pre_tool_use


def _build_post_tool_use_hook(engine: _ClaudeAuthZ):
    """The ``PostToolUse`` hook applying the AFTER_TOOL strategy to a tool response.

    Returns an empty dict when the strategy changed nothing, so a tool whose output holds
    no records passes through untouched.
    """
    async def post_tool_use(input_data: dict[str, Any], tool_use_id: Any, context: Any):
        tool_name, server = _split_tool_name(input_data.get("tool_name", ""))
        try:
            new_output = engine.filter_result(
                tool_name,
                input_data.get("tool_input", {}) or {},
                input_data.get("tool_response"),
                server,
            )
        except Exception:   # the CLI would pass the unfiltered output on: withhold it instead
            logger.exception("PostToolUse [%s]: authorization failed", tool_name)
            new_output = _UNAVAILABLE
        if new_output is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": new_output,
            }
        }

    return post_tool_use


def authorized_claude_options(
    options: Any,
    config: AuthZConfig | None = None,
    *,
    identity: dict[str, Any] | None,
):
    """Return a copy of ``ClaudeAgentOptions`` with authorization wired in.

    Adds only what the config enforces: a ``UserPromptSubmit`` hook (the BEFORE_MODEL hard
    block, and/or keeping each session's prompts for intent), a ``PreToolUse`` hook (tool
    invocation, placed first) and a ``PostToolUse`` hook (post-invocation). Hooks already on
    ``options`` are kept. Tool visibility and AFTER_MODEL are not supported on this path; an
    Enforcer at either is ignored, with a warning.

    Args:
        options (ClaudeAgentOptions): The options to copy.
        config (AuthZConfig, optional): What to enforce and where the PDP is.
        identity (dict[str, Any] | None): The caller's identity, handed to the config's
            context provider (``DefaultClientContextProvider`` by default). It is captured by
            closure, so build one options object per user.

    Returns:
        ClaudeAgentOptions: The copy, with the hooks added.
    """
    config = config or AuthZConfig()
    engine = _ClaudeAuthZ(config, _as_dict(identity))
    for point in _UNSUPPORTED_POINTS:
        if config.enforces(point):
            logger.warning(
                "authorized_claude_options: %s is not supported on the Claude Agent SDK and is "
                "not enforced; enforce it at BEFORE_TOOL / AFTER_TOOL instead.", point.value,
            )

    hooks = dict(getattr(options, "hooks", None) or {})
    hooks_modified = False

    if config.enforces(EnforcementPoint.BEFORE_MODEL) or engine.captures_prompts:
        hooks["UserPromptSubmit"] = [
            *hooks.get("UserPromptSubmit", []),
            HookMatcher(hooks=[_build_user_prompt_submit_hook(engine)]),
        ]
        hooks_modified = True

    if config.enforces(EnforcementPoint.BEFORE_TOOL):
        hooks["PreToolUse"] = [
            # first, and no matcher → every tool
            HookMatcher(hooks=[_build_pre_tool_use_hook(engine)]),
            *hooks.get("PreToolUse", []),
        ]
        hooks_modified = True

    if config.enforces(EnforcementPoint.AFTER_TOOL):
        hooks["PostToolUse"] = [
            *hooks.get("PostToolUse", []),
            # no matcher → every tool; non-records pass through
            HookMatcher(hooks=[_build_post_tool_use_hook(engine)]),
        ]
        hooks_modified = True

    if not hooks_modified:
        return options
    return dataclasses.replace(options, hooks=hooks)
