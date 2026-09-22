"""Configuration: what to enforce, where the PDP is, and how identity becomes attributes.

``AuthZConfig`` is the one object every adapter takes. It holds the PDP connection
(``PDPConnection``), the decision-based enforcement points (``EnforcementPoint`` →
``Enforcer``), RAG retrieval filtering (``RetrievalConfig``), the context provider that turns
the caller's identity into PDP attributes (``ContextProvider``), hooks, and the
obligations-and-advice handling. Nothing is enforced unless configured.
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from axiomatics_ai.core.hooks import AuthzHook, HookDispatcher
from axiomatics_ai.core.pdp import ObligationsAndAdviceConfig, PdpEndpoints
from axiomatics_ai.core.pdp.obligations import ApprovalMemory, PausedCalls
from axiomatics_ai.core.post_invocation import PostInvocationStrategy

if TYPE_CHECKING:
    from axiomatics_ai.core.pdp import PdpClient, HealthCheckResult
    from axiomatics_ai.core.cache import AuthzDecisionCacheWithTTL
    from axiomatics_ai.core.llm_enrichment import LLMAttributeEnricher

logger = logging.getLogger(__name__)

# Whether DefaultServerContextProvider has warned that it does not verify tokens (once).
_warned_unverified_tokens = False


class EnforcementPoint(str, Enum):
    """A point where decision-based authorization is enforced.

    The five agent-lifecycle points plus A2A invocation. RAG retrieval filtering is a different
    kind of enforcement (a constraint-based metadata pre-filter), configured separately via
    ``AuthZConfig.retrieval``.

    Attributes:
        BEFORE_MODEL: The user's prompt, before it reaches the model (content guard).
        TOOL_VISIBILITY: The tool list the model sees.
        BEFORE_TOOL: Each tool call, before it runs.
        AFTER_TOOL: Each tool result, before the model sees it (post-invocation).
        AFTER_MODEL: The model's answer, before the user sees it (content guard).
        A2A_INVOCATION: A call to or from another agent over A2A.
    """

    TOOL_VISIBILITY = "tool_visibility"   # which tools the model may see
    BEFORE_MODEL = "before_model"         # the assembled prompt / model input
    BEFORE_TOOL = "before_tool"           # a tool call + its arguments (the model's intent)
    AFTER_TOOL = "after_tool"             # a tool's output
    AFTER_MODEL = "after_model"           # the final answer to the user
    A2A_INVOCATION = "a2a_invocation"     # sending a message to a remote agent


@dataclass
class Enforcer:
    """Declarative configuration for one enforcement point, with no runtime logic of its own.

    Placed in ``AuthZConfig.enforcement[point]``; the adapters read it. Presence at a point is
    the on-switch: an absent point (or ``enforcement=None``) is not enforced.

    Attributes:
        action_name (str): The PDP action name sent for this point (through the provider's
            ``get_action``).
        strategy (PostInvocationStrategy | None): The post-invocation transform, **AFTER_TOOL
            only**, e.g. ``RecordFilterStrategy()`` or ``ColumnMaskStrategy("ssn")``.
        action_overrides (dict[str, str]): Per-tool action names, **AFTER_TOOL only**
            (``{tool_name: action_name}``); others use ``action_name``.
        enrichers (list[LLMAttributeEnricher]): LLM attribute enrichers whose attributes join
            this point's PDP request (BEFORE_MODEL, AFTER_MODEL, BEFORE_TOOL). The PDP still
            decides.
        on_unavailable (str): When an enricher cannot classify: ``"fail_closed"`` (default)
            blocks without asking the PDP; ``"fail_open"`` asks the PDP with the availability
            sentinel, so the policy decides.
        block_message (str | None): The refusal text for blocked content at BEFORE_MODEL /
            AFTER_MODEL; None uses the point's default.
        on_unrecognized_output (str): What happens to tool output the strategy can't inspect
            (plain text, a scalar), **AFTER_TOOL only**: ``"fail_open"`` (default) returns it
            unchanged, with a warning and a ``skipped`` event; ``"fail_closed"`` withholds it.
    """

    action_name: str
    strategy: PostInvocationStrategy | None = None
    action_overrides: dict[str, str] = field(default_factory=dict)
    enrichers: list[LLMAttributeEnricher] = field(default_factory=list)
    on_unavailable: Literal["fail_closed", "fail_open"] = "fail_closed"
    block_message: str | None = None
    on_unrecognized_output: Literal["fail_open", "fail_closed"] = "fail_open"


# Default refusal text substituted for blocked content at the content-guard points, when an
# Enforcer sets no ``block_message``. Kept here (not per-surface) so every framework shares
# one wording; resolve via ``block_message_for``.
DEFAULT_BLOCK_MESSAGES: dict[EnforcementPoint, str] = {
    EnforcementPoint.BEFORE_MODEL: (
        "I can't help with that request — it was blocked by your organization's authorization "
        "policy before reaching the model."
    ),
    EnforcementPoint.AFTER_MODEL: (
        "I generated a response, but it was withheld by your organization's authorization "
        "policy and can't be shown."
    ),
}

_GENERIC_BLOCK_MESSAGE = "This action was blocked by your organization's authorization policy."


def block_message_for(enforcer: Enforcer, point: EnforcementPoint) -> str:
    """Pick the refusal text for content an Enforcer blocks at a content-guard point.

    Args:
        enforcer (Enforcer): The point's Enforcer.
        point (EnforcementPoint): The point.

    Returns:
        str: The Enforcer's ``block_message`` if set, else the point's shared default.
    """
    return enforcer.block_message or DEFAULT_BLOCK_MESSAGES.get(point, _GENERIC_BLOCK_MESSAGE)


# Points whose Enforcer runs its enrichers; anywhere else they are never called.
_POINTS_THAT_RUN_ENRICHERS = frozenset({
    EnforcementPoint.BEFORE_MODEL, EnforcementPoint.AFTER_MODEL, EnforcementPoint.BEFORE_TOOL,
})


def _enricher_problems(point: EnforcementPoint, enf: Enforcer) -> list[str]:
    """Check one Enforcer's enrichers for ``validate()``.

    Each must sit at a point that runs enrichers and that it is meant for
    (``EnricherSpec.allowed_points``).
    """
    if not enf.enrichers:
        return []
    if point not in _POINTS_THAT_RUN_ENRICHERS:
        return [f"Enforcer at {point.value} sets enrichers, which never run at this point."]
    problems = []
    for enricher in enf.enrichers:
        spec = enricher.spec
        if spec.allowed_points is not None and point not in spec.allowed_points:
            meant_for = ", ".join(sorted(p.value for p in spec.allowed_points))
            problems.append(f"Enricher {spec.name!r} is meant for {meant_for}, not {point.value}.")
        if point is EnforcementPoint.BEFORE_TOOL and spec.num_user_messages < 1:
            problems.append(f"Enricher {spec.name!r} at before_tool needs num_user_messages >= 1.")
    return problems


@dataclass
class RetrievalConfig:
    """Configuration for RAG retrieval filtering.

    A constraint-based enforcement: a partial-evaluation (CAQ) reverse query whose result
    becomes a vector-store metadata filter. Setting ``AuthZConfig.retrieval`` turns it on.

    Attributes:
        action_name (str): The PDP action for the reverse query.
        attribute_map (dict[str, str] | None): PDP attribute name (e.g. ``"resource.dept"``) →
            vector-store metadata key. Names not listed fall back to stripping the ``doc.`` /
            ``resource.`` prefix.
    """

    action_name: str = "retrieval"
    attribute_map: dict[str, str] | None = None


def standard_tool_enforcement(
    *,
    visibility: bool = True,
    invocation: bool = True,
    post_invocation: bool = True,
    visibility_action: str = "tool_visibility",
    invocation_action: str = "tool_invocation",
    post_invocation_action: str = "post_invocation",
    strategy: PostInvocationStrategy | None = None,
    action_overrides: dict[str, str] | None = None,
) -> dict[EnforcementPoint, Enforcer]:
    """Build the standard tool enforcement: visibility, invocation and post-invocation.

    Pass the result as ``AuthZConfig(enforcement=standard_tool_enforcement())``.

    Args:
        visibility (bool): Enforce TOOL_VISIBILITY.
        invocation (bool): Enforce BEFORE_TOOL.
        post_invocation (bool): Enforce AFTER_TOOL.
        visibility_action (str): The action name for TOOL_VISIBILITY.
        invocation_action (str): The action name for BEFORE_TOOL.
        post_invocation_action (str): The action name for AFTER_TOOL.
        strategy (PostInvocationStrategy, optional): The AFTER_TOOL strategy;
            ``RecordFilterStrategy()`` when None.
        action_overrides (dict[str, str], optional): Per-tool AFTER_TOOL action names.

    Returns:
        dict[EnforcementPoint, Enforcer]: The enforcement map.
    """
    # Lazy import: RecordFilterStrategy lives in core.presets, which imports back into core —
    # deferring to call time keeps module import order robust.
    from axiomatics_ai.core.presets import RecordFilterStrategy

    enforcement: dict[EnforcementPoint, Enforcer] = {}
    if visibility:
        enforcement[EnforcementPoint.TOOL_VISIBILITY] = Enforcer(action_name=visibility_action)
    if invocation:
        enforcement[EnforcementPoint.BEFORE_TOOL] = Enforcer(action_name=invocation_action)
    if post_invocation:
        enforcement[EnforcementPoint.AFTER_TOOL] = Enforcer(
            action_name=post_invocation_action,
            strategy=strategy or RecordFilterStrategy(),
            action_overrides=action_overrides or {},
        )
    return enforcement


@dataclass
class PDPConnection:
    """How to reach the PDP and cache its decisions.

    Attributes:
        base_url (str): The PDP's base URL.
        timeout (float): HTTP timeout per request, in seconds.
        protocol (str): ``"xacml"`` (XACML 3.0 JSON profile, default) or ``"authzen"`` (OpenID
            AuthZEN) for permission checks. RAG retrieval always uses the XACML-based reverse
            query.
        fail_open (bool): An unreachable PDP or inconclusive decision: True permits, False
            (default) denies.
        endpoints (PdpEndpoints): Where the PDP mounts each call; ADS's paths by default.
        cache_ttl_seconds (float): How long a decision is cached (never past a verified token's
            expiry); 0 disables the cache.
        cache_maxsize (int): How many decisions the cache holds.
        headers (dict[str, str]): Headers sent with every PDP request, e.g. an API key.
        auth (Any): Anything httpx accepts: a ``(user, password)`` tuple for Basic, or an
            ``httpx.Auth`` that sets (and refreshes) a bearer token.
        verify (Any): TLS verification: True, False, a CA bundle path, or an ``ssl.SSLContext``.
        cert (Any): A client certificate for mTLS: a path, or a ``(cert, key)`` tuple.
        http_client (httpx.Client | None): Your own client, instead of ``headers``, ``auth``,
            ``verify`` and ``cert``.

    Credentials (``headers``, ``auth``, ``cert``, ``http_client``) are left out of ``repr``.
    """

    base_url: str = "http://localhost:8080"
    timeout: float = 10.0
    protocol: Literal["xacml", "authzen"] = "xacml"
    fail_open: bool = False                 # behaviour when the PDP is unreachable
    # Request paths relative to base_url — override for a PDP that mounts them elsewhere.
    endpoints: PdpEndpoints = field(default_factory=PdpEndpoints)
    # Decision cache — keyed on (subject, action, resource, environment); set 0 to disable.
    cache_ttl_seconds: float = 300.0
    cache_maxsize: int = 1024
    # Authentication and TLS (see the class docstring).
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    auth: Any = field(default=None, repr=False)
    verify: Any = True
    cert: Any = field(default=None, repr=False)
    http_client: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class AttributeNames:
    """The attributes the SDK's own names (of a tool, an agent, an MCP server) are sent as.

    Attributes:
        tool (str): Resource: the tool's name at tool visibility, invocation and
            post-invocation: the same attribute at all three, so one policy attribute
            identifies the tool everywhere.
        agent (str): Resource: the target agent's name at A2A invocation.
        mcp_server (str): Resource: the MCP server a tool belongs to, where the framework names
            it (the Claude Agent SDK's ``mcp__<server>__<tool>``), so a policy can tell two
            servers' same-named tools apart.
        acting_agent (str | None): Subject: the name of the agent in the app making the call
            (the main agent, or a sub-agent), so one policy can treat sub-agents differently.
            A name from code, not an authenticated identity (that is ``agent_id`` /
            ``actor_id``). Sent where the agent has a name (ADK, OpenAI, Strands, a named
            LangChain agent, a deepagents sub-agent), and only with a caller identity; None
            leaves it out.
    """

    tool: str = "tool_name"
    agent: str = "agent_name"
    mcp_server: str = "mcp_server"
    acting_agent: str | None = "agent_name"


class _ConfigState:
    """Runtime state shared by everything built from one config; not configuration.

    The decision caches (weakly held, so ``invalidate_decisions()`` reaches them all), the human
    approvals remembered for their ttl, and the tool calls paused for approval.
    """

    def __init__(self) -> None:
        self.caches: weakref.WeakSet = weakref.WeakSet()
        self.approvals = ApprovalMemory()
        self.paused = PausedCalls()


@dataclass
class AuthZConfig:
    """Configuration for authorization: the one object every adapter takes.

    Nothing is enforced unless you populate ``enforcement`` (and/or ``retrieval``): there are no
    hidden defaults. ``standard_tool_enforcement()`` builds the usual tool set. Misconfigurations
    that are always wrong are logged at construction (see ``validate``).

    Attributes:
        connection (PDPConnection): How to reach the PDP and cache its decisions.
        enforcement (dict[EnforcementPoint, Enforcer] | None): The decision-based enforcement
            points and their Enforcers.
        retrieval (RetrievalConfig | None): RAG retrieval filtering; None is off.
        context_provider (ContextProvider | None): Turns the caller's identity into PDP
            attributes; None uses the path's default (``DefaultClientContextProvider`` or
            ``DefaultServerContextProvider``).
        hooks (list[AuthzHook]): Observe-only hooks (logging, metrics, audit).
        obligations_and_advice (ObligationsAndAdviceConfig | None): How obligations and advice
            are handled, including human approval; None is the default (unknown obligations
            deny).
        attribute_names (AttributeNames): The attributes a tool's, an agent's and an MCP
            server's names are sent as, and the acting agent's.
    """

    connection: PDPConnection = field(default_factory=PDPConnection)

    # Decision-based enforcement points → their Enforcer. Presence of a point = enforced
    # there; an absent point (or None) = not enforced.
    enforcement: dict[EnforcementPoint, Enforcer] | None = None

    # Constraint-based RAG retrieval filtering (a CAQ reverse-query metadata pre-filter).
    # Set to enable it on wrapped vector stores; None = off.
    retrieval: RetrievalConfig | None = None

    # Context provider (None → DefaultClientContextProvider or DefaultServerContextProvider)
    context_provider: ContextProvider | None = None

    # Observe-only hooks (see core.hooks, e.g. LoggingHook). Empty (default) => zero overhead.
    hooks: list[AuthzHook] = field(default_factory=list)

    # What the PEP does with the obligations and advice the PDP returns, including human
    # approval of tool calls (see core.pdp.obligations). None → the default: unknown
    # obligations deny (XACML), and "require-approval" is recognised.
    obligations_and_advice: ObligationsAndAdviceConfig | None = None

    # The attributes a tool's, an agent's and an MCP server's names are sent as.
    attribute_names: AttributeNames = field(default_factory=AttributeNames)

    # Runtime state shared by everything built from this config (caches, approvals).
    _state: _ConfigState = field(default_factory=_ConfigState, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Surface always-wrong misconfigurations early, at construction, instead of failing
        # deep in a middleware call at request time. These are logged (not raised) so a
        # partially-built config in a REPL/notebook still constructs.
        for problem in self.validate():
            logger.warning("AuthZConfig: %s", problem)

    def validate(self) -> list[str]:
        """List the misconfigurations that are always wrong, whatever the path.

        Called automatically at construction (problems are logged, not raised); also useful in a
        test or a preflight check. An empty ``enforcement`` is not flagged: the A2A and
        tool_decorator paths, and retrieval-only configs, legitimately use a bare config.

        Returns:
            list[str]: Human-readable problems; empty when the config looks fine.
        """
        problems: list[str] = []
        for field_name in ("tool", "agent", "mcp_server"):
            if not getattr(self.attribute_names, field_name):
                problems.append(
                    f"attribute_names.{field_name} is empty; it must name the attribute "
                    "the resource is sent as."
                )
        for point, enf in (self.enforcement or {}).items():
            after_tool_only = (
                enf.strategy is not None or enf.action_overrides
                or enf.on_unrecognized_output != "fail_open"
            )
            if point is not EnforcementPoint.AFTER_TOOL and after_tool_only:
                problems.append(
                    f"Enforcer at {point.value} sets strategy/action_overrides/"
                    "on_unrecognized_output, which are AFTER_TOOL-only and will be ignored here."
                )
            problems.extend(_enricher_problems(point, enf))
        if self.enforces(EnforcementPoint.TOOL_VISIBILITY) and not self.enforces(EnforcementPoint.BEFORE_TOOL):
            problems.append(
                "TOOL_VISIBILITY is enforced without BEFORE_TOOL. Visibility only hides tools "
                "from the model's tool list; a hidden tool can still be called (an MCP client "
                "can call any tool by name). Add a BEFORE_TOOL Enforcer to block the calls."
            )
        before_tool = self.enforcer(EnforcementPoint.BEFORE_TOOL)
        if before_tool is not None and before_tool.enrichers and self.connection.protocol == "authzen":
            problems.append(
                "BEFORE_TOOL has enrichers (e.g. intent_enricher) with protocol='authzen'. Intent "
                "attributes are lists, and Axiomatics ADS's AuthZEN mapping does not read JSON "
                "lists as multi-valued attributes, so intent rules would deny intended calls. "
                "Use protocol='xacml' with ADS; other AuthZEN PDPs may handle lists."
            )
        if self.retrieval is not None and self.connection.protocol == "authzen":
            problems.append(
                "retrieval is configured with protocol='authzen', but RAG retrieval filtering "
                "always uses XACML (AuthZEN has no partial-evaluation endpoint) — the reverse "
                "query will use XACML regardless."
            )
        return problems

    def summary(self) -> str:
        """Describe what this config enforces, in one log-friendly line.

        Returns:
            str: The PDP URL, protocol, fail_open, enforced points and extras.
        """
        points = ", ".join(p.value for p in (self.enforcement or {})) or "none"
        extras = []
        if self.retrieval is not None:
            extras.append("retrieval:on")
        provider = type(self.context_provider).__name__ if self.context_provider else "default"
        extra = (", " + ", ".join(extras)) if extras else ""
        return (
            f"AuthZConfig(pdp={self.connection.base_url} [{self.connection.protocol}, "
            f"fail_open={self.connection.fail_open}], enforces=[{points}]{extra}, "
            f"context_provider={provider}, hooks={len(self.hooks)})"
        )

    def check_connection(self) -> HealthCheckResult:
        """Probe the PDP this config points at.

        Returns:
            HealthCheckResult: ``ok`` and the status, latency, or error (see
            ``PdpClient.healthcheck``).
        """
        return self.build_pdp_client().healthcheck()

    def enforcer(self, point: EnforcementPoint) -> Enforcer | None:
        """Get the Enforcer configured at a point.

        Args:
            point (EnforcementPoint): The point.

        Returns:
            Enforcer | None: The Enforcer, or None when the point is not enforced.
        """
        return (self.enforcement or {}).get(point)

    def enforces(self, point: EnforcementPoint) -> bool:
        """Whether a point is enforced (an Enforcer is configured there).

        Args:
            point (EnforcementPoint): The point.

        Returns:
            bool: True when the point is enforced.
        """
        return self.enforcer(point) is not None

    def build_pdp_client(self) -> PdpClient:
        """Build a PDP client from this config's connection, hooks and obligations config.

        Returns:
            PdpClient: A new client.
        """
        from axiomatics_ai.core.pdp import PdpClient
        return PdpClient(
            base_url=self.connection.base_url,
            timeout=self.connection.timeout,
            fail_open=self.connection.fail_open,
            protocol=self.connection.protocol,
            dispatcher=self.build_hook_dispatcher(),
            endpoints=self.connection.endpoints,
            obligations=self.obligations_and_advice,
            headers=self.connection.headers,
            auth=self.connection.auth,
            verify=self.connection.verify,
            cert=self.connection.cert,
            http_client=self.connection.http_client,
        )

    def build_hook_dispatcher(self) -> HookDispatcher:
        """Build a dispatcher for this config's hooks.

        Returns:
            HookDispatcher: A new dispatcher.
        """
        return HookDispatcher(self.hooks)

    def build_cache(self) -> AuthzDecisionCacheWithTTL:
        """Build a decision cache from this config's connection settings.

        The cache is registered with the config, so ``invalidate_decisions()`` reaches it.

        Returns:
            AuthzDecisionCacheWithTTL: A new cache.
        """
        from axiomatics_ai.core.cache import AuthzDecisionCacheWithTTL
        cache = AuthzDecisionCacheWithTTL(
            ttl=self.connection.cache_ttl_seconds,
            maxsize=self.connection.cache_maxsize,
        )
        self._state.caches.add(cache)  # so invalidate_decisions() reaches every surface's cache
        return cache

    def invalidate_decisions(self, reason: str = "", **subject_attributes: Any) -> int:
        """Drop cached decisions, e.g. on a revocation or risk signal.

        Cached decisions must not outlive such a signal (AIMS §11). Reaches every cache built from
        this config, the remembered human approvals and the calls paused for approval (a paused
        call is checked anew when it resumes), and emits an ``on_invalidation`` event.

        Args:
            reason (str): Why, for the event.
            **subject_attributes (Any): Drop only decisions for this subject, e.g.
                ``user_id="alice"`` or ``agent_id="spiffe://…"``; all decisions when none are given.

        Returns:
            int: How many decisions were dropped.
        """
        # A snapshot: the WeakSet can shrink mid-loop when a cache is garbage-collected.
        removed = sum(cache.invalidate(**subject_attributes) for cache in self._state.caches.copy())
        self._state.approvals.invalidate(**subject_attributes)
        self._state.paused.invalidate(**subject_attributes)
        self.build_hook_dispatcher().invalidation(
            subject=dict(subject_attributes), removed=removed, reason=reason,
        )
        return removed


@runtime_checkable
class ContextProvider(Protocol):
    """The protocol for turning the caller's identity and request into PDP attributes.

    The SDK calls these methods to build the subject, action, resource and environment of each
    request. Each receives the runtime context the caller passed (client-side paths) or the
    request's HTTP headers (server-side paths). A method may return an empty dict to omit that
    category.
    """

    def get_subject(self, context: Any) -> dict[str, Any]:
        """Build the subject attributes (XACML AccessSubject).

        Args:
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).

        Returns:
            dict[str, Any]: The subject attributes. Empty means no identity: the request is denied
            (resolved by ``fail_open``) without asking the PDP.
        """
        ...

    def get_action(self, action_name: str, context: Any) -> dict[str, Any]:
        """Build the action attributes.

        Args:
            action_name (str): The point's action name (``Enforcer.action_name``).
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).

        Returns:
            dict[str, Any]: The action attributes, usually ``{"name": action_name}``.
        """
        ...

    def get_resource(
        self,
        resource_name: str,
        context: Any,
        *,
        args: dict[str, Any] | None = None,
        resource_name_attribute: str | None,
    ) -> dict[str, Any]:
        """Build the resource attributes.

        Include ``args`` when the policy matches on argument values, but never let one overwrite
        the resource identifier. A provider may send the name under a different attribute id, as
        long as the policy uses the same one.

        Args:
            resource_name (str): The resource: a tool, an agent, or what an ``Authorizer`` caller
                passes.
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).
            args (dict[str, Any], optional): The tool call's arguments, when checking a tool call.
            resource_name_attribute (str | None): The attribute the SDK suggests sending the name
                as (``attribute_names.tool`` at the tool points, ``.agent`` for A2A); None
                where the resource has no identifier (the content guards).

        Returns:
            dict[str, Any]: The resource attributes.
        """
        ...

    def get_environment(self, context: Any) -> dict[str, Any]:
        """Build the environment attributes.

        Only include stable, session-scoped values: volatile data (timestamps, IPs, nonces) makes
        every request a new cache key, defeating the cache.

        Args:
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).

        Returns:
            dict[str, Any]: The environment attributes.
        """
        ...


def _default_resource(
    resource_name: str,
    args: dict[str, Any] | None,
    resource_name_attribute: str | None,
) -> dict[str, Any]:
    """The default providers' resource: ``{resource_name_attribute: resource_name, **args}``.

    The identifier always wins. Arguments come from the model, so an argument whose name
    matches ``resource_name_attribute`` would otherwise let the model decide which resource
    the policy sees; it is dropped instead, with a warning.
    """
    resource: dict[str, Any] = {}
    if resource_name_attribute is not None:
        resource[resource_name_attribute] = resource_name
    for key, value in (args or {}).items():
        if key == resource_name_attribute:
            logger.warning(
                "Argument %r of %r has the same name as the resource identifier attribute; "
                "it is not sent to the PDP. Rename the argument, or set a different attribute "
                "id (e.g. AuthZConfig.attribute_names).",
                key, resource_name,
            )
            continue
        resource[key] = value
    return resource


class DefaultClientContextProvider:
    """The default provider for client-side paths (agents, langchain_mcp, RAG).

    Reads these attributes (or dict keys) from the runtime context: ``username``, ``user_id``,
    ``tenant_id``, ``agent_id`` (subject) and ``session_id`` (environment). Missing fields are
    omitted. ``agent_id`` is the acting agent's own identity (e.g. its SPIFFE ID); it comes from
    the context, so one application can run several agents.
    """

    def _get(self, context: Any, key: str) -> str | None:
        if isinstance(context, dict):
            return context.get(key)
        return getattr(context, key, None)

    def get_subject(self, context: Any) -> dict[str, Any]:
        """Read ``username``, ``user_id``, ``tenant_id`` and ``agent_id`` from the context.

        Args:
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).

        Returns:
            dict[str, Any]: The subject attributes present in the context.
        """
        return {
            k: v
            for k, v in {
                "username": self._get(context, "username"),
                "user_id": self._get(context, "user_id"),
                "tenant_id": self._get(context, "tenant_id"),
                "agent_id": self._get(context, "agent_id"),
            }.items()
            if v is not None
        }

    def get_action(self, action_name: str, context: Any) -> dict[str, Any]:
        """Build ``{"name": action_name}``.

        Args:
            action_name (str): The point's action name.
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).

        Returns:
            dict[str, Any]: The action attributes.
        """
        return {"name": action_name}

    def get_resource(
        self,
        resource_name: str,
        context: Any,
        *,
        args: dict[str, Any] | None = None,
        resource_name_attribute: str | None,
    ) -> dict[str, Any]:
        """Build the resource: the name under ``resource_name_attribute``, plus the arguments.

        The arguments never overwrite the name.

        Args:
            resource_name (str): The resource name.
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).
            args (dict[str, Any], optional): The tool call's arguments.
            resource_name_attribute (str | None): The attribute the name is sent as; None sends
                only the arguments.

        Returns:
            dict[str, Any]: The resource attributes.
        """
        return _default_resource(resource_name, args, resource_name_attribute)

    def get_environment(self, context: Any) -> dict[str, Any]:
        """Read ``session_id`` from the context.

        Args:
            context (Any): The runtime context the caller passed (e.g. ``agent.invoke(..., context=ctx)``).

        Returns:
            dict[str, Any]: ``{"session_id": ...}`` when present, else empty.
        """
        session_id = self._get(context, "session_id")
        return {"session_id": session_id} if session_id else {}


class DefaultServerContextProvider:
    """The default provider for server-side paths (fastmcp, tool_decorator, a2a).

    Every method receives the HTTP headers. All JWT claims from ``Authorization: Bearer`` become
    subject attributes; the environment is empty.

    **Authentication is assumed to have happened upstream.** Claims are read without signature
    verification: deploy behind an ingress, API gateway or auth middleware that verifies the
    token first, or use ``AgentIdentityContextProvider`` with a ``JwtVerification`` to verify
    here (the gateway requires one). Subclass to customise::

        class MyProvider(DefaultServerContextProvider):
            def get_environment(self, headers):
                return {"session_id": headers.get("x-session-id", "")}
    """

    def get_subject(self, headers: dict[str, Any]) -> dict[str, Any]:
        """Read the bearer token's claims, without verifying its signature.

        Args:
            headers (dict[str, Any]): The request's HTTP headers.

        Returns:
            dict[str, Any]: The token's claims; empty without a bearer token or when it doesn't
            decode.
        """
        from axiomatics_ai.core.authz_helpers import decode_bearer_jwt_unverified
        global _warned_unverified_tokens
        if not _warned_unverified_tokens:
            _warned_unverified_tokens = True
            logger.warning(
                "Bearer tokens are read WITHOUT verification (DefaultServerContextProvider): "
                "authentication must happen upstream (ingress, API gateway). To verify them "
                "here, use AgentIdentityContextProvider with a JwtVerification."
            )
        return decode_bearer_jwt_unverified(headers)

    def get_environment(self, headers: dict[str, Any]) -> dict[str, Any]:
        """Return no environment attributes.

        Args:
            headers (dict[str, Any]): The request's HTTP headers.

        Returns:
            dict[str, Any]: An empty dict.
        """
        return {}

    def get_action(self, action_name: str, headers: dict[str, Any]) -> dict[str, Any]:
        """Build ``{"name": action_name}``.

        Args:
            action_name (str): The point's action name.
            headers (dict[str, Any]): The request's HTTP headers.

        Returns:
            dict[str, Any]: The action attributes.
        """
        return {"name": action_name}

    def get_resource(
        self,
        resource_name: str,
        headers: dict[str, Any],
        *,
        args: dict[str, Any] | None = None,
        resource_name_attribute: str | None,
    ) -> dict[str, Any]:
        """Build the resource: the name under ``resource_name_attribute``, plus the arguments.

        The arguments never overwrite the name.

        Args:
            resource_name (str): The resource name.
            headers (dict[str, Any]): The request's HTTP headers.
            args (dict[str, Any], optional): The tool call's arguments.
            resource_name_attribute (str | None): The attribute the name is sent as.

        Returns:
            dict[str, Any]: The resource attributes.
        """
        return _default_resource(resource_name, args, resource_name_attribute)
