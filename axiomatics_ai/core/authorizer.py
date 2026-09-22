"""Direct PDP access for applications the SDK has no adapter for.

Every other path in this SDK hooks a framework (LangChain, FastMCP, ADK, ...) and decides
*where* to ask the PDP. ``Authorizer`` is the same machinery with no framework: you decide
where to ask, it builds the request from your ``AuthZConfig`` and returns the answer. What
you get over calling ``PdpClient`` yourself:

- the configured **context provider** turns your application context into subject /
  action / resource / environment attributes (or pass the dicts explicitly);
- the **decision cache** (``PDPConnection.cache_ttl_seconds``) is honoured;
- **hooks** fire — transport events for every wire call and a ``DecisionEvent`` with
  ``surface="direct"`` for every answer, so audit and metrics work like on any other path;
- ``fail_open`` and the wire ``protocol`` come from the connection, not from call-site
  defaults.

Enforcement is yours: the ``Authorizer`` only answers; what you do with a ``False`` is up to
your application. Example::

    from axiomatics_ai import Authorizer, AuthZConfig, PDPConnection

    authz = Authorizer(AuthZConfig(connection=PDPConnection(base_url="http://localhost:8080")))

    if not authz.is_permitted(action="read_document", resource="doc-42",
                              args={"classification": "internal"},
                              context={"username": "alice"}):
        raise PermissionError("not allowed")

    visible = authz.is_permitted_batch(action="tool_visibility", resource_name_attribute="tool_name",
                                       resources=["search_docs", "send_email"],
                                       context={"username": "alice"})

    # Partial evaluation (PDPs that support it): the condition under which alice may read,
    # as a constraint tree you can translate into a query filter, or as display text.
    tree = authz.partial_evaluation(action="read_document", context={"username": "alice"})
    text = authz.partial_evaluation_text(action="read_document", context={"username": "alice"})

The raw client stays reachable as ``authz.pdp`` for anything the façade does not cover.
"""
from __future__ import annotations

from typing import Any

from axiomatics_ai.core.config import AuthZConfig, DefaultClientContextProvider
from axiomatics_ai.core.enforcement import authorized_call
from axiomatics_ai.core.pdp import (
    HealthCheckResult,
    PdpClient,
    flatten_to_provided_attributes,
    strip_none,
)

DIRECT_SURFACE = "direct"


class Authorizer:
    """Ask the PDP directly, through the config's context provider, cache and hooks.

    Args:
        config (AuthZConfig): The PDP connection, provider, cache settings and hooks.
        surface (str): The label of the decision events this instance emits; give each call
            site its own to tell them apart in a hook.
    """

    def __init__(self, config: AuthZConfig, *, surface: str = DIRECT_SURFACE) -> None:
        self.config = config
        self.surface = surface
        self.pdp: PdpClient = config.build_pdp_client()
        self._cache = config.build_cache()
        self._dispatcher = config.build_hook_dispatcher()
        self._provider = config.context_provider or DefaultClientContextProvider()

    def __repr__(self) -> str:
        return f"Authorizer(surface={self.surface!r}, pdp={self.pdp!r})"

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def check_connection(self) -> HealthCheckResult:
        """Probe the PDP's health endpoint; never raises.

        Returns:
            HealthCheckResult: ``ok`` and the status, latency, or error.
        """
        return self.pdp.healthcheck()

    def clear_cache(self) -> None:
        """Drop cached decisions; call after a policy change so stale answers aren't served."""
        self._cache.clear()

    # ── Decisions ──────────────────────────────────────────────────────────────

    def is_permitted(
        self,
        *,
        action: str,
        resource: str,
        args: dict[str, Any] | None = None,
        resource_name_attribute: str = "resource_name",
        context: Any = None,
        subject: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
    ) -> bool:
        """Ask whether the subject may perform ``action`` on ``resource``: one decision.

        The context provider turns the names into attribute dicts (the default sends the action as
        ``{"name": action}`` and the resource as ``{resource_name_attribute: resource}`` with
        ``args`` merged in). Permit → True, Deny → False, anything else (NotApplicable,
        Indeterminate, PDP unreachable) → ``connection.fail_open``. An empty subject never reaches
        the PDP.

        Args:
            action (str): The action name.
            resource (str): The resource name.
            args (dict[str, Any], optional): Extra resource attributes, e.g. ``{"classification":
                "internal"}``.
            resource_name_attribute (str): The attribute the resource name is sent as.
            context (Any): Your application context, handed to the config's context provider.
            subject (dict[str, Any], optional): Subject attributes, instead of the provider's.
            environment (dict[str, Any], optional): Environment attributes, instead of the
                provider's.

        Returns:
            bool: Whether it is permitted.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        subject_attrs, environment_attrs = self._identity(context, subject, environment)
        return authorized_call(
            pdp=self.pdp,
            cache=self._cache,
            config=self.config,
            surface=self.surface,
            enforcement_point=action,
            subject=subject_attrs,
            action=self._provider.get_action(action, context),
            resource=self._provider.get_resource(
                resource, context, args=args, resource_name_attribute=resource_name_attribute,
            ),
            environment=environment_attrs,
        )

    def is_permitted_batch(
        self,
        *,
        action: str,
        resources: list[str],
        resource_name_attribute: str = "resource_name",
        context: Any = None,
        subject: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
    ) -> set[str]:
        """Ask which of ``resources`` the subject may perform ``action`` on, in one batched request.

        Each name is sent as ``resource_name_attribute`` of its own sub-request (XACML Multiple
        Decision Profile, or an AuthZEN ``evaluations`` batch). Undecided names follow
        ``connection.fail_open``; an empty subject never reaches the PDP. A result resolved by
        ``fail_open`` is not cached.

        Args:
            action (str): The action name.
            resources (list[str]): The resource names to decide.
            resource_name_attribute (str): The attribute each name is sent as.
            context (Any): Your application context, handed to the config's context provider.
            subject (dict[str, Any], optional): Subject attributes, instead of the provider's.
            environment (dict[str, Any], optional): Environment attributes, instead of the
                provider's.

        Returns:
            set[str]: The permitted names.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        subject_attrs, environment_attrs = self._identity(context, subject, environment)
        action_attrs = self._provider.get_action(action, context)
        # Cache key covers the whole batch: the sorted names plus the attribute they're sent as.
        cache_resource = {resource_name_attribute: sorted(resources)}
        with self._dispatcher.correlation():
            cached = self._cache.get(subject_attrs, action_attrs, cache_resource, environment_attrs)
            if cached is not None:
                self._emit_batch(
                    action, subject_attrs, action_attrs, cache_resource, cached, resources,
                    cached=True, environment=environment_attrs,
                )
                return cached
            permitted = self.pdp.check_permission_batch(
                subject=subject_attrs,
                action=action_attrs,
                resource_names=list(resources),
                environment=environment_attrs,
                resource_attr_id=resource_name_attribute,
                fail_open=self.config.connection.fail_open,
            )
            if getattr(permitted, "answered", True):   # a fail_open fallback must not outlive the outage
                self._cache.set(subject_attrs, action_attrs, cache_resource, environment_attrs, permitted)
            self._emit_batch(
                action, subject_attrs, action_attrs, cache_resource, permitted, resources,
                cached=False, environment=environment_attrs,
            )
            return permitted

    # ── Partial evaluation (reverse query) ─────────────────────────────────────

    def partial_evaluation(
        self,
        *,
        action: str,
        context: Any = None,
        subject: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        mocked_attributes: list[dict] | None = None,
        decision_set: list[str] | None = None,
        exclude_indeterminate: bool = False,
    ) -> dict | None:
        """Ask not "is it permitted?" but "under what condition would it be?".

        The subject, action and environment are sent as provided; everything else the policy
        references (typically resource attributes) is left unbound, and the PDP returns the
        condition on those attributes under which the decision would be in ``decision_set``, as a
        Boolean Expression Tree to translate into a database or vector-store filter. This is what
        ``to_authorized_vectorstore`` does. Only PDPs with partial evaluation serve it (Axiomatics
        ADS's CAQ). Not cached.

        Args:
            action (str): The action name.
            context (Any): Your application context, handed to the config's context provider.
            subject (dict[str, Any], optional): Subject attributes, instead of the provider's.
            environment (dict[str, Any], optional): Environment attributes, instead of the
                provider's.
            mocked_attributes (list[dict], optional): Attributes the PDP should treat as given.
            decision_set (list[str], optional): The decisions to solve for; ``["PERMIT"]`` by default.
            exclude_indeterminate (bool): Leave Indeterminate branches out.

        Returns:
            dict | None: The constraint tree, or None on a transient failure, so you can apply your
            own fail-open or fail-closed rule.

        Raises:
            PdpRequestError: The PDP rejected the request, e.g. HTTP 404 from a PDP without
                partial evaluation.
        """
        provided = self._provided(action, context, subject, environment)
        return self.pdp.get_constraints(
            provided_attributes=provided,
            mocked_attributes=mocked_attributes,
            pdp_decision_set=decision_set,
            exclude_indeterminate=exclude_indeterminate,
        )

    def partial_evaluation_text(
        self,
        *,
        action: str,
        context: Any = None,
        subject: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        mocked_attributes: list[dict] | None = None,
        decision_set: list[str] | None = None,
        exclude_indeterminate: bool = False,
    ) -> str | None:
        """Get the same condition as ``partial_evaluation``, as the PDP's own display text.

        E.g. ``"department = finance AND level <= 3"``. For display and debugging only; never drive
        enforcement from it.

        Args:
            action (str): The action name.
            context (Any): Your application context, handed to the config's context provider.
            subject (dict[str, Any], optional): Subject attributes, instead of the provider's.
            environment (dict[str, Any], optional): Environment attributes, instead of the
                provider's.
            mocked_attributes (list[dict], optional): Attributes the PDP should treat as given.
            decision_set (list[str], optional): The decisions to solve for; ``["PERMIT"]`` by default.
            exclude_indeterminate (bool): Leave Indeterminate branches out.

        Returns:
            str | None: The rendering, or None (or an empty string) on a transient failure.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        provided = self._provided(action, context, subject, environment)
        return self.pdp.get_simplified_constraints(
            provided_attributes=provided,
            mocked_attributes=mocked_attributes,
            pdp_decision_set=decision_set,
            exclude_indeterminate=exclude_indeterminate,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _identity(
        self, context: Any, subject: dict | None, environment: dict | None,
    ) -> tuple[dict, dict]:
        subject_attrs = subject if subject is not None else self._provider.get_subject(context)
        environment_attrs = (
            environment if environment is not None else self._provider.get_environment(context)
        )
        return strip_none(subject_attrs), strip_none(environment_attrs)

    def _provided(
        self, action: str, context: Any, subject: dict | None, environment: dict | None,
    ) -> list[dict]:
        subject_attrs, environment_attrs = self._identity(context, subject, environment)
        action_attrs = self._provider.get_action(action, context)
        return flatten_to_provided_attributes(subject_attrs, action_attrs, environment_attrs)

    def _emit_batch(
        self, action, subject, action_attrs, resource, permitted, resources, *, cached, environment,
    ) -> None:
        self._dispatcher.decision(
            surface=self.surface,
            enforcement_point=action,
            subject=subject,
            action=action_attrs,
            resource=resource,
            decision="partial",
            cached=cached,
            detail={"permitted": len(permitted), "total": len(resources)},
            environment=environment,
        )
