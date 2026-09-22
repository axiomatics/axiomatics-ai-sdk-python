"""PdpClient: the PDP transport and the decision rule shared by both wire protocols.

The protocol-specific request builders and response parsers live in ``pdp.xacml`` and
``pdp.authzen``; both parse into a ``PdpDecision``, which this client turns into a Permit or
Deny (``fail_open``, then the obligations and advice, see ``pdp.obligations``).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from axiomatics_ai.core.hooks import HookDispatcher
from axiomatics_ai.core.pdp import authzen, xacml
from axiomatics_ai.core.pdp.attributes import sendable_attributes, strip_none
from axiomatics_ai.core.pdp.decision import Decision, PdpDecision
from axiomatics_ai.core.pdp.obligations import (
    ObligationContext,
    ObligationsAndAdviceConfig,
    Resolution,
    resolve as resolve_obligations,
)

AuthorizationDecision = dict[str, Any]
AuthorizationRequest = dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Outcome of a PDP connectivity probe.

    Returned by ``PdpClient.healthcheck`` and ``AuthZConfig.check_connection``. Truthy iff
    ``ok``.

    Attributes:
        ok (bool): True only on an HTTP 200 from the health endpoint.
        base_url (str): The PDP base URL that was probed.
        protocol (str): The configured wire protocol (``"xacml"`` or ``"authzen"``).
        latency_ms (float | None): How long the probe took.
        status_code (int | None): The HTTP status, or None when the PDP could not be reached.
        error (str | None): What went wrong, or None when ``ok``.
    """

    ok: bool
    base_url: str
    protocol: str
    latency_ms: float | None = None
    status_code: int | None = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True)
class PdpEndpoints:
    """Request paths appended to ``PDPConnection.base_url`` for each kind of PDP call.

    Fields are named by what the call does, not by any vendor's URL. The defaults are the
    paths Axiomatics ADS serves; a PDP that mounts the same protocols elsewhere needs only
    the differing fields overridden::

        PDPConnection(base_url="https://pdp.example", endpoints=PdpEndpoints(decision="/pdp"))

    Attributes:
        decision (str): XACML JSON decision requests, single and batch.
        authzen_evaluation (str): AuthZEN single evaluation.
        authzen_evaluations (str): AuthZEN batch evaluations.
        partial_evaluation (str): The reverse query returning a constraint tree (RAG retrieval
            filtering). Only PDPs that support partial evaluation (ADS's CAQ) serve it.
        partial_evaluation_text (str): The same reverse query as display text.
        health (str): The liveness probe used by ``healthcheck``.
    """

    decision: str = "/authorize"                       # XACML JSON: single request + MultiRequest
    health: str = "/health"                            # GET liveness probe
    authzen_evaluation: str = "/access/v1/evaluation"  # AuthZEN: single evaluation
    authzen_evaluations: str = "/access/v1/evaluations"  # AuthZEN: batch evaluations
    partial_evaluation: str = "/authorization/constraints"  # reverse query → constraint tree
    partial_evaluation_text: str = "/authorization/simplified-constraints"  # reverse query → display string


class PdpRequestError(Exception):
    """The PDP rejected the request with a deterministic error (HTTP 3xx or 4xx, or no JSON).

    Raised for misconfiguration: an undeclared attribute (AuthZEN maps attributes strictly),
    the AuthZEN endpoints not being enabled on ADS, a wrong path, bad credentials, a redirect
    (to a login page or another host), or a success response that isn't a JSON decision. These
    are NOT masked by ``fail_open``: retrying won't help, and silently degrading could hide a
    real bug (or wrongly permit access when ``fail_open=True``). Transient failures
    (connection errors, timeouts, HTTP 5xx) are swallowed instead, so ``fail_open`` can
    govern.
    """


def _elapsed_ms(start: float) -> float:
    """Milliseconds since ``start`` (a ``time.perf_counter()`` reading)."""
    elapsed = time.perf_counter() - start
    return elapsed * 1000


def _serialize_constraint_attributes(attributes: list[dict] | None) -> list[dict]:
    """Normalize CAQ ``providedAttributes``/``mockedAttributes`` entries.

    Each entry must carry ``attribute`` and ``values``; ``issuer`` is optional and only
    emitted when set. Values are always sent as a list, preserving native types.
    """
    out: list[dict] = []
    for entry in attributes or []:
        name = entry.get("attribute")
        if name is None:
            continue
        raw = entry.get("values")
        values = raw if isinstance(raw, list) else [raw]
        item = {"attribute": name, "values": values}
        if entry.get("issuer") is not None:
            item["issuer"] = entry["issuer"]
        out.append(item)
    return out


def flatten_to_provided_attributes(*category_dicts: dict | None) -> list[dict]:
    """Flatten attribute dicts into the entries a partial-evaluation query takes.

    Args:
        *category_dicts (dict | None): Subject, action and environment attribute dicts, in any
            number. ``None`` dicts and ``None`` values are skipped.

    Returns:
        list[dict]: One ``{"attribute": id, "values": [...]}`` entry per attribute; a scalar
        value is wrapped in a list.
    """
    provided: list[dict] = []
    for category in category_dicts:
        for name, value in sendable_attributes(category).items():
            provided.append({"attribute": name, "values": value if isinstance(value, list) else [value]})
    return provided


_SUPPORTED_PROTOCOLS = ("xacml", "authzen")


def _decision_to_bool(decision: Decision | None, *, fail_open: bool) -> bool:
    """Resolve a single-request decision: Permit → True, Deny → False, anything else → fail_open.

    "Anything else" is NotApplicable, Indeterminate, unknown, or PDP unreachable. Shared by both
    protocols so XACML and AuthZEN resolve to bool identically.
    """
    if decision is Decision.DENY:
        return False
    return decision is Decision.PERMIT or fail_open


class PermittedNames(set):
    """The permitted names of a batch check: a ``set`` that also says whether the PDP answered.

    Attributes:
        answered (bool): False when the PDP gave no decision for some name (unreachable, or a
            gap in its response), so those names were resolved by ``fail_open``. A cache must
            not keep such a result.
    """

    answered: bool = True

    @classmethod
    def of(cls, names, *, answered: bool) -> PermittedNames:
        """Build a ``PermittedNames``.

        Args:
            names (Iterable[str]): The permitted names.
            answered (bool): Whether the PDP decided every name.

        Returns:
            PermittedNames: The set, with ``answered`` set.
        """
        result = cls(names)
        result.answered = answered
        return result


class PdpClient:
    """The PDP wire client (XACML JSON or AuthZEN).

    Transport: one pooled ``httpx.Client`` per ``PdpClient``, built on first use from
    ``headers``, ``auth``, ``verify`` and ``cert``. Pass ``http_client`` instead to bring your
    own; its settings are then used as they are. ``close()`` (or a ``with`` block) releases a
    client this one built.

    Args:
        base_url (str): The PDP's base URL, e.g. ``"http://localhost:8080"``.
        timeout (float): HTTP timeout per request, in seconds.
        fail_open (bool): The default for an inconclusive decision or an unreachable PDP:
            True permits, False denies.
        protocol (str): ``"xacml"`` (XACML 3.0 JSON profile) or ``"authzen"`` (OpenID AuthZEN)
            for decision requests. Partial evaluation always uses XACML.
        dispatcher (HookDispatcher, optional): Receives the transport events
            (``on_pdp_request`` / ``on_pdp_response``).
        endpoints (PdpEndpoints, optional): Request paths; ADS's by default.
        obligations (ObligationsAndAdviceConfig, optional): How obligations and advice are
            handled; unknown obligations fail closed by default.
        headers (dict[str, str], optional): Headers sent with every request, e.g. an API key.
        auth (Any): Anything httpx accepts: a ``(user, password)`` tuple for Basic, or an
            ``httpx.Auth`` that sets (and refreshes) a bearer token.
        verify (Any): TLS verification: True, False, a CA bundle path, or an ``ssl.SSLContext``.
        cert (Any): A client certificate for mTLS: a path, or a ``(cert, key)`` tuple.
        http_client (httpx.Client, optional): Your own client, used instead of building one.

    Raises:
        ValueError: ``protocol`` is unsupported, or ``http_client`` is given together with
            ``headers``, ``auth``, ``verify`` or ``cert``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        fail_open: bool = False,
        protocol: str = "xacml",
        dispatcher: HookDispatcher | None = None,
        endpoints: PdpEndpoints | None = None,
        obligations: ObligationsAndAdviceConfig | None = None,
        headers: dict[str, str] | None = None,
        auth: Any = None,
        verify: Any = True,
        cert: Any = None,
        http_client: httpx.Client | None = None,
    ):
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported PDP protocol {protocol!r}; expected one of {_SUPPORTED_PROTOCOLS}"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Default for check_permission / check_permission_batch when the caller passes no
        # explicit fail_open: what to answer when the PDP is unreachable or inconclusive.
        self.fail_open = fail_open
        # Where each kind of call goes, relative to base_url (ADS paths by default).
        self.endpoints = endpoints or PdpEndpoints()
        # Wire protocol for check_permission / check_permission_batch. "xacml" posts XACML
        # JSON to /authorize; "authzen" posts to the AuthZEN /access/v1/evaluation(s)
        # endpoints. get_constraints (CAQ reverse query) is XACML-only regardless — AuthZEN
        # has no partial-evaluation endpoint.
        self.protocol = protocol
        # Observe-only hooks: every wire call emits on_pdp_request / on_pdp_response. A
        # no-op dispatcher (no hooks) short-circuits, so this is free when unused.
        self._dispatcher = dispatcher or HookDispatcher()
        # What to do with the obligations and advice a decision carries. The default fails
        # closed on unknown obligations (XACML).
        self.obligations = obligations or ObligationsAndAdviceConfig()
        if http_client is not None and (headers or auth is not None or verify is not True or cert is not None):
            raise ValueError("pass either http_client or headers/auth/verify/cert, not both")
        self._client_settings = {"headers": headers or {}, "auth": auth, "verify": verify, "cert": cert}
        self._client = http_client
        self._owns_client = http_client is None
        self._client_lock = threading.Lock()

    def __repr__(self) -> str:
        # No headers or auth: they hold credentials.
        return (
            f"PdpClient(base_url={self.base_url!r}, protocol={self.protocol!r}, "
            f"fail_open={self.fail_open}, timeout={self.timeout})"
        )

    def _http(self) -> httpx.Client:
        """The pooled HTTP client, built on first use (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=self.timeout, **self._client_settings)
        return self._client

    def close(self) -> None:
        """Close the HTTP client this ``PdpClient`` built.

        A client passed in as ``http_client`` is left open; its owner closes it.
        """
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> PdpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def healthcheck(self, *, endpoint: str | None = None) -> HealthCheckResult:
        """Probe PDP connectivity with a single ``GET {base_url}{endpoints.health}``.

        A liveness check independent of any policy: is ``base_url`` right, and can the PDP be
        reached? Never raises: every failure is reported in the result.

        Args:
            endpoint (str, optional): A health path to probe instead of ``endpoints.health``.

        Returns:
            HealthCheckResult: ``ok`` and the status, latency, or error.
        """
        url = f"{self.base_url}{endpoint or self.endpoints.health}"
        start = time.perf_counter()
        try:
            response = self._http().get(url, timeout=self.timeout)
            latency_ms = _elapsed_ms(start)
            ok = response.status_code == 200
            return HealthCheckResult(
                ok=ok, base_url=self.base_url, protocol=self.protocol,
                latency_ms=latency_ms, status_code=response.status_code,
                error=None if ok else f"PDP returned HTTP {response.status_code} at {url}",
            )
        except Exception as exc:  # noqa: BLE001
            # The health check reports every failure mode as ok=False (fail-closed);
            # narrowing to httpx errors would let an unexpected exception escape a
            # diagnostic call that must never raise.
            latency_ms = _elapsed_ms(start)
            return HealthCheckResult(
                ok=False, base_url=self.base_url, protocol=self.protocol,
                latency_ms=latency_ms, status_code=None,
                error=f"could not reach PDP at {url}: {exc!r}",
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        subject: dict,
        action: dict,
        resource: dict,
        environment: dict,
    ) -> PdpDecision:
        """Ask the PDP for one decision, as returned.

        Neither ``fail_open`` nor the obligations are applied; ``resolve`` does both. A subject
        with no usable attributes (empty, or all ``None``: no token, or one that failed
        verification) is no identity to authorize, so it never reaches the PDP.

        Args:
            subject (dict): Subject attributes, e.g. ``{"username": "alice"}``.
            action (dict): Action attributes, e.g. ``{"name": "tool_invocation"}``.
            resource (dict): Resource attributes, e.g. ``{"tool_name": "search_docs"}``.
            environment (dict): Environment attributes.

        Returns:
            PdpDecision: The decision with its obligations and advice. Its ``decision`` is None
            when the PDP was unreachable, answered something unrecognised, or was not asked
            (empty subject).

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        if not strip_none(subject):
            logger.debug("PDP request skipped: no subject attributes to authorize")
            return PdpDecision(None)
        if self.protocol == "authzen":
            request = authzen.build_request(
                subject=subject, action=action, resource=resource, environment=environment,
            )
            return authzen.parse_result(self._authorize(request, endpoint=self.endpoints.authzen_evaluation))
        request = xacml.build_request(
            subject=subject, action=action, resource=resource, environment=environment,
        )
        return xacml.parse_response(self._authorize(request))

    def resolve(
        self,
        decision: PdpDecision,
        *,
        subject: dict,
        action: dict,
        resource: dict,
        environment: dict,
        fail_open: bool | None = None,
        approval_possible: bool = False,
        approval_given: bool = False,
        run_handlers: bool = True,
    ) -> Resolution:
        """Turn a PDP decision into what the enforcement point does.

        Applies ``fail_open`` to an inconclusive decision, then this client's obligations-and-advice
        config (see ``pdp.obligations.resolve``).

        Args:
            decision (PdpDecision): The decision to resolve.
            subject (dict): The request's subject attributes (handed to obligation handlers).
            action (dict): The request's action attributes.
            resource (dict): The request's resource attributes.
            environment (dict): The request's environment attributes.
            fail_open (bool, optional): Overrides the client's default for this call.
            approval_possible (bool): True only where the call can wait for human approval.
            approval_given (bool): True when a human has already approved the call.
            run_handlers (bool): False only looks: obligation handlers are not run.

        Returns:
            Resolution: The outcome (permit, deny, or approval required) with event detail.
        """
        fail_open = self.fail_open if fail_open is None else fail_open
        return resolve_obligations(
            decision,
            permitted=_decision_to_bool(decision.decision, fail_open=fail_open),
            context=ObligationContext(subject=subject, action=action, resource=resource, environment=environment),
            config=self.obligations,
            approval_possible=approval_possible,
            approval_given=approval_given,
            run_handlers=run_handlers,
        )

    def check_permission(
        self,
        *,
        subject: dict,
        action: dict,
        resource: dict,
        environment: dict,
        fail_open: bool | None = None,
    ) -> bool:
        """Ask the PDP for one decision and resolve it to a bool.

        Permit → True, Deny → False; anything else (NotApplicable, Indeterminate, PDP unreachable)
        → ``fail_open``. A Permit whose obligations can't all be fulfilled is a Deny, and so is one
        that asks for human approval (only a tool call can wait for it). A subject with no usable
        attributes never reaches the PDP and resolves to ``fail_open``.

        Args:
            subject (dict): Subject attributes, e.g. ``{"username": "alice"}``.
            action (dict): Action attributes, e.g. ``{"name": "tool_invocation"}``.
            resource (dict): Resource attributes, e.g. ``{"tool_name": "search_docs"}``.
            environment (dict): Environment attributes.
            fail_open (bool, optional): Overrides the client's default for this call.

        Returns:
            bool: Whether the request is permitted.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        decision = self.evaluate(subject=subject, action=action, resource=resource, environment=environment)
        return self.resolve(
            decision, subject=subject, action=action, resource=resource, environment=environment,
            fail_open=fail_open,
        ).permitted

    def evaluate_batch(
        self,
        *,
        subject: dict,
        action: dict,
        resource_names: list[str],
        environment: dict,
        resource_attr_id: str,
        resource_extra_attrs: list[dict] | None = None,
    ) -> dict[str, PdpDecision]:
        """Ask the PDP for one decision per resource name, in one batched request.

        Uses the XACML Multiple Decision Profile, or an AuthZEN ``evaluations`` batch. Like
        ``evaluate``, nothing is resolved.

        Args:
            subject (dict): Subject attributes, shared by every sub-request.
            action (dict): Action attributes, shared by every sub-request.
            resource_names (list[str]): One sub-request per name.
            environment (dict): Environment attributes, shared by every sub-request.
            resource_attr_id (str): The resource attribute each name is sent as, e.g.
                ``"tool_name"``; also how decisions are matched back.
            resource_extra_attrs (list[dict], optional): A list parallel to ``resource_names``
                adding attributes to each sub-request's resource (e.g. record fields).

        Returns:
            dict[str, PdpDecision]: Each name's decision. A name the PDP gave no decision for, or
            every name when it was unreachable, maps to a decision of None.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        if self.protocol == "authzen":
            request = authzen.build_batch_request(
                subject=subject,
                action=action,
                resource_attr_id=resource_attr_id,
                resource_names=resource_names,
                environment=environment,
                resource_extra_attrs=resource_extra_attrs,
            )
            response = self._authorize(request, endpoint=self.endpoints.authzen_evaluations)
            # AuthZEN returns an ordered list with no echoed attributes → correlate by position.
            return authzen.parse_batch_response(response, resource_names)
        request = xacml.build_multi_request(
            subject=subject,
            action=action,
            resource_attr_id=resource_attr_id,
            resource_names=resource_names,
            environment=environment,
            resource_extra_attrs=resource_extra_attrs,
        )
        # XACML MultiRequest echoes the resource attribute back → match decisions by value.
        return xacml.parse_multi_response(self._authorize(request), resource_attr_id, resource_names)

    def check_permission_batch(
        self,
        *,
        subject: dict,
        action: dict,
        resource_names: list[str],
        environment: dict,
        resource_attr_id: str,
        resource_extra_attrs: list[dict] | None = None,
        fail_open: bool | None = None,
    ) -> PermittedNames:
        """Ask the PDP which resource names are permitted, in one batched request.

        Each name is decided like ``check_permission``: ``fail_open``, then its obligations. A
        subject with no usable attributes never reaches the PDP and resolves to ``fail_open``.

        Args:
            subject (dict): Subject attributes, shared by every sub-request.
            action (dict): Action attributes, e.g. ``{"name": "tool_visibility"}``; its keys become
                the action attribute ids.
            resource_names (list[str]): One sub-request per name.
            environment (dict): Environment attributes, shared by every sub-request.
            resource_attr_id (str): The resource attribute each name is sent as, e.g.
                ``"tool_name"``.
            resource_extra_attrs (list[dict], optional): A list parallel to ``resource_names``
                adding attributes to each sub-request's resource (e.g. record fields).
            fail_open (bool, optional): Overrides the client's default for this call.

        Returns:
            PermittedNames: The permitted names; ``answered`` is False when some name was resolved
            by ``fail_open`` because the PDP gave it no decision.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        fail_open = self.fail_open if fail_open is None else fail_open
        subject = strip_none(subject)
        if not resource_names:
            return PermittedNames.of((), answered=True)
        if not subject:
            return PermittedNames.of(resource_names if fail_open else (), answered=False)
        decisions = self.evaluate_batch(
            subject=subject,
            action=action,
            resource_names=resource_names,
            environment=environment,
            resource_attr_id=resource_attr_id,
            resource_extra_attrs=resource_extra_attrs,
        )
        extras = resource_extra_attrs or [None] * len(resource_names)
        permitted: set[str] = set()
        for name, extra in zip(resource_names, extras, strict=True):
            resource = {resource_attr_id: name, **(extra or {})}
            if self.resolve(
                decisions[name], subject=subject, action=action, resource=resource,
                environment=environment, fail_open=fail_open,
            ).permitted:
                permitted.add(name)
        # An Indeterminate (or no answer) resolved by fail_open must not be cached.
        answered = all(
            decisions[name].decision in (Decision.PERMIT, Decision.DENY, Decision.NOT_APPLICABLE)
            for name in resource_names
        )
        return PermittedNames.of(permitted, answered=answered)

    def get_constraints(
        self,
        *,
        provided_attributes: list[dict],
        mocked_attributes: list[dict] | None = None,
        pdp_decision_set: list[str] | None = None,
        exclude_indeterminate: bool = False,
        context_type: str = "NONE",
    ) -> dict | None:
        """Ask the PDP for the condition under which a partial request is permitted.

        A reverse query (partial evaluation, ADS's CAQ): instead of a decision, the PDP returns the
        condition on the *unbound* attributes (e.g. document metadata) as a Boolean Expression
        Tree, which RAG retrieval filtering turns into a vector-store filter. Always XACML-based.

        Args:
            provided_attributes (list[dict]): The known attributes (subject, action, environment),
                as ``{"attribute": id, "values": [...], "issuer": optional}`` entries (see
                ``flatten_to_provided_attributes``).
            mocked_attributes (list[dict], optional): Attributes the PDP should treat as given.
            pdp_decision_set (list[str], optional): The decisions to solve for; ``["PERMIT"]`` by
                default.
            exclude_indeterminate (bool): Leave Indeterminate branches out of the tree.
            context_type (str): The ADS context type of the query.

        Returns:
            dict | None: The Boolean Expression Tree, or None on a transient failure, so the
            caller can apply ``fail_open``.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        body = self._build_constraints_body(
            provided_attributes=provided_attributes,
            mocked_attributes=mocked_attributes,
            pdp_decision_set=pdp_decision_set,
            exclude_indeterminate=exclude_indeterminate,
            context_type=context_type,
        )
        raw = self._authorize(body, endpoint=self.endpoints.partial_evaluation)
        if raw is None:
            return None
        return raw.get("response", {}) if isinstance(raw, dict) else {}

    def get_simplified_constraints(
        self,
        *,
        provided_attributes: list[dict],
        mocked_attributes: list[dict] | None = None,
        pdp_decision_set: list[str] | None = None,
        exclude_indeterminate: bool = False,
        context_type: str = "NONE",
    ) -> str | None:
        """Ask the PDP for the same condition as ``get_constraints``, as display text.

        Returns ADS's own rendering (e.g. ``"department = finance AND level <= 3"``) to show next
        to the vector-store filter it translates into. It must never drive enforcement: a missing
        rendering means "nothing to display", never "permit everything".

        Args:
            provided_attributes (list[dict]): As for ``get_constraints``.
            mocked_attributes (list[dict], optional): As for ``get_constraints``.
            pdp_decision_set (list[str], optional): As for ``get_constraints``.
            exclude_indeterminate (bool): As for ``get_constraints``.
            context_type (str): As for ``get_constraints``.

        Returns:
            str | None: The rendering, or None (or an empty string) on a transient failure.

        Raises:
            PdpRequestError: The PDP rejected the request (HTTP 4xx).
        """
        body = self._build_constraints_body(
            provided_attributes=provided_attributes,
            mocked_attributes=mocked_attributes,
            pdp_decision_set=pdp_decision_set,
            exclude_indeterminate=exclude_indeterminate,
            context_type=context_type,
        )
        raw = self._authorize(body, endpoint=self.endpoints.partial_evaluation_text)
        if raw is None:
            return None
        # ADS returns a JSON envelope ``{"response": "<expr>"}`` (verified on 26.1.6); the
        # bare-string branch tolerates a plain-text response from other PDP versions.
        return raw.get("response", "") if isinstance(raw, dict) else raw

    # ── Private plumbing ───────────────────────────────────────────────────────

    def _build_constraints_body(
        self,
        *,
        provided_attributes: list[dict],
        mocked_attributes: list[dict] | None,
        pdp_decision_set: list[str] | None,
        exclude_indeterminate: bool,
        context_type: str,
    ) -> dict:
        """Shared request body for the two ``/authorization/*constraints`` reverse queries."""
        return {
            "providedAttributes": _serialize_constraint_attributes(provided_attributes),
            "mockedAttributes": _serialize_constraint_attributes(mocked_attributes),
            "pdpDecisionSet": pdp_decision_set or ["PERMIT"],
            "excludeIndeterminate": exclude_indeterminate,
            "contextType": context_type,
        }

    def _authorize(
        self,
        request: AuthorizationRequest,
        *,
        endpoint: str | None = None,
    ) -> AuthorizationDecision | None:
        """Post the request to PDP and return the parsed decision.

        A deterministic error (HTTP 3xx or 4xx, or a 2xx without a JSON body — misconfiguration)
        raises ``PdpRequestError`` so it is never masked by ``fail_open``. Transient failures
        (connection errors, timeouts, HTTP 5xx) return None so callers can apply
        ``fail_open``.
        """
        endpoint = endpoint or self.endpoints.decision
        url = f"{self.base_url}{endpoint}"
        try:   # encoded as httpx will: a value JSON can't carry (NaN, a lone surrogate) is not an
            # outage for fail_open to permit, but a request that can never be sent
            json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError) as exc:  # UnicodeEncodeError is a ValueError
            raise PdpRequestError(f"The PDP request to {url} can't be encoded as JSON: {exc}") from exc
        # Bodies are not logged: requests carry the subject's claims. Hooks
        # (on_pdp_request / on_pdp_response) see the full bodies when you opt in.
        logger.debug("PDP request → %s", url)
        self._dispatcher.pdp_request(endpoint=endpoint, url=url, protocol=self.protocol, request=request)
        start = time.perf_counter()
        try:
            response = self._http().post(url, json=request, timeout=self.timeout)
        except Exception as exc:
            latency_ms = _elapsed_ms(start)
            logger.exception("PDP request to %s failed (fail_open=%s)", url, self.fail_open)
            self._dispatcher.pdp_response(
                endpoint=endpoint, url=url, protocol=self.protocol,
                status_code=None, latency_ms=latency_ms, error=repr(exc),
            )
            return None

        latency_ms = _elapsed_ms(start)
        if 300 <= response.status_code < 500:
            # Misconfiguration (undeclared attribute, AuthZEN disabled, wrong path/creds, or a
            # redirect to a login page or another host). Fail loudly — fail_open must not mask
            # a request the PDP will always reject.
            self._fail_misconfigured(
                endpoint, url, response, latency_ms,
                f"PDP rejected the request at {url} with HTTP {response.status_code}: {response.text.strip()}",
            )
        if 200 <= response.status_code < 300 and not _is_json_body(response):
            # A 2xx that isn't a JSON decision (an HTML page from a proxy or the wrong service)
            # is a misconfiguration too, not an outage.
            self._fail_misconfigured(
                endpoint, url, response, latency_ms,
                f"PDP at {url} answered HTTP {response.status_code} without a JSON decision; "
                "check the base URL and endpoints.",
            )

        try:
            response.raise_for_status()  # 5xx → transient → fail_open
            decision = response.json()
            logger.debug("PDP response ← %s: HTTP %s in %.1f ms", url, response.status_code, latency_ms)
            self._dispatcher.pdp_response(
                endpoint=endpoint, url=url, protocol=self.protocol,
                status_code=response.status_code, latency_ms=latency_ms, decision=decision,
            )
            return decision
        except Exception as exc:
            logger.exception("PDP request to %s failed (fail_open=%s)", url, self.fail_open)
            self._dispatcher.pdp_response(
                endpoint=endpoint, url=url, protocol=self.protocol,
                status_code=response.status_code, latency_ms=latency_ms, error=repr(exc),
            )
            return None

    def _fail_misconfigured(self, endpoint: str, url: str, response: Any, latency_ms: float, message: str) -> None:
        self._dispatcher.pdp_response(
            endpoint=endpoint, url=url, protocol=self.protocol,
            status_code=response.status_code, latency_ms=latency_ms, error=response.text.strip(),
        )
        raise PdpRequestError(message)


def _is_json_body(response: Any) -> bool:
    """Whether a response body is JSON: a decision object, or a CAQ answer, which can be a string."""
    try:
        response.json()
    except ValueError:
        return False
    return True
