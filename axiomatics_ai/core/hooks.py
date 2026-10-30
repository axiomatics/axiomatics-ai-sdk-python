"""Observe-only hooks for SDK enforcement.

Users attach ``AuthzHook`` instances via ``AuthZConfig.hooks`` to react to what the
SDK is doing — logging, metrics, tracing, audit. Hooks NEVER change a decision; they
only observe. All mutation stays in ``ContextProvider`` (inputs) and the
post-invocation strategies / guards (outputs).

Two levels of events:

- **Transport** — emitted by ``PdpClient`` for every PDP wire call
  (``on_pdp_request`` / ``on_pdp_response``). Path-independent: every surface uses the
  one client, so these fire everywhere for free. Use them for PDP latency/error
  dashboards.
- **Decision** — emitted by the enforcement engines (``on_decision``): the *semantic*
  outcome at an enforcement point ("tool X denied for subject Y"), tagged with the
  ``surface`` it came from and a ``cached`` flag. A cache hit emits a decision with **no**
  transport event; one batched transport call maps to a single summary decision. Use
  these for audit trails.

Events emitted within a ``dispatcher.correlation()`` block share a ``correlation_id`` so
a single turn (visibility + invocation + post-invocation) can be stitched together.

A hook that raises is isolated and logged — it can never break enforcement. Hooks run
inline on the request path, so keep them cheap.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from inspect import iscoroutinefunction
from typing import Any

from axiomatics_ai._extras import requires_extra

logger = logging.getLogger(__name__)

# Set within HookDispatcher.correlation(); read when building every event so related
# events can be grouped. Module-level so it is shared across dispatcher instances (the
# PDP client and each engine build their own dispatcher over the same hooks list).
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "axiomatics_correlation_id", default=None
)


@dataclass(frozen=True)
class PdpRequestEvent:
    """One PDP wire request is about to be sent.

    Attributes:
        endpoint (str): The request path, e.g. ``"/authorize"``.
        url (str): The full URL.
        protocol (str): ``"xacml"`` or ``"authzen"``.
        request (dict): The request body. It carries the subject's attributes: treat it as
            sensitive.
        correlation_id (str | None): Shared by the events of one enforcement step.
    """

    endpoint: str
    url: str
    protocol: str
    request: dict
    correlation_id: str | None = None


@dataclass(frozen=True)
class PdpResponseEvent:
    """One PDP wire request has returned, or failed.

    Attributes:
        endpoint (str): The request path.
        url (str): The full URL.
        protocol (str): ``"xacml"`` or ``"authzen"``.
        status_code (int | None): The HTTP status, or None when the PDP could not be reached.
        latency_ms (float): How long the request took.
        decision (Any): The parsed response body, on success.
        error (str | None): What went wrong: a transient failure, or a 4xx's body.
        correlation_id (str | None): Shared by the events of one enforcement step.
    """

    endpoint: str
    url: str
    protocol: str
    status_code: int | None
    latency_ms: float
    decision: Any = None
    error: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True)
class DecisionEvent:
    """A semantic enforcement outcome at one enforcement point.

    ``decision`` is a coarse label: ``"permit"``, ``"deny"``, ``"partial"`` (batch
    visibility), ``"filtered"`` / ``"passthrough"`` / ``"skipped"`` (post-invocation). Human
    approval of a tool call adds ``"approval_required"``, then ``"approved"`` / ``"rejected"``
    (the answer), then the call's final ``"permit"`` / ``"deny"``.

    Attributes:
        surface (str): The adapter that decided, e.g. ``"langchain_agents"``.
        enforcement_point (str): The point, e.g. ``"tool_invocation"``.
        subject (dict): The subject attributes sent.
        action (dict): The action attributes sent.
        resource (Any): The resource attributes sent (for a batch, the names decided).
        decision (str): The outcome label (see above).
        cached (bool): True when the outcome came from the decision cache, without a PDP call.
        detail (dict): Point-specific extras: permitted/total counts, the obligations and advice
            the decision carried (each ``{"id": ..., "attributes": {...}}``), an approval's
            ``request_id`` / ``approver`` / ``note`` / ``source``.
        correlation_id (str | None): Shared by the events of one enforcement step (one tool
            call, across a pause and its resume).
        environment (dict): The environment attributes sent (a transaction id, risk or posture
            attributes, the classified intent).
        timestamp (float): When the decision was made (epoch seconds).
    """

    surface: str
    enforcement_point: str
    subject: dict
    action: dict
    resource: Any
    decision: str
    cached: bool = False
    detail: dict = field(default_factory=dict)
    correlation_id: str | None = None
    environment: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class InvalidationEvent:
    """Cached decisions were dropped, e.g. on a revocation or risk signal.

    A remediation, in AIMS §11 terms.

    Attributes:
        subject (dict): The attributes that selected the dropped decisions (empty: all).
        removed (int): How many were dropped.
        reason (str): The caller's explanation.
        correlation_id (str | None): The active correlation id, if any.
        timestamp (float): When it happened (epoch seconds).
    """

    subject: dict
    removed: int
    reason: str = ""
    correlation_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class EnrichmentEvent:
    """One LLM attribute enricher ran at an enforcement point.

    The enricher never decides: this is the classification that feeds the PDP decision.

    Attributes:
        point (str): The enforcement point, e.g. ``"before_model"``.
        enricher (str): The enricher's name (``EnricherSpec.name``).
        content (str): The text it classified. It may hold user data: treat it as sensitive.
        attributes (dict): The attributes it produced for the PDP.
        available (bool): Whether it classified successfully.
        latency_ms (float): How long the classification took.
        error (str | None): What went wrong, when not available.
        correlation_id (str | None): Shared by the events of one enforcement step.
    """

    point: str            # EnforcementPoint value
    enricher: str         # EnricherSpec.name
    content: str          # the input the enricher inspected
    attributes: dict      # the attributes it produced for PDP
    available: bool
    latency_ms: float
    error: str | None = None
    correlation_id: str | None = None


class AuthzHook:
    """Base class for observe-only enforcement hooks.

    Override only the methods you care about; the defaults do nothing. Hooks never change a
    decision. Don't raise (the dispatcher isolates exceptions, but a raising hook still does no
    work) and keep each method cheap: they run synchronously on the enforcement path.
    """

    def on_pdp_request(self, event: PdpRequestEvent) -> None:
        """Called before every PDP wire request.

        Args:
            event (PdpRequestEvent): The request about to be sent.
        """

    def on_pdp_response(self, event: PdpResponseEvent) -> None:
        """Called after every PDP wire request, on success or failure.

        Args:
            event (PdpResponseEvent): The response, or the error.
        """

    def on_decision(self, event: DecisionEvent) -> None:
        """Called for every enforcement outcome, cached or not.

        Args:
            event (DecisionEvent): The outcome.
        """

    def on_enrichment(self, event: EnrichmentEvent) -> None:
        """Called after each LLM attribute enricher runs.

        Args:
            event (EnrichmentEvent): The classification and its attributes.
        """

    def on_invalidation(self, event: InvalidationEvent) -> None:
        """Called when cached decisions are dropped.

        Args:
            event (InvalidationEvent): Which decisions, and why.
        """


class LoggingHook(AuthzHook):
    """Built-in hook that logs every event: a ready-made answer to "log all PDP calls".

    Logs the URL, status, decision label and correlation id, never a request body.

    Args:
        logger_ (logging.Logger, optional): Where to log; ``axiomatics_ai.hooks`` by default.
        level (int): The log level for every record.
    """

    def __init__(self, logger_: logging.Logger | None = None, level: int = logging.INFO) -> None:
        self._log = logger_ or logging.getLogger("axiomatics_ai.hooks")
        self._level = level

    def on_pdp_request(self, event: PdpRequestEvent) -> None:
        """Log the request's URL, protocol and correlation id.

        Args:
            event (PdpRequestEvent): The request.
        """
        self._log.log(self._level, f"PDP request → {event.url} [{event.protocol}] cid={event.correlation_id}")

    def on_pdp_response(self, event: PdpResponseEvent) -> None:
        """Log the response's status and latency, or its error.

        Args:
            event (PdpResponseEvent): The response.
        """
        if event.error is not None:
            self._log.log(
                self._level,
                f"PDP response ← {event.url} failed in {event.latency_ms:.1f}ms: {event.error} cid={event.correlation_id}",
            )
        else:
            self._log.log(
                self._level,
                f"PDP response ← {event.url} status={event.status_code} in {event.latency_ms:.1f}ms cid={event.correlation_id}",
            )

    def on_decision(self, event: DecisionEvent) -> None:
        """Log the decision: surface, point, outcome, cached flag, subject and detail.

        Args:
            event (DecisionEvent): The decision.
        """
        self._log.log(
            self._level,
            f"Decision [{event.surface}/{event.enforcement_point}] {event.decision} "
            f"cached={event.cached} subject={event.subject} detail={event.detail} cid={event.correlation_id}",
        )

    def on_enrichment(self, event: EnrichmentEvent) -> None:
        """Log the enricher's outcome and the attributes it produced.

        Args:
            event (EnrichmentEvent): The classification.
        """
        preview = event.content if len(event.content) <= 200 else event.content[:200] + "…"
        self._log.log(
            self._level,
            f"Enrichment [{event.point}/{event.enricher}] available={event.available} "
            f"attributes={event.attributes} content={preview!r} in {event.latency_ms:.1f}ms "
            f"cid={event.correlation_id}",
        )

    def on_invalidation(self, event: InvalidationEvent) -> None:
        """Log how many cached decisions were dropped, and why.

        Args:
            event (InvalidationEvent): The invalidation.
        """
        self._log.log(
            self._level,
            f"Cache invalidation subject={event.subject or 'ALL'} removed={event.removed} "
            f"reason={event.reason!r}",
        )


class AuditLogHook(AuthzHook):
    """Built-in audit trail: one structured JSON record per decision and per cache invalidation.

    Each record has the fields AIMS §11 asks for: agent, delegated user, current actor, tool or
    resource, action, decision, cached flag, timestamp, correlation id, and the environment
    attributes (transaction id, risk or posture, intent). Durability belongs to the sink.

    With ``hash_chain`` each record carries ``chain_id``, ``seq``, ``previous_hash`` and
    ``hash``, so an edited, removed or reordered record is detectable with
    :func:`verify_audit_trail`. With a ``signing_key`` each record is also signed, so whoever can
    rewrite the stored trail still can't recompute it.

    Args:
        sink (logging.Logger | Callable[[dict], None], optional): Receives each record: a
            logger (default ``axiomatics_ai.audit``; the record is one JSON line) or a callable
            taking the record dict.
        hash_chain (bool): Chain the records (see above).
        signing_key (Any, optional): A private key (PEM or a ``cryptography`` key object) that
            signs each chained record. Needs the ``jwt`` extra.
        signing_algorithm (str): The JWS algorithm: ``ES256`` (ECDSA P-256, the default),
            ``EdDSA`` (Ed25519), ``PS256`` or the other asymmetric ones.
        signing_key_id (str, optional): The ``kid`` put in each signature's header.
        agent_attribute (str): The subject attribute lifted to the record's ``agent``.
        user_attribute (str): The subject attribute lifted to the record's ``user``.
        actor_attribute (str): The subject attribute lifted to the record's ``actor``.
        agent_name_attribute (str): The subject attribute lifted to the record's
            ``agent_name``: which agent in the app made the call (the main agent or a
            sub-agent), as sent by ``AuthZConfig.attribute_names.acting_agent``.
    """

    def __init__(
        self,
        sink: logging.Logger | Callable[[dict], None] | None = None,
        *,
        hash_chain: bool = False,
        signing_key: Any = None,
        signing_algorithm: str = "ES256",
        signing_key_id: str | None = None,
        agent_attribute: str = "agent_id",
        user_attribute: str = "user_id",
        actor_attribute: str = "actor_id",
        agent_name_attribute: str = "agent_name",
    ) -> None:
        self._sink = sink or logging.getLogger("axiomatics_ai.audit")
        self._hash_chain = hash_chain
        self._names = {
            "agent_id": agent_attribute, "user_id": user_attribute, "actor_id": actor_attribute,
            "agent_name": agent_name_attribute,
        }
        if signing_key is not None and not hash_chain:
            raise ValueError("AuditLogHook signs chained records only: set hash_chain=True")
        if signing_algorithm not in _SIGNING_ALGORITHMS:
            raise ValueError(f"signing_algorithm must be one of {sorted(_SIGNING_ALGORITHMS)}")
        self._signing = (signing_key, signing_algorithm, {"kid": signing_key_id} if signing_key_id else None)
        if signing_key is not None:
            self._sign({"probe": True})  # a wrong key or algorithm fails here, not on the first decision
        self._pid = -1   # the chain starts on the first record, in the process that writes it
        self._lock = threading.Lock()

    def on_decision(self, event: DecisionEvent) -> None:
        """Write one audit record for the decision.

        Args:
            event (DecisionEvent): The decision.
        """
        subject = event.subject or {}
        self._emit({
            "event": "decision",
            "timestamp": _iso(event.timestamp),
            "correlation_id": event.correlation_id,
            **{field_: subject.get(name) for field_, name in self._names.items()},
            "surface": event.surface,
            "enforcement_point": event.enforcement_point,
            "action": (event.action or {}).get("name"),
            "resource": event.resource,
            "decision": event.decision,
            "cached": event.cached,
            "subject": subject,
            "environment": event.environment,
            "detail": event.detail,
        })

    def on_invalidation(self, event: InvalidationEvent) -> None:
        """Write one audit record for the invalidation (a remediation).

        Args:
            event (InvalidationEvent): The invalidation.
        """
        self._emit({
            "event": "remediation",
            "timestamp": _iso(event.timestamp),
            "correlation_id": event.correlation_id,
            "remediation": "cache_invalidation",
            "subject": event.subject,
            "removed": event.removed,
            "reason": event.reason,
        })

    def _emit(self, record: dict) -> None:
        with self._lock:
            if self._hash_chain:
                # The record as it will be stored (JSON values only), so the hash is recomputable.
                record = json.loads(json.dumps(record, default=str))
                if self._pid != os.getpid():   # first record, or a forked worker: a chain of its own
                    self._pid, self._chain_id, self._seq, self._previous_hash = os.getpid(), str(uuid.uuid4()), 0, ""
                seq = self._seq + 1
                record.update(chain_id=self._chain_id, seq=seq, previous_hash=self._previous_hash)
                record["hash"] = _record_hash(record)
                if self._signing[0] is not None:
                    record["signature"] = self._sign({"chain_id": self._chain_id, "seq": seq, "hash": record["hash"]})
                self._seq, self._previous_hash = seq, record["hash"]
            if isinstance(self._sink, logging.Logger):
                self._sink.info(json.dumps(record, default=str))
            else:
                self._sink(record)


    def _sign(self, payload: dict) -> str:
        with requires_extra("jwt", "Signing audit records"):
            import jwt

        key, algorithm, headers = self._signing
        return jwt.encode(payload, key, algorithm=algorithm, headers=headers)


_SIGNING_ALGORITHMS = frozenset({"ES256", "ES384", "ES512", "EdDSA", "PS256", "PS384", "PS512", "RS256"})


def _record_hash(record: dict) -> str:
    """SHA-256 over the record without ``hash`` and ``signature``: JSON, sorted keys, no spaces, ASCII."""
    body = {k: v for k, v in record.items() if k not in ("hash", "signature")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")   # parsers disagree on which value wins
    return dict(pairs)


def verify_audit_trail(records: Iterable[dict | str], public_key: Any = None) -> list[str]:
    """Check a stored ``AuditLogHook`` trail written with ``hash_chain=True``.

    Records of several chains (one per process or worker) may be interleaved. Per chain, the
    ``seq`` numbers must run 1, 2, 3…, each ``previous_hash`` must be the hash of the record
    before, and each ``hash`` must match the record's content. With ``public_key``, every
    record's ``signature`` must verify and name the same chain, ``seq`` and hash. Records cut off
    the end of a chain, or a whole chain removed, can't be detected from the trail alone: keep
    each chain's id and latest hash elsewhere.

    Args:
        records (Iterable[dict | str]): The records, in the order they were written: the stored
            JSON lines (preferred: a line with a duplicate key is refused, since other parsers
            may read another value) or parsed dicts.
        public_key (Any, optional): The public key matching the hook's ``signing_key``.

    Returns:
        list[str]: What is wrong, one line per problem; empty when the trail is intact.
    """
    problems: list[str] = []
    last: dict[str, tuple[int, str]] = {}   # chain_id -> (seq, hash) of its latest record
    for index, line in enumerate(records):
        record = _parsed_record(line, index, problems)
        if record is None:
            continue
        chain, seq, claimed = record.get("chain_id"), record.get("seq"), record.get("hash")
        where = f"record {index} (chain {chain}, seq {seq})"
        if not isinstance(chain, str) or not isinstance(seq, int) or not isinstance(claimed, str):
            problems.append(f"{where}: not a chained record")
            continue
        problems.extend(f"{where}: {p}" for p in _chain_problems(record, last.get(chain, (0, "")), public_key))
        last[chain] = (seq, claimed)
    return problems


def _chain_problems(record: dict, previous: tuple[int, str], public_key: Any) -> list[str]:
    """What is wrong with one chained record, given its chain's previous (seq, hash)."""
    problems = []
    if record["seq"] != previous[0] + 1:
        problems.append(f"expected seq {previous[0] + 1}; records are missing or out of order")
    if record.get("previous_hash") != previous[1]:
        problems.append("previous_hash is not the hash of the record before")
    if _record_hash(record) != record["hash"]:
        problems.append("content does not match its hash")
    if public_key is not None:
        problems.extend(_signature_problems(record, public_key))
    return problems


def _parsed_record(line: dict | str, index: int, problems: list[str]) -> dict | None:
    """The record as a dict, or None (with the problem noted) when it isn't one."""
    if isinstance(line, str):
        try:
            line = json.loads(line, object_pairs_hook=_no_duplicate_keys)
        except ValueError as exc:
            problems.append(f"record {index}: not a valid JSON record ({exc})")
            return None
    if not isinstance(line, dict):
        problems.append(f"record {index}: not a JSON object")
        return None
    return line


def _signature_problems(record: dict, public_key: Any) -> list[str]:
    with requires_extra("jwt", "Verifying audit records"):
        import jwt

    try:
        signed = jwt.decode(record.get("signature") or "", public_key, algorithms=sorted(_SIGNING_ALGORITHMS))
    except jwt.PyJWTError as exc:
        return [f"signature does not verify ({exc})"]
    if signed != {"chain_id": record["chain_id"], "seq": record["seq"], "hash": record["hash"]}:
        return ["signature is for another record"]
    return []


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


class HookDispatcher:
    """Fans events out to the configured hooks, isolating their exceptions.

    Cheap when no hooks are configured: every emit and the ``correlation()`` block
    short-circuit, so the enforcement path pays nothing extra.

    Args:
        hooks (list[AuthzHook], optional): The hooks to notify, in order.
    """

    def __init__(self, hooks: list[AuthzHook] | None = None) -> None:
        self._hooks: list[AuthzHook] = list(hooks or [])

    @property
    def active(self) -> bool:
        """Whether any hook is configured.

        Returns:
            bool: True when events are delivered to at least one hook.
        """
        return bool(self._hooks)

    @contextlib.contextmanager
    def correlation(self, correlation_id: str | None = None):
        """Group the events emitted in the block under one correlation id.

        Reentrant: if a scope is already active and no explicit id is given, the existing id is
        reused, so an engine can wrap several core helpers in one outer scope while a standalone
        call still gets its own id. Contextvars propagate across ``await`` and are copied into
        ``asyncio.to_thread`` workers, so a block spanning both shares one id. A no-op when no
        hooks are configured.

        Args:
            correlation_id (str, optional): An explicit id, e.g. one derived from a tool-call id;
                a fresh random id otherwise.

        Yields:
            str | None: The active correlation id (None without hooks).
        """
        if not self._hooks:
            yield None
            return
        existing = _correlation_id.get()
        if existing is not None and correlation_id is None:
            yield existing
            return
        token = _correlation_id.set(correlation_id or uuid.uuid4().hex)
        try:
            yield _correlation_id.get()
        finally:
            _correlation_id.reset(token)

    def _dispatch(self, method: str, event: Any) -> None:
        for hook in self._hooks:
            try:
                getattr(hook, method)(event)
            except Exception:
                logger.exception(f"AuthzHook {type(hook).__name__}.{method} raised; ignoring")

    def pdp_request(self, *, endpoint: str, url: str, protocol: str, request: dict) -> None:
        """Emit a ``PdpRequestEvent``.

        Args:
            endpoint (str): The request path.
            url (str): The full URL.
            protocol (str): ``"xacml"`` or ``"authzen"``.
            request (dict): The request body.
        """
        if not self._hooks:
            return
        self._dispatch("on_pdp_request", PdpRequestEvent(
            endpoint=endpoint, url=url, protocol=protocol, request=request,
            correlation_id=_correlation_id.get(),
        ))

    def pdp_response(
        self, *, endpoint: str, url: str, protocol: str, status_code: int | None,
        latency_ms: float, decision: Any = None, error: str | None = None,
    ) -> None:
        """Emit a ``PdpResponseEvent``.

        Args:
            endpoint (str): The request path.
            url (str): The full URL.
            protocol (str): ``"xacml"`` or ``"authzen"``.
            status_code (int | None): The HTTP status, or None when unreachable.
            latency_ms (float): How long the request took.
            decision (Any): The parsed response body, on success.
            error (str, optional): What went wrong.
        """
        if not self._hooks:
            return
        self._dispatch("on_pdp_response", PdpResponseEvent(
            endpoint=endpoint, url=url, protocol=protocol, status_code=status_code,
            latency_ms=latency_ms, decision=decision, error=error,
            correlation_id=_correlation_id.get(),
        ))

    def decision(
        self, *, surface: str, enforcement_point: str, subject: dict, action: dict,
        resource: Any, decision: str, cached: bool = False, detail: dict | None = None,
        environment: dict | None = None,
    ) -> None:
        """Emit a ``DecisionEvent``.

        Args:
            surface (str): The adapter that decided.
            enforcement_point (str): The point.
            subject (dict): The subject attributes sent.
            action (dict): The action attributes sent.
            resource (Any): The resource attributes sent.
            decision (str): The outcome label.
            cached (bool): Whether the outcome came from the cache.
            detail (dict, optional): Point-specific extras.
            environment (dict, optional): The environment attributes sent.
        """
        if not self._hooks:
            return
        self._dispatch("on_decision", DecisionEvent(
            surface=surface, enforcement_point=enforcement_point, subject=subject,
            action=action, resource=resource, decision=decision, cached=cached,
            detail=detail or {}, correlation_id=_correlation_id.get(),
            environment=environment or {},
        ))

    def invalidation(self, *, subject: dict, removed: int, reason: str = "") -> None:
        """Emit an ``InvalidationEvent``.

        Args:
            subject (dict): The attributes that selected the dropped decisions.
            removed (int): How many were dropped.
            reason (str): The caller's explanation.
        """
        if not self._hooks:
            return
        self._dispatch("on_invalidation", InvalidationEvent(
            subject=subject, removed=removed, reason=reason, correlation_id=_correlation_id.get(),
        ))

    def enrichment(
        self, *, point: str, enricher: str, content: str, attributes: dict,
        available: bool, latency_ms: float, error: str | None = None,
    ) -> None:
        """Emit an ``EnrichmentEvent``.

        Args:
            point (str): The enforcement point.
            enricher (str): The enricher's name.
            content (str): The text it classified.
            attributes (dict): The attributes it produced.
            available (bool): Whether it classified successfully.
            latency_ms (float): How long it took.
            error (str, optional): What went wrong.
        """
        if not self._hooks:
            return
        self._dispatch("on_enrichment", EnrichmentEvent(
            point=point, enricher=enricher, content=content, attributes=attributes,
            available=available, latency_ms=latency_ms, error=error,
            correlation_id=_correlation_id.get(),
        ))


def correlate_events(method):
    """Decorate a method so every event it emits shares one fresh correlation id.

    Expects the owning instance to expose ``self._hook_dispatcher``. Works on sync and async
    methods; on async the scope is entered before any ``asyncio.to_thread`` inside, so the id
    is copied into the worker context. No overhead when no hooks are configured.

    Args:
        method (Callable): The method to wrap.

    Returns:
        Callable: The wrapped method.
    """
    if iscoroutinefunction(method):
        @functools.wraps(method)
        async def _async(self, *args, **kwargs):
            with self._hook_dispatcher.correlation():
                return await method(self, *args, **kwargs)
        return _async

    @functools.wraps(method)
    def _sync(self, *args, **kwargs):
        with self._hook_dispatcher.correlation():
            return method(self, *args, **kwargs)
    return _sync
