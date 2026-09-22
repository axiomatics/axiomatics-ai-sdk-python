"""Obligations and advice: what the PDP asks the PEP to do alongside its decision.

A XACML PDP can attach **obligations** (the PEP must carry them out) and **advice** (the PEP
may) to a decision. ADS returns them on the XACML path (``Obligations`` /
``AssociatedAdvice``) and on the AuthZEN path (``context.xacml.obligations`` / ``advice``);
the PDP client reads both into ``Obligation`` / ``Advice`` objects on a ``PdpDecision``.

``resolve`` turns a decision and its obligations and advice into what the PEP does:

- **Unknown obligation on Permit** → Deny by default (XACML: a PEP that can't fulfil an
  obligation must not permit). ``on_unknown="ignore"`` permits anyway, logging each id once.
- **Known obligation** → its handler runs; a handler that returns ``False`` or raises turns
  the Permit into a Deny. A ``None`` handler marks the id as handled elsewhere.
- **Advice** is advisory: unknown advice is ignored.
- **Human approval** (the obligation or advice ``require-approval``, configurable) is built
  in: Deny + advice, or Permit + obligation, means "approval required". Only a tool call can
  wait for approval; anywhere else it can't be fulfilled, so it is a Deny.

Example::

    AuthZConfig(
        obligations_and_advice=ObligationsAndAdviceConfig(
            handlers={"log-approval": audit, "notify-owner": None},
            on_unknown="deny",
            approval=ApprovalConfig(approver=my_approver),
        ),
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from axiomatics_ai.core.pdp.decision import Advice, Obligation, PdpDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObligationContext:
    """The request a decision was made for: what an obligation handler gets to act on.

    Attributes:
        subject (dict): The subject attributes sent.
        action (dict): The action attributes sent; they name the enforcement point, e.g.
            ``{"name": "tool_invocation"}``.
        resource (dict): The resource attributes sent.
        environment (dict): The environment attributes sent.
    """

    subject: dict
    action: dict
    resource: dict
    environment: dict


ObligationHandler = Callable[[Obligation, ObligationContext], "bool | None"]


@dataclass(frozen=True)
class ApprovalRequest:
    """A tool call waiting for a human's approval, as shown to the approver.

    Attributes:
        request_id (str): Identifies this request in the decision events; stable across a
            framework pause and its resume.
        tool_name (str): The tool to approve.
        args (dict): The tool call's own arguments (or, when the adapter doesn't know them, the
            resource attributes sent to the PDP).
        args_hash (str): Binds the approval to these arguments and this PDP resource: edited
            arguments need a new approval.
        reasons (tuple[str, ...]): The ``reason`` attributes of the obligations and advice that
            asked.
        ttl_seconds (float | None): How long an approval may be reused for the same caller and
            arguments: the smallest ``ttl`` of the asking rules. None means single use.
        subject (dict): The caller's subject attributes (left out of ``to_payload``).
        asked_by (tuple[Obligation | Advice, ...]): Every obligation or advice that asked for
            approval; one answer covers them all.
    """

    request_id: str
    tool_name: str
    args: dict
    args_hash: str
    reasons: tuple[str, ...]
    ttl_seconds: float | None
    subject: dict
    asked_by: tuple[Obligation | Advice, ...]

    @property
    def reason(self) -> str | None:
        """The reasons joined into one string.

        Returns:
            str | None: ``"; "``-joined reasons, or None when there are none.
        """
        return "; ".join(self.reasons) or None

    def to_payload(self) -> dict[str, Any]:
        """What a framework interrupt carries to the app.

        JSON-safe, and without the subject: it may hold token claims the app's UI has no business
        showing.

        Returns:
            dict[str, Any]: ``type``, ``request_id``, ``tool_name``, ``args``, ``reason``,
            ``reasons`` and ``ttl_seconds``.
        """
        return {
            "type": APPROVAL_PAYLOAD_TYPE,
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "args": json.loads(json.dumps(self.args, default=str)),
            "reason": self.reason,
            "reasons": list(self.reasons),
            "ttl_seconds": self.ttl_seconds,
        }


# The ``type`` of an approval-request payload, so an app handling several kinds of framework
# interrupt can tell this one apart.
APPROVAL_PAYLOAD_TYPE = "axiomatics.approval_request"


@dataclass(frozen=True)
class ApprovalDecision:
    """The answer to an approval request.

    Attributes:
        approved (bool): Whether the call is approved.
        approver (str | None): Who answered, for the audit trail (never sent to the PDP).
        note (str | None): Why, for the audit trail.
        asked (bool): False for a refusal where no human could be asked (no approver
            configured, an MCP client without elicitation). It is logged with
            ``source: "none"``, so the audit trail tells it apart from a human's rejection.
    """

    approved: bool
    approver: str | None = None
    note: str | None = None
    asked: bool = True

    @classmethod
    def from_answer(cls, answer: Any) -> ApprovalDecision:
        """Read the decision in an app's answer to a framework interrupt.

        Only an explicit ``True`` approves.

        Args:
            answer (Any): An ``ApprovalDecision``, a bool, or a dict ``{"approved": bool,
                "approver": ..., "note": ...}``. Anything else is a rejection.

        Returns:
            ApprovalDecision: The decision.
        """
        if isinstance(answer, cls):
            return answer
        if isinstance(answer, bool):
            return cls(approved=answer)
        if isinstance(answer, dict):
            approver, note = answer.get("approver"), answer.get("note")
            return cls(
                approved=answer.get("approved") is True,
                approver=str(approver) if approver is not None else None,
                note=str(note) if note is not None else None,
            )
        return cls(approved=False, note="unrecognised answer")


Approver = Callable[[ApprovalRequest], "ApprovalDecision | Awaitable[ApprovalDecision]"]


@dataclass(frozen=True)
class ApprovalConfig:
    """Human approval of tool calls.

    Attributes:
        approver (Approver | None): Called with an ``ApprovalRequest``, returns an
            ``ApprovalDecision`` (sync or async). Unset, the framework's own pause is used where
            it has one; elsewhere the call is denied.
        obligation_or_advice_ids (frozenset[str]): The ids policies use to ask for approval,
            as an obligation on a Permit or advice on a Deny.
        attribute (str): The boolean environment attribute sent as ``True`` when the call is
            decided again after approval. A policy must stop asking once it is set.
    """

    approver: Approver | None = None
    obligation_or_advice_ids: frozenset[str] = frozenset({"require-approval"})
    attribute: str = "approval_granted"

    def __post_init__(self) -> None:
        ids = self.obligation_or_advice_ids
        if isinstance(ids, str):
            # A bare string would silently become a set of its characters.
            raise ValueError("obligation_or_advice_ids must be a set of ids, not a single string")
        ids = frozenset(ids)
        if not ids or not all(isinstance(i, str) and i for i in ids):
            raise ValueError("obligation_or_advice_ids must be a non-empty set of non-empty ids")
        object.__setattr__(self, "obligation_or_advice_ids", ids)
        if not self.attribute:
            raise ValueError("ApprovalConfig.attribute must not be empty")


@dataclass(frozen=True)
class ObligationsAndAdviceConfig:
    """How the PEP handles the obligations and advice the PDP returns.

    The default fails closed on unknown obligations and recognises approval (see the module
    docstring).

    Attributes:
        handlers (dict[str, ObligationHandler | None]): Obligation id → handler. A handler
            returning ``False`` or raising turns the Permit into a Deny; ``None`` marks the id
            as handled elsewhere.
        on_unknown (str): ``"deny"`` (default): an unknown obligation on a Permit is a Deny;
            ``"ignore"`` permits anyway, logging each id once.
        approval (ApprovalConfig): Human approval of tool calls.
    """

    handlers: dict[str, ObligationHandler | None] = field(default_factory=dict)
    on_unknown: Literal["deny", "ignore"] = "deny"
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)

    def __post_init__(self) -> None:
        if self.on_unknown not in ("deny", "ignore"):
            raise ValueError(f"on_unknown must be 'deny' or 'ignore'; got {self.on_unknown!r}")
        clash = sorted(self.approval.obligation_or_advice_ids & self.handlers.keys())
        if clash:
            raise ValueError(
                f"{clash} are approval ids, handled by the SDK; remove them from handlers"
            )
        for obligation_id, handler in self.handlers.items():
            if handler is not None and not callable(handler):
                raise ValueError(f"Handler for {obligation_id!r} must be callable or None")


class Outcome(str, Enum):
    """What the PEP does with a decision.

    Attributes:
        PERMIT: Let the request through.
        DENY: Block it.
        APPROVAL_REQUIRED: Wait for a human's approval (tool calls only).
    """
    PERMIT = "permit"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class Resolution:
    """What the PEP does with a decision, and why.

    Attributes:
        outcome (Outcome): Permit, deny, or approval required.
        approval (tuple[Obligation | Advice, ...]): Every obligation or advice that asked for
            approval (with ``APPROVAL_REQUIRED``, and with ``DENY`` when approval was asked for
            where it can't be given).
        detail (dict[str, Any]): What happened to the obligations and advice, for the
            decision event.
        decision (PdpDecision | None): The decision this was resolved from, so it can be
            resolved again after approval without asking the PDP again.
    """

    outcome: Outcome
    approval: tuple[Obligation | Advice, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)
    # The decision this was resolved from, so it can be resolved again after approval
    # without asking the PDP again.
    decision: PdpDecision | None = None

    @property
    def permitted(self) -> bool:
        """Whether the outcome is a permit.

        Returns:
            bool: True for ``PERMIT``.
        """
        return self.outcome is Outcome.PERMIT


# Obligation ids already warned about under on_unknown="ignore", so each is logged once.
_warned_unknown: set[str] = set()


def resolve(
    decision: PdpDecision,
    *,
    permitted: bool,
    context: ObligationContext,
    config: ObligationsAndAdviceConfig,
    approval_possible: bool,
    approval_given: bool = False,
    run_handlers: bool = True,
) -> Resolution:
    """Apply the obligations-and-advice config to a decision.

    Args:
        decision (PdpDecision): The PDP's decision.
        permitted (bool): The base Permit/Deny reading of the decision, ``fail_open`` already
            applied.
        context (ObligationContext): The request, handed to obligation handlers.
        config (ObligationsAndAdviceConfig): How to handle obligations and advice.
        approval_possible (bool): True only where the call can wait for approval (a tool call
            not yet approved).
        approval_given (bool): A human has approved: a Permit's approval obligations are then
            fulfilled, and its other obligations are handled as usual.
        run_handlers (bool): False only looks, for a pre-check that must have no side effects;
            the outcome assumes the handlers succeed.

    Returns:
        Resolution: The outcome and its detail.
    """
    detail = _describe(decision)
    approval_ids = config.approval.obligation_or_advice_ids
    handle = _run_handlers if run_handlers else _skip_handlers
    if not permitted:
        handle(decision.obligations, context, config, approval_ids)
        asked = _find_all(decision.advice, approval_ids) + _find_all(decision.obligations, approval_ids)
        return _approval_or_deny(decision, asked, approval_possible, detail)

    unknown = [
        o.id for o in decision.obligations if o.id not in approval_ids and o.id not in config.handlers
    ]
    if unknown and config.on_unknown == "deny":
        detail["unknown_obligations"] = unknown
        return Resolution(Outcome.DENY, detail=detail, decision=decision)
    for obligation_id in set(unknown) - _warned_unknown:
        _warned_unknown.add(obligation_id)
        logger.warning("Ignoring obligation %r: no handler for it (on_unknown='ignore').", obligation_id)

    asked = _find_all(decision.obligations, approval_ids)
    if asked and not approval_given:
        return _approval_or_deny(decision, asked, approval_possible, detail)

    failed = handle(decision.obligations, context, config, approval_ids)
    if failed:
        detail["failed_obligations"] = failed
        return Resolution(Outcome.DENY, detail=detail, decision=decision)
    return Resolution(Outcome.PERMIT, detail=detail, decision=decision)


class ApprovalMemory:
    """Approvals that may be reused until their ttl expires.

    An approval whose request carried a ``ttl`` holds for the same caller and the same
    ``args_hash`` until it expires. Single-use approvals (no ``ttl``) are never stored.
    Thread-safe.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[float, dict, ApprovalDecision]] = {}
        self._lock = threading.Lock()

    def remember(self, request: ApprovalRequest, decision: ApprovalDecision) -> None:
        """Store an approval, if it is reusable.

        Only an approval (not a rejection) whose request has a ``ttl`` is stored.

        Args:
            request (ApprovalRequest): The request that was approved.
            decision (ApprovalDecision): The answer.
        """
        if request.ttl_seconds is None or not decision.approved:
            return
        key = (args_hash(request.subject), request.args_hash)
        with self._lock:
            self._entries[key] = (time.monotonic() + request.ttl_seconds, dict(request.subject), decision)

    def recall(self, request: ApprovalRequest) -> ApprovalDecision | None:
        """Find a still-valid approval for the same caller and arguments.

        Args:
            request (ApprovalRequest): The request to answer.

        Returns:
            ApprovalDecision | None: The remembered approval, or None.
        """
        key = (args_hash(request.subject), request.args_hash)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry[0] <= time.monotonic():
                del self._entries[key]
                return None
            return entry[2]

    def invalidate(self, **subject_attributes: Any) -> int:
        """Drop remembered approvals.

        Args:
            **subject_attributes (Any): Drop the approvals whose subject has all of these; all of
                them when none are given.

        Returns:
            int: How many were dropped.
        """
        with self._lock:
            doomed = [
                key for key, (_, subject, _) in self._entries.items()
                if all(subject.get(k) == v for k, v in subject_attributes.items())
            ]
            for key in doomed:
                del self._entries[key]
            return len(doomed)


class PausedCalls:
    """Tool calls paused for a human's approval in this process, with the decision that asked.

    Keyed by the caller, the framework's tool-call id and the arguments' binding, so another
    caller reusing a call id never gets someone else's pending request. Frameworks that resume
    by running the paused step again (LangGraph, Strands) check the call once more before they
    hand over the answer, and every framework checks it once more with the answer. With the
    pause remembered, neither re-asks the PDP for the decision it already gave. A pause
    expires after ``ttl_seconds`` (the call is then checked again from scratch), and
    ``invalidate`` drops a caller's pauses together with its cached decisions. Bounded (least
    recently used dropped first) and thread-safe; a call resumed in another process isn't here
    and is simply checked again.

    Args:
        maxsize (int): How many paused calls are kept.
        ttl_seconds (float): How long a pause is honoured.
    """

    def __init__(self, maxsize: int = 1024, ttl_seconds: float = 3600) -> None:
        self._entries: OrderedDict[tuple, tuple[float, ApprovalRequest, PdpDecision | None]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def put(self, key: tuple, request: ApprovalRequest, decision: PdpDecision | None) -> None:
        """Remember a paused call.

        Args:
            key (tuple): Caller, surface, call id and argument binding.
            request (ApprovalRequest): The pending approval request.
            decision (PdpDecision | None): The PDP decision that asked for approval.
        """
        with self._lock:
            self._entries[key] = (time.monotonic() + self._ttl, request, decision)
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def get(self, key: tuple) -> tuple[ApprovalRequest, PdpDecision | None] | None:
        """Look up a paused call that hasn't expired.

        Args:
            key (tuple): Caller, surface, call id and argument binding.

        Returns:
            tuple[ApprovalRequest, PdpDecision | None] | None: The request and decision, or None.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry[0] <= time.monotonic():
                del self._entries[key]
                return None
            return entry[1], entry[2]

    def invalidate(self, **subject_attributes: Any) -> int:
        """Drop paused calls, e.g. on a revocation signal.

        Args:
            **subject_attributes (Any): Drop the pauses whose caller has all of these; all of
                them when none are given.

        Returns:
            int: How many were dropped.
        """
        with self._lock:
            doomed = [
                key for key, (_, request, _) in self._entries.items()
                if all(request.subject.get(k) == v for k, v in subject_attributes.items())
            ]
            for key in doomed:
                del self._entries[key]
            return len(doomed)

    def pop(self, key: tuple) -> None:
        """Forget a paused call (it was answered).

        Args:
            key (tuple): Caller, surface, call id and argument binding.
        """
        with self._lock:
            self._entries.pop(key, None)

    def paused_otherwise(self, key: tuple, subject: dict, *, ignore: str | None = None) -> bool:
        """Whether this caller's same call (surface and call id) is paused with other arguments.

        The arguments a human was shown can't change before the answer arrives (an edited
        checkpoint or RunState): a resumed step keeps its arguments and parallel calls have their
        own ids. Another caller's pause under the same id is not a conflict (providers may number
        ids per run); which caller may resume a paused run is the application's to enforce.

        Args:
            key (tuple): Caller, surface, call id and argument binding.
            subject (dict): This call's subject.
            ignore (str, optional): A subject attribute that doesn't make another caller (the
                acting agent: one call can pause on one of the app's agents and run on another).

        Returns:
            bool: True when an answer for ``key`` would really be for arguments nobody showed.
        """
        def caller(attributes: dict) -> dict:
            return {k: v for k, v in attributes.items() if k != ignore}

        now = time.monotonic()
        with self._lock:
            return any(
                k[1:3] == key[1:3] and k[3] != key[3] and expiry > now
                and caller(request.subject) == caller(subject)
                for k, (expiry, request, _) in self._entries.items()
            )


def args_hash(args: Any) -> str:
    """Hash tool-call arguments stably, binding an approval to exactly these values.

    Args:
        args (Any): JSON-like arguments; key order doesn't matter.

    Returns:
        str: A SHA-256 hex digest.
    """
    serialized = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def ttl_seconds(asked_by: tuple[Obligation | Advice, ...]) -> float | None:
    """Work out how long an approval may be reused.

    Args:
        asked_by (tuple[Obligation | Advice, ...]): The obligations and advice that asked.

    Returns:
        float | None: The smallest positive ``ttl`` (seconds) among them, or None (single use)
        if any of them has no valid ``ttl``, so no approval outlives what one of the asking
        rules allows.
    """
    ttls = [_ttl(item) for item in asked_by]
    if not ttls or None in ttls:
        return None
    return min(ttls)


def _ttl(item: Obligation | Advice) -> float | None:
    value = item.attributes.get("ttl")
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def reasons(asked_by: tuple[Obligation | Advice, ...]) -> tuple[str, ...]:
    """Collect the ``reason`` attributes of the obligations and advice that asked for approval.

    Args:
        asked_by (tuple[Obligation | Advice, ...]): The obligations and advice that asked.

    Returns:
        tuple[str, ...]: Their reasons, in order.
    """
    found: list[str] = []
    for item in asked_by:
        value = item.attributes.get("reason")
        for reason in value if isinstance(value, list) else [value]:
            if reason is not None and str(reason) not in found:
                found.append(str(reason))
    return tuple(found)


def _approval_or_deny(
    decision: PdpDecision, asked: tuple[Obligation | Advice, ...], approval_possible: bool, detail: dict,
) -> Resolution:
    if not asked:
        return Resolution(Outcome.DENY, detail=detail, decision=decision)
    if approval_possible:
        return Resolution(Outcome.APPROVAL_REQUIRED, approval=asked, detail=detail, decision=decision)
    detail["approval"] = "can't be given here"
    return Resolution(Outcome.DENY, approval=asked, detail=detail, decision=decision)


def _run_handlers(
    obligations: tuple[Obligation, ...],
    context: ObligationContext,
    config: ObligationsAndAdviceConfig,
    approval_ids: frozenset[str],
) -> list[str]:
    """Run the handler of each known obligation; return the ids whose handler failed."""
    failed: list[str] = []
    for obligation in obligations:
        if obligation.id in approval_ids:
            continue
        handler = config.handlers.get(obligation.id)
        if handler is None:
            continue
        try:
            ok = handler(obligation, context)
        except Exception:
            # A handler is user code; whatever it raises means the obligation wasn't met.
            logger.exception("Obligation handler for %r failed", obligation.id)
            ok = False
        if ok is False:
            failed.append(obligation.id)
    return failed


def _skip_handlers(*_args: Any) -> list[str]:
    """``_run_handlers`` for a pre-check: runs nothing, reports no failures."""
    return []


def _find_all(items: tuple[Obligation, ...] | tuple[Advice, ...], ids: frozenset[str]) -> tuple:
    return tuple(item for item in items if item.id in ids)


def _describe(decision: PdpDecision) -> dict[str, Any]:
    """Describe the decision's obligations and advice for the decision event.

    Each with its id and attribute assignments (e.g. an approval's ``reason``).
    """
    detail: dict[str, Any] = {}
    if decision.obligations:
        detail["obligations"] = [{"id": o.id, "attributes": dict(o.attributes)} for o in decision.obligations]
    if decision.advice:
        detail["advice"] = [{"id": a.id, "attributes": dict(a.attributes)} for a in decision.advice]
    return detail
