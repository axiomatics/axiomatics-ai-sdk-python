"""What a PDP returns for one request: the decision, plus any obligations and advice.

These are wire types, shared by the PDP client (which parses them) and ``core.obligations``
(which decides what the PEP does with them).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """The XACML decision.

    AuthZEN's boolean maps to PERMIT / DENY; ADS also echoes the full XACML decision on its
    AuthZEN responses.

    Attributes:
        PERMIT: Permit.
        DENY: Deny.
        NOT_APPLICABLE: No policy applied.
        INDETERMINATE: The PDP could not decide (e.g. an error evaluating the policy).
    """

    PERMIT = "Permit"
    DENY = "Deny"
    NOT_APPLICABLE = "NotApplicable"
    INDETERMINATE = "Indeterminate"

    @classmethod
    def parse(cls, value: Any) -> Decision | None:
        """Read a decision from a wire value.

        Case is ignored (``"deny"`` is a Deny), so another PDP's spelling never turns a Deny
        into an inconclusive decision that ``fail_open`` would permit.

        Args:
            value (Any): The wire value, e.g. ``"Permit"``.

        Returns:
            Decision | None: The decision, or None when the value isn't one.
        """
        if isinstance(value, str):
            for decision in cls:
                if decision.value.lower() == value.strip().lower():
                    return decision
        return None


@dataclass(frozen=True)
class Obligation:
    """An obligation from the PDP: something the PEP must carry out.

    Attributes:
        id (str): The obligation id.
        attributes (dict[str, Any]): Its attribute assignments, as the PDP sent them (ADS sends
            every value as a string, e.g. ``"300"``); an attribute assigned more than once holds
            a list.
    """

    id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Advice:
    """Advice from the PDP: like an ``Obligation``, but the PEP may ignore it.

    Attributes:
        id (str): The advice id.
        attributes (dict[str, Any]): Its attribute assignments, as the PDP sent them.
    """

    id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PdpDecision:
    """A PDP decision with its obligations and advice.

    Attributes:
        decision (Decision | None): The decision, or None when none could be read (PDP
            unreachable, an unrecognised value, or not asked).
        obligations (tuple[Obligation, ...]): The obligations.
        advice (tuple[Advice, ...]): The advice.
    """

    decision: Decision | None
    obligations: tuple[Obligation, ...] = ()
    advice: tuple[Advice, ...] = ()
