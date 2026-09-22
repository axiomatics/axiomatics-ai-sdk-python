"""Attribute values a PDP request can carry.

The rules shared by the XACML and AuthZEN request builders, the partial-evaluation query
and the context providers.
"""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def _value_kind(value: Any) -> type | None:
    """Get the PDP value kind of one scalar, or None if a PDP attribute can't carry it.

    ``bool`` is kept apart from ``int``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None   # JSON can't carry NaN or infinity
    if isinstance(value, str) and not value.isascii():
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return None   # a lone surrogate
    for kind in (bool, int, float, str):
        if isinstance(value, kind):
            return kind
    return None


def is_sendable(value: Any) -> bool:
    """Whether a PDP attribute can carry ``value``.

    Objects, nested lists and mixed-kind lists can't be expressed as one typed attribute (XACML
    needs one DataType per attribute; ADS rejects mixed lists).

    Args:
        value (Any): The value to test.

    Returns:
        bool: True for one scalar, or a list of scalars of a single kind.
    """
    if isinstance(value, list):
        kinds = {_value_kind(v) for v in value}
        return None not in kinds and len(kinds) <= 1
    return _value_kind(value) is not None


# Attribute ids already warned about in sendable_attributes, so each is reported once.
_warned_unsendable: set[str] = set()


def sendable_attributes(attributes: dict | None) -> dict:
    """Drop ``None`` values and values a PDP attribute can't carry, warning once per attribute.

    Applied by every request builder, so tool arguments, token claims, record fields and
    enricher output all get the same rule: a policy never receives a value it can't evaluate,
    and one malformed value (e.g. a model-chosen tool argument) can't make the whole request
    fail.

    Args:
        attributes (dict | None): The attributes.

    Returns:
        dict: The attributes a PDP request can carry.
    """
    sendable: dict = {}
    for key, value in (attributes or {}).items():
        if value is None:
            continue
        if not is_sendable(value):
            if key not in _warned_unsendable:
                _warned_unsendable.add(key)
                logger.warning(
                    "Attribute %r is not sent to the PDP: its value is %s, and an attribute can "
                    "only hold a scalar or a list of scalars of one type. Map a field inside it "
                    "(e.g. with a context provider) if a policy needs it.",
                    key, type(value).__name__ if not isinstance(value, list) else "a mixed or nested list",
                )
            continue
        sendable[key] = value
    return sendable


def strip_none(attributes: dict | None) -> dict:
    """Drop the entries whose value is ``None``.

    A context provider signals "omit this attribute" with ``None``. Applying the rule as soon as
    a provider's output is consumed means a subject of all-``None`` values is recognised as no
    identity, not a truthy dict that slips past ``if not subject``.

    Args:
        attributes (dict | None): The attributes.

    Returns:
        dict: The attributes without ``None`` values.
    """
    return {k: v for k, v in (attributes or {}).items() if v is not None}


def paired_extras(
    resource_names: list[str], resource_extra_attrs: list[dict] | None
) -> list[dict | None]:
    """Align per-resource extra attributes 1:1 with ``resource_names``.

    A short list would otherwise drop resources from the batch, and their missing decisions
    would read as a legitimate deny.

    Args:
        resource_names (list[str]): The resource names.
        resource_extra_attrs (list[dict] | None): Extra attributes per name, or None.

    Returns:
        list[dict | None]: One entry per name.

    Raises:
        ValueError: The two lists differ in length.
    """
    if resource_extra_attrs is None:
        return [None] * len(resource_names)
    if len(resource_extra_attrs) != len(resource_names):
        raise ValueError(
            f"resource_extra_attrs has {len(resource_extra_attrs)} entries but "
            f"resource_names has {len(resource_names)}; they must correspond 1:1"
        )
    return list(resource_extra_attrs)
