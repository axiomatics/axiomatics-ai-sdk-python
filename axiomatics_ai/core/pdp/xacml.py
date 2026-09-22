"""XACML 3.0 JSON Profile — request builders and response parsers.

A single request carries one ``Category`` entry per non-empty category. A batch is a
``MultiRequests`` request pairing the shared subject/action/environment with each resource;
each resource marks its correlation attribute ``IncludeInResult``, so every result echoes the
resource it is about. Responses carry the decision plus ``Obligations`` / ``AssociatedAdvice``
(``[{Id, AttributeAssignment: [{AttributeId, Value, DataType}]}]``).
"""
from __future__ import annotations

from typing import Any

from axiomatics_ai.core.pdp.attributes import paired_extras, sendable_attributes
from axiomatics_ai.core.pdp.decision import Advice, Decision, Obligation, PdpDecision

# XACML DataType identifiers. These are the XML Schema namespace URIs fixed by the XACML 3.0
# spec and compared as opaque strings by every PDP; nothing is ever fetched from them, and
# `https` would be a different (invalid) identifier. Sonar's clear-text-protocol hotspot
# (S5332) does not apply, so it is suppressed on these four lines.
_XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"  # NOSONAR
_XSD_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"  # NOSONAR
_XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"  # NOSONAR
_XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"  # NOSONAR

_SUBJECT_CAT = "urn:oasis:names:tc:xacml:1.0:subject-category:access-subject"
_ACTION_CAT = "urn:oasis:names:tc:xacml:3.0:attribute-category:action"
_RESOURCE_CAT = "urn:oasis:names:tc:xacml:3.0:attribute-category:resource"
_ENV_CAT = "urn:oasis:names:tc:xacml:3.0:attribute-category:environment"


# ── Request builders ──────────────────────────────────────────────────────────────

def _typed_value(value: Any) -> tuple[str, list]:
    """Infer the XACML DataType from the Python type, returning ``(datatype, value_list)``.

    Supports ``str``, ``int``, ``float``, ``bool``, and lists of those (multi-valued
    attributes). Native values are preserved (not stringified) so JSON encoding emits
    the correct type. ``bool`` is checked before ``int`` because ``bool`` is a subclass
    of ``int`` in Python.
    """
    values = value if isinstance(value, list) else [value]
    sample = values[0] if values else ""
    if isinstance(sample, bool):
        datatype = _XSD_BOOLEAN
    elif isinstance(sample, int):
        datatype = _XSD_INTEGER
    elif isinstance(sample, float):
        datatype = _XSD_DOUBLE
    else:
        datatype = _XSD_STRING
    return datatype, values


def _category(category_id: str, attributes: dict) -> dict:
    xacml_attributes = []
    for key, value in sendable_attributes(attributes).items():
        datatype, values = _typed_value(value)
        xacml_attributes.append(
            {"AttributeId": key, "DataType": datatype, "Value": values}
        )
    return {"CategoryId": category_id, "Attribute": xacml_attributes}


def _attr(attr_id: str, value: Any, *, include_in_result: bool = False) -> dict:
    """One attribute entry of a MultiRequest category, with the inferred XACML datatype."""
    datatype, values = _typed_value(value)
    return {"AttributeId": attr_id, "DataType": datatype, "Value": values, "IncludeInResult": include_in_result}


def _category_entry(entry_id: str, attributes: dict) -> dict:
    """Build a named category entry (``AccessSubject`` / ``Action`` / ``Environment``).

    For a MultiRequest; drops ``None`` values and values an attribute can't carry.
    """
    return {"Id": entry_id, "Attribute": [_attr(k, v) for k, v in sendable_attributes(attributes).items()]}


def _resource_entry(name: str, resource_attr_id: str, extra: dict | None) -> dict:
    """Build one ``Resource`` entry of a MultiRequest.

    The correlation attribute (echoed back via ``IncludeInResult``) plus any extra per-resource
    attributes.
    """
    attrs = [_attr(resource_attr_id, name, include_in_result=True)]
    attrs += [_attr(k, v) for k, v in sendable_attributes(extra).items()]
    return {"Id": name, "Attribute": attrs}


def build_request(*, subject: dict, action: dict, resource: dict, environment: dict) -> dict:
    """Build a single XACML JSON request.

    Args:
        subject (dict): Subject attributes.
        action (dict): Action attributes.
        resource (dict): Resource attributes.
        environment (dict): Environment attributes.

    Returns:
        dict: The request body.
    """
    categories = []
    if subject:
        categories.append(_category(_SUBJECT_CAT, subject))
    if action:
        categories.append(_category(_ACTION_CAT, action))
    if resource:
        categories.append(_category(_RESOURCE_CAT, resource))
    if environment:
        categories.append(_category(_ENV_CAT, environment))
    return {"Request": {"Category": categories}}


def build_multi_request(
    *,
    subject: dict,
    action: dict,
    resource_attr_id: str,
    resource_names: list[str],
    environment: dict | None = None,
    resource_extra_attrs: list[dict] | None = None,
) -> dict:
    """Build a MultiRequest (Multiple Decision Profile) pairing the subject with each resource.

    Each Resource entry marks its ``resource_attr_id`` attribute ``IncludeInResult: True`` (and
    nothing else), so decisions can be matched back by value.

    Args:
        subject (dict): Subject attributes, shared by every sub-request.
        action (dict): Action attributes, shared by every sub-request.
        resource_attr_id (str): The attribute each resource name is sent as.
        resource_names (list[str]): One sub-request per name.
        environment (dict, optional): Environment attributes, shared by every sub-request.
        resource_extra_attrs (list[dict], optional): A list parallel to ``resource_names``
            adding attributes to each resource (e.g. ``doc_type`` for record filtering).

    Returns:
        dict: The request body.
    """
    env = environment or {}
    ref_ids = ["subject", "action"] + (["env"] if env else [])
    extras = paired_extras(resource_names, resource_extra_attrs)
    request: dict = {
        "MultiRequests": {
            "RequestReference": [{"ReferenceId": [*ref_ids, name]} for name in resource_names]
        },
        "AccessSubject": [_category_entry("subject", subject)],
        "Action": [_category_entry("action", action)],
        "Resource": [
            _resource_entry(name, resource_attr_id, extra)
            for name, extra in zip(resource_names, extras, strict=True)
        ],
    }
    if env:
        request["Environment"] = [_category_entry("env", env)]
    return {"Request": request}


# ── Response parsers ──────────────────────────────────────────────────────────────

def _add_value(attributes: dict[str, Any], key: str, value: Any) -> None:
    """Add one assigned value; an attribute assigned more than once becomes a list."""
    if key not in attributes:
        attributes[key] = value
    elif isinstance(attributes[key], list):
        attributes[key].append(value)
    else:
        attributes[key] = [attributes[key], value]


def _items(entries: Any, kind: type[Obligation] | type[Advice]) -> tuple:
    """``Obligations`` / ``AssociatedAdvice`` → ``Obligation`` / ``Advice`` objects."""
    items = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("Id"):
            continue
        attributes: dict[str, Any] = {}
        assignments = entry.get("AttributeAssignment")
        for assignment in assignments if isinstance(assignments, list) else []:
            if isinstance(assignment, dict) and assignment.get("AttributeId"):
                _add_value(attributes, assignment["AttributeId"], assignment.get("Value"))
        items.append(kind(entry["Id"], attributes))
    return tuple(items)


def parse_result(result: Any) -> PdpDecision:
    """Read one XACML JSON result.

    Args:
        result (Any): One element of the response's ``Response`` list.

    Returns:
        PdpDecision: The decision, obligations and advice.
    """
    if not isinstance(result, dict):
        return PdpDecision(None)
    return PdpDecision(
        Decision.parse(result.get("Decision")),
        _items(result.get("Obligations"), Obligation),
        _items(result.get("AssociatedAdvice"), Advice),
    )


def parse_response(response: Any) -> PdpDecision:
    """Read a single-request response: its first result's decision.

    Args:
        response (Any): The response body, or None when the PDP was unreachable.

    Returns:
        PdpDecision: The decision; no decision when there was no response or no result.
    """
    results = response.get("Response", []) if isinstance(response, dict) else []
    return parse_result(results[0] if isinstance(results, list) and results else None)


def _echoed_resource_names(result: dict, attr_id: str) -> list[str]:
    """The resource names the PDP echoed back in one MultiRequest result.

    The request marks the resource attribute ``IncludeInResult``, so each sub-response carries
    the resource it decided on; this is how a decision is correlated to its resource on the
    XACML path (AuthZEN correlates by position instead). Returns a list rather than a single
    name so a result echoing the attribute in more than one category is fully accounted for.
    """
    names: list[str] = []
    # A malformed shape echoes nothing (so its name gets no decision) rather than raising.
    for category in _as_list(result.get("Category")):
        for attribute in _as_list(category.get("Attribute") if isinstance(category, dict) else None):
            value = _echoed_value(attribute, attr_id)
            if value is not None:
                names.append(value)
    return names


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _echoed_value(attribute: Any, attr_id: str) -> str | None:
    """The name one echoed attribute carries, or None if it isn't ``attr_id`` or is malformed."""
    if not isinstance(attribute, dict) or attribute.get("AttributeId") != attr_id:
        return None
    values = attribute.get("Value")
    value = values[0] if isinstance(values, list) and values else values
    return value if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None


def parse_multi_response(
    response: Any, attr_id: str, resource_names: list[str],
) -> dict[str, PdpDecision]:
    """Read a MultiRequest response: one decision per name, matched by the echoed attribute.

    Args:
        response (Any): The response body, or None when the PDP was unreachable.
        attr_id (str): The attribute each result echoes.
        resource_names (list[str]): The names that were asked about.

    Returns:
        dict[str, PdpDecision]: Each name's decision; a name no result is about has no
        decision.
    """
    decisions = {name: PdpDecision(None) for name in resource_names}
    results = response.get("Response", []) if isinstance(response, dict) else []
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        parsed = parse_result(result)
        for name in _echoed_resource_names(result, attr_id):
            if name in decisions:
                decisions[name] = parsed
    return decisions
