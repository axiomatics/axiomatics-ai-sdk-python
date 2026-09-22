"""OpenID AuthZEN Authorization API 1.0 — request builders and response parsers.

ADS maps AuthZEN into XACML server-side (see AuthZenToXacmlRequestMapper): every
``properties`` key is looked up in the domain dictionary and must be a declared attribute,
while the action ``name`` becomes a raw XACML AttributeId "name" (no lookup) — mirroring the
XACML path where action {"name": x} → AttributeId "name". Structural subject/resource
id/type are also raw and ignored unless declared. ``environment`` attributes ride in the
request-level ``context`` map (same strict-dictionary rule).

A result is ``{"decision": bool, "context": {...}}``. ADS adds the underlying XACML result
under ``context.xacml``: ``{decision, obligations|advice: [{id, attributes: {name: value}}]}``.
"""
from __future__ import annotations

from typing import Any

from axiomatics_ai.core.pdp.attributes import paired_extras, sendable_attributes
from axiomatics_ai.core.pdp.decision import Advice, Decision, Obligation, PdpDecision


# ── Request builders ──────────────────────────────────────────────────────────────

def _properties(attributes: dict | None) -> dict[str, Any]:
    """Build an AuthZEN ``properties`` map from a category dict.

    Drops ``None`` values and values an attribute can't carry (see ``sendable_attributes``).
    AuthZEN carries attributes as a flat ``properties`` map rather than XACML's ``Category`` /
    ``Attribute`` arrays. Unlike the lenient XACML path, ADS maps AuthZEN ``properties``
    strictly against the domain dictionary: every key must be a declared attribute (its ALFA
    FQN) or ADS returns HTTP 400. Structural fields (subject/resource ``id``/``type``, action
    ``name``) are exempt.
    """
    return sendable_attributes(attributes)


def _subject(subject: dict) -> dict:
    # id/type are required by the AuthZEN schema but carry no policy meaning here;
    # the authorization-relevant subject attributes live in `properties`.
    return {"type": "subject", "id": "subject", "properties": _properties(subject)}


def _action(action: dict) -> dict:
    props = _properties(action)
    name = props.pop("name", "")
    return {"name": name, "properties": props}


def _resource(resource: dict, *, resource_id: str = "resource") -> dict:
    """Build one AuthZEN ``resource``.

    ``id`` must identify the resource: PDPs treat entries of a batch that share an id as the
    same resource (some collapse them to one decision, some return them out of order), so batch
    callers pass each entry's own name as ``resource_id``.
    """
    return {"type": "resource", "id": resource_id, "properties": _properties(resource)}


def build_request(*, subject: dict, action: dict, resource: dict, environment: dict) -> dict:
    """Build a single AuthZEN evaluation (``/access/v1/evaluation``) request.

    Args:
        subject (dict): Subject attributes.
        action (dict): Action attributes.
        resource (dict): Resource attributes.
        environment (dict): Environment attributes (the AuthZEN ``context``).

    Returns:
        dict: The request body.
    """
    request: dict = {
        "subject": _subject(subject),
        "action": _action(action),
        "resource": _resource(resource),
    }
    context = _properties(environment)
    if context:
        request["context"] = context
    return request


def build_batch_request(
    *,
    subject: dict,
    action: dict,
    resource_attr_id: str,
    resource_names: list[str],
    environment: dict | None = None,
    resource_extra_attrs: list[dict] | None = None,
) -> dict:
    """Build an AuthZEN batch (``/access/v1/evaluations``) request.

    The top-level ``subject`` / ``action`` are shared defaults; each entry in ``evaluations``
    carries its own resource. Results come back in ``evaluations`` order, so callers correlate
    by position (see ``parse_batch_response``).

    Args:
        subject (dict): Subject attributes, shared by every evaluation.
        action (dict): Action attributes, shared by every evaluation.
        resource_attr_id (str): The attribute each resource name is sent as.
        resource_names (list[str]): One evaluation per name.
        environment (dict, optional): Environment attributes, shared by every evaluation.
        resource_extra_attrs (list[dict], optional): A list parallel to ``resource_names``
            adding attributes to each resource.

    Returns:
        dict: The request body.
    """
    extras = paired_extras(resource_names, resource_extra_attrs)
    evaluations = []
    for name, extra in zip(resource_names, extras, strict=True):
        resource = {resource_attr_id: name}
        resource.update(_properties(extra))
        evaluations.append({"resource": _resource(resource, resource_id=str(name))})

    request: dict = {
        "subject": _subject(subject),
        "action": _action(action),
        "evaluations": evaluations,
    }
    context = _properties(environment)
    if context:
        request["context"] = context
    return request


# ── Response parsers ──────────────────────────────────────────────────────────────

def _xacml_block(result: Any) -> dict:
    """The ``context.xacml`` block ADS adds to a result, or ``{}`` when the PDP sent none."""
    context = result.get("context") if isinstance(result, dict) else None
    xacml = context.get("xacml") if isinstance(context, dict) else None
    return xacml if isinstance(xacml, dict) else {}


def _decision(result: Any) -> Decision | None:
    """Normalise one result into a XACML ``Decision``, so both protocols share one decision rule.

    Returns None when no decision can be read. The source, in order of preference:
    ``context.xacml.decision`` (ADS echoes the underlying XACML decision there, which keeps the
    full Permit/Deny/NotApplicable/Indeterminate distinction and exact ``fail_open``
    semantics); then the plain boolean ``decision``, the only field a standards-compliant
    AuthZEN PDP must return (true → Permit, false → Deny: a bare false is a real deny, so
    ``fail_open`` won't flip it).
    """
    if not isinstance(result, dict):
        return None
    xacml = _xacml_block(result)
    if xacml.get("decision"):
        return Decision.parse(xacml["decision"])
    decision = result.get("decision")
    if decision is True:
        return Decision.PERMIT
    if decision is False:
        return Decision.DENY
    return None


def _items(entries: Any, kind: type[Obligation] | type[Advice]) -> tuple:
    """``context.xacml.obligations`` / ``advice`` → ``Obligation`` / ``Advice`` objects."""
    items = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        attributes = entry.get("attributes")
        items.append(kind(entry["id"], dict(attributes) if isinstance(attributes, dict) else {}))
    return tuple(items)


def parse_result(result: Any) -> PdpDecision:
    """Read one AuthZEN evaluation result.

    Obligations and advice are read from ADS's ``context.xacml`` block; a PDP that sends none
    gives a bare decision.

    Args:
        result (Any): The result body, or None when the PDP was unreachable.

    Returns:
        PdpDecision: The decision; no decision when there was no result.
    """
    xacml = _xacml_block(result)
    return PdpDecision(
        _decision(result),
        _items(xacml.get("obligations"), Obligation),
        _items(xacml.get("advice"), Advice),
    )


def parse_batch_response(response: Any, resource_names: list[str]) -> dict[str, PdpDecision]:
    """Read a batch (``/evaluations``) response: one decision per name, by position.

    AuthZEN returns decisions as an ordered list with no echoed attributes, so results are
    correlated to ``resource_names`` by position (the XACML path parses the echoed attribute
    instead).

    Args:
        response (Any): The response body, or None when the PDP was unreachable.
        resource_names (list[str]): The names that were asked about, in request order.

    Returns:
        dict[str, PdpDecision]: Each name's decision; none has a decision when the response
        doesn't hold exactly one evaluation per name.
    """
    evaluations = response.get("evaluations", []) if isinstance(response, dict) else []
    if not isinstance(evaluations, list) or len(evaluations) != len(resource_names):
        # Matched by position: a list of another length can't be lined up with the names
        # (a missing entry would shift every later decision), so none is read.
        evaluations = []
    return {
        name: parse_result(evaluations[i] if i < len(evaluations) else None)
        for i, name in enumerate(resource_names)
    }
