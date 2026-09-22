"""Translate an ADS CAQ Boolean Expression Tree (BET) into a small, store-neutral filter tree.

ADS ``/authorization/constraints`` returns, for a partial request, the condition on
the *unbound* attributes (here the document metadata) under which the decision is
Permit. We walk that BET into a small store-neutral tree — our IR;
``ir_to_vectorstore_filter.py`` renders that IR to whichever vector store is in use (via
LangChain's per-store query translators, or a predicate for in-memory stores).

This module is the source of truth for parsing ADS CAQ output and intentionally has
**no LangChain dependency** — it emits plain dicts so other repos can reuse it.

IR nodes::

    {"cmp": <op>, "key": <k>, "value": <v>}   # leaf comparison
    {"and": [<node>, ...]}                      # conjunction
    {"or":  [<node>, ...]}                      # disjunction

``caq_to_ir`` returns a :class:`FilterResult`:

* ``tree=None, deny_all=False`` — no constraint (retrieve everything)
* ``tree=<dict>, deny_all=False`` — apply this filter
* ``tree=None, deny_all=True``  — the subject may read nothing (return no documents)
* ``untranslatable=True`` — part of the condition has no faithful rendering: a comparison
  over a document attribute that could not be reduced to a constant (e.g. an under-specified
  subject leaks unbound subject attributes into the residual), or a function or node shape
  this module does not recognise. The caller must fail closed rather than treat it as "no
  constraint" (which would over-permit).

The translation assumes single-valued document metadata: ``all*`` bag functions (which only
differ from ``any*`` on multi-valued attributes) are untranslatable, and ``not`` is rendered
by negating the comparison (``eq`` → ``ne``, ``lt`` → ``ge``, …).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

# Sentinels for boolean simplification during the walk. ``_UNKNOWN`` marks a subtree we
# could not translate — it must never silently reduce to ``_TRUE`` (permit) or the search
# would come back unfiltered. ``_SIZE_GUARD`` is a cardinality guard (``size(X) == n``): it
# can be dropped from a conjunction, where omitting it only loosens a single-valued check,
# but anywhere else it is untranslatable.
_TRUE = object()
_FALSE = object()
_UNKNOWN = object()
_SIZE_GUARD = object()

# PDP comparison function → neutral op.
_CMP = {
    "equal": "eq",
    "notEqual": "ne",
    "lessThanOrEqual": "le",
    "greaterThanOrEqual": "ge",
    "lessThan": "lt",
    "greaterThan": "gt",
}

# Higher-order bag comparison functions. XACML compares attribute *bags*, so an ALFA
# ``resource.department == "x"`` compiles to ``anyOfAny(equal, resource.department,
# list("x"))`` rather than a bare ``equal``. Hand-written XACML may instead use
# ``anyOf``; we accept both, pulling the comparison op from the embedded functionElement.
# The all* variants are not listed, so they fall through to untranslatable: rendering
# them as "any" would over-permit on multi-valued metadata.
_BAG_CMP = {"anyOf", "anyOfAny"}

# One-and-only functions that unwrap a single-valued bag to its value (``one(X)``).
_ONE_AND_ONLY = "one"

# Negation of each comparison, used to render ``not(...)``. It is exact for single-valued
# metadata that is present; for a missing attribute the negated filter excludes the
# document (the in-memory predicate treats every comparison on a missing key as false), which
# is stricter than XACML and so fails closed.
_NEGATED_CMP = {"eq": "ne", "ne": "eq", "lt": "ge", "ge": "lt", "le": "gt", "gt": "le"}

# Default attribute-name prefixes stripped to reach the bare metadata key.
_DEFAULT_PREFIXES = ("doc.", "resource.")


@dataclass(frozen=True)
class FilterResult:
    """The outcome of translating a BET.

    Attributes:
        tree (dict | None): The neutral filter tree (see the module docstring), or None.
        deny_all (bool): The subject may read nothing.
        untranslatable (bool): Part of the condition has no faithful rendering; the caller must
            fail closed.
    """

    tree: dict | None
    deny_all: bool
    untranslatable: bool = False


def _default_resolve_key(attribute_name: str) -> str:
    """``resource.department`` → ``department`` (metadata key)."""
    for prefix in _DEFAULT_PREFIXES:
        if attribute_name.startswith(prefix):
            return attribute_name[len(prefix):]
    return attribute_name


def _make_resolve_key(attribute_map: dict[str, str] | None) -> Callable[[str], str]:
    if not attribute_map:
        return _default_resolve_key

    def resolve(attribute_name: str) -> str:
        if attribute_name in attribute_map:
            return attribute_map[attribute_name]
        return _default_resolve_key(attribute_name)

    return resolve


def _typed(constant: dict) -> Any:
    value, kind = constant.get("value"), constant.get("type")
    if kind == "integer":
        return int(value)
    if kind == "double":
        return float(value)
    if kind == "boolean":
        return value == "true"
    return value


def _designator_name(node: Any) -> str | None:
    """Attribute name from a designator, unwrapping a one-and-only (``one``) apply.

    Any other wrapping function (``size``, string functions, …) changes the value being
    compared, so it is not treated as the attribute itself.
    """
    if not isinstance(node, dict):
        return None
    if "attributeDesignator" in node:
        return node["attributeDesignator"].get("attributeName")
    apply_node = node.get("apply", {})
    if apply_node.get("function") != _ONE_AND_ONLY:
        return None
    inner = apply_node.get("arguments", [])
    if inner and isinstance(inner[0], dict) and "attributeDesignator" in inner[0]:
        return inner[0]["attributeDesignator"].get("attributeName")
    return None


def _constants_in(args: list) -> list:
    """All constant values among ``args``, unwrapping a ``list(...)`` apply."""
    out = []
    for a in args:
        if not isinstance(a, dict):
            continue
        if "constant" in a:
            out.append(_typed(a["constant"]))
        elif a.get("apply", {}).get("function") == "list":
            for c in a["apply"].get("arguments", []):
                if isinstance(c, dict) and "constant" in c:
                    out.append(_typed(c["constant"]))
    return out


def _is_size_guard(node: dict) -> bool:
    """True for ``equal(size(X), n)`` — a cardinality guard we discard."""
    ap = node.get("apply") if isinstance(node, dict) else None
    if not ap or ap.get("function") != "equal":
        return False
    for arg in ap.get("arguments", []):
        inner = arg.get("apply") if isinstance(arg, dict) else None
        if inner and inner.get("function") == "size":
            return True
    return False


# Asymmetric comparisons read differently depending on which side the attribute is on. Our IR
# always puts the attribute on the left (``{"cmp": "le", "key": k, "value": v}`` means
# ``k <= v``), so when PDP wrote the constant first the operator must be flipped to preserve
# meaning: ``3 <= level`` is ``level >= 3``, not ``level <= 3``.
_INVERTED_CMP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}


def _operator_with_attribute_on_left(op: str, *, attribute_first: bool) -> str:
    """Rewrite ``op`` so it reads correctly with the attribute on the left-hand side.

    ``eq``/``ne`` are symmetric and pass through unchanged; the ordering operators flip when
    PDP put the constant first.
    """
    return op if attribute_first else _INVERTED_CMP.get(op, op)


def _build_comparison_node(op: str, key: str, values: list):
    """IR node for a resolved comparison, or ``_UNKNOWN`` when it has no faithful rendering."""
    if len(values) == 1:
        return {"cmp": op, "key": key, "value": values[0]}
    if op == "eq":
        # Membership ``attr in (a, b, ...)`` → OR of equals. We deliberately avoid an ``in``
        # operator: not every store's translator supports it (e.g. Chroma rejects IN),
        # whereas eq+or is universal.
        return {"or": [{"cmp": "eq", "key": key, "value": v} for v in values]}
    # Several values under an ordering operator cannot be expressed as one comparison.
    return _UNKNOWN


def _walk_bag_comparison(args: list, resolve_key: Callable[[str], str]):
    """Translate ``anyOf``/``anyOfAny``/… — the operator is an embedded ``functionElement``.

    XACML compares attribute *bags*, so ALFA's ``resource.department == "x"`` arrives as
    ``anyOfAny(equal, resource.department, list("x"))`` rather than a bare ``equal``.
    """
    function_name = next(
        (a["functionElement"].get("name") for a in args
         if isinstance(a, dict) and "functionElement" in a),
        None,
    )
    op = _CMP.get(function_name)
    operands = [a for a in args if not (isinstance(a, dict) and "functionElement" in a)]
    attribute_name = next((_designator_name(a) for a in operands if _designator_name(a)), None)
    values = _constants_in(operands)

    if not (op and attribute_name and values):
        # An unknown comparison function, or a comparison we could not reduce to attribute vs
        # constant (e.g. the other operand is itself an unbound designator from an
        # under-specified subject). Cannot be translated; must not silently become _TRUE.
        return _UNKNOWN

    attribute_first = bool(operands) and _designator_name(operands[0]) is not None
    return _build_comparison_node(
        _operator_with_attribute_on_left(op, attribute_first=attribute_first),
        resolve_key(attribute_name),
        values,
    )


def _walk_direct_comparison(
    function_name: str, node: dict, args: list, resolve_key: Callable[[str], str]
):
    """Translate a bare ``equal``/``lessThan``/… comparing a designator against a constant."""
    if function_name == "equal" and _is_size_guard(node):
        return _SIZE_GUARD
    attribute_index = next((i for i, a in enumerate(args) if _designator_name(a)), None)
    constant_index = next(
        (i for i, a in enumerate(args) if isinstance(a, dict) and "constant" in a), None
    )
    if attribute_index is None or constant_index is None:
        # No plain attribute operand (e.g. ``size(X) > 0``), or a designator compared to a
        # non-constant → untranslatable.
        return _UNKNOWN
    return {
        "cmp": _operator_with_attribute_on_left(
            _CMP[function_name], attribute_first=attribute_index < constant_index
        ),
        "key": resolve_key(_designator_name(args[attribute_index])),
        "value": _typed(args[constant_index]["constant"]),
    }


def _walk(node: Any, resolve_key: Callable[[str], str]):
    """Dispatch one BET node; each leaf shape is handled by a ``_walk_*`` helper above.

    Only the shapes handled here are translated. Anything else (an unknown function, a node
    without ``apply``, a non-dict) is ``_UNKNOWN``, so the caller fails closed.
    """
    if not isinstance(node, dict):
        return _UNKNOWN

    if "constant" in node:
        return _TRUE if node["constant"].get("value") == "true" else _FALSE

    apply_node = node.get("apply")
    if not apply_node:
        return _UNKNOWN
    function_name = apply_node.get("function")
    args = apply_node.get("arguments", [])

    if function_name in ("and", "or"):
        return _combine([_walk(a, resolve_key) for a in args], op=function_name)
    if function_name == "not":
        if len(args) != 1:
            return _UNKNOWN
        return _negate(_walk(args[0], resolve_key))
    if function_name in _BAG_CMP:
        return _walk_bag_comparison(args, resolve_key)
    if function_name in _CMP:
        return _walk_direct_comparison(function_name, node, args, resolve_key)
    return _UNKNOWN


def _negate(result):
    """``not`` of a walk result: swap the constants, De Morgan over the IR tree."""
    if result is _TRUE:
        return _FALSE
    if result is _FALSE:
        return _TRUE
    if result is _UNKNOWN or result is _SIZE_GUARD:
        return _UNKNOWN
    if "and" in result:
        return {"or": [_negate(k) for k in result["and"]]}
    if "or" in result:
        return {"and": [_negate(k) for k in result["or"]]}
    return {**result, "cmp": _NEGATED_CMP[result["cmp"]]}


def _resolve_size_guards(results: list, *, op: str) -> list:
    """A size guard is TRUE (dropped) in a conjunction and UNKNOWN in a disjunction."""
    replacement = _TRUE if op == "and" else _UNKNOWN
    return [replacement if r is _SIZE_GUARD else r for r in results]


def _combine(results: list, *, op: str):
    """Fold child results under ``and`` / ``or`` with three-valued logic.

    The two operators are duals: ``and`` is absorbed by FALSE and has TRUE as its identity,
    ``or`` the other way round. An UNKNOWN child (untranslatable) can never be dropped, so it
    poisons the whole node unless the absorbing constant is present. A size guard is dropped
    from a conjunction and is UNKNOWN under a disjunction.
    """
    absorbing, identity = (_FALSE, _TRUE) if op == "and" else (_TRUE, _FALSE)
    results = _resolve_size_guards(results, op=op)
    if any(r is absorbing for r in results):
        return absorbing  # F ∧ X = F and T ∨ X = T, even if some X is unknown
    if any(r is _UNKNOWN for r in results):
        return _UNKNOWN  # an untranslatable child can't be dropped
    kids = [r for r in results if r is not _TRUE and r is not _FALSE]
    if not kids:
        return identity
    return kids[0] if len(kids) == 1 else {op: kids}


def caq_to_ir(
    bet: dict | None,
    *,
    attribute_map: dict[str, str] | None = None,
) -> FilterResult:
    """Translate an ADS CAQ Boolean Expression Tree into the store-neutral filter tree.

    Anything the translator doesn't recognise (an unknown function, the ``all*`` bag functions,
    an unusual node shape) is untranslatable, never "no constraint".

    Args:
        bet (dict | None): The ``/authorization/constraints`` response. A falsy value is a CAQ
            failure and fails closed (``deny_all``); apply ``fail_open`` before calling this if
            you want otherwise.
        attribute_map (dict[str, str], optional): ADS attribute name (e.g. ``resource.dept``) →
            metadata key; others fall back to stripping the ``doc.`` / ``resource.`` prefix.

    Returns:
        FilterResult: No constraint (``tree=None``), a filter tree, ``deny_all``, or
        ``untranslatable``.
    """
    if not bet:
        return FilterResult(tree=None, deny_all=True)
    resolve_key = _make_resolve_key(attribute_map)
    r = _walk(bet, resolve_key)
    if r is _TRUE:
        return FilterResult(tree=None, deny_all=False)
    if r is _FALSE:
        return FilterResult(tree=None, deny_all=True)
    if r is _UNKNOWN or r is _SIZE_GUARD:
        return FilterResult(tree=None, deny_all=False, untranslatable=True)
    return FilterResult(tree=r, deny_all=False)
