"""Unit tests for the ADS CAQ BET → neutral filter tree translation.

No ADS instance required — these exercise the parser against representative BET shapes.
"""
from __future__ import annotations

import pytest

from axiomatics_ai.rag.caq_to_ir import caq_to_ir


# ── BET construction helpers (mirroring ADS /authorization/constraints output) ──

def _bag_eq(attr: str, value: str) -> dict:
    """ALFA-style ``attr == value`` → anyOfAny(equal, designator, list(value))."""
    return {"apply": {"function": "anyOfAny", "arguments": [
        {"functionElement": {"name": "equal"}},
        {"attributeDesignator": {"attributeName": attr}},
        {"apply": {"function": "list", "arguments": [{"constant": {"value": value, "type": "string"}}]}},
    ]}}


def _le(attr: str, value: str) -> dict:
    return {"apply": {"function": "lessThanOrEqual", "arguments": [
        {"apply": {"function": "one", "returnType": "integer",
                   "arguments": [{"attributeDesignator": {"attributeName": attr}}]}},
        {"constant": {"value": value, "type": "integer"}},
    ]}}


def _and(*args) -> dict:
    return {"apply": {"function": "and", "arguments": list(args)}}


def _or(*args) -> dict:
    return {"apply": {"function": "or", "arguments": list(args)}}


# ── constant / empty handling ────────────────────────────────────────────────────

def test_permit_all_constant_yields_no_filter():
    result = caq_to_ir({"constant": {"value": "true"}})
    assert result.tree is None
    assert result.deny_all is False


def test_deny_all_constant():
    result = caq_to_ir({"constant": {"value": "false"}})
    assert result.tree is None
    assert result.deny_all is True


def test_none_bet_fails_closed():
    assert caq_to_ir(None).deny_all is True


def test_empty_bet_fails_closed():
    assert caq_to_ir({}).deny_all is True


# ── leaf comparisons ──────────────────────────────────────────────────────────────

def test_department_match_strips_resource_prefix():
    result = caq_to_ir(_bag_eq("resource.department", "engineering"))
    assert result.tree == {"cmp": "eq", "key": "department", "value": "engineering"}


def test_doc_prefix_also_stripped():
    result = caq_to_ir(_bag_eq("doc.department", "hr"))
    assert result.tree == {"cmp": "eq", "key": "department", "value": "hr"}


def test_integer_less_than_or_equal_keeps_native_int():
    result = caq_to_ir(_le("resource.level", "1"))
    assert result.tree == {"cmp": "le", "key": "level", "value": 1}
    assert isinstance(result.tree["value"], int)


# ── boolean composition ───────────────────────────────────────────────────────────

def test_public_or_department_and_level():
    bet = _and(
        _or(_bag_eq("resource.department", "engineering"), _bag_eq("resource.department", "public")),
        _le("resource.level", "1"),
    )
    result = caq_to_ir(bet)
    assert result.tree == {"and": [
        {"or": [{"cmp": "eq", "key": "department", "value": "engineering"},
                {"cmp": "eq", "key": "department", "value": "public"}]},
        {"cmp": "le", "key": "level", "value": 1},
    ]}


def test_membership_multiple_values_becomes_or_of_equals():
    bet = {"apply": {"function": "anyOfAny", "arguments": [
        {"functionElement": {"name": "equal"}},
        {"attributeDesignator": {"attributeName": "resource.region"}},
        {"apply": {"function": "list", "arguments": [
            {"constant": {"value": "eu", "type": "string"}},
            {"constant": {"value": "us", "type": "string"}},
        ]}},
    ]}}
    result = caq_to_ir(bet)
    assert result.tree == {"or": [
        {"cmp": "eq", "key": "region", "value": "eu"},
        {"cmp": "eq", "key": "region", "value": "us"},
    ]}


# ── structural noise ───────────────────────────────────────────────────────────────

def test_size_cardinality_guard_is_discarded():
    size_guard = {"apply": {"function": "equal", "arguments": [
        {"apply": {"function": "size", "arguments": [{"attributeDesignator": {"attributeName": "resource.department"}}]}},
        {"constant": {"value": "1", "type": "integer"}},
    ]}}
    bet = _and(size_guard, _bag_eq("resource.department", "engineering"))
    result = caq_to_ir(bet)
    assert result.tree == {"cmp": "eq", "key": "department", "value": "engineering"}


# ── untranslatable comparisons (unbound non-document operand) ────────────────────────
#
# When the reverse query is issued with an under-specified subject, ADS leaves the missing
# subject attributes unbound and they surface in the residual as designator-vs-designator
# comparisons (e.g. ``resource.department == department``, no constant to bind). These cannot
# be rendered into a metadata filter and must be flagged ``untranslatable`` — never silently
# reduced to permit-all, which would return the whole corpus.

def _bag_eq_designator(attr: str, other_attr: str) -> dict:
    """``attr == other_attr`` — both operands are designators (no constant)."""
    return {"apply": {"function": "anyOfAny", "arguments": [
        {"functionElement": {"name": "equal"}},
        {"attributeDesignator": {"attributeName": attr}},
        {"attributeDesignator": {"attributeName": other_attr}},
    ]}}


def test_designator_vs_designator_is_untranslatable():
    result = caq_to_ir(_bag_eq_designator("resource.department", "department"))
    assert result.untranslatable is True
    assert result.tree is None
    assert result.deny_all is False


def test_untranslatable_propagates_through_and():
    bet = _and(_bag_eq("resource.department", "public"),
               _bag_eq_designator("resource.department", "department"))
    assert caq_to_ir(bet).untranslatable is True


def test_untranslatable_propagates_through_or():
    bet = _or(_bag_eq("resource.department", "public"),
              _bag_eq_designator("resource.department", "department"))
    assert caq_to_ir(bet).untranslatable is True


def test_or_true_short_circuits_over_untranslatable():
    # true ∨ <unknown> = true → genuine permit-all wins; not flagged untranslatable.
    bet = _or({"constant": {"value": "true"}},
              _bag_eq_designator("resource.department", "department"))
    result = caq_to_ir(bet)
    assert result.untranslatable is False
    assert result.tree is None and result.deny_all is False


def test_and_false_short_circuits_over_untranslatable():
    # false ∧ <unknown> = false → deny wins; not flagged untranslatable.
    bet = _and({"constant": {"value": "false"}},
               _bag_eq_designator("resource.department", "department"))
    result = caq_to_ir(bet)
    assert result.untranslatable is False
    assert result.deny_all is True


def test_translatable_result_is_not_flagged_untranslatable():
    assert caq_to_ir(_bag_eq("resource.department", "engineering")).untranslatable is False


# ── attribute_map override ─────────────────────────────────────────────────────────

def test_attribute_map_overrides_key():
    result = caq_to_ir(
        _bag_eq("resource.dept_code", "ENG"),
        attribute_map={"resource.dept_code": "department"},
    )
    assert result.tree == {"cmp": "eq", "key": "department", "value": "ENG"}


# ── operand order for asymmetric comparisons ───────────────────────────────────────
# Our IR always writes the attribute on the left, so an ordering operator must be flipped
# when ADS put the constant first: ``3 <= level`` is ``level >= 3``. Getting this wrong
# inverts a retrieval filter silently — it returns documents policy meant to withhold —
# so both operand orders are pinned here for the bag and the bare-comparison paths.

def _bag_cmp_ordered(op: str, attr: str, value: str, *, attribute_first: bool) -> dict:
    designator = {"attributeDesignator": {"attributeName": attr}}
    constant = {"apply": {"function": "list",
                          "arguments": [{"constant": {"value": value, "type": "integer"}}]}}
    operands = [designator, constant] if attribute_first else [constant, designator]
    return {"apply": {"function": "anyOfAny",
                      "arguments": [{"functionElement": {"name": op}}, *operands]}}


def _direct_cmp_ordered(op: str, attr: str, value: str, *, attribute_first: bool) -> dict:
    designator = {"attributeDesignator": {"attributeName": attr}}
    constant = {"constant": {"value": value, "type": "integer"}}
    operands = [designator, constant] if attribute_first else [constant, designator]
    return {"apply": {"function": op, "arguments": operands}}


@pytest.mark.parametrize("ads_function,attribute_first,expected_op", [
    ("lessThanOrEqual",    True,  "le"),
    ("lessThanOrEqual",    False, "ge"),   # 3 <= level  ==  level >= 3
    ("greaterThanOrEqual", True,  "ge"),
    ("greaterThanOrEqual", False, "le"),
    ("lessThan",           True,  "lt"),
    ("lessThan",           False, "gt"),
    ("greaterThan",        True,  "gt"),
    ("greaterThan",        False, "lt"),
])
def test_bag_comparison_respects_operand_order(ads_function, attribute_first, expected_op):
    bet = _bag_cmp_ordered(ads_function, "resource.level", "3", attribute_first=attribute_first)
    assert caq_to_ir(bet).tree == {"cmp": expected_op, "key": "level", "value": 3}


@pytest.mark.parametrize("ads_function,attribute_first,expected_op", [
    ("lessThan",    True,  "lt"),
    ("lessThan",    False, "gt"),
    ("greaterThan", True,  "gt"),
    ("greaterThan", False, "lt"),
])
def test_direct_comparison_respects_operand_order(ads_function, attribute_first, expected_op):
    bet = _direct_cmp_ordered(ads_function, "resource.level", "3", attribute_first=attribute_first)
    assert caq_to_ir(bet).tree == {"cmp": expected_op, "key": "level", "value": 3}


@pytest.mark.parametrize("attribute_first", [True, False])
def test_equality_is_symmetric_regardless_of_operand_order(attribute_first):
    # eq/ne mean the same thing from either side — they must NOT be flipped.
    bet = _bag_cmp_ordered("equal", "resource.level", "3", attribute_first=attribute_first)
    assert caq_to_ir(bet).tree == {"cmp": "eq", "key": "level", "value": 3}


def test_ordering_operator_with_several_values_is_untranslatable():
    # ``level <= (1, 2)`` has no faithful single-comparison rendering; it must not silently
    # collapse to one of the values (which would widen or narrow the filter).
    bet = {"apply": {"function": "anyOfAny", "arguments": [
        {"functionElement": {"name": "lessThanOrEqual"}},
        {"attributeDesignator": {"attributeName": "resource.level"}},
        {"apply": {"function": "list", "arguments": [
            {"constant": {"value": "1", "type": "integer"}},
            {"constant": {"value": "2", "type": "integer"}}]}},
    ]}}
    result = caq_to_ir(bet)
    assert result.untranslatable is True
    assert result.tree is None


# ── fail closed on anything unrecognised ─────────────────────────────────────────────
#
# A shape the translator does not understand must never reduce to "no constraint": under an
# AND it would drop a condition, under an OR it would drop the whole filter.

def _apply(function: str, *args) -> dict:
    return {"apply": {"function": function, "arguments": list(args)}}


def _size(attr: str) -> dict:
    return _apply("size", {"attributeDesignator": {"attributeName": attr}})


def _int(value: str) -> dict:
    return {"constant": {"value": value, "type": "integer"}}


_UNRECOGNISED = [
    pytest.param(_apply("stringStartsWith", {"constant": {"value": "x", "type": "string"}},
                        {"attributeDesignator": {"attributeName": "resource.title"}}), id="unknown-function"),
    pytest.param(_apply("stringRegexpMatch", {"constant": {"value": "^x", "type": "string"}},
                        {"attributeDesignator": {"attributeName": "resource.title"}}), id="regex"),
    pytest.param(_apply("stringIsIn", {"constant": {"value": "x", "type": "string"}},
                        {"attributeDesignator": {"attributeName": "resource.tags"}}), id="is-in"),
    pytest.param({"foo": 1}, id="dict-without-apply"),
    pytest.param(_apply("anyOfAny", {"functionElement": {"name": "stringStartsWith"}},
                        {"attributeDesignator": {"attributeName": "resource.title"}},
                        {"constant": {"value": "x", "type": "string"}}), id="bag-with-unknown-op"),
    pytest.param(_apply("allOfAny", {"functionElement": {"name": "equal"}},
                        {"attributeDesignator": {"attributeName": "resource.tags"}},
                        _apply("list", {"constant": {"value": "a", "type": "string"}})), id="allOfAny"),
    pytest.param(_apply("allOfAll", {"functionElement": {"name": "equal"}},
                        {"attributeDesignator": {"attributeName": "resource.tags"}},
                        _apply("list", {"constant": {"value": "a", "type": "string"}})), id="allOfAll"),
    pytest.param(_apply("anyOfAll", {"functionElement": {"name": "equal"}},
                        {"attributeDesignator": {"attributeName": "resource.tags"}},
                        _apply("list", {"constant": {"value": "a", "type": "string"}})), id="anyOfAll"),
    pytest.param(_apply("greaterThan", _size("resource.tags"), _int("0")), id="size-comparison"),
    pytest.param(_apply("equal", _size("resource.tags"), _int("0")), id="bare-size-guard"),
]


@pytest.mark.parametrize("node", _UNRECOGNISED)
def test_unrecognised_node_is_untranslatable(node):
    result = caq_to_ir(node)
    assert result.untranslatable is True and result.tree is None


@pytest.mark.parametrize("node", [n for n in _UNRECOGNISED if n.id != "bare-size-guard"])
def test_unrecognised_node_poisons_an_and(node):
    # (A size guard is the one exception: dropping it from an AND only loosens a
    # single-valued check — see test_size_cardinality_guard_is_discarded.)
    assert caq_to_ir(_and(_bag_eq("resource.department", "eng"), node)).untranslatable is True


@pytest.mark.parametrize("node", _UNRECOGNISED)
def test_unrecognised_node_poisons_an_or(node):
    assert caq_to_ir(_or(_bag_eq("resource.department", "eng"), node)).untranslatable is True


def test_size_guard_inside_an_or_is_untranslatable():
    # or(eng, size(tags) == 0) means "eng, or no tags" — dropping the guard would permit all.
    guard = _apply("equal", _size("resource.tags"), _int("0"))
    assert caq_to_ir(_or(_bag_eq("resource.department", "eng"), guard)).untranslatable is True


def test_size_is_not_read_as_the_attribute():
    # ``size(tags) > 0`` is not ``tags > 0``.
    result = caq_to_ir(_apply("greaterThan", _size("resource.tags"), _int("0")))
    assert result.tree is None and result.untranslatable is True


# ── not ───────────────────────────────────────────────────────────────────────────

def _not(node: dict) -> dict:
    return _apply("not", node)


def test_not_negates_a_comparison():
    result = caq_to_ir(_not(_bag_eq("resource.classification", "secret")))
    assert result.tree == {"cmp": "ne", "key": "classification", "value": "secret"}


def test_not_is_kept_inside_an_and():
    result = caq_to_ir(_and(_bag_eq("resource.department", "eng"),
                            _not(_bag_eq("resource.classification", "secret"))))
    assert result.tree == {"and": [
        {"cmp": "eq", "key": "department", "value": "eng"},
        {"cmp": "ne", "key": "classification", "value": "secret"},
    ]}


def test_not_applies_de_morgan():
    result = caq_to_ir(_not(_or(_bag_eq("resource.department", "eng"), _le("resource.level", "2"))))
    assert result.tree == {"and": [
        {"cmp": "ne", "key": "department", "value": "eng"},
        {"cmp": "gt", "key": "level", "value": 2},
    ]}


def test_double_not_restores_the_comparison():
    result = caq_to_ir(_not(_not(_bag_eq("resource.department", "eng"))))
    assert result.tree == {"cmp": "eq", "key": "department", "value": "eng"}


def test_not_of_a_constant():
    assert caq_to_ir(_not({"constant": {"value": "true"}})).deny_all is True
    assert caq_to_ir(_not({"constant": {"value": "false"}})).tree is None


def test_not_of_something_untranslatable_stays_untranslatable():
    assert caq_to_ir(_not(_bag_eq_designator("resource.department", "department"))).untranslatable


def test_not_with_the_wrong_arity_is_untranslatable():
    node = _apply("not", _bag_eq("resource.a", "x"), _bag_eq("resource.b", "y"))
    assert caq_to_ir(node).untranslatable is True
