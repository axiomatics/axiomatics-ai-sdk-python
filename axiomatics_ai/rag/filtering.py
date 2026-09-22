"""Orchestration: turn a caller's identity into a native vector-store filter.

``compute_filter`` ties the pieces together for one retrieval:

    context ─(ContextProvider)→ subject/action/environment
            ─(flatten)→ CAQ providedAttributes
            ─(ADS /authorization/constraints)→ BET
            ─(bet_to_filter)→ neutral tree
            ─(render_native_filter)→ native filter for this store

The result is a :class:`FilterOutcome` the retrieval layer applies: inject ``native_filter``,
or short-circuit to no documents when ``deny_all`` is set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from axiomatics_ai.core.pdp import flatten_to_provided_attributes, strip_none
from axiomatics_ai.rag.caq_to_ir import caq_to_ir
from axiomatics_ai.rag.ir_to_vectorstore_filter import (
    UnsupportedVectorStoreError,
    ir_to_filter_kwargs,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterOutcome:
    """What the retrieval layer does for one search.

    Attributes:
        filter_kwargs (dict | None): The store's filter, as search keyword arguments to merge
            into the call; None (and not ``deny_all``) means no constraint.
        deny_all (bool): The subject may read nothing: return no documents.
    """

    filter_kwargs: dict | None
    deny_all: bool


def fail_safe_outcome(config: Any, reason: str) -> FilterOutcome:
    """Get the outcome when no trustworthy filter can be built.

    No documents, unless ``config.connection.fail_open`` opts into an unfiltered search (the
    same availability trade-off as an unreachable PDP). Used for a missing identity and for a
    reverse query that couldn't be fully translated, which would otherwise leak the whole
    corpus.

    Args:
        config (AuthZConfig): The config (its ``fail_open``).
        reason (str): Why, for the log.

    Returns:
        FilterOutcome: ``deny_all``, or no constraint under ``fail_open``.
    """
    if config.connection.fail_open:
        logger.warning("%s; fail_open → searching unfiltered", reason)
        return FilterOutcome(filter_kwargs=None, deny_all=False)
    logger.warning("%s; fail_closed → returning no documents", reason)
    return FilterOutcome(filter_kwargs=None, deny_all=True)


def compute_filter(
    *,
    vectorstore: Any,
    provider: Any,
    context: Any,
    config: Any,
    pdp: Any,
) -> FilterOutcome:
    """Reverse-query the PDP for the caller and render the result for the store.

    Emits one ``retrieval`` decision event (``deny_all`` / ``filtered`` / ``unfiltered``),
    grouped with the reverse query's transport events. Fails closed (unless ``fail_open``)
    whenever no trustworthy filter can be built: no identity, the PDP unreachable, an
    untranslatable condition, or a store with no filter renderer.

    Args:
        vectorstore (VectorStore): The store, which decides the filter dialect.
        provider (ContextProvider): Turns ``context`` into attributes.
        context (Any): The caller's identity.
        config (AuthZConfig): The config (``retrieval``, ``fail_open``).
        pdp (PdpClient): The PDP client.

    Returns:
        FilterOutcome: What to do for the search.
    """
    dispatcher = config.build_hook_dispatcher()
    with dispatcher.correlation():
        outcome = _compute_filter(
            vectorstore=vectorstore, provider=provider, context=context, config=config, pdp=pdp,
        )
        if dispatcher.active:
            if outcome.deny_all:
                decision = "deny_all"
            elif outcome.filter_kwargs is not None:
                decision = "filtered"
            else:
                decision = "unfiltered"
            dispatcher.decision(
                surface="rag",
                enforcement_point="retrieval",
                subject=strip_none(provider.get_subject(context)),
                action=provider.get_action(config.retrieval.action_name, context),
                resource={},
                decision=decision,
                environment=strip_none(provider.get_environment(context)),
            )
        return outcome


def _compute_filter(
    *,
    vectorstore: Any,
    provider: Any,
    context: Any,
    config: Any,
    pdp: Any,
) -> FilterOutcome:
    """Reverse-query PDP for ``context`` and render the result for ``vectorstore``.

    Fails **closed** (``deny_all``) — unless ``config.connection.fail_open`` is set — in every case
    where a trustworthy filter cannot be built: no caller identity, PDP unreachable, a reverse
    query that could not be fully translated (an under-specified subject leaks unbound subject
    attributes into the residual), or a store with no known filter renderer. Each of these
    would otherwise silently return the whole corpus.
    """
    subject = strip_none(provider.get_subject(context))
    if not subject:
        # No usable subject (empty, or all-None) → the reverse query would be unscoped and
        # come back unfiltered. Deny (or pass through under fail_open) instead — never leak
        # the whole corpus.
        return fail_safe_outcome(config, "No caller identity for retrieval")

    action = provider.get_action(config.retrieval.action_name, context)
    environment = strip_none(provider.get_environment(context))
    provided = flatten_to_provided_attributes(subject, action, environment)

    bet = pdp.get_constraints(provided_attributes=provided)
    if bet is None:
        return fail_safe_outcome(config, "PDP constraints unavailable")

    result = caq_to_ir(bet, attribute_map=config.retrieval.attribute_map)
    if result.untranslatable:
        # PDP returned a condition referencing attributes we can't bind to a document filter
        # (typically an under-specified subject, whose unbound attributes leak into the
        # residual). Reducing it would drop those terms and over-permit — fail closed instead.
        return fail_safe_outcome(config, "Reverse query could not be fully translated")
    if result.deny_all:
        return FilterOutcome(filter_kwargs=None, deny_all=True)
    if result.tree is None:
        return FilterOutcome(filter_kwargs=None, deny_all=False)

    try:
        kwargs = ir_to_filter_kwargs(result.tree, vectorstore)
    except UnsupportedVectorStoreError:
        return fail_safe_outcome(config, "Unsupported vector store for filtering")
    return FilterOutcome(filter_kwargs=kwargs, deny_all=False)
