"""Wrap a vector store so its searches are filtered by an external PDP (CAQ reverse query).

``to_authorized_vectorstore(vector_store, config)`` returns the *same* store with its
search methods (``similarity_search`` and friends) replaced — on this instance only — by
wrappers that reverse-query PDP for a metadata filter and push it into each search.
Nothing global is patched; only stores you explicitly wrap are governed::

    vector_store = to_authorized_vectorstore(vector_store, config)

    # retrieval code is unchanged — including inside agent tools and as_retriever():
    docs = vector_store.similarity_search(query, k=4)

Caller identity is read at search time from, in order:

1. an explicit ``with authz_context(user): ...`` block — for static (non-agent) pipelines
   using ``as_retriever()`` or bare searches. It is backed by a ``contextvar``, so it is
   task-local and safe when one shared store serves many concurrent users; then
2. the ambient LangGraph agent runtime (``get_runtime().context`` — the ``context=``
   passed to ``agent.invoke``), so deepagents and ``create_agent`` RAG need no extra wiring.

When neither yields an identity there is no subject to authorize, so the search **fails
closed** (returns no documents) — leaving it unfiltered would return the whole corpus. This,
like an PDP error or a reverse query that cannot be translated, is governed by
``AuthZConfig.fail_open`` (set it to opt into an unfiltered pass-through instead).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
from typing import Any

from axiomatics_ai.rag.filtering import compute_filter, fail_safe_outcome

logger = logging.getLogger(__name__)

# Explicit per-request identity for non-agent flows. A contextvar keeps it task-local, so a
# shared wrapped store serving concurrent users never leaks one user's identity into another
# request. Takes precedence over the ambient agent runtime.
_explicit_context: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "axio_authz_context", default=None
)


@contextlib.contextmanager
def authz_context(context: Any):
    """Bind the caller's identity for the retrievals inside this block (non-agent pipelines).

    Backed by a contextvar, so it is task-local and safe when one shared store serves many
    concurrent users. Example::

        store = to_authorized_vectorstore(store, config)   # wrap once, shared
        with authz_context(current_user):
            docs = store.as_retriever().invoke(query)

    Args:
        context (Any): What your ``ContextProvider`` expects, e.g. a dict or object with
            ``username`` / ``department`` for the default provider.

    Yields:
        None: While the identity is bound.
    """
    token = _explicit_context.set(context)
    try:
        yield
    finally:
        _explicit_context.reset(token)


def ambient_identity_bound() -> bool:
    """Whether an identity is currently bound with ``authz_context``.

    An identity bridge (e.g. the FastMCP middleware) uses this to avoid re-binding when an outer
    scope already established one.

    Returns:
        bool: True when an identity is bound.
    """
    return _explicit_context.get() is not None


def _search_method_names(vectorstore: Any) -> list[str]:
    """Every public method on the store's class with ``search`` in its name.

    Discovered rather than listed, so store-specific variants (``*_by_vector``, Chroma's
    ``similarity_search_by_image``, …) are filtered too instead of silently bypassing the
    filter. Each returns a list (of Documents or (Document, score) tuples), so a deny
    short-circuits to [].
    """
    return [
        name for name in dir(type(vectorstore))
        if "search" in name and not name.startswith("_")
        and inspect.isfunction(inspect.getattr_static(type(vectorstore), name, None))
    ]

# Re-entrancy guard: once a filter is injected for an outer search, nested wrapped calls
# (a store's similarity_search delegating to similarity_search_with_score) pass straight
# through instead of re-querying PDP. A contextvar (not a global) keeps this task-local so
# a concurrent search is never accidentally left unfiltered.
_bypass: contextvars.ContextVar[bool] = contextvars.ContextVar("axio_authz_bypass", default=False)


def to_authorized_vectorstore(vectorstore: Any, config=None):
    """Wrap a vector store's search methods so every search is filtered by the PDP.

    Every public method with ``search`` in its name is wrapped, on this instance only (nothing
    global is patched): the ``*_by_vector`` variants and store-specific ones included. Each
    search reverse-queries the PDP for the condition under which the caller may read, and
    passes it to the store as a metadata filter. The caller's identity comes from an
    ``authz_context`` block, else the LangGraph runtime context; with none, the search returns
    no documents (unless ``fail_open``). Changes the store in place, so existing references are
    governed too.

    Args:
        vectorstore (VectorStore): The LangChain vector store.
        config (AuthZConfig): Must set ``retrieval`` (a ``RetrievalConfig``); its
            ``context_provider`` turns the identity into attributes.

    Returns:
        VectorStore: ``vectorstore``, with its searches wrapped.

    Raises:
        ValueError: ``config`` is missing or doesn't set ``retrieval``, so nothing would be
            filtered.
    """
    from axiomatics_ai.core.config import DefaultClientContextProvider

    if config is None or config.retrieval is None:
        raise ValueError(
            "to_authorized_vectorstore needs AuthZConfig(retrieval=RetrievalConfig(...)); "
            "without it the store's searches would not be filtered."
        )
    provider = config.context_provider or DefaultClientContextProvider()
    pdp = config.build_pdp_client()

    for name in _search_method_names(vectorstore):
        original = getattr(vectorstore, name, None)  # bound method on this instance
        if not callable(original):
            continue
        wrapped = _wrap(original, vectorstore, provider, pdp, config)
        _set_instance_attr(vectorstore, name, wrapped)

    return vectorstore


# ── internals ─────────────────────────────────────────────────────────────────────

def _set_instance_attr(obj: Any, name: str, value: Any) -> None:
    """Shadow a class method with a per-instance attribute, tolerating frozen models."""
    try:
        object.__setattr__(obj, name, value)
    except Exception:  # noqa: BLE001
        # Frozen/slotted models reject __setattr__ in varied ways across pydantic versions;
        # the plain setattr path handles the rest.
        setattr(obj, name, value)


def _resolve_context() -> Any:
    """Caller identity: explicit authz_context first, else the ambient agent runtime."""
    explicit = _explicit_context.get()
    if explicit is not None:
        return explicit
    try:
        from langgraph.runtime import get_runtime

        return getattr(get_runtime(), "context", None)
    except Exception:  # noqa: BLE001
        # No LangGraph installed, or no runtime bound outside an agent turn. Both mean
        # "no ambient identity", which callers handle as unauthenticated.
        return None


def _outcome_for(vectorstore, provider, pdp, config):
    """Filter outcome for this search."""
    context = _resolve_context()
    if context is None:
        # No caller identity resolved. Leaving the search unfiltered would return the whole
        # corpus, so fail closed (or pass through only under fail_open).
        return fail_safe_outcome(config, "No caller identity for retrieval")
    return compute_filter(
        vectorstore=vectorstore, provider=provider, context=context, config=config, pdp=pdp
    )


def _merge_filter_kwargs(call_kwargs: dict, filter_kwargs: dict, vectorstore: Any) -> dict:
    """Merge the store's filter kwargs into the caller's search kwargs.

    Keys the caller didn't set are added as-is. For a key both sides set: compose two
    in-memory predicates with AND; otherwise the authorization filter wins (a caller
    filter in the same dialect can't be safely combined) and we warn.
    """
    merged = dict(call_kwargs)
    for key, value in filter_kwargs.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = value
        elif callable(existing) and callable(value):  # in-memory predicates → AND
            merged[key] = lambda doc, e=existing, v=value: e(doc) and v(doc)
        else:
            logger.warning(
                "An explicit '%s' was passed to %s alongside the authorization filter; the "
                "authorization filter takes precedence and the explicit one is ignored.",
                key, type(vectorstore).__name__,
            )
            merged[key] = value
    return merged


# How a single search should run, decided before the store is touched.
_RUN_UNCHANGED = "run_unchanged"   # no filtering applies — call the store as the caller asked
_RETURN_NOTHING = "return_nothing"  # policy permits no documents — never reach the store
_RUN_FILTERED = "run_filtered"     # call the store with the authorization filter applied


def _plan_search(vectorstore, provider, pdp, config, kwargs: dict):
    """Decide how one search should run, without performing it.

    Returns ``(plan, kwargs)`` — the kwargs carry the merged authorization filter when the
    plan is ``_RUN_FILTERED``. Shared by the sync and async wrappers below so the two cannot
    drift apart; a divergence between them would mean one call path silently skips filtering.
    """
    if _bypass.get():
        return _RUN_UNCHANGED, kwargs
    outcome = _outcome_for(vectorstore, provider, pdp, config)
    if outcome.deny_all:
        return _RETURN_NOTHING, kwargs
    if outcome.filter_kwargs is not None:
        kwargs = _merge_filter_kwargs(kwargs, outcome.filter_kwargs, vectorstore)
    return _RUN_FILTERED, kwargs


def _wrap(original, vectorstore, provider, pdp, config):
    """Wrap a bound search method. ``original`` is already bound, so wrappers take no self."""
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def awrapper(*args, **kwargs):
            # The reverse query blocks; run it on a worker thread (which keeps the contextvars,
            # so the caller's identity and the bypass flag) instead of stalling the loop.
            plan, kwargs = await asyncio.to_thread(_plan_search, vectorstore, provider, pdp, config, kwargs)
            if plan == _RUN_UNCHANGED:
                return await original(*args, **kwargs)
            if plan == _RETURN_NOTHING:
                return []
            # The bypass flag stops a nested search (the store calling itself) from being
            # filtered a second time.
            token = _bypass.set(True)
            try:
                return await original(*args, **kwargs)
            finally:
                _bypass.reset(token)

        return awrapper

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        plan, kwargs = _plan_search(vectorstore, provider, pdp, config, kwargs)
        if plan == _RUN_UNCHANGED:
            return original(*args, **kwargs)
        if plan == _RETURN_NOTHING:
            return []
        token = _bypass.set(True)
        try:
            return original(*args, **kwargs)
        finally:
            _bypass.reset(token)

    return wrapper
