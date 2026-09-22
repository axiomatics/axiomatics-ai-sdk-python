"""Friendly errors for a missing optional dependency: name the extra that installs it.

Wrap the import statement itself, so editors and type checkers still see it::

    with requires_extra("jwt", "Verifying tokens"):
        import jwt

A subpackage that needs an extra wraps its imports in its ``__init__``; core code that imports
an optional module lazily does the same inside the function.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator


@contextlib.contextmanager
def requires_extra(extra: str, feature: str) -> Iterator[None]:
    """Turn a missing third-party module into an ImportError that says which extra to install.

    Args:
        extra (str): The extra, e.g. ``"langchain-agents"``.
        feature (str): What needs it, e.g. ``"axiomatics_ai.langchain_agents"``.

    Raises:
        ImportError: A module the block imports is missing; a missing ``axiomatics_ai`` module
            is re-raised as it is (that is a bug, not a missing extra).
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        if name == "axiomatics_ai" or name.startswith("axiomatics_ai."):
            raise
        raise ImportError(
            f"{feature} needs the '{extra}' extra of axiomatics-ai-sdk; add it to the extras "
            f"you install (missing module: {exc.name})."
        ) from exc
