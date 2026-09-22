"""The decision cache: PDP decisions kept for a TTL, keyed on the full request.

The key is the subject, action, resource and environment, so any attribute change is a new
entry. An entry never outlives a verified token's expiry, and ``AuthZConfig.invalidate_decisions``
reaches every cache built from a config.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import time
from typing import Any, NamedTuple

from cachetools import TLRUCache

# When the caller's identity came from a verified credential (e.g. a JWT access token), the
# context provider records its expiry (epoch seconds) here for the current request, so a
# decision cached for that caller never outlives the credential (AIMS §11). None = unknown.
credential_expires_at: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "axiomatics_credential_expires_at", default=None
)


def _stable_hash(value: Any) -> str:
    try:
        serialized = json.dumps(value, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        # repr() is a sound cache-key fallback for anything json can't serialise; the key
        # only has to be stable and collision-resistant.
        serialized = repr(value)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class _Entry(NamedTuple):
    value: Any
    subject: dict      # kept so invalidate() can find a subject's decisions
    lifetime: float    # seconds: the TTL, capped at the caller's credential expiry


class AuthzDecisionCacheWithTTL:
    """A thread-safe in-memory cache of PDP decisions.

    Keys are derived from the full request (subject, action, resource, environment), so a
    decision is only reused for the exact same request. Only put stable, session-scoped values
    in the environment: volatile ones (timestamps) defeat the cache. A decision never outlives
    the caller's credential: when the context provider recorded one
    (``credential_expires_at``, e.g. a verified JWT's ``exp``), the entry expires at the earlier
    of that and the TTL.

    Args:
        ttl (float): How long an entry lives, in seconds; 0 disables the cache.
        maxsize (int): How many entries are kept; 0 disables the cache.
    """

    def __init__(self, ttl: float, maxsize: int):
        self._disabled = ttl <= 0 or maxsize <= 0
        self._ttl = ttl
        self._cache: TLRUCache | None = None if self._disabled else TLRUCache(
            maxsize=maxsize, ttu=lambda _key, entry, now: now + entry.lifetime,
        )
        self._lock = threading.Lock()

    def _make_key(
        self,
        subject: dict,
        action: dict,
        resource: dict,
        environment: dict,
    ) -> tuple:
        return (
            _stable_hash(subject),
            _stable_hash(action),
            _stable_hash(resource),
            _stable_hash(environment),
        )

    def get(
        self,
        subject: dict,
        action: dict,
        resource: dict,
        environment: dict,
    ) -> Any | None:
        """Look up the decision for a request.

        Args:
            subject (dict): Subject attributes.
            action (dict): Action attributes.
            resource (dict): Resource attributes.
            environment (dict): Environment attributes.

        Returns:
            Any | None: The cached decision, or None (missing, expired, or caching disabled).
        """
        if self._disabled:
            return None
        key = self._make_key(subject, action, resource, environment)
        with self._lock:
            entry = self._cache.get(key)
        return entry.value if entry is not None else None

    def set(
        self,
        subject: dict,
        action: dict,
        resource: dict,
        environment: dict,
        value: Any,
    ) -> None:
        """Cache the decision for a request.

        Args:
            subject (dict): Subject attributes.
            action (dict): Action attributes.
            resource (dict): Resource attributes.
            environment (dict): Environment attributes.
            value (Any): The decision to cache.
        """
        if self._disabled:
            return
        lifetime = self._ttl
        expires_at = credential_expires_at.get()
        if expires_at is not None:
            lifetime = min(lifetime, expires_at - time.time())
        if lifetime <= 0:
            return  # the credential has already expired: nothing to cache
        key = self._make_key(subject, action, resource, environment)
        with self._lock:
            self._cache[key] = _Entry(value, dict(subject or {}), lifetime)

    def invalidate(self, **subject_attributes: Any) -> int:
        """Drop cached decisions, e.g. on a revocation signal (AIMS §11).

        Args:
            **subject_attributes (Any): Drop the decisions whose subject has all of these, e.g.
                ``invalidate(user_id="alice")``; every decision when none are given.

        Returns:
            int: How many were dropped.
        """
        if self._disabled:
            return 0
        with self._lock:
            if not subject_attributes:
                removed = len(self._cache)
                self._cache.clear()
                return removed
            doomed = [
                key for key, entry in self._cache.items()
                if all(entry.subject.get(k) == v for k, v in subject_attributes.items())
            ]
            for key in doomed:
                del self._cache[key]
            return len(doomed)

    def clear(self) -> None:
        """Drop all cached decisions.

        Call this after a policy change so stale decisions aren't served for up to the TTL. A
        no-op when caching is disabled.
        """
        if self._disabled:
            return
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Number of cached decisions currently held (0 when disabled)."""
        if self._disabled:
            return 0
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        if self._disabled:
            return "AuthzDecisionCacheWithTTL(disabled)"
        return (
            f"AuthzDecisionCacheWithTTL(size={len(self)}, maxsize={self._cache.maxsize}, "
            f"ttl={self._ttl}s)"
        )
