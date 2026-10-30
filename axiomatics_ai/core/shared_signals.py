"""Receive Shared Signals (OpenID SSF / CAEP / RISC) and drop cached decisions they affect.

An identity provider (the transmitter) pushes a Security Event Token (SET, RFC 8417) to an HTTPS
endpoint of yours whenever something changes about a subject: a session revoked, credentials
changed, risk raised, an account disabled. Push delivery is RFC 8935: one signed SET per POST,
answered with 202, or 400 and an error code.

``SsfReceiver`` is that endpoint. It verifies the SET (signature against the transmitter's
published keys, ``iss``, ``aud``, ``typ``) and calls ``config.invalidate_decisions()`` for the
subject, so no cached decision outlives the signal (AIMS §11). The decision itself still comes
from the PDP: it must learn of the change too, for example through an attribute it looks up.

Deliberately not covered: poll delivery (RFC 8936), creating or managing the stream at the
transmitter (done once, in its console or API), and acting on events beyond the cache (a callback
gets each event for that).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from axiomatics_ai._extras import requires_extra
from axiomatics_ai.core.config import AuthZConfig

logger = logging.getLogger(__name__)

# Events that are about the stream itself, not about a subject.
_STREAM_EVENTS = frozenset({
    "https://schemas.openid.net/secevent/ssf/event-type/verification",
    "https://schemas.openid.net/secevent/ssf/event-type/stream-updated",
})

# Subject identifier formats (RFC 9493) and the member holding the identifier.
_IDENTIFIER_MEMBERS = {
    "iss_sub": "sub", "opaque": "id", "email": "email", "phone_number": "phone_number",
    "account": "uri", "uri": "uri", "did": "url",
}

_MAX_BODY_BYTES = 64 * 1024
_LEEWAY_SECONDS = 60          # tolerated transmitter clock skew on iat
_KEYS_RETRY_SECONDS = 30      # after the transmitter's keys couldn't be fetched


class SignalRejected(Exception):
    """The SET was refused; ``error`` is the RFC 8935 error code."""

    def __init__(self, error: str, description: str, status: int = 400) -> None:
        super().__init__(f"{error}: {description}")
        self.error, self.description, self.status = error, description, status


class SsfReceiver:
    """Push endpoint for Shared Signals: verifies each SET and drops the subject's cached decisions.

    Every event about a subject drops that subject's cached decisions (an extra PDP call is the
    only cost of dropping too much). The subject is matched by ``subject_attributes``: which
    subject attribute, as your context provider sends it, holds each identifier format. A subject
    the receiver can't match drops every cached decision, since it may be any of them.

    Serve it at exactly the URL registered with the transmitter, e.g. in Starlette
    ``Route("/ssf/events", receiver)`` (not ``Mount``, which redirects to a trailing slash that a
    transmitter won't follow), or call ``handle(token)`` from any web framework.
    Each process has its own cache, so with several replicas each must get the signal.

    Args:
        config (AuthZConfig): The config whose caches to invalidate.
        issuer (str): The transmitter's issuer, as configured there.
        audience (str | list[str]): This receiver's audience, as registered with the transmitter.
        jwks_url (str, optional): The transmitter's keys; otherwise read from its
            ``/.well-known/ssf-configuration``.
        authorization (str, optional): The exact ``Authorization`` header value the transmitter
            must send (set when creating the stream). Recommended: without it, anyone can make
            the receiver fetch keys and verify tokens.
        subject_attributes (dict[str, str], optional): Identifier format → subject attribute.
            Default: ``iss_sub`` and ``opaque`` → ``user_id``. Add ``email`` only if your context
            provider sends the user's email as a subject attribute.
        on_event (Callable[[str, dict, dict], None], optional): Called for each event with its
            type, the matched subject attributes and the SET's claims; e.g. to end sessions.
        algorithms (tuple[str, ...]): Accepted signing algorithms.
        http_timeout (float): Timeout for fetching the transmitter's configuration and keys.
    """

    def __init__(
        self,
        config: AuthZConfig,
        *,
        issuer: str,
        audience: str | list[str],
        jwks_url: str | None = None,
        authorization: str | None = None,
        subject_attributes: dict[str, str] | None = None,
        on_event: Callable[[str, dict, dict], None] | None = None,
        algorithms: tuple[str, ...] = ("RS256", "PS256", "ES256", "EdDSA"),
        http_timeout: float = 5.0,
    ) -> None:
        if not issuer or not audience:
            raise ValueError("SsfReceiver needs the transmitter's issuer and this receiver's audience")
        if "reason" in (subject_attributes or {}).values():
            raise ValueError("'reason' can't be a subject attribute here (invalidate_decisions takes it)")
        self._config = config
        self._issuer = issuer
        self._audience = audience
        self._jwks_url = jwks_url
        self._authorization = authorization
        self._subject_attributes = subject_attributes or {"iss_sub": "user_id", "opaque": "user_id"}
        self._on_event = on_event
        self._algorithms = list(algorithms)
        self._http_timeout = http_timeout
        self._keys: Any = None
        self._keys_failed_at = -_KEYS_RETRY_SECONDS

    def handle(self, token: str, authorization: str | None = None) -> None:
        """Verify one SET and act on its events.

        Args:
            token (str): The SET, as POSTed.
            authorization (str, optional): The request's ``Authorization`` header.

        Raises:
            SignalRejected: The SET was refused (to be answered with its status and error).
        """
        if self._authorization is not None and not hmac.compare_digest(
            (authorization or "").encode(), self._authorization.encode(),
        ):
            raise SignalRejected("authentication_failed", "missing or wrong Authorization header", 401)
        claims = self._verify(token)
        event_types = [t for t in claims["events"] if t not in _STREAM_EVENTS]
        if not event_types:
            return   # a verification or stream event: acknowledged, nothing to drop
        subjects = self._subjects(claims)
        for event_type in event_types:
            reason = f"{event_type.rsplit('/', 1)[-1]} (SET {claims['jti']})"
            for subject in subjects:
                self._config.invalidate_decisions(reason=reason, **subject)
            if self._on_event is not None:
                try:
                    self._on_event(event_type, subjects[0], claims)
                except Exception:   # the cache is already dropped; a retry would not help
                    logger.exception("Shared Signals: on_event failed for %s", event_type)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        """ASGI app: one SET per POST (RFC 8935)."""
        if scope["type"] != "http":
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        try:
            if scope["method"] != "POST":
                raise SignalRejected("invalid_request", "only POST is accepted", 405)
            if not headers.get("content-type", "").lower().startswith("application/secevent+jwt"):
                raise SignalRejected("invalid_request", "content type must be application/secevent+jwt")
            body = await _read_body(receive)
            await asyncio.to_thread(self.handle, body.decode("ascii", "replace").strip(), headers.get("authorization"))
        except SignalRejected as exc:
            logger.warning("Shared Signals: SET refused: %s (%s)", exc, exc.__cause__ or "")
            payload = json.dumps({"err": exc.error, "description": exc.description}).encode()
            await _respond(send, exc.status, payload)
            return
        await _respond(send, 202, b"")

    def _verify(self, token: str) -> dict:
        with requires_extra("jwt", "Receiving Shared Signals"):
            import jwt

        for key in self._signing_keys(token):
            try:
                verified = jwt.decode_complete(
                    token, key, algorithms=self._algorithms, audience=self._audience, issuer=self._issuer,
                    leeway=_LEEWAY_SECONDS, options={"require": ["iss", "aud", "iat", "jti", "events"]},
                )
                break
            except jwt.InvalidSignatureError:
                continue   # a token without kid is tried against each key
            except jwt.InvalidIssuerError as exc:
                raise SignalRejected("invalid_issuer", "issuer not accepted") from exc
            except jwt.InvalidAudienceError as exc:
                raise SignalRejected("invalid_audience", "audience not accepted") from exc
            except jwt.PyJWTError as exc:
                raise SignalRejected("invalid_request", "not a valid SET") from exc
        else:
            raise SignalRejected("invalid_key", "signature does not verify")
        if str(verified["header"].get("typ", "")).lower() not in ("secevent+jwt", "application/secevent+jwt"):
            raise SignalRejected("invalid_request", "token type is not secevent+jwt")
        claims = verified["payload"]
        if not isinstance(claims["events"], dict) or not claims["events"]:
            raise SignalRejected("invalid_request", "events must be a non-empty object")
        return claims

    def _signing_keys(self, token: str) -> list:
        """The key named by the token's ``kid``, or every signing key when it names none.

        After a failed fetch, keys aren't fetched again for ``_KEYS_RETRY_SECONDS``, so unverified
        requests can't make the receiver call the transmitter on each one.
        """
        import jwt

        if time.monotonic() - self._keys_failed_at < _KEYS_RETRY_SECONDS:
            raise SignalRejected("invalid_key", "transmitter keys unavailable")
        try:
            if self._keys is None:
                url = self._jwks_url or self._discover_jwks_url()
                self._keys = jwt.PyJWKClient(url, cache_keys=True, timeout=self._http_timeout)
            if jwt.get_unverified_header(token).get("kid") is not None:
                return [self._keys.get_signing_key_from_jwt(token).key]
            return [k.key for k in self._keys.get_jwk_set().keys if k.public_key_use in ("sig", None)]
        except (jwt.PyJWKClientConnectionError, httpx.HTTPError, ValueError) as exc:
            self._keys_failed_at = time.monotonic()
            raise SignalRejected("invalid_key", "transmitter keys unavailable") from exc
        except jwt.DecodeError as exc:
            raise SignalRejected("invalid_request", "not a valid SET") from exc
        except jwt.PyJWTError as exc:
            raise SignalRejected("invalid_key", "no signing key for this SET") from exc

    def _discover_jwks_url(self) -> str:
        """The ``jwks_uri`` from the transmitter's configuration (SSF §7.1)."""
        parts = urlsplit(self._issuer)
        url = f"{parts.scheme}://{parts.netloc}/.well-known/ssf-configuration{parts.path.rstrip('/')}"
        document = httpx.get(url, timeout=self._http_timeout).json()
        jwks_uri = document.get("jwks_uri") if isinstance(document, dict) else None
        if not isinstance(jwks_uri, str) or not jwks_uri.startswith("https://"):
            raise ValueError("transmitter configuration has no https jwks_uri")
        return jwks_uri

    def _subjects(self, claims: dict) -> list[dict]:
        """Subject attributes for each matched identifier; ``[{}]`` (drop everything) when none match."""
        identifiers = _identifiers(claims)
        subjects = []
        for identifier in identifiers:
            fmt = identifier.get("format")
            attribute = self._subject_attributes.get(fmt) if isinstance(fmt, str) else None
            value = identifier.get(_IDENTIFIER_MEMBERS.get(fmt, "")) if attribute else None
            if isinstance(value, str) and value:
                subjects.append({attribute: value})
        if not subjects:
            logger.warning("Shared Signals: subject %r not matched; dropping every cached decision", identifiers)
        return subjects or [{}]


def _identifiers(claims: dict) -> list[dict]:
    """The SET's subject identifiers.

    ``sub_id``, or a CAEP event's ``subject``; the ``complex`` format's user and the ``aliases``
    format's list are unwrapped.
    """
    identifier = claims.get("sub_id")
    if identifier is None:
        identifier = next((e.get("subject") for e in claims["events"].values() if isinstance(e, dict)), None)
    if isinstance(identifier, dict) and identifier.get("format") == "complex":
        identifier = identifier.get("user")
    if isinstance(identifier, dict) and identifier.get("format") == "aliases":
        candidates = identifier.get("identifiers")
        return [c for c in candidates if isinstance(c, dict)] if isinstance(candidates, list) else []
    return [identifier] if isinstance(identifier, dict) else []


async def _read_body(receive: Callable) -> bytes:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if len(body) > _MAX_BODY_BYTES:
            raise SignalRejected("invalid_request", "body too large")
        if not message.get("more_body"):
            return body


async def _respond(send: Callable, status: int, body: bytes) -> None:
    headers = [(b"content-type", b"application/json")] if body else []
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
