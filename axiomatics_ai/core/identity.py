"""Agent identity from verified OAuth access tokens.

For the server-side paths (FastMCP, the tool decorator, the A2A server, the gateway), and for
client paths where the agent RECEIVES the user's token, e.g. behind your own API
(``token_from=``), or its platform already verified it (``claims_from=``). A token the agent
obtained to call another API is not for the agent to read (RFC 9068 §6); this is only for
tokens addressed to it.

``AgentIdentityContextProvider`` verifies the caller's bearer token as a resource server must
(RFC 9068 §4: signature against the issuer's published keys, ``iss``, ``aud``, ``exp``, the
``at+jwt`` type, an asymmetric-algorithm allowlist) and turns its claims into named PDP
attributes instead of forwarding every raw claim:

- ``user_id``: ``sub``, only when a user is present (with Client Credentials ``sub`` is the
  client itself, RFC 9068 §2.2 / OAuth 2.1 §7.4)
- ``agent_id``: ``client_id``, the agent (AIMS §10.3)
- ``actor_id``: the current actor, the outermost ``act.sub`` (RFC 8693 §4.1; earlier actors
  are informational only and are logged, never sent)
- ``scopes``, ``acr``, ``amr``, ``auth_time``, and ``transaction_id`` (``txn``) in the
  environment

Extra claims are mapped explicitly by dotted path (``claims={"realm_access.roles": "roles"}``),
and an OAuth Rich Authorization Request grant (RFC 9396 ``authorization_details``) can be
mapped with ``RarMapping``. Only mapped values reach the PDP, which keeps the request valid
(no objects) and usable over AuthZEN (only declared attributes). ``preview()`` shows what a
claim set turns into, without verifying or calling the PDP.

Verification needs the ``pyjwt[crypto]`` extra; it is imported only when a token is verified.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Literal

import httpx
from cachetools import TTLCache

from axiomatics_ai._extras import requires_extra

from axiomatics_ai.core.cache import credential_expires_at
from axiomatics_ai.core.client_metadata import ClientIdMetadata
from axiomatics_ai.core.config import DefaultServerContextProvider
from axiomatics_ai.core.pdp.attributes import is_sendable

logger = logging.getLogger(__name__)

AttributeType = Literal["string", "integer", "double", "boolean"]

# Asymmetric JWS algorithms only: a resource server verifies with the issuer's PUBLIC keys, so
# a shared-secret (HS*) or unsigned ("none") token is never acceptable (RFC 9068 §4).
_ASYMMETRIC_ALGORITHMS = frozenset({
    "RS256", "RS384", "RS512", "PS256", "PS384", "PS512",
    "ES256", "ES384", "ES512", "EdDSA",
})
_ACCESS_TOKEN_TYPES = frozenset({"at+jwt", "application/at+jwt"})
# How long a failed issuer discovery, and a rejected token, are remembered before retrying.
_DISCOVERY_RETRY_SECONDS = 30
_REJECTED_TOKEN_SECONDS = 30


class TokenVerificationError(Exception):
    """The bearer token failed verification; the message says why (never the token)."""


@dataclass
class JwtVerification:
    """How to verify a JWT access token (RFC 9068 §4).

    ``issuer`` and ``audience`` are what THIS deployment trusts, never taken from the token:
    whoever sends a token controls its contents. The signing keys come from ``jwks_url``, or by
    default from the issuer's discovery document (``/.well-known/openid-configuration``, then
    ``/.well-known/oauth-authorization-server``).

    Attributes:
        issuer (str | list[str]): The identity provider's issuer URL (a list if several are
            trusted).
        audience (str | list[str]): This resource server's identifier as registered there (a
            list if it is known by several names).
        jwks_url (str | None): The signing keys' URL, instead of discovering it.
        algorithms (tuple[str, ...]): The accepted signing algorithms (asymmetric only).
        leeway_seconds (float): Tolerated clock skew on ``exp`` / ``nbf``.
        require_typ (bool): Accept only access tokens (``typ`` ``at+jwt``), so an ID token can't
            be passed off as one. Set False for an identity provider that types access tokens
            plain ``JWT``.
        http_timeout (float): Timeout for discovery and key fetches, in seconds.
    """

    issuer: str | list[str]
    audience: str | list[str]
    jwks_url: str | None = None
    algorithms: tuple[str, ...] = ("RS256", "PS256", "ES256", "EdDSA")
    leeway_seconds: float = 30
    require_typ: bool = True
    http_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.issuer or not self.audience:
            raise ValueError("JwtVerification needs a trusted issuer and this server's audience")
        unsafe = [a for a in self.algorithms if a not in _ASYMMETRIC_ALGORITHMS]
        if unsafe or not self.algorithms:
            raise ValueError(
                f"JwtVerification accepts only asymmetric algorithms {sorted(_ASYMMETRIC_ALGORITHMS)}; "
                f"got {list(self.algorithms)}"
            )

    @property
    def issuers(self) -> list[str]:
        """The trusted issuers, as a list.

        Returns:
            list[str]: ``issuer`` normalised to a list.
        """
        return [self.issuer] if isinstance(self.issuer, str) else list(self.issuer)

    @property
    def audiences(self) -> list[str]:
        """The accepted audiences, as a list.

        Returns:
            list[str]: ``audience`` normalised to a list.
        """
        return [self.audience] if isinstance(self.audience, str) else list(self.audience)

    def with_audience(self, audience: str | list[str]) -> JwtVerification:
        """Copy this verification for another resource server, e.g. one gateway route.

        Args:
            audience (str | list[str]): The other server's audience.

        Returns:
            JwtVerification: The copy, with only the audience changed.
        """
        return JwtVerification(
            issuer=self.issuer, audience=audience, jwks_url=self.jwks_url,
            algorithms=self.algorithms, leeway_seconds=self.leeway_seconds,
            require_typ=self.require_typ, http_timeout=self.http_timeout,
        )


class _KeySource:
    """Signing keys per trusted issuer.

    A ``PyJWKClient`` (which caches keys) per JWKS URL, the URL coming from ``jwks_url`` or the
    issuer's discovery document.
    """

    def __init__(self, verification: JwtVerification) -> None:
        self._verification = verification
        self._clients: dict[str, Any] = {}
        self._discovery_failed_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def signing_key(self, token: str, issuer: str) -> Any:
        return self._client(issuer).get_signing_key_from_jwt(token).key

    def _client(self, issuer: str) -> Any:
        with requires_extra("jwt", "Verifying tokens"):
            import jwt

        with self._lock:
            if issuer not in self._clients:
                url = self._verification.jwks_url or self._discover_jwks_url(issuer)
                self._clients[issuer] = jwt.PyJWKClient(
                    url, cache_keys=True, timeout=self._verification.http_timeout,
                )
            return self._clients[issuer]

    def _discover_jwks_url(self, issuer: str) -> str:
        """Find the issuer's ``jwks_uri`` in its discovery document.

        A failure is remembered for ``_DISCOVERY_RETRY_SECONDS``, so an unreachable identity
        provider doesn't add two HTTP timeouts to every request meanwhile.
        """
        failed_at = self._discovery_failed_at.get(issuer)
        if failed_at is not None and time.monotonic() - failed_at < _DISCOVERY_RETRY_SECONDS:
            raise TokenVerificationError(f"signing keys of issuer {issuer!r} unavailable (discovery failed recently)")
        base = issuer.rstrip("/")
        for path in ("/.well-known/openid-configuration", "/.well-known/oauth-authorization-server"):
            try:
                response = httpx.get(base + path, timeout=self._verification.http_timeout)
                document = response.json() if response.status_code == 200 else None
            except (httpx.HTTPError, ValueError):
                continue
            jwks_uri = document.get("jwks_uri") if isinstance(document, dict) else None
            if isinstance(jwks_uri, str) and jwks_uri:
                self._discovery_failed_at.pop(issuer, None)
                return jwks_uri
        self._discovery_failed_at[issuer] = time.monotonic()
        raise TokenVerificationError(f"could not discover the signing keys (jwks_uri) of issuer {issuer!r}")


def verify_access_token(token: str, verification: JwtVerification, keys: _KeySource | None = None) -> dict:
    """Verify a JWT access token per RFC 9068 §4 and return its claims.

    Nothing in the token is used before its signature is verified: for each trusted issuer,
    the key matching the token's key id is taken from that issuer's published keys, and the
    token is fully verified against it (signature, the algorithm allowlist, ``iss``, ``aud``,
    ``exp``). Keys are therefore only ever fetched from a trusted issuer, and a token signed by
    anyone else finds no key. The ``at+jwt`` type is checked on the verified header.

    Args:
        token (str): The encoded JWT.
        verification (JwtVerification): What this deployment trusts.
        keys (_KeySource, optional): A key source to reuse (its key cache); a new one otherwise.

    Returns:
        dict: The verified claims.

    Raises:
        TokenVerificationError: The token failed verification; the message says why (never the
            token).
    """
    with requires_extra("jwt", "Verifying tokens"):
        import jwt

    keys = keys or _KeySource(verification)
    reasons: list[str] = []
    for issuer in verification.issuers:
        try:
            key = keys.signing_key(token, issuer)
        except (jwt.PyJWTError, TokenVerificationError) as exc:
            reasons.append(f"{issuer}: {exc}")   # no matching key at this issuer
            continue
        try:
            verified = jwt.decode_complete(
                token,
                key,
                algorithms=list(verification.algorithms),
                audience=verification.audiences,
                issuer=issuer,
                leeway=verification.leeway_seconds,
                options={"require": ["iss", "aud", "exp", "sub"]},
            )
        except jwt.InvalidIssuerError as exc:
            reasons.append(f"{issuer}: {exc}")   # a key this issuer shares, but not its token
            continue
        except jwt.PyJWTError as exc:
            raise TokenVerificationError(str(exc)) from exc
        typ = str(verified["header"].get("typ", ""))
        if verification.require_typ and typ.lower() not in _ACCESS_TOKEN_TYPES:
            raise TokenVerificationError(f"token type {typ!r} is not an access token (at+jwt)")
        return verified["payload"]
    raise TokenVerificationError("not signed by a trusted issuer (" + "; ".join(reasons) + ")")


@dataclass(frozen=True)
class Claim:
    """Where a mapped claim goes: the PDP attribute id and, optionally, a type to convert it to.

    Attributes:
        attribute (str): The PDP attribute id.
        type (AttributeType | None): The type to convert to (e.g. RAR amounts arrive as strings:
            ``Claim("approved_amount", type="double")``); without one the value is sent as is.
    """

    attribute: str
    type: AttributeType | None = None

    def __post_init__(self) -> None:
        if not self.attribute:
            raise ValueError("Claim needs an attribute name")
        if self.type not in (None, "string", "integer", "double", "boolean"):
            raise ValueError(f"Claim type must be string, integer, double or boolean; got {self.type!r}")


@dataclass(frozen=True)
class AgentIdentityAttributes:
    """The PDP attribute ids the standard claims are sent as; None to not send one.

    Attributes:
        user_id (str | None): ``sub``, when a user is present (subject).
        agent_id (str | None): ``client_id``, the agent (subject).
        actor_id (str | None): The current actor, the outermost ``act.sub`` (subject).
        scopes (str | None): The token's scopes (environment).
        acr (str | None): The authentication context class (environment).
        amr (str | None): The authentication methods (environment).
        auth_time (str | None): When the user authenticated (environment).
        transaction_id (str | None): ``txn``, the transaction id (environment).
    """

    user_id: str | None = "user_id"
    agent_id: str | None = "agent_id"
    actor_id: str | None = "actor_id"
    scopes: str | None = "scopes"
    acr: str | None = "acr"
    amr: str | None = "amr"
    auth_time: str | None = "auth_time"
    transaction_id: str | None = "transaction_id"  # environment


@dataclass(frozen=True)
class RarMapping:
    """Map one approved OAuth Rich Authorization Request entry (RFC 9396) to attributes.

    From the token's ``authorization_details``, the entry of this ``type`` whose ``locations``
    include ``location`` (default: the verification audience) and, if ``action`` is set, whose
    ``actions`` include it, is selected. An entry without ``locations`` counts only when the
    token has a single audience (this server); with several, it might be meant for another API.
    Its common fields are sent as ``{prefix}type``, ``{prefix}actions``, ``{prefix}locations``,
    ``{prefix}datatypes``, ``{prefix}identifier`` and ``{prefix}privileges``.
    ``{prefix}available`` says whether exactly one entry matched. With none or several, no other
    ``{prefix}*`` attribute is sent: separate grants are never merged, since merging could
    permit a combination nobody approved.

    Attributes:
        type (str): The ``authorization_details`` entry type to select.
        fields (dict[str, str | Claim]): Type-specific fields, by dotted path → attribute.
        location (str | None): This server's identifier in ``locations``; the verification
            audience by default.
        action (str | None): An action the entry's ``actions`` must include.
        prefix (str): The prefix of every attribute this mapping sends.
    """

    type: str
    fields: dict[str, str | Claim] = field(default_factory=dict)
    location: str | None = None
    action: str | None = None
    prefix: str = "approved_"

    def __post_init__(self) -> None:
        for path, mapping in self.fields.items():
            if not isinstance(mapping, (str, Claim)):
                raise ValueError(f"RarMapping field {path!r} must map to an attribute name or a Claim")

    def apply(self, claims: dict, location: str, subject: dict, dropped: dict[str, str]) -> None:
        """Add the approved grant's attributes to ``subject``.

        Args:
            claims (dict): The token's claims.
            location (str): This server's identifier (used unless ``self.location`` is set).
            subject (dict): The subject attributes to add to.
            dropped (dict[str, str]): Collects anything left out, with the reason.
        """
        location = self.location or location
        audience = claims.get("aud")
        single_audience = isinstance(audience, str) or (isinstance(audience, list) and len(audience) == 1)
        entries = [
            e for e in claims.get("authorization_details") or []
            if isinstance(e, dict) and e.get("type") == self.type
            and (isinstance(e["locations"], list) and location in e["locations"] if e.get("locations") else single_audience)
            and (self.action is None or self.action in (e.get("actions") or []))
        ]
        subject[f"{self.prefix}available"] = len(entries) == 1
        if len(entries) > 1:
            dropped["authorization_details"] = f"{len(entries)} entries match; separate grants are never merged"
        if len(entries) != 1:
            return
        _map_claims(entries[0], {f: Claim(f"{self.prefix}{f}") for f in _RAR_COMMON_FIELDS}, subject, dropped)
        _map_claims(entries[0], {p: m if isinstance(m, Claim) else Claim(m) for p, m in self.fields.items()},
                    subject, dropped)


_RAR_COMMON_FIELDS = ("type", "actions", "locations", "datatypes", "identifier", "privileges")


@dataclass(frozen=True)
class IdentityPreview:
    """What a claim set turns into.

    Attributes:
        subject (dict[str, Any]): The subject attributes.
        environment (dict[str, Any]): The environment attributes.
        dropped (dict[str, str]): Each claim or mapping that was dropped, with the reason.
    """

    subject: dict[str, Any]
    environment: dict[str, Any]
    dropped: dict[str, str]


def _resolve_path(claims: dict, path: str) -> tuple[bool, Any]:
    """Walk a dotted path through nested objects: ``(found, value)``."""
    value: Any = claims
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _convert(value: Any, to: AttributeType | None) -> Any:
    """Convert a scalar or a list of scalars to ``to``; raise ValueError if it can't be."""
    if to is None:
        return value
    one = {"string": str, "integer": int, "double": float, "boolean": _as_bool}[to]
    values = value if isinstance(value, list) else [value]
    if any(isinstance(v, (dict, list)) for v in values):
        raise ValueError("it is not a scalar or a list of scalars")
    converted = [one(v) for v in values]
    return converted if isinstance(value, list) else converted[0]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{value!r} is not a boolean")


def _map_claims(source: dict, mapping: dict[str, Claim], target: dict, dropped: dict[str, str]) -> None:
    """Copy each mapped path of ``source`` into ``target`` as its attribute.

    Converted to its type; a value that can't be converted or sent goes to ``dropped`` with the
    reason.
    """
    for path, claim in mapping.items():
        found, value = _resolve_path(source, path)
        if not found:
            continue
        try:
            value = _convert(value, claim.type)
        except (TypeError, ValueError) as exc:
            dropped[path] = f"can't be sent as {claim.attribute!r}: {exc}"
            continue
        if not is_sendable(value):
            dropped[path] = f"can't be sent as {claim.attribute!r}: not a scalar or a list of scalars of one type"
            continue
        target[claim.attribute] = value


class AgentIdentityContextProvider(DefaultServerContextProvider):
    """A context provider that verifies the caller's access token and maps its claims.

    Works on server paths (the context is the request headers) and client paths (the context is
    whatever the app passes, e.g. to ``agent.invoke(..., context=...)``). Pass exactly one claim
    source: ``verification`` (verify the token here) or ``claims_from`` (claims verified
    upstream). A token that fails verification yields an empty subject, which is denied; the
    reason is logged, never the token. Verified claims are cached per token until it expires.

    Args:
        verification (JwtVerification, optional): Verify the token here. Use it on a server
            path, or for an agent behind your own API that receives the user's token.
        token_from (Callable[[Any], str | None], optional): Finds the token in the context; by
            default the ``Authorization: Bearer`` header of a headers dict. Only with
            ``verification``.
        claims_from (Callable[[Any], dict | None], optional): Returns claims ALREADY verified
            upstream (LangGraph Platform auth, AgentCore Runtime), mapped without verifying
            again. Only use it when something trustworthy did verify them.
        claims (dict[str, str | Claim], optional): Extra claims to map, by dotted path, e.g.
            ``{"realm_access.roles": "roles"}``.
        rar (RarMapping, optional): Map an approved Rich Authorization Request grant.
        attributes (AgentIdentityAttributes, optional): The attribute ids the standard claims
            are sent as.
        extra_subject (Callable[[dict], dict[str, Any]], optional): Adds derived subject
            attributes from the claims.
        client_metadata (ClientIdMetadata, optional): For a ``client_id`` that is a Client ID
            Metadata Document URL (CIMD), send the client's host and, optionally, fields of its
            document.

    Raises:
        ValueError: Not exactly one of ``verification`` and ``claims_from``; ``token_from``
            without ``verification``; or a ``RarMapping`` without ``location`` with
            ``claims_from``.
    """

    def __init__(
        self,
        verification: JwtVerification | None = None,
        *,
        token_from: Callable[[Any], str | None] | None = None,
        claims_from: Callable[[Any], dict | None] | None = None,
        claims: dict[str, str | Claim] | None = None,
        rar: RarMapping | None = None,
        attributes: AgentIdentityAttributes | None = None,
        extra_subject: Callable[[dict], dict[str, Any]] | None = None,
        client_metadata: ClientIdMetadata | None = None,
    ) -> None:
        if (verification is None) == (claims_from is None):
            raise ValueError(
                "AgentIdentityContextProvider needs exactly one of verification= (verify the token "
                "here) or claims_from= (claims already verified upstream)"
            )
        if token_from is not None and verification is None:
            raise ValueError("token_from= only applies with verification=; claims_from= supplies claims directly")
        if claims_from is not None and rar is not None and rar.location is None:
            raise ValueError("With claims_from=, RarMapping needs location= (this server's identifier)")
        self.verification = verification
        self.token_from = token_from or _bearer_token
        self.claims_from = claims_from
        self.claims = {path: (m if isinstance(m, Claim) else Claim(m)) for path, m in (claims or {}).items()}
        self.rar = rar
        self.attributes = attributes or AgentIdentityAttributes()
        self.extra_subject = extra_subject
        self.client_metadata = client_metadata
        self._keys = _KeySource(verification) if verification is not None else None
        self._verified: TTLCache = TTLCache(maxsize=1024, ttl=300)
        self._rejected: TTLCache = TTLCache(maxsize=1024, ttl=_REJECTED_TOKEN_SECONDS)
        self._lock = threading.Lock()
        if claims_from is not None:
            logger.info("AgentIdentityContextProvider maps claims verified upstream (claims_from); "
                        "tokens are not verified here.")

    def for_audience(self, audience: str | list[str]) -> AgentIdentityContextProvider:
        """Copy this provider to verify for another resource server (used per gateway route).

        Args:
            audience (str | list[str]): The other server's audience.

        Returns:
            AgentIdentityContextProvider: The copy.
        """
        if self.verification is None:
            raise ValueError("for_audience() needs verification=; with claims_from= the audience was checked upstream")
        return type(self)(
            self.verification.with_audience(audience), token_from=self.token_from,
            claims=dict(self.claims), rar=self.rar, attributes=self.attributes,
            extra_subject=self.extra_subject, client_metadata=self.client_metadata,
        )

    # ── ContextProvider ───────────────────────────────────────────────────────

    def get_subject(self, context: Any) -> dict[str, Any]:
        """Verify the caller's token and map its claims to subject attributes.

        Args:
            context (Any): The request headers (server paths) or the app's context (client paths).

        Returns:
            dict[str, Any]: The subject attributes; empty when there is no token or it failed
            verification.
        """
        claims = self._verified_claims(context)
        return self.subject_from_claims(claims) if claims else {}

    def get_environment(self, context: Any) -> dict[str, Any]:
        """Verify the caller's token and map its claims to environment attributes.

        Args:
            context (Any): The request headers (server paths) or the app's context (client paths).

        Returns:
            dict[str, Any]: The environment attributes; empty when there is no token or it failed
            verification.
        """
        claims = self._verified_claims(context)
        return self.environment_from_claims(claims) if claims else {}

    # ── Mapping (public so tests and preview() can use it without a token) ────

    def subject_from_claims(self, claims: dict) -> dict[str, Any]:
        """Map already-verified claims to subject attributes.

        Args:
            claims (dict): Verified claims.

        Returns:
            dict[str, Any]: The subject attributes.
        """
        return self.preview(claims).subject

    def environment_from_claims(self, claims: dict) -> dict[str, Any]:
        """Map already-verified claims to environment attributes.

        Args:
            claims (dict): Verified claims.

        Returns:
            dict[str, Any]: The environment attributes.
        """
        return self.preview(claims).environment

    def is_user_present(self, claims: dict) -> bool:
        """Whether ``sub`` is a user rather than the client itself.

        With Client Credentials the client is its own subject, so ``sub == client_id`` means no
        user. Override for an identity provider that marks this differently.

        Args:
            claims (dict): Verified claims.

        Returns:
            bool: True when a user is present.
        """
        sub = claims.get("sub")
        return bool(sub) and sub != claims.get("client_id")

    def preview(self, claims: dict) -> IdentityPreview:
        """Show what a claim set turns into, without verifying or calling the PDP.

        Args:
            claims (dict): Claims to map.

        Returns:
            IdentityPreview: The subject and environment attributes, and what was dropped and
            why.
        """
        names = self.attributes
        subject: dict[str, Any] = {}
        environment: dict[str, Any] = {}
        dropped: dict[str, str] = {}

        def put(target: dict, name: str | None, value: Any) -> None:
            if name is not None and value is not None:
                target[name] = value

        put(subject, names.user_id, claims.get("sub") if self.is_user_present(claims) else None)
        put(subject, names.agent_id, claims.get("client_id"))
        put(subject, names.actor_id, self._current_actor(claims))
        scope = claims.get("scope", claims.get("scp"))
        put(subject, names.scopes, scope.split() if isinstance(scope, str) else scope)
        put(subject, names.acr, claims.get("acr"))
        put(subject, names.amr, claims.get("amr"))
        put(subject, names.auth_time, claims.get("auth_time"))
        put(environment, names.transaction_id, claims.get("txn"))
        if self.client_metadata is not None:
            subject.update(self.client_metadata.attributes(claims.get("client_id")))
        _map_claims(claims, self.claims, subject, dropped)
        if self.rar is not None:
            location = self.verification.audiences[0] if self.verification is not None else ""
            self.rar.apply(claims, location, subject, dropped)   # rar.location wins when set
        if self.extra_subject is not None:
            subject.update(self.extra_subject(claims))
        for key in [k for k, v in subject.items() if not is_sendable(v)]:
            dropped[key] = "not a scalar or a list of scalars of one type"
            del subject[key]
        return IdentityPreview(subject=subject, environment=environment, dropped=dropped)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _verified_claims(self, context: Any) -> dict | None:
        """Get the caller's verified claims, or None.

        From ``claims_from`` (verified upstream), or by verifying the token ``token_from`` finds.
        Both verified claims and rejections are remembered per token (a rejection briefly, and
        logged once), so a request's subject and environment, and repeated calls with the same
        token, verify once.
        """
        credential_expires_at.set(None)  # this request's caller, not a previous one's
        if self.claims_from is not None:
            return self._upstream_claims(context)
        token = self.token_from(context)
        if not token:
            return None
        key = hashlib.sha256(token.encode()).hexdigest()
        with self._lock:
            claims = self._verified.get(key)
            rejected = key in self._rejected
        if rejected:
            return None
        if claims is None:
            try:
                claims = verify_access_token(token, self.verification, self._keys)
            except TokenVerificationError as exc:
                with self._lock:
                    self._rejected[key] = True
                logger.warning("Caller identity not verified, so the request is denied: %s", exc)
                return None
            with self._lock:
                self._verified[key] = claims
        if claims.get("exp", 0) <= time.time() - self.verification.leeway_seconds:
            return None  # expired since it was cached
        credential_expires_at.set(float(claims["exp"]))
        if isinstance(claims.get("act"), dict) and isinstance(claims["act"].get("act"), dict):
            logger.debug("Delegation chain (prior actors are informational only): %s", claims["act"])
        return claims

    def _upstream_claims(self, context: Any) -> dict | None:
        """Get the claims verified upstream (``claims_from``).

        An ``exp`` in them is still honoured: expired claims are refused, and cached decisions never
        outlive them.
        """
        claims = self.claims_from(context)
        if not isinstance(claims, dict) or not claims:
            return None
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            if exp <= time.time():
                return None
            credential_expires_at.set(float(exp))
        return claims

    @staticmethod
    def _current_actor(claims: dict) -> Any:
        act = claims.get("act")
        return act.get("sub") if isinstance(act, dict) else None

def _bearer_token(headers: dict[str, Any]) -> str | None:
    auth = next((v for k, v in (headers or {}).items() if k.lower() == "authorization"), "")
    if not isinstance(auth, str) or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip() or None
