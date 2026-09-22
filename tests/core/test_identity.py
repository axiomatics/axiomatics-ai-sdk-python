"""
Unit tests for agent identity from verified OAuth access tokens (``core/identity.py``).

Verification uses REAL crypto: RSA/EC keys are generated with ``cryptography`` and tokens are
signed with PyJWT. Signing keys are injected by replacing the provider's ``_keys`` (or passing
``keys=`` to ``verify_access_token``) with an object whose ``signing_key(token, issuer)``
returns the public key. ``_KeySource`` discovery is covered by monkeypatching ``httpx.get``
and ``jwt.PyJWKClient``. No identity provider or ADS is required.
"""
from __future__ import annotations

import contextvars
import hashlib
import logging
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from axiomatics_ai.core.cache import credential_expires_at
from axiomatics_ai.core.identity import (
    AgentIdentityAttributes,
    AgentIdentityContextProvider,
    Claim,
    JwtVerification,
    RarMapping,
    TokenVerificationError,
    _KeySource,
    verify_access_token,
)

_LOGGER = "axiomatics_ai.core.identity"
ISSUER = "https://idp.example"
AUDIENCE = "https://tools.example"

_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_EC = ec.generate_private_key(ec.SECP256R1())


class StaticKeys:
    """Stand-in for ``_KeySource``: returns one public key and records every fetch."""

    def __init__(self, public_key) -> None:
        self.public_key = public_key
        self.fetched: list[str] = []

    def signing_key(self, token: str, issuer: str):
        self.fetched.append(issuer)
        return self.public_key


@pytest.fixture(autouse=True)
def _reset_credential_expiry():
    """The provider sets ``credential_expires_at`` in the current context; keep it from leaking
    into other tests (which share pytest's context)."""
    reset = credential_expires_at.set(None)
    yield
    credential_expires_at.reset(reset)


def _claims(**overrides) -> dict:
    now = int(time.time())
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": "alice", "client_id": "agent-1",
        "exp": now + 300, "iat": now,
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


def _token(claims: dict | None = None, *, key=_RSA, alg: str = "RS256", typ: str | None = "at+jwt") -> str:
    return jwt.encode(claims if claims is not None else _claims(), key, algorithm=alg, headers={"typ": typ})


def _verification(**kw) -> JwtVerification:
    return JwtVerification(**{"issuer": ISSUER, "audience": AUDIENCE, **kw})


def _verify(token: str, verification: JwtVerification | None = None, public_key=None) -> dict:
    return verify_access_token(
        token, verification or _verification(), StaticKeys(public_key or _RSA.public_key()),
    )


def _provider(verification: JwtVerification | None = None, **kw) -> tuple[AgentIdentityContextProvider, StaticKeys]:
    provider = AgentIdentityContextProvider(verification or _verification(), **kw)
    keys = StaticKeys(_RSA.public_key())
    provider._keys = keys
    return provider, keys


def _headers(token: str) -> dict:
    return {"authorization": f"Bearer {token}"}


# ── JwtVerification ─────────────────────────────────────────────────────────────

class TestJwtVerification:
    def test_needs_issuer_and_audience(self):
        with pytest.raises(ValueError):
            JwtVerification(issuer="", audience=AUDIENCE)
        with pytest.raises(ValueError):
            JwtVerification(issuer=ISSUER, audience=[])

    @pytest.mark.parametrize("algorithms", [("HS256",), ("RS256", "none"), ()])
    def test_only_asymmetric_algorithms(self, algorithms):
        with pytest.raises(ValueError):
            _verification(algorithms=algorithms)

    def test_issuers_and_audiences_normalized_to_lists(self):
        v = _verification(issuer=["a", "b"])
        assert v.issuers == ["a", "b"]
        assert v.audiences == [AUDIENCE]

    def test_with_audience_copies_everything_else(self):
        v = _verification(leeway_seconds=5, require_typ=False, jwks_url="https://idp.example/jwks")
        copy = v.with_audience("https://other.example")
        assert copy.audiences == ["https://other.example"]
        assert (copy.issuer, copy.jwks_url, copy.leeway_seconds, copy.require_typ, copy.algorithms) == (
            v.issuer, v.jwks_url, v.leeway_seconds, v.require_typ, v.algorithms,
        )
        assert v.audiences == [AUDIENCE]  # original untouched


# ── verify_access_token ─────────────────────────────────────────────────────────

class TestVerifyAccessToken:
    def test_valid_rsa_token(self):
        claims = _verify(_token())
        assert claims["sub"] == "alice" and claims["client_id"] == "agent-1"

    def test_valid_ec_token(self):
        assert _verify(_token(key=_EC, alg="ES256"), public_key=_EC.public_key())["sub"] == "alice"

    @pytest.mark.parametrize("typ", ["at+jwt", "application/at+jwt", "AT+JWT"])
    def test_access_token_types_accepted(self, typ):
        assert _verify(_token(typ=typ))["sub"] == "alice"

    @pytest.mark.parametrize("typ", ["JWT", None])
    def test_id_token_type_rejected(self, typ):
        with pytest.raises(TokenVerificationError, match="not an access token"):
            _verify(_token(typ=typ))

    def test_require_typ_false_accepts_plain_jwt(self):
        assert _verify(_token(typ="JWT"), _verification(require_typ=False))["sub"] == "alice"

    def test_wrong_audience(self):
        with pytest.raises(TokenVerificationError):
            _verify(_token(_claims(aud="https://someone-else.example")))

    def test_untrusted_issuer_rejected_and_keys_only_from_trusted_issuers(self):
        keys = StaticKeys(_RSA.public_key())
        with pytest.raises(TokenVerificationError, match="not signed by a trusted issuer"):
            verify_access_token(_token(_claims(iss="https://evil.example")), _verification(), keys)
        assert keys.fetched == [ISSUER]              # never the token's own (untrusted) issuer

    def test_trusted_issuer_claimed_but_signed_by_another_key(self):
        with pytest.raises(TokenVerificationError, match="Signature verification failed"):
            _verify(_token(key=_OTHER_RSA))          # iss says ISSUER, but ISSUER's key disagrees

    def test_nothing_read_before_verification(self, monkeypatch):
        calls = []
        monkeypatch.setattr(jwt, "get_unverified_header", lambda *a, **k: calls.append("header"))
        real_decode = jwt.decode
        monkeypatch.setattr(jwt, "decode", lambda *a, **k: calls.append("decode") or real_decode(*a, **k))
        assert _verify(_token())["sub"] == "alice"
        assert calls == []                           # only the verifying decode_complete is used

    def test_no_key_found_at_any_trusted_issuer(self):
        class NoKeys:
            def signing_key(self, token, issuer):
                raise jwt.PyJWKClientError("Unable to find a signing key that matches")

        with pytest.raises(TokenVerificationError, match="not signed by a trusted issuer"):
            verify_access_token(_token(), _verification(issuer=["https://a.example", ISSUER]), NoKeys())

    def test_expired(self):
        with pytest.raises(TokenVerificationError, match="expired"):
            _verify(_token(_claims(exp=int(time.time()) - 120)))

    def test_expired_within_leeway_accepted(self):
        token = _token(_claims(exp=int(time.time()) - 10))
        assert _verify(token, _verification(leeway_seconds=30))["sub"] == "alice"
        with pytest.raises(TokenVerificationError):
            _verify(token, _verification(leeway_seconds=0))

    def test_hs256_rejected(self):
        token = jwt.encode(_claims(), "shared-secret-that-is-long-enough-for-hs256", algorithm="HS256",
                           headers={"typ": "at+jwt"})
        with pytest.raises(TokenVerificationError, match="not allowed"):
            _verify(token)

    def test_alg_none_rejected(self):
        token = jwt.encode(_claims(), None, algorithm="none", headers={"typ": "at+jwt"})
        keys = StaticKeys(_RSA.public_key())
        with pytest.raises(TokenVerificationError, match="not allowed"):
            verify_access_token(token, _verification(), keys)
        assert keys.fetched == [ISSUER]              # a trusted issuer's key; alg none still refused

    def test_allowed_asymmetric_but_not_configured_alg_rejected(self):
        with pytest.raises(TokenVerificationError, match="not allowed"):
            _verify(_token(alg="RS512"))  # asymmetric, but not in the default allowlist

    def test_wrong_key(self):
        with pytest.raises(TokenVerificationError):
            _verify(_token(key=_OTHER_RSA))

    @pytest.mark.parametrize("missing", ["sub", "exp"])
    def test_missing_required_claim(self, missing):
        with pytest.raises(TokenVerificationError):
            _verify(_token(_claims(**{missing: None})))

    @pytest.mark.parametrize("token", ["not-a-jwt", "a.b.c", ""])
    def test_malformed(self, token):
        with pytest.raises(TokenVerificationError):
            _verify(token)

    def test_multiple_trusted_issuers(self):
        keys = StaticKeys(_RSA.public_key())
        verification = _verification(issuer=["https://a.example", "https://b.example"])
        claims = verify_access_token(_token(_claims(iss="https://b.example")), verification, keys)
        assert claims["iss"] == "https://b.example"
        # Each trusted issuer is tried in turn; only the token's issuer verifies.
        assert keys.fetched == ["https://a.example", "https://b.example"]

    def test_multiple_audiences(self):
        verification = _verification(audience=["https://x.example", AUDIENCE])
        assert _verify(_token(), verification)["aud"] == AUDIENCE
        assert _verify(_token(_claims(aud=["https://z.example", "https://x.example"])), verification)


# ── _KeySource (JWKS discovery) ─────────────────────────────────────────────────

class _FakeJWKClient:
    created: list[dict] = []

    def __init__(self, url, **kw):
        _FakeJWKClient.created.append({"url": url, **kw})

    def get_signing_key_from_jwt(self, token):
        return type("SigningKey", (), {"key": _RSA.public_key()})()


def _response(status: int, body: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body or {}, request=httpx.Request("GET", "http://x"))


@pytest.fixture
def fake_jwk_client(monkeypatch):
    _FakeJWKClient.created = []
    monkeypatch.setattr(jwt, "PyJWKClient", _FakeJWKClient)
    return _FakeJWKClient


class TestKeySource:
    def test_openid_discovery(self, monkeypatch, fake_jwk_client):
        urls = []

        def get(url, **kw):
            urls.append(url)
            return _response(200, {"jwks_uri": "https://idp.example/keys"})

        monkeypatch.setattr(httpx, "get", get)
        source = _KeySource(_verification(http_timeout=2.0))
        assert verify_access_token(_token(), _verification(), source)["sub"] == "alice"
        assert urls == ["https://idp.example/.well-known/openid-configuration"]
        assert fake_jwk_client.created[0]["url"] == "https://idp.example/keys"
        assert fake_jwk_client.created[0]["timeout"] == 2.0

    def test_falls_back_to_oauth_authorization_server(self, monkeypatch, fake_jwk_client):
        urls = []

        def get(url, **kw):
            urls.append(url)
            if url.endswith("openid-configuration"):
                return _response(404)
            return _response(200, {"jwks_uri": "https://idp.example/oauth-keys"})

        monkeypatch.setattr(httpx, "get", get)
        _KeySource(_verification()).signing_key("t", ISSUER + "/")  # trailing slash stripped
        assert urls == [
            "https://idp.example/.well-known/openid-configuration",
            "https://idp.example/.well-known/oauth-authorization-server",
        ]
        assert fake_jwk_client.created[0]["url"] == "https://idp.example/oauth-keys"

    def test_http_error_falls_through(self, monkeypatch, fake_jwk_client):
        def get(url, **kw):
            if url.endswith("openid-configuration"):
                raise httpx.ConnectError("down")
            return _response(200, {"jwks_uri": "https://idp.example/k"})

        monkeypatch.setattr(httpx, "get", get)
        _KeySource(_verification()).signing_key("t", ISSUER)
        assert fake_jwk_client.created[0]["url"] == "https://idp.example/k"

    def test_discovery_failure_raises(self, monkeypatch, fake_jwk_client):
        monkeypatch.setattr(httpx, "get", lambda url, **kw: _response(200, {"issuer": ISSUER}))  # no jwks_uri
        with pytest.raises(TokenVerificationError, match="jwks_uri"):
            _KeySource(_verification()).signing_key("t", ISSUER)
        assert fake_jwk_client.created == []

    def test_non_json_discovery_document_is_a_clean_failure(self, monkeypatch, fake_jwk_client):
        html = httpx.Response(200, text="<html>login</html>", request=httpx.Request("GET", "http://x"))
        monkeypatch.setattr(httpx, "get", lambda url, **kw: html)
        with pytest.raises(TokenVerificationError, match="jwks_uri"):
            _KeySource(_verification()).signing_key("t", ISSUER)

    def test_non_json_discovery_denies_rather_than_errors(self, monkeypatch, fake_jwk_client):
        html = httpx.Response(200, text="not json", request=httpx.Request("GET", "http://x"))
        monkeypatch.setattr(httpx, "get", lambda url, **kw: html)
        provider = AgentIdentityContextProvider(_verification())
        assert provider.get_subject(_headers(_token())) == {}

    def test_failed_discovery_is_remembered_then_retried(self, monkeypatch, fake_jwk_client):
        import axiomatics_ai.core.identity as identity

        calls = []
        monkeypatch.setattr(httpx, "get", lambda url, **kw: calls.append(url) or _response(404))
        clock = [1000.0]
        monkeypatch.setattr(identity.time, "monotonic", lambda: clock[0])
        source = _KeySource(_verification())
        for _ in range(3):
            with pytest.raises(TokenVerificationError):
                source.signing_key("t", ISSUER)
        assert len(calls) == 2                        # both well-known paths, tried once
        clock[0] += identity._DISCOVERY_RETRY_SECONDS + 1
        monkeypatch.setattr(httpx, "get", lambda url, **kw: calls.append(url) or _response(200, {"jwks_uri": "k"}))
        source.signing_key("t", ISSUER)                # retried after the window, and recovers
        assert fake_jwk_client.created[0]["url"] == "k"

    def test_explicit_jwks_url_skips_discovery(self, monkeypatch, fake_jwk_client):
        def get(url, **kw):
            raise AssertionError("discovery must not run")

        monkeypatch.setattr(httpx, "get", get)
        _KeySource(_verification(jwks_url="https://idp.example/static-keys")).signing_key("t", ISSUER)
        assert fake_jwk_client.created[0]["url"] == "https://idp.example/static-keys"

    def test_client_cached_per_issuer(self, monkeypatch, fake_jwk_client):
        calls = []
        monkeypatch.setattr(httpx, "get", lambda url, **kw: calls.append(url) or _response(200, {"jwks_uri": "k"}))
        source = _KeySource(_verification(issuer=["https://a.example", "https://b.example"]))
        for _ in range(3):
            source.signing_key("t", "https://a.example")
        source.signing_key("t", "https://b.example")
        assert len(fake_jwk_client.created) == 2
        assert len(calls) == 2


# ── Claim / RarMapping validation ───────────────────────────────────────────────

class TestClaimValidation:
    def test_valid_types(self):
        for t in (None, "string", "integer", "double", "boolean"):
            assert Claim("x", type=t).type == t

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            Claim("x", type="float")

    def test_empty_attribute(self):
        with pytest.raises(ValueError):
            Claim("")

    def test_rar_field_must_be_name_or_claim(self):
        with pytest.raises(ValueError):
            AgentIdentityContextProvider(_verification(), rar=RarMapping(type="payment", fields={"amount": 5}))


# ── Claim mapping (no token) ────────────────────────────────────────────────────

class TestClaimMapping:
    def test_standard_claims(self):
        provider, _ = _provider()
        claims = _claims(
            scope="read write", acr="urn:mace:incommon:iap:silver", amr=["pwd", "mfa"],
            auth_time=1700000000, txn="tx-42", act={"sub": "agent-orchestrator"},
        )
        assert provider.subject_from_claims(claims) == {
            "user_id": "alice", "agent_id": "agent-1", "actor_id": "agent-orchestrator",
            "scopes": ["read", "write"], "acr": "urn:mace:incommon:iap:silver",
            "amr": ["pwd", "mfa"], "auth_time": 1700000000,
        }
        assert provider.environment_from_claims(claims) == {"transaction_id": "tx-42"}

    def test_client_credentials_has_no_user(self):
        provider, _ = _provider()
        subject = provider.subject_from_claims(_claims(sub="agent-1"))
        assert "user_id" not in subject
        assert subject["agent_id"] == "agent-1"

    def test_is_user_present(self):
        provider, _ = _provider()
        assert provider.is_user_present({"sub": "alice", "client_id": "agent-1"}) is True
        assert provider.is_user_present({"sub": "agent-1", "client_id": "agent-1"}) is False
        assert provider.is_user_present({"client_id": "agent-1"}) is False
        assert provider.is_user_present({"sub": "", "client_id": "x"}) is False

    @pytest.mark.parametrize("scp,expected", [("a b", ["a", "b"]), (["a", "b"], ["a", "b"])])
    def test_scp_claim(self, scp, expected):
        provider, _ = _provider()
        assert provider.subject_from_claims(_claims(scp=scp))["scopes"] == expected

    def test_scope_wins_over_scp(self):
        provider, _ = _provider()
        assert provider.subject_from_claims(_claims(scope="a", scp="b"))["scopes"] == ["a"]

    def test_nested_actor_sends_only_outermost(self):
        provider, _ = _provider()
        subject = provider.subject_from_claims(_claims(act={"sub": "current", "act": {"sub": "earlier"}}))
        assert subject["actor_id"] == "current"
        assert "earlier" not in str(subject)

    def test_raw_claims_not_forwarded(self):
        provider, _ = _provider()
        subject = provider.subject_from_claims(_claims(email="a@b.c", realm_access={"roles": ["x"]}))
        assert "email" not in subject and "realm_access" not in subject

    def test_renamed_and_omitted_attributes(self):
        names = AgentIdentityAttributes(user_id="uid", agent_id=None, scopes=None, transaction_id="txid")
        provider, _ = _provider(attributes=names)
        claims = _claims(scope="a", txn="t1")
        subject = provider.subject_from_claims(claims)
        assert subject == {"uid": "alice"}
        assert provider.environment_from_claims(claims) == {"txid": "t1"}

    def test_dotted_path_and_typed_claims(self):
        provider, _ = _provider(claims={
            "realm_access.roles": "roles",
            "tenant.id": Claim("tenant_id"),
            "limits.max": Claim("max_amount", type="double"),
            "flags.beta": Claim("beta", type="boolean"),
            "missing.path": "never",
        })
        subject = provider.subject_from_claims(_claims(
            realm_access={"roles": ["admin", "user"]}, tenant={"id": "t-9"},
            limits={"max": "12.5"}, flags={"beta": "true"},
        ))
        assert subject["roles"] == ["admin", "user"]
        assert subject["tenant_id"] == "t-9"
        assert subject["max_amount"] == 12.5
        assert subject["beta"] is True
        assert "never" not in subject

    def test_unconvertible_and_unsendable_claims_dropped(self):
        provider, _ = _provider(claims={
            "amount": Claim("amount", type="integer"),
            "flag": Claim("flag", type="boolean"),
            "obj": "obj",
            "mixed": "mixed",
        })
        preview = provider.preview(_claims(amount="lots", flag="maybe", obj={"a": 1}, mixed=[1, "a"]))
        for name in ("amount", "flag", "obj", "mixed"):
            assert name not in preview.subject
            assert name in preview.dropped

    def test_extra_subject_and_unsendable_dropped(self):
        provider = AgentIdentityContextProvider(
            _verification(), extra_subject=lambda c: {"is_admin": c.get("sub") == "alice", "blob": {"x": 1}},
        )
        preview = provider.preview(_claims())
        assert preview.subject["is_admin"] is True
        assert "blob" not in preview.subject and "blob" in preview.dropped

    def test_is_user_present_override(self):
        class Provider(AgentIdentityContextProvider):
            def is_user_present(self, claims):
                return claims.get("user_type") == "human"

        provider = Provider(_verification())
        assert "user_id" not in provider.subject_from_claims(_claims())
        assert provider.subject_from_claims(_claims(user_type="human"))["user_id"] == "alice"


# ── Rich Authorization Requests (RFC 9396) ─────────────────────────────────────

def _payment(**overrides) -> dict:
    entry = {
        "type": "payment_initiation", "actions": ["initiate"], "locations": [AUDIENCE],
        "instructedAmount": {"currency": "EUR", "amount": "123.50"}, "creditorName": "Merchant A",
    }
    entry.update(overrides)
    return {k: v for k, v in entry.items() if v is not None}


_PAYMENT_FIELDS = {
    "instructedAmount.amount": Claim("approved_amount", type="double"),
    "instructedAmount.currency": "approved_currency",
    "creditorName": "approved_creditor",
}


class TestRarMapping:
    def _preview(self, details, **rar_kw):
        provider, _ = _provider(rar=RarMapping(type="payment_initiation", fields=_PAYMENT_FIELDS, **rar_kw))
        return provider.preview(_claims(authorization_details=details))

    def _approved(self, subject):
        return {k: v for k, v in subject.items() if k.startswith("approved_")}

    def test_single_match(self):
        preview = self._preview([_payment(), {"type": "account_information", "actions": ["read"]}])
        assert self._approved(preview.subject) == {
            "approved_available": True,
            "approved_type": "payment_initiation",
            "approved_actions": ["initiate"],
            "approved_locations": [AUDIENCE],
            "approved_amount": 123.5,
            "approved_currency": "EUR",
            "approved_creditor": "Merchant A",
        }

    def test_entry_without_locations_matches(self):
        preview = self._preview([_payment(locations=None)])
        assert preview.subject["approved_available"] is True
        assert "approved_locations" not in preview.subject

    def test_entry_without_locations_ignored_for_multi_audience_token(self):
        provider, _ = _provider(rar=RarMapping(type="payment_initiation", fields=_PAYMENT_FIELDS))
        claims = _claims(aud=[AUDIENCE, "https://other.example"], authorization_details=[_payment(locations=None)])
        assert self._approved(provider.preview(claims).subject) == {"approved_available": False}

    def test_entry_without_locations_matches_single_audience_list(self):
        provider, _ = _provider(rar=RarMapping(type="payment_initiation", fields=_PAYMENT_FIELDS))
        claims = _claims(aud=[AUDIENCE], authorization_details=[_payment(locations=None)])
        assert provider.preview(claims).subject["approved_available"] is True

    def test_invalid_field_mapping_rejected_at_construction(self):
        with pytest.raises(ValueError, match="attribute name or a Claim"):
            RarMapping(type="payment_initiation", fields={"x": 1})

    def test_location_defaults_to_first_audience(self):
        other = "https://other.example"
        provider, _ = _provider(
            _verification(audience=[other, AUDIENCE]),
            rar=RarMapping(type="payment_initiation", fields=_PAYMENT_FIELDS),
        )
        details = [_payment(), _payment(locations=[other], creditorName="Merchant B")]
        subject = provider.subject_from_claims(_claims(aud=other, authorization_details=details))
        assert subject["approved_creditor"] == "Merchant B"

    def test_explicit_location(self):
        preview = self._preview([_payment(locations=["https://elsewhere"])], location="https://elsewhere")
        assert preview.subject["approved_available"] is True

    def test_other_location_does_not_match(self):
        preview = self._preview([_payment(locations=["https://elsewhere"])])
        assert self._approved(preview.subject) == {"approved_available": False}

    def test_action_filter(self):
        details = [_payment(actions=["status"]), _payment(actions=["initiate", "cancel"], creditorName="B")]
        preview = self._preview(details, action="cancel")
        assert preview.subject["approved_creditor"] == "B"
        assert self._approved(self._preview([_payment(actions=["status"])], action="cancel").subject) == {
            "approved_available": False,
        }

    def test_multiple_matches_are_never_merged(self):
        preview = self._preview([_payment(), _payment(creditorName="Merchant B")])
        assert self._approved(preview.subject) == {"approved_available": False}
        assert "authorization_details" in preview.dropped

    def test_no_details(self):
        preview = self._preview(None)
        assert self._approved(preview.subject) == {"approved_available": False}
        assert "authorization_details" not in preview.dropped

    def test_custom_prefix(self):
        provider, _ = _provider(rar=RarMapping(type="payment_initiation", prefix="grant_"))
        subject = provider.subject_from_claims(_claims(authorization_details=[_payment()]))
        assert subject["grant_available"] is True and subject["grant_type"] == "payment_initiation"


# ── Provider: verification, caching, expiry ─────────────────────────────────────

class TestProviderVerification:
    def test_valid_token(self):
        provider, _ = _provider()
        token = _token(_claims(txn="tx-1", scope="read"))
        assert provider.get_subject(_headers(token)) == {
            "user_id": "alice", "agent_id": "agent-1", "scopes": ["read"],
        }
        assert provider.get_environment(_headers(token)) == {"transaction_id": "tx-1"}

    def test_header_name_case_insensitive(self):
        provider, _ = _provider()
        assert provider.get_subject({"Authorization": f"bearer {_token()}"})["user_id"] == "alice"

    @pytest.mark.parametrize("headers", [{}, {"authorization": "Basic abc"}, {"authorization": "Bearer "}, None])
    def test_no_bearer_token(self, headers):
        provider, keys = _provider()
        assert provider.get_subject(headers) == {}
        assert provider.get_environment(headers) == {}
        assert keys.fetched == []

    def test_invalid_token_yields_empty_and_warns_without_token(self, caplog):
        provider, _ = _provider()
        token = _token(key=_OTHER_RSA)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert provider.get_subject(_headers(token)) == {}
            assert provider.get_environment(_headers(token)) == {}
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "not verified" in warnings[0]      # subject + environment: one check
        assert not any(token in w for w in warnings)

    def test_rejected_token_is_remembered_and_logged_once(self, caplog):
        provider, keys = _provider()
        token = _token(key=_OTHER_RSA)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            for _ in range(3):
                assert provider.get_subject(_headers(token)) == {}
        assert len(keys.fetched) == 1                                        # verified once
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_rejection_is_retried_after_it_expires(self, monkeypatch):
        import axiomatics_ai.core.identity as identity

        monkeypatch.setattr(identity, "_REJECTED_TOKEN_SECONDS", 0.05)
        provider, keys = _provider()
        token = _token(key=_OTHER_RSA)
        provider.get_subject(_headers(token))
        time.sleep(0.1)
        provider.get_subject(_headers(token))
        assert len(keys.fetched) == 2

    def test_expiry_cleared_for_a_later_unverified_request(self):
        provider, _ = _provider()
        claims = _claims()
        assert provider.get_subject(_headers(_token(claims)))
        assert credential_expires_at.get() == float(claims["exp"])
        assert provider.get_subject(_headers(_token(key=_OTHER_RSA))) == {}
        assert credential_expires_at.get() is None
        assert provider.get_subject({}) == {}
        assert credential_expires_at.get() is None

    def test_untrusted_issuer_is_denied_and_keys_only_from_trusted(self):
        provider, keys = _provider()
        assert provider.get_subject(_headers(_token(_claims(iss="https://evil.example")))) == {}
        assert keys.fetched == [ISSUER]

    def test_for_audience_keeps_extra_subject(self):
        extra = lambda c: {"x": 1}
        provider, _ = _provider(extra_subject=extra)
        assert provider.for_audience("https://other.example").extra_subject is extra

    def test_verified_claims_cached_per_token(self):
        provider, keys = _provider()
        token = _token()
        for _ in range(3):
            provider.get_subject(_headers(token))
            provider.get_environment(_headers(token))
        assert len(keys.fetched) == 1
        provider.get_subject(_headers(_token(_claims(sub="bob"))))
        assert len(keys.fetched) == 2

    def test_sets_credential_expiry(self):
        provider, _ = _provider()
        exp = int(time.time()) + 120

        def run():
            provider.get_subject(_headers(_token(_claims(exp=exp))))
            return credential_expires_at.get()

        assert contextvars.copy_context().run(run) == float(exp)

    def test_failed_verification_leaves_expiry_unset(self):
        provider, _ = _provider()

        def run():
            provider.get_subject(_headers(_token(key=_OTHER_RSA)))
            return credential_expires_at.get()

        assert contextvars.copy_context().run(run) is None

    def test_expired_since_cached(self):
        provider, keys = _provider(_verification(leeway_seconds=0))
        token = _token()
        provider._verified[hashlib.sha256(token.encode()).hexdigest()] = _claims(exp=int(time.time()) - 1)
        assert provider.get_subject(_headers(token)) == {}
        assert keys.fetched == []

    def test_for_audience(self):
        other = "https://other.example"
        provider, _ = _provider(
            claims={"tenant": "tenant_id"}, rar=RarMapping(type="payment_initiation"),
            attributes=AgentIdentityAttributes(user_id="uid"),
        )
        copy = provider.for_audience(other)
        copy._keys = StaticKeys(_RSA.public_key())
        assert isinstance(copy, AgentIdentityContextProvider)
        assert copy.verification.audiences == [other]
        assert copy.claims == provider.claims and copy.rar is provider.rar
        assert copy.attributes is provider.attributes

        token = _token(_claims(aud=other, tenant="t1"))
        assert copy.get_subject(_headers(token))["tenant_id"] == "t1"
        assert copy.get_subject(_headers(token))["uid"] == "alice"
        assert provider.get_subject(_headers(token)) == {}  # wrong audience for the original

    def test_for_audience_keeps_subclass(self):
        class Provider(AgentIdentityContextProvider):
            pass

        assert type(Provider(_verification()).for_audience("x")) is Provider


# ── Where the claims come from: token_from / claims_from (client paths, platforms) ─────

class TestTokenFrom:
    def test_token_from_runtime_context(self):
        provider, _ = _provider(token_from=lambda ctx: ctx.get("access_token"))
        assert provider.get_subject({"access_token": _token()})["user_id"] == "alice"

    def test_token_from_object_context(self):
        class Ctx:
            access_token = _token()

        provider, _ = _provider(token_from=lambda ctx: ctx.access_token)
        assert provider.get_subject(Ctx())["agent_id"] == "agent-1"

    def test_no_token_found_is_empty_without_verifying(self):
        provider, keys = _provider(token_from=lambda ctx: None)
        assert provider.get_subject({"authorization": f"Bearer {_token()}"}) == {}
        assert keys.fetched == []

    def test_default_still_reads_the_authorization_header(self):
        provider, _ = _provider()
        assert provider.get_subject(_headers(_token()))["user_id"] == "alice"

    def test_for_audience_keeps_token_from(self):
        token_from = lambda ctx: ctx["t"]
        provider, _ = _provider(token_from=token_from)
        assert provider.for_audience("https://other.example").token_from is token_from


class TestClaimsFrom:
    def _provider(self, **kw):
        return AgentIdentityContextProvider(claims_from=lambda ctx: ctx.get("claims"), **kw)

    def test_maps_upstream_claims_without_verifying(self):
        provider = self._provider()
        assert provider._keys is None
        subject = provider.get_subject({"claims": _claims(scope="a b")})
        assert subject == {"user_id": "alice", "agent_id": "agent-1", "scopes": ["a", "b"]}

    def test_expiry_is_honoured(self):
        provider = self._provider()
        claims = _claims()
        assert provider.get_subject({"claims": claims})
        assert credential_expires_at.get() == float(claims["exp"])
        assert provider.get_subject({"claims": _claims(exp=int(time.time()) - 1)}) == {}
        assert credential_expires_at.get() is None

    def test_claims_without_exp_are_used_and_leave_expiry_unset(self):
        provider = self._provider()
        assert provider.get_subject({"claims": _claims(exp=None)})["user_id"] == "alice"
        assert credential_expires_at.get() is None

    @pytest.mark.parametrize("value", [None, {}, "not-a-dict", ["x"]])
    def test_nothing_usable_is_empty(self, value):
        assert self._provider().get_subject({"claims": value}) == {}

    def test_environment_and_rar_with_explicit_location(self):
        provider = self._provider(rar=RarMapping(type="payment_initiation", fields=_PAYMENT_FIELDS, location=AUDIENCE))
        context = {"claims": _claims(txn="tx-1", authorization_details=[_payment()])}
        assert provider.get_environment(context) == {"transaction_id": "tx-1"}
        assert provider.get_subject(context)["approved_amount"] == 123.5

    def test_logs_that_tokens_are_not_verified_here(self, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            self._provider()
        assert any("not verified here" in r.getMessage() for r in caplog.records)

    def test_for_audience_is_not_available(self):
        with pytest.raises(ValueError, match="for_audience"):
            self._provider().for_audience("x")


class TestProviderSourceValidation:
    def test_needs_verification_or_claims_from(self):
        with pytest.raises(ValueError, match="exactly one"):
            AgentIdentityContextProvider()

    def test_not_both(self):
        with pytest.raises(ValueError, match="exactly one"):
            AgentIdentityContextProvider(_verification(), claims_from=lambda ctx: {})

    def test_token_from_needs_verification(self):
        with pytest.raises(ValueError, match="token_from"):
            AgentIdentityContextProvider(claims_from=lambda ctx: {}, token_from=lambda ctx: "t")

    def test_rar_needs_location_with_claims_from(self):
        with pytest.raises(ValueError, match="location"):
            AgentIdentityContextProvider(claims_from=lambda ctx: {}, rar=RarMapping(type="payment_initiation"))
