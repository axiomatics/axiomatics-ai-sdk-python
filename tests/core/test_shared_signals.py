"""SsfReceiver: verifying pushed Security Event Tokens and dropping the subject's cached decisions."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from axiomatics_ai import AuditLogHook, AuthZConfig, PDPConnection, SsfReceiver
from axiomatics_ai.core.shared_signals import SignalRejected

ISSUER = "https://idp.example.com"
AUDIENCE = "https://mcp.example.com/ssf"
CAEP = "https://schemas.openid.net/secevent/caep/event-type/"
VERIFICATION = "https://schemas.openid.net/secevent/ssf/event-type/verification"
KEY = ec.generate_private_key(ec.SECP256R1())


def _set(events=None, *, key=KEY, typ="secevent+jwt", kid="k1", **claims) -> str:
    payload = {
        "iss": ISSUER, "aud": AUDIENCE, "iat": int(time.time()), "jti": "set-1",
        "sub_id": {"format": "iss_sub", "iss": ISSUER, "sub": "alice"},
        "events": events if events is not None else {CAEP + "session-revoked": {"event_timestamp": 1}},
    }
    payload.update(claims)
    headers = {"typ": typ, **({"kid": kid} if kid else {})}
    return jwt.encode({k: v for k, v in payload.items() if v is not None}, key, algorithm="ES256", headers=headers)


def _receiver(**kwargs) -> tuple[SsfReceiver, list[dict]]:
    """A receiver whose key lookup returns KEY's public key, and the remediation records it causes."""
    records: list[dict] = []
    config = AuthZConfig(connection=PDPConnection(base_url="http://pdp"), hooks=[AuditLogHook(records.append)])
    receiver = SsfReceiver(config, issuer=ISSUER, audience=AUDIENCE, jwks_url="https://idp.example.com/jwks", **kwargs)
    other = SimpleNamespace(key=ec.generate_private_key(ec.SECP256R1()).public_key(), public_key_use="sig")
    mine = SimpleNamespace(key=KEY.public_key(), public_key_use="sig")
    receiver._keys = SimpleNamespace(get_signing_key_from_jwt=lambda token: mine,
                                     get_jwk_set=lambda: SimpleNamespace(keys=[other, mine]))
    return receiver, records


class TestHandle:
    def test_session_revoked_drops_the_users_decisions_and_is_audited(self):
        receiver, records = _receiver()
        receiver.handle(_set())
        (record,) = records
        assert record["event"] == "remediation"
        assert record["subject"] == {"user_id": "alice"}
        assert record["reason"] == "session-revoked (SET set-1)"

    @pytest.mark.parametrize(("sub_id", "expected"), [
        ({"format": "email", "email": "alice@example.com"}, [{}]),   # no default mapping: drops everything
        ({"format": "opaque", "id": "u-1"}, [{"user_id": "u-1"}]),
        ({"format": "complex", "user": {"format": "iss_sub", "iss": ISSUER, "sub": "bob"}}, [{"user_id": "bob"}]),
        ({"format": "aliases", "identifiers": [{"format": "email", "email": "c@example.com"},
                                               {"format": "iss_sub", "iss": ISSUER, "sub": "carol"}]}, [{"user_id": "carol"}]),
        ({"format": ["iss_sub"], "sub": "x"}, [{}]),                  # malformed: no crash, drops everything
    ])
    def test_subject_formats(self, sub_id, expected):
        receiver, records = _receiver()
        receiver.handle(_set(sub_id=sub_id))
        assert [r["subject"] for r in records] == expected

    def test_aliases_drop_every_matched_identifier(self):
        receiver, records = _receiver(subject_attributes={"email": "email", "iss_sub": "user_id"})
        receiver.handle(_set(sub_id={"format": "aliases", "identifiers": [
            {"format": "email", "email": "c@example.com"}, {"format": "iss_sub", "iss": ISSUER, "sub": "carol"}]}))
        assert [r["subject"] for r in records] == [{"email": "c@example.com"}, {"user_id": "carol"}]

    def test_clock_skew_typ_variant_and_kid_less_token_are_accepted(self):
        receiver, records = _receiver()
        receiver.handle(_set(iat=int(time.time()) + 30))
        receiver.handle(_set(typ="application/secevent+jwt"))
        receiver.handle(_set(kid=None))
        assert len(records) == 3

    def test_legacy_subject_inside_the_event(self):
        receiver, records = _receiver()
        receiver.handle(_set(sub_id=None, events={CAEP + "credential-change": {"subject": {"format": "opaque", "id": "u-2"}}}))
        assert records[0]["subject"] == {"user_id": "u-2"}

    def test_unmatched_subject_drops_everything(self):
        receiver, records = _receiver()
        receiver.handle(_set(sub_id={"format": "did", "url": "did:example:1"}))
        assert records[0]["subject"] == {}

    def test_custom_subject_attributes(self):
        receiver, records = _receiver(subject_attributes={"email": "mail"})
        receiver.handle(_set(sub_id={"format": "email", "email": "a@example.com"}))
        assert records[0]["subject"] == {"mail": "a@example.com"}

    def test_verification_event_is_acknowledged_only(self):
        receiver, records = _receiver()
        receiver.handle(_set(sub_id={"format": "opaque", "id": "stream-1"}, events={VERIFICATION: {"state": "x"}}))
        assert records == []

    def test_on_event_gets_each_event_and_its_failure_is_contained(self):
        seen = []

        def on_event(event_type, subject, claims):
            seen.append((event_type, subject, claims["jti"]))
            raise RuntimeError("callback bug")

        receiver, records = _receiver(on_event=on_event)
        receiver.handle(_set(events={CAEP + "session-revoked": {}, CAEP + "risk-level-change": {}}))
        assert [s[0].rsplit("/", 1)[-1] for s in seen] == ["session-revoked", "risk-level-change"]
        assert len(records) == 2

    @pytest.mark.parametrize(("token", "error"), [
        (lambda: _set(key=ec.generate_private_key(ec.SECP256R1())), "invalid_key"),
        (lambda: _set(iss="https://evil.example.com"), "invalid_issuer"),
        (lambda: _set(aud="https://other.example.com"), "invalid_audience"),
        (lambda: _set(typ="JWT"), "invalid_request"),
        (lambda: _set(jti=None), "invalid_request"),
        (lambda: _set(events={}), "invalid_request"),
        (lambda: _set(iat=int(time.time()) + 3600), "invalid_request"),
        (lambda: _set(kid=None, key=ec.generate_private_key(ec.SECP256R1())), "invalid_key"),
        (lambda: "not-a-jwt", "invalid_request"),
    ])
    def test_refused_sets(self, token, error):
        receiver, records = _receiver()
        with pytest.raises(SignalRejected) as exc:
            receiver.handle(token())
        assert exc.value.error == error and exc.value.status == 400
        assert records == []

    def test_authorization_header_is_required_when_configured(self):
        receiver, records = _receiver(authorization="Bearer stream-secret")
        with pytest.raises(SignalRejected) as exc:
            receiver.handle(_set(), authorization="Bearer wrong")
        assert (exc.value.error, exc.value.status) == ("authentication_failed", 401)
        receiver.handle(_set(), authorization="Bearer stream-secret")
        assert len(records) == 1

    def test_needs_issuer_and_audience_and_refuses_reason_attribute(self):
        config = AuthZConfig(connection=PDPConnection(base_url="http://pdp"))
        with pytest.raises(ValueError):
            SsfReceiver(config, issuer="", audience=AUDIENCE)
        with pytest.raises(ValueError, match="reason"):
            SsfReceiver(config, issuer=ISSUER, audience=AUDIENCE, subject_attributes={"opaque": "reason"})


class TestDiscovery:
    def _receiver(self, issuer: str) -> SsfReceiver:
        return SsfReceiver(AuthZConfig(connection=PDPConnection(base_url="http://pdp")), issuer=issuer, audience=AUDIENCE)

    def test_jwks_uri_from_ssf_configuration(self, monkeypatch):
        fetched = []

        def fake_get(url, timeout):
            fetched.append(url)
            return httpx.Response(200, json={"jwks_uri": "https://idp.example.com/keys"})

        monkeypatch.setattr(httpx, "get", fake_get)
        assert self._receiver("https://idp.example.com/tenant-1/")._discover_jwks_url() == "https://idp.example.com/keys"
        assert fetched == ["https://idp.example.com/.well-known/ssf-configuration/tenant-1"]

    @pytest.mark.parametrize("document", [{}, {"jwks_uri": "http://idp.example.com/keys"}, ["not", "a", "dict"]])
    def test_missing_or_insecure_jwks_uri_is_refused(self, monkeypatch, document):
        monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(200, json=document))
        with pytest.raises(ValueError, match="jwks_uri"):
            self._receiver(ISSUER)._discover_jwks_url()

    def test_failed_key_fetch_is_not_retried_per_request(self, monkeypatch):
        fetches = []

        def down(url, timeout):
            fetches.append(url)
            raise httpx.ConnectError("down")

        monkeypatch.setattr(httpx, "get", down)
        receiver = self._receiver(ISSUER)
        for _ in range(5):
            with pytest.raises(SignalRejected) as exc:
                receiver.handle(_set())
            assert (exc.value.error, exc.value.description) == ("invalid_key", "transmitter keys unavailable")
        assert len(fetches) == 1


def _asgi(receiver, *, method="POST", body=b"", content_type="application/secevent+jwt", authorization=None):
    headers = [(b"content-type", content_type.encode())]
    if authorization:
        headers.append((b"authorization", authorization.encode()))
    scope = {"type": "http", "method": method, "headers": headers}
    sent: list[dict] = []
    chunks = [{"type": "http.request", "body": body[:10], "more_body": True},
              {"type": "http.request", "body": body[10:], "more_body": False}]

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(receiver(scope, receive, send))
    status = sent[0]["status"]
    payload = sent[1]["body"]
    return status, json.loads(payload) if payload else None


class TestAsgi:
    def test_accepted_set_is_202(self):
        receiver, records = _receiver()
        assert _asgi(receiver, body=_set().encode()) == (202, None)
        assert len(records) == 1

    def test_refused_set_is_400_with_rfc8935_error(self):
        receiver, _ = _receiver()
        status, body = _asgi(receiver, body=_set(aud="https://other.example.com").encode())
        assert status == 400 and body["err"] == "invalid_audience"

    def test_method_content_type_and_size(self):
        receiver, _ = _receiver()
        assert _asgi(receiver, method="GET")[0] == 405
        assert _asgi(receiver, body=_set().encode(), content_type="application/json")[1]["err"] == "invalid_request"
        assert _asgi(receiver, body=b"x" * (65 * 1024))[1]["description"] == "body too large"

    def test_wrong_authorization_is_401(self):
        receiver, _ = _receiver(authorization="Bearer s")
        assert _asgi(receiver, body=_set().encode(), authorization="Bearer x")[0] == 401
