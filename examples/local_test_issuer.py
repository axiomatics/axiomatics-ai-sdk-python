"""
A LOCAL TEST ISSUER for the verified-identity examples: a stand-in for your identity provider
(Okta, Entra, Keycloak, Auth0...) so the examples run without an account.

It generates an RSA key, serves an OIDC discovery document and its JWKS on localhost, and signs
RFC 9068 access tokens (``typ: at+jwt``). A real identity provider does the same after the user
logs in and consents; here the token is simply minted. Everything that consumes the token — the
SDK's verification (keys fetched over HTTP through discovery), the claim mapping, and ADS's
decisions — is real.

Each run uses a fresh key id, so a long-running server fetches the new key instead of trusting
a cached one.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa


class LocalTestIssuer:
    """Discovery document, JWKS and signed access tokens on ``http://127.0.0.1:<port>``."""

    def __init__(self, port: int = 0) -> None:
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._kid = uuid.uuid4().hex
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self._key.public_key()))
        jwks = {"keys": [{**jwk, "kid": self._kid, "use": "sig", "alg": "RS256"}]}
        issuer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = {
                    "/.well-known/openid-configuration": {"issuer": issuer.url, "jwks_uri": issuer.url + "/jwks.json"},
                    "/jwks.json": jwks,
                }.get(self.path)
                self.send_response(200 if body else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body or {}).encode())

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def access_token(self, *, audience: str, lifetime: int = 300, typ: str = "at+jwt", **claims) -> str:
        """Sign an access token for ``audience`` carrying ``claims`` (sub, client_id, scope...)."""
        now = int(time.time())
        payload = {"iss": self.url, "aud": audience, "iat": now, "exp": now + lifetime,
                   "jti": uuid.uuid4().hex, **claims}
        return jwt.encode(payload, self._key, algorithm="RS256", headers={"typ": typ, "kid": self._kid})

    def close(self) -> None:
        self._server.shutdown()
