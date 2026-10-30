"""ClientIdMetadata: CIMD client_id URLs to subject attributes, and the guarded document fetch."""

from __future__ import annotations

import json
import socket
import time

import httpx
import pytest

from axiomatics_ai import AgentIdentityContextProvider, ClientIdMetadata, JwtVerification
from axiomatics_ai.core import client_metadata
from axiomatics_ai.core.client_metadata import _cimd_host

CLIENT = "https://app.example.com/oauth/client.json"
DOCUMENT = {"client_id": CLIENT, "client_name": "Example App", "token_endpoint_auth_method": "private_key_jwt",
            "redirect_uris": ["https://app.example.com/cb"], "jwks": {"keys": []}}


@pytest.fixture
def network(monkeypatch):
    """Fake DNS and HTTP: ``network.addresses`` resolve the host, ``network.respond`` answers."""
    state = type("Network", (), {})()
    state.addresses = ["93.184.216.34"]
    state.respond = lambda request: httpx.Response(200, json=DOCUMENT)
    state.requests = []
    state.lookups = 0

    def getaddrinfo(host, port, type=0):
        state.lookups += 1
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in state.addresses]

    def handler(request):
        state.requests.append(request)
        response = state.respond(request)   # unread, as off the network: the fetch streams the raw bytes
        return httpx.Response(response.status_code, headers=response.headers, stream=httpx.ByteStream(response.content))

    real_client = httpx.Client
    monkeypatch.setattr(client_metadata.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(client_metadata.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    return state


@pytest.mark.parametrize(("client_id", "host"), [
    (CLIENT, "app.example.com"),
    ("https://App.Example.com:8443/c", "app.example.com"),
    ("https://app.example.com", None),             # no path
    ("https://app.example.com/", None),
    ("http://app.example.com/c", None),            # not https
    ("https://app.example.com/a/../c", None),      # dot segment
    ("https://u:p@app.example.com/c", None),       # credentials
    ("https://app.example.com/c#x", None),         # fragment
    ("https://app.example.com:99999/c", None),     # bad port
    ("https://@app.example.com/c", None),          # empty credentials
    ("https://app.example.com/a/%2e%2e/c", None),  # encoded dot segment
    ("my-client", None),
    (None, None),
])
def test_cimd_host(client_id, host):
    assert _cimd_host(client_id) == host


def test_host_only_by_default_without_network(network):
    assert ClientIdMetadata().attributes(CLIENT) == {"client_host": "app.example.com"}
    assert ClientIdMetadata().attributes("my-client") == {}
    assert ClientIdMetadata(host_attribute=None).attributes(CLIENT) == {}
    assert network.lookups == 0 and network.requests == []


def test_fetch_maps_fields_and_caches(network):
    metadata = ClientIdMetadata(fetch=True)
    expected = {"client_host": "app.example.com", "client_name": "Example App", "client_auth_method": "private_key_jwt"}
    assert metadata.attributes(CLIENT) == expected
    assert metadata.attributes(CLIENT) == expected
    assert len(network.requests) == 1


def test_request_goes_to_the_checked_address_with_tls_for_the_host(network):
    ClientIdMetadata(fetch=True).attributes("https://app.example.com:8443/oauth/client.json?v=1")
    (request,) = network.requests
    assert str(request.url) == "https://93.184.216.34:8443/oauth/client.json?v=1"
    assert request.headers["host"] == "app.example.com:8443"
    assert request.extensions["sni_hostname"] == "app.example.com"


def test_custom_fields(network):
    metadata = ClientIdMetadata(fetch=True, fields={"redirect_uris": "redirect_uris", "jwks": "jwks"})
    assert metadata.attributes(CLIENT)["redirect_uris"] == ["https://app.example.com/cb"]


@pytest.mark.parametrize("addresses", [["10.0.0.5"], ["127.0.0.1"], ["169.254.169.254"], ["::1"],
                                       ["93.184.216.34", "192.168.1.1"], ["224.0.0.1"],
                                       ["64:ff9b::a9fe:a9fe"], ["::7f00:1"]])   # NAT64, IPv4-compatible
def test_non_public_address_is_never_fetched(network, addresses, caplog):
    network.addresses = addresses
    assert ClientIdMetadata(fetch=True).attributes(CLIENT) == {"client_host": "app.example.com"}
    assert network.requests == []
    assert "non-public address" in caplog.text


@pytest.mark.parametrize("response", [
    httpx.Response(302, headers={"location": "https://10.0.0.1/"}),
    httpx.Response(404),
    httpx.Response(200, content=b"<html>"),
    httpx.Response(200, json=[DOCUMENT]),
    httpx.Response(200, json={**DOCUMENT, "client_id": "https://other.example.com/c.json"}),
    httpx.Response(200, content=json.dumps({**DOCUMENT, "pad": "x" * 6000}).encode()),
    httpx.Response(200, headers={"content-encoding": "gzip"}, content=b"\x1f\x8b"),   # compressed: never expanded
    httpx.Response(200, content=b"[" * 1100 + b"]" * 1100),                          # too deep for json on 3.11
])
def test_unusable_document_adds_no_fields_and_is_not_retried_at_once(network, response):
    network.respond = lambda request: response
    metadata = ClientIdMetadata(fetch=True)
    assert metadata.attributes(CLIENT) == {"client_host": "app.example.com"}
    assert metadata.attributes(CLIENT) == {"client_host": "app.example.com"}
    assert len(network.requests) == 1


def test_a_slow_server_never_holds_the_request_past_the_timeout(network, monkeypatch):
    monkeypatch.setattr(client_metadata, "_fetch_document", lambda url, timeout: time.sleep(5))
    started = time.monotonic()
    assert ClientIdMetadata(fetch=True, timeout=0.2).attributes(CLIENT) == {"client_host": "app.example.com"}
    assert time.monotonic() - started < 1


def test_allowed_hosts(network):
    assert ClientIdMetadata(fetch=True, allowed_hosts=("*.example.com",)).attributes(CLIENT)["client_name"] == "Example App"
    assert "client_name" not in ClientIdMetadata(fetch=True, allowed_hosts=("other.com",)).attributes(CLIENT)
    assert len(network.requests) == 1


class TestProvider:
    def test_attributes_join_the_subject(self, network):
        provider = AgentIdentityContextProvider(claims_from=lambda ctx: ctx, client_metadata=ClientIdMetadata(fetch=True))
        subject = provider.get_subject({"sub": "alice", "client_id": CLIENT})
        assert subject["agent_id"] == CLIENT
        assert (subject["client_host"], subject["client_name"]) == ("app.example.com", "Example App")

    def test_unsendable_fields_are_dropped_like_other_attributes(self, network):
        metadata = ClientIdMetadata(fetch=True, fields={"jwks": "client_jwks"})
        preview = AgentIdentityContextProvider(claims_from=lambda ctx: ctx, client_metadata=metadata).preview(
            {"sub": "alice", "client_id": CLIENT},
        )
        assert "client_jwks" not in preview.subject and "client_jwks" in preview.dropped

    def test_opaque_client_id_unchanged_and_for_audience_keeps_it(self):
        metadata = ClientIdMetadata()
        provider = AgentIdentityContextProvider(
            JwtVerification(issuer="https://idp.example.com", audience="a"), client_metadata=metadata,
        )
        assert provider.subject_from_claims({"sub": "alice", "client_id": "my-client"}) == {
            "user_id": "alice", "agent_id": "my-client",
        }
        assert provider.for_audience("b").client_metadata is metadata
