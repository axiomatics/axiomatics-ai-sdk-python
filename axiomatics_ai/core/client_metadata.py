"""Policy attributes from OAuth Client ID Metadata Documents (CIMD).

With CIMD (draft-ietf-oauth-client-id-metadata-document) a client's ``client_id`` is an HTTPS URL,
and the document at that URL describes the client: its name, how it authenticates, its redirect
URIs. MCP recommends it for client registration. The authorization server fetches the document
when it issues the token; by the time a token reaches this SDK, its ``client_id`` is that URL.

``ClientIdMetadata`` turns such a ``client_id`` into subject attributes: the client's host (no
network), and, with ``fetch=True``, chosen fields of the document. The fetch follows the draft's
security rules: HTTPS only, no redirects, never an address that isn't public (SSRF), at most 5 KB.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import threading
import time
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_MAX_DOCUMENT_BYTES = 5 * 1024
_FAILURE_SECONDS = 60
# Fetches in flight at once. A slow server holds one until its timeout; the rest of the app's
# threads never wait on a fetch for longer than ``timeout``.
_FETCH_SLOTS = threading.BoundedSemaphore(4)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class ClientIdMetadata:
    """Send a CIMD client's host, and optionally fields of its metadata document, as attributes.

    Pass it to ``AgentIdentityContextProvider(client_metadata=...)``. Only a ``client_id`` that is
    a valid CIMD URL (``https``, a path, no fragment, credentials or dot segments) gets
    attributes; any other ``client_id`` is left as it is.

    Args:
        fetch (bool): Fetch the document and map ``fields``. Off by default: the host needs no
            network.
        fields (dict[str, str], optional): Document field → subject attribute. Default:
            ``client_name`` → ``client_name``, ``token_endpoint_auth_method`` →
            ``client_auth_method``.
        host_attribute (str | None): The subject attribute for the client's host; None to not
            send it.
        allowed_hosts (tuple[str, ...]): Fetch only from these hosts (``example.com`` or
            ``*.example.com``); empty fetches from any public host.
        cache_seconds (float): How long a fetched document is reused.
        timeout (float): Overall time allowed for one fetch, in seconds.
    """

    def __init__(
        self,
        *,
        fetch: bool = False,
        fields: dict[str, str] | None = None,
        host_attribute: str | None = "client_host",
        allowed_hosts: tuple[str, ...] = (),
        cache_seconds: float = 300,
        timeout: float = 2.0,
    ) -> None:
        self.fetch = fetch
        self.fields = fields or {"client_name": "client_name", "token_endpoint_auth_method": "client_auth_method"}
        self.host_attribute = host_attribute
        self.allowed_hosts = tuple(h.lower() for h in allowed_hosts)
        self.timeout = timeout
        self._documents: TTLCache = TTLCache(maxsize=1024, ttl=cache_seconds)
        self._failures: TTLCache = TTLCache(maxsize=1024, ttl=_FAILURE_SECONDS)
        self._lock = threading.Lock()

    def attributes(self, client_id: Any) -> dict[str, Any]:
        """The subject attributes for this ``client_id``.

        Args:
            client_id (Any): The token's ``client_id``.

        Returns:
            dict[str, Any]: The attributes; empty when ``client_id`` isn't a CIMD URL. A document
            that can't be fetched or doesn't match adds none of its fields (logged).
        """
        host = _cimd_host(client_id)
        if host is None:
            return {}
        attributes: dict[str, Any] = {self.host_attribute: host} if self.host_attribute else {}
        if self.fetch and self._host_allowed(host):
            document = self._document(client_id)
            for field, attribute in self.fields.items():
                if document.get(field) is not None:
                    attributes[attribute] = document[field]
        return attributes

    def _host_allowed(self, host: str) -> bool:
        if not self.allowed_hosts:
            return True
        return any(host == h or (h.startswith("*.") and host.endswith(h[1:])) for h in self.allowed_hosts)

    def _document(self, url: str) -> dict:
        """The verified document at ``url``, cached; empty when it can't be had."""
        with self._lock:
            if url in self._documents:
                return self._documents[url]
            if url in self._failures:
                return {}
        try:
            document = _fetch_within(url, self.timeout)
        except Exception as exc:  # noqa: BLE001
            # Whatever the server sends, the request goes on without the document's fields.
            logger.warning("Client metadata document %s not used: %s", url, exc)
            with self._lock:
                self._failures[url] = True
            return {}
        with self._lock:
            self._documents[url] = document
        return document


def _cimd_host(client_id: Any) -> str | None:
    """The host of a valid CIMD ``client_id`` URL, or None."""
    if not isinstance(client_id, str) or not client_id.startswith("https://"):
        return None
    try:
        parts = urlsplit(client_id)
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        return None
    segments = {unquote(segment) for segment in parts.path.split("/")}
    if (not parts.hostname or parts.path in ("", "/") or parts.fragment or "@" in parts.netloc
            or "." in segments or ".." in segments):
        return None
    return parts.hostname.lower()


def _fetch_within(url: str, timeout: float) -> dict:
    """``_fetch_document`` with a hard wall-clock limit: it runs in a daemon thread, abandoned on timeout."""
    if not _FETCH_SLOTS.acquire(blocking=False):
        raise TimeoutError("too many client metadata fetches in progress")
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["document"] = _fetch_document(url, timeout)
        except Exception as exc:  # noqa: BLE001
            # Handed to the waiting caller; anything not caught here still reads as a failed fetch.
            outcome["error"] = exc
        finally:
            _FETCH_SLOTS.release()

    thread = threading.Thread(target=run, name="cimd-fetch", daemon=True)
    thread.start()
    thread.join(timeout)
    if "document" in outcome:
        return outcome["document"]
    raise outcome.get("error") or TimeoutError(f"no document within {timeout}s")


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 6 and (ip in _NAT64 or int(ip) >> 32 == 0):   # embeds an IPv4 address
        return _is_public(ipaddress.ip_address(int(ip) & 0xFFFFFFFF))
    return ip.is_global and not ip.is_multicast


def _fetch_document(url: str, timeout: float) -> dict:
    """GET the document from a checked public address, without redirects, proxies or more than 5 KB.

    The host is resolved here and every address must be public; the request then goes to that
    address (TLS still verified for the host name), so DNS can't point it elsewhere in between.
    """
    deadline = time.monotonic() + timeout
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port or 443
    addresses = {info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    for address in addresses:
        if not _is_public(ipaddress.ip_address(address)):
            raise ValueError(f"{host} resolves to non-public address {address}")
    address = sorted(addresses)[0]
    target = f"https://{f'[{address}]' if ':' in address else address}:{port}{parts.path}"
    if parts.query:
        target += f"?{parts.query}"
    headers = {"Host": parts.netloc, "Accept": "application/json", "Accept-Encoding": "identity"}
    with (
        httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client,
        client.stream("GET", target, headers=headers, extensions={"sni_hostname": host}) as response,
    ):
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")
        if response.headers.get("content-encoding", "identity") != "identity":
            raise ValueError("compressed document")   # its size can't be capped before it is expanded
        body = b""
        for chunk in response.iter_raw():
            body += chunk
            if len(body) > _MAX_DOCUMENT_BYTES:
                raise ValueError("document larger than 5 KB")
            if time.monotonic() > deadline:
                raise ValueError("fetch took too long")
    document = json.loads(body)
    if not isinstance(document, dict) or document.get("client_id") != url:
        raise ValueError("the document's client_id is not its URL")
    return document
