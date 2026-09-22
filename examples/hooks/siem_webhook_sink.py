"""
Example: send the SDK's audit trail to a SIEM over HTTP.

`AuditLogHook` produces one structured record per decision and per cache invalidation (AIMS
§11 fields: agent, delegated user, actor, tool, action, decision, cached, timestamp,
correlation id, environment). Its `sink` can be any callable — here a `SiemWebhookSink` that
POSTs the records to a collector URL (Splunk HEC, Elastic, Datadog, a Logstash HTTP input...).

Delivery is the sink's job, not the enforcement path's, so the sink never blocks a decision:
records go on a bounded queue and a background thread sends them in batches, retrying with
backoff; if the queue is full (the SIEM is down for a long time) the oldest records are
dropped and counted. For guaranteed delivery, point the default logger sink at a durable log
pipeline instead (`AuditLogHook()` logs JSON lines on "axiomatics_ai.audit").

Set SIEM_URL (and optionally SIEM_TOKEN, sent as a Bearer token) to send to your SIEM. If
SIEM_URL is not set, this script starts a LOCAL COLLECTOR on localhost that just prints what
arrives, so you can see the batches. The decisions themselves are real ADS decisions.

Run with ADS at http://localhost:8080 and domain YAML loaded:
  examples/policies/domain_all_authz_enabled.yaml
"""
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from axiomatics_ai import AuditLogHook, AuthZConfig, Authorizer, PDPConnection


class SiemWebhookSink:
    """A non-blocking audit sink: batches records and POSTs them as a JSON array."""

    def __init__(self, url: str, *, token: str | None = None, batch_size: int = 50,
                 flush_seconds: float = 2.0, max_queue: int = 10_000, retries: int = 3) -> None:
        self.url, self.batch_size, self.flush_seconds, self.retries = url, batch_size, flush_seconds, retries
        self.dropped = 0
        self._headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def __call__(self, record: dict) -> None:
        """Called by AuditLogHook on the enforcement path: enqueue and return at once."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop the oldest to make room
                self.dropped += 1
            except queue.Empty:
                pass
            self._queue.put_nowait(record)

    def close(self) -> None:
        """Send what is still queued, then stop."""
        self._stop.set()
        self._thread.join(timeout=self.flush_seconds * 2 + 5)

    def _run(self) -> None:
        with httpx.Client(timeout=5.0) as client:
            while not (self._stop.is_set() and self._queue.empty()):
                batch = self._next_batch()
                if batch:
                    self._send(client, batch)

    def _next_batch(self) -> list[dict]:
        batch, deadline = [], time.monotonic() + self.flush_seconds
        while len(batch) < self.batch_size:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                break
        return batch

    def _send(self, client: httpx.Client, batch: list[dict]) -> None:
        for attempt in range(self.retries + 1):
            try:
                if client.post(self.url, content=json.dumps(batch, default=str), headers=self._headers).is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(min(2 ** attempt, 10))
        self.dropped += len(batch)


def start_local_collector() -> str:
    """A stand-in for a SIEM's HTTP input: prints each batch it receives."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            batch = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            print(f"[local collector] received {len(batch)} record(s):")
            for record in batch:
                who = record.get("subject", {}).get("username")
                print(f"   {record['event']:<11} {record.get('enforcement_point') or record.get('remediation')} "
                      f"user={who} decision={record.get('decision', '-')} cached={record.get('cached', '-')} "
                      f"hash={record['hash'][:12]}…")
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/audit"


def main() -> None:
    url = os.getenv("SIEM_URL") or start_local_collector()
    print(f"Sending audit records to {url}" + ("" if os.getenv("SIEM_URL") else "  (local collector: set SIEM_URL for a real SIEM)"))
    sink = SiemWebhookSink(url, token=os.getenv("SIEM_TOKEN"))
    config = AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        hooks=[AuditLogHook(sink=sink, hash_chain=True)],
    )
    authz = Authorizer(config)

    for user in ("alice", "bob"):
        for account in ("ACC-001", "ACC-999"):
            authz.is_permitted(action="tool_invocation", resource="get_account_balance",
                               resource_name_attribute="tool_name", args={"account_id": account},
                               context={"username": user})
    authz.is_permitted(action="tool_invocation", resource="get_account_balance",   # a cache hit
                       resource_name_attribute="tool_name", args={"account_id": "ACC-001"},
                       context={"username": "bob"})
    # A revocation or risk signal for bob: his cached decisions go, and a remediation is logged.
    removed = config.invalidate_decisions(reason="risk signal: session revoked", username="bob")
    print(f"invalidated {removed} cached decision(s) for bob")

    sink.close()
    if sink.dropped:
        print(f"{sink.dropped} record(s) could not be delivered")


if __name__ == "__main__":
    main()
