"""
Shared pytest fixtures for ADS integration tests.

Fixtures here are available to all test subdirectories (langchain_agents/, mcp/, ...).
Integration tests run automatically as part of the normal test suite as long as
the ADS jar is present in examples/policies/. They are skipped only when the jar is
missing. Use -m ads_integration to run ONLY the integration tests.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from axiomatics_ai.core.pdp import PdpClient

REPO_ROOT = Path(__file__).resolve().parent.parent
ADS_DIR = REPO_ROOT / "examples" / "policies"
DEPLOYMENT_YAML = ADS_DIR / "deployment.yaml"
ADS_URL = "http://localhost:8080"
ADS_CAQ_LICENSE = ADS_DIR / "axiomatics_ADS_CAQ.license"
ADS_LICENSE = ADS_DIR / "axiomatics_ADS.license"


# ── ADS lifecycle helpers ──────────────────────────────────────────────────────

def _find_jar() -> Path | None:
    """Return the newest ADS jar by semantic version."""
    jars = list(ADS_DIR.glob("access-decision-service-*.jar"))
    if not jars:
        return None

    def version_key(path: Path) -> tuple[int, ...]:
        match = re.search(r"access-decision-service-(\d+(?:\.\d+)*)\.jar$", path.name)
        if not match:
            return ()
        return tuple(int(part) for part in match.group(1).split("."))

    return max(jars, key=version_key)


def _find_license() -> Path:
    return ADS_CAQ_LICENSE if ADS_CAQ_LICENSE.exists() else ADS_LICENSE


def _kill_port(port: int, wait: float = 2.0) -> None:
    """Kill any process listening on the given port, then wait for it to release."""
    try:
        # LISTEN only: the PDP client keeps a pooled keep-alive connection to ADS, so a plain
        # `lsof -ti :port` also matches the pytest process itself, which would then kill itself.
        result = subprocess.run(["lsof", "-ti", f":{port}", "-sTCP:LISTEN"], capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split() if p]
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
        if pids:
            time.sleep(wait)
    except FileNotFoundError:
        pass  # lsof not available; ignore


def _write_deployment_yaml(domain_filename: str) -> None:
    """Overwrite deployment.yaml to point at the given domain file."""
    license_file = _find_license()
    lines = [
        f"license: file:./{license_file.name}",
        "",
        "engine:",
        "  mode: ads_caq",
        "",
    ]
    lines.extend([
        "domain:",
        f"  path: ./{domain_filename}",
        "  refreshInterval: 30 seconds",
        "",
        "server:",
        "  port: 8080",
        "",
        "logging:",
        "  level.com.axiomatics.audit: info",
        "",
        "authentication:",
        "  enabled: false",
        "",
        # AuthZEN is enabled so integration tests can exercise the AuthZEN wire protocol
        # (connection.protocol="authzen") alongside XACML. It only exposes the extra
        # /access/v1/evaluation(s) endpoints; the XACML /authorize path is unaffected.
        "authzen:",
        "  enabled: true",
        "",
        "identity:",
        "  constant: ads-test-local",
        "",
    ])
    DEPLOYMENT_YAML.write_text("\n".join(lines))


def _ads_is_healthy(timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(f"{ADS_URL}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_ads(max_wait: float = 30.0, poll: float = 1.0) -> bool:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if _ads_is_healthy():
            return True
        time.sleep(poll)
    return False


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def ads_server(request):
    """
    Start ADS with the domain YAML specified by the test class attribute
    ``domain_yaml``. Stops ADS after the test class completes.

    Usage in a test class:
        domain_yaml = "domain_tool_visibility_only.yaml"
    """
    jar = _find_jar()
    if jar is None:
        pytest.skip(f"ADS jar not found in {ADS_DIR}")

    domain_yaml: str = getattr(request.cls, "domain_yaml", None)
    if domain_yaml is None:
        pytest.skip("Test class does not define domain_yaml")

    _kill_port(8080)
    _write_deployment_yaml(domain_yaml)

    process = subprocess.Popen(
        ["java", "-jar", str(jar)],
        cwd=str(ADS_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _wait_for_ads(max_wait=45.0):
        process.terminate()
        process.wait(timeout=10)
        pytest.fail(f"ADS did not become healthy within 45s using {domain_yaml}")

    yield

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(scope="class")
def ads(ads_server: None):
    return PdpClient(base_url=ADS_URL, timeout=10.0, fail_open=False)
