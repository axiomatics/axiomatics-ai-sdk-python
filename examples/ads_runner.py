"""
Helpers for starting and stopping ADS from example scripts.

    from ads_runner import start_ads, kill_ads

    ads_url = start_ads("domain_tool_visibility_only.yaml")
    config = AuthZConfig(connection=PDPConnection(base_url=ads_url), enforcement=standard_tool_enforcement(), ...)
    ...
    kill_ads()

start_ads:
  1. Patches the domain.path line in the existing examples/policies/deployment.yaml.
  2. Launches the ADS jar as a background subprocess (CWD = examples/policies/).
  3. Polls GET /health until 200 (default timeout: 60 s).
  4. Returns the base URL (e.g. "http://localhost:8080").

kill_ads:
  Terminates the process started by start_ads.

The domain_yaml argument is a filename only — it must exist inside examples/policies/.
"""
import re
import subprocess
import time
import urllib.request
from pathlib import Path

_ADS_DIR = Path(__file__).resolve().parent / "policies"
_DEPLOYMENT_YAML = _ADS_DIR / "deployment.yaml"
_proc: subprocess.Popen | None = None


def _find_jar() -> Path:
    jars = sorted(_ADS_DIR.glob("access-decision-service-*.jar"), reverse=True)
    if not jars:
        raise FileNotFoundError(f"No ADS jar found in {_ADS_DIR}")
    return jars[0]


def start_ads(domain_yaml: str, *, port: int = 8080, timeout: int = 60) -> str:
    """Patch deployment.yaml, start ADS, wait for /health, return the base URL."""
    global _proc

    domain_path = _ADS_DIR / domain_yaml
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain file not found: {domain_path}")

    text = _DEPLOYMENT_YAML.read_text()
    text = re.sub(r"(  path: )\./\S+", rf"\1./{domain_yaml}", text)
    _DEPLOYMENT_YAML.write_text(text)

    _proc = subprocess.Popen(
        ["java", "-jar", str(_find_jar())],
        cwd=_ADS_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    base_url = f"http://localhost:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # The URL is a literal localhost address with only the port interpolated; this
            # polls the ADS jar that this script itself just started.
            # nosemgrep: dynamic-urllib-use-detected
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    return base_url
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(
        f"ADS did not become healthy within {timeout}s "
        f"(domain: {domain_yaml}, port: {port})"
    )


def kill_ads() -> None:
    """Terminate the ADS process started by start_ads."""
    global _proc
    if _proc is not None:
        _proc.terminate()
        _proc = None
