#!/usr/bin/env bash
# Install the SDK the way a user does, then test that install.
#
#   scripts/fresh-install-check.sh            # Python 3.11, the oldest supported
#   FRESH_PYTHON=3.13 scripts/fresh-install-check.sh
#
# The wheel is built from the working tree (uncommitted changes included) and installed with
# every extra into an empty venv. Every dependency is resolved fresh from PyPI: no uv.lock, no
# [tool.uv] constraints, as for a pip user. The tests' own dependencies (the dev and examples
# groups, unpinned) are resolved fresh too. Then every integration is imported and the unit tests
# run from a copy outside the repository, so they exercise the installed wheel, not the source
# tree. This catches what the locked environment hides: a new major of a framework that moved an
# API the SDK uses (fastmcp 4 / mcp 2 did).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON_VERSION="${FRESH_PYTHON:-3.11}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

uv build -q --wheel --out-dir "$WORK/dist"
WHEEL="$(ls "$WORK"/dist/*.whl)"
uv venv -q --python "$PYTHON_VERSION" "$WORK/venv"
PY="$WORK/venv/bin/python"

# The dev and examples groups' specs, as written (unpinned), one per line.
GROUP_DEPS=()
while IFS= read -r spec; do GROUP_DEPS+=("$spec"); done < <("$PY" - "$ROOT/pyproject.toml" <<'EOF'
import sys, tomllib
groups = tomllib.load(open(sys.argv[1], "rb"))["dependency-groups"]
for name in ("dev", "examples"):
    for spec in groups.get(name, []):
        if isinstance(spec, str):
            print(spec)
EOF
)

uv pip install -q --python "$PY" "axiomatics-ai-sdk[all] @ file://$WHEEL" "${GROUP_DEPS[@]}"
echo "resolved (python $("$PY" -c 'import platform; print(platform.python_version())')):"
uv pip list --python "$PY" 2>/dev/null \
  | grep -iE '^(fastmcp|mcp|langchain|langchain-core|langgraph|deepagents|langchain-mcp-adapters|strands-agents|claude-agent-sdk|openai-agents|google-adk|litellm|starlette|pyjwt|pydantic|httpx) ' \
  | sed 's/^/  /'

# The tests and examples (some tests load example modules), tracked or new, never ignored files.
mkdir -p "$WORK/repo"
git ls-files -co --exclude-standard -- tests examples pyproject.toml | while IFS= read -r f; do
  [[ -e "$f" ]] || continue   # deleted in the working tree
  mkdir -p "$WORK/repo/$(dirname "$f")"
  cp "$f" "$WORK/repo/$f"
done
cd "$WORK/repo"

"$PY" - <<'EOF'
import importlib
import axiomatics_ai
assert "site-packages" in axiomatics_ai.__file__, f"testing the source tree, not the wheel: {axiomatics_ai.__file__}"
failed = []
for name in ("langchain_agents", "langchain_mcp", "rag", "fastmcp", "tool_decorator", "a2a", "gateway",
             "strands", "claude_agent", "openai_agents", "google_adk"):
    try:
        importlib.import_module(f"axiomatics_ai.{name}")
    except Exception as exc:  # report every broken integration, not just the first
        failed.append(f"axiomatics_ai.{name}: {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit("integrations that don't import on a fresh install:\n  " + "\n  ".join(failed))
print("every integration imports")
EOF

# The pytest script, not `python -m pytest`: that would put the working directory on sys.path.
"$WORK/venv/bin/pytest" tests -m "not ads_integration" -q -p no:cacheprovider
