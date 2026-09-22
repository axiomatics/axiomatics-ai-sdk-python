#!/usr/bin/env bash
# Local pre-push checks. Runs every gate, then reports; it does NOT stop at the first failure,
# because the point of running this before pushing is to see everything at once. Exit status is
# non-zero if any gate failed. See docs/checks.md for what each gate stands in for.
#
#   scripts/scan-local.sh              # all gates
#   scripts/scan-local.sh deps         # one gate: lint|complexity|sast|deps|secrets|config|tests|fresh
#   AUDIT=1 scripts/scan-local.sh      # audit mode: disable suppressions, drop severity floor
#
# AUDIT=1 is for periodically re-examining what has been accepted: it ignores `# noqa`, the ruff
# per-file-ignores and global ignore list, `nosemgrep` directives, and lowers the dependency
# severity floor so moderates surface. It is EXPECTED to fail; it is a review tool, not a gate.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
REPORTS="$ROOT/build/reports"
mkdir -p "$REPORTS"

ONLY="${1:-}"
AUDIT="${AUDIT:-0}"
FAILED=(); PASSED=(); SKIPPED=(); INFOED=()
RUN_START=$SECONDS

# Dependency gates are HIGH+, matching the Snyk threshold in the Jenkinsfile.
SEVERITY="HIGH,CRITICAL"
[[ "$AUDIT" == 1 ]] && SEVERITY="LOW,MEDIUM,HIGH,CRITICAL"

RUFF_AUDIT=(); SEMGREP_AUDIT=(); GITLEAKS_AUDIT=()
if [[ "$AUDIT" == 1 ]]; then
  RUFF_AUDIT=(--ignore-noqa --config "lint.per-file-ignores = {}" --config "lint.ignore = []")
  SEMGREP_AUDIT=(--disable-nosem)
  GITLEAKS_AUDIT=(--ignore-gitleaks-allow)
fi

# The product extras customers install (`all` is every integration extra). Mirrors SNYK_EXTRAS
# in the Jenkinsfile; the dev and examples dependency groups are left out with
# --no-default-groups. Scanning the shipped surface on its own is what makes the dependency
# numbers meaningful; the examples group alone pulls in streamlit, pillow, tornado...
SHIPPED_EXTRAS=(--extra all)

# Cognitive-complexity ceiling. SonarQube's default rule (python:S3776) fails a function above
# 15; complexipy implements the same metric so we can see it before Sonar does.
COMPLEXITY_MAX=15

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
have() { command -v "$1" >/dev/null 2>&1; }
stamp() { date +%H:%M:%S; }
phase() { printf '\n\033[1;36m▶ [%s] PHASE %s\033[0m\n' "$(stamp)" "$1"; }

run() {  # run <label> <command...>: records pass/fail with timing, never aborts the script.
  local label="$1"; shift
  local start=$SECONDS rc
  printf '\033[1m  ┌ [%s] START  %s\033[0m\n' "$(stamp)" "$label"
  "$@"; rc=$?
  local elapsed=$((SECONDS - start))
  if [[ $rc -eq 0 ]]; then
    printf '\033[32m  └ [%s] PASS   %s  (%ss)\033[0m\n' "$(stamp)" "$label" "$elapsed"; PASSED+=("$label (${elapsed}s)")
  else
    printf '\033[31m  └ [%s] FAIL   %s  (%ss, exit %s)\033[0m\n' "$(stamp)" "$label" "$elapsed" "$rc"; FAILED+=("$label — exit $rc, ${elapsed}s")
  fi
}

info() {  # info <label> <command...>: runs and reports, never fails the gate.
  local label="$1"; shift
  local start=$SECONDS
  printf '\033[1m  ┌ [%s] START  %s\033[0m\n' "$(stamp)" "$label"
  "$@"
  printf '\033[36m  └ [%s] INFO   %s  (%ss)\033[0m\n' "$(stamp)" "$label" "$((SECONDS - start))"; INFOED+=("$label")
}

skip_unless() {  # skip_unless <label> <tool>: a missing scanner must not look like a clean scan.
  local label="$1" tool="$2"
  if have "$tool"; then return 0; fi
  printf '\033[33m  · [%s] SKIP   %s — %s not installed\033[0m\n' "$(stamp)" "$label" "$tool"
  SKIPPED+=("$label — '$tool' not installed"); return 1
}

# The dev dependency group (ruff, pytest-cov) is a default group, so `uv run` includes it.
UV=(uv run --frozen)

printf '\033[1m═══ scan-local [%s] gates=%s ═══\033[0m\n' "$(stamp)" "${ONLY:-all}"
[[ "$AUDIT" == 1 ]] && printf '\033[1;35m═══ AUDIT MODE — suppressions disabled, severity floor removed. Failures expected. ═══\033[0m\n'

# ── lint: ruff = Python lint + Bandit security rules (`S`). Same command as the Jenkins Lint stage.
if want lint; then
  phase "lint"
  run "ruff (python lint + bandit rules)" "${UV[@]}" ruff check ${RUFF_AUDIT[@]+"${RUFF_AUDIT[@]}"} .
fi

# ── complexity: cognitive complexity per function, SonarQube's threshold, on the shipped package only.
if want complexity; then
  phase "complexity"
  if have complexipy; then
    run "complexipy (cognitive complexity ≤ $COMPLEXITY_MAX)" complexipy axiomatics_ai --max-complexity-allowed "$COMPLEXITY_MAX" --quiet
  else
    run "complexipy via uvx (cognitive complexity ≤ $COMPLEXITY_MAX)" uvx complexipy axiomatics_ai --max-complexity-allowed "$COMPLEXITY_MAX" --quiet
  fi
fi

# ── sast: semgrep. p/jwt is the ruleset that catches unverified JWT decoding; ruff's `S` set does not.
if want sast; then
  phase "sast"
  if skip_unless "semgrep" semgrep; then
    run "semgrep (python, security-audit, owasp, jwt, dockerfile)" \
      semgrep scan --error --quiet --metrics=off ${SEMGREP_AUDIT[@]+"${SEMGREP_AUDIT[@]}"} \
        --config p/python --config p/security-audit --config p/owasp-top-ten --config p/jwt --config p/dockerfile \
        --exclude examples/policies --json-output="$REPORTS/semgrep.json" axiomatics_ai examples tests
  fi
fi

# ── deps: pip-audit on the shipped surface is the gate (stands in for Snyk); trivy on the full
#          lockfile is informational, because the examples/dev extras carry vulnerabilities users
#          never install.
if want deps; then
  phase "deps"
  # --no-deps: uv export already emits the pinned transitive set. --disable-pip: stops pip-audit
  # building a throwaway venv, whose ensurepip aborts under uv-managed Python 3.14.
  PIP_AUDIT=(pip-audit); have pip-audit || PIP_AUDIT=(uvx pip-audit)
  run "pip-audit (shipped extras, HIGH+ gate)" bash -c '
    uv export --no-default-groups --no-hashes --no-annotate --no-header --no-emit-project "$@" -o "'"$REPORTS"'/requirements-shipped.raw.txt" \
    && sed -E "s/[[:space:]]*;.*$//" "'"$REPORTS"'/requirements-shipped.raw.txt" | grep -E "^[A-Za-z0-9_.-]+==" > "'"$REPORTS"'/requirements-shipped.txt" \
    && '"${PIP_AUDIT[*]}"' --strict --no-deps --disable-pip --progress-spinner off -r "'"$REPORTS"'/requirements-shipped.txt"' _ "${SHIPPED_EXTRAS[@]}"
  if skip_unless "trivy" trivy; then
    info "trivy fs (full uv.lock incl. examples/dev, informational)" \
      trivy fs --severity "$SEVERITY" --ignore-unfixed --exit-code 0 --scanners vuln,misconfig \
        --skip-dirs ".venv,build,examples/policies" --format table --output "$REPORTS/trivy.txt" . \
      && { grep -E '^│ [A-Za-z0-9_.-]+ +│ (CVE|GHSA|PYSEC)-' "$REPORTS/trivy.txt" | awk -F'│' '{gsub(/ /,"",$2); print $2}' | sort -u | paste -sd, - | sed 's/^/    packages with HIGH+ findings: /'; true; }
  fi
fi

# ── secrets: git history is the gate; the working tree is informational (it reads .env and licences on purpose).
if want secrets; then
  phase "secrets"
  if skip_unless "gitleaks" gitleaks; then
    run "gitleaks (git history)" gitleaks git --redact --no-banner ${GITLEAKS_AUDIT[@]+"${GITLEAKS_AUDIT[@]}"} --report-path="$REPORTS/gitleaks-history.json" .
    if [[ "$AUDIT" == 1 ]]; then
      run "gitleaks (working tree)" gitleaks dir --redact --no-banner "${GITLEAKS_AUDIT[@]}" --report-path="$REPORTS/gitleaks-worktree.json" .
    else
      info "gitleaks (working tree, informational)" gitleaks dir --redact --no-banner --exit-code 0 --report-path="$REPORTS/gitleaks-worktree.json" .
    fi
  fi
fi

# ── config: Dockerfile and shell scripts.
if want config; then
  phase "config"
  if skip_unless "hadolint" hadolint; then run "hadolint (examples/gateway/Dockerfile)" hadolint examples/gateway/Dockerfile; fi
  if skip_unless "shellcheck" shellcheck; then run "shellcheck (scripts)" shellcheck scripts/*.sh; fi
fi

# ── tests: unit only, same selection and coverage as the Jenkins Tests stage. Integration tests need the ADS jar.
if want tests; then
  phase "tests"
  run "pytest (unit, with coverage)" "${UV[@]}" pytest tests/ -m "not ads_integration" -q \
    --cov=axiomatics_ai --cov-report=term-missing:skip-covered --cov-report="xml:$REPORTS/coverage.xml"
fi

# ── fresh: install the built wheel the way a user does (empty venv, oldest Python, every
#           dependency resolved fresh from PyPI, no lockfile), import every integration and run
#           the unit tests against that install. Catches a new framework release that breaks us.
if want fresh; then
  phase "fresh"
  run "fresh install (wheel, Python ${FRESH_PYTHON:-3.11}, latest deps) + unit tests" bash scripts/fresh-install-check.sh
fi

# ── summary
printf '\n\033[1m════ scan summary [%s] total %ss ════\033[0m\n' "$(stamp)" "$((SECONDS - RUN_START))"
for p in "${PASSED[@]-}";  do [[ -n "$p" ]] && printf '  \033[32mPASS\033[0m  %s\n' "$p"; done
for i in "${INFOED[@]-}";  do [[ -n "$i" ]] && printf '  \033[36mINFO\033[0m  %s — see report, not gating\n' "$i"; done
for s in "${SKIPPED[@]-}"; do [[ -n "$s" ]] && printf '  \033[33mSKIP\033[0m  %s\n' "$s"; done
for f in "${FAILED[@]-}";  do [[ -n "$f" ]] && printf '  \033[31mFAIL\033[0m  %s\n' "$f"; done
printf '\nreports: %s\n' "${REPORTS#"$ROOT"/}"
if [[ "$AUDIT" == 1 ]]; then exit 0; fi
[[ ${#FAILED[@]} -gt 0 ]] && exit 1
[[ ${#SKIPPED[@]} -gt 0 ]] && printf '\033[33mAll run gates passed, but some were skipped.\033[0m\n'
exit 0
