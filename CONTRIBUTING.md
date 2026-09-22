# Contributing

Thanks for your interest. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/axiomatics/axiomatics-ai-sdk-python.git
cd axiomatics-ai-sdk-python
uv sync --extra all
uv run pytest tests/
```

Unit tests need no PDP. Integration tests (marked `ads_integration`) require an Axiomatics ADS
jar and licence in `examples/policies/` and are skipped automatically when absent; see
[docs/running-ads.md](docs/running-ads.md).

## Before opening a pull request

```bash
scripts/scan-local.sh
```

This runs ruff (lint and Bandit rules), a cognitive-complexity check, semgrep, a dependency
audit of the shipped extras, gitleaks, the unit tests, and the unit tests again against a fresh
install (the built wheel in an empty Python 3.11 venv, dependencies resolved from PyPI with no
lockfile), and reports at the end. Tools that
are not installed are reported as skipped; install them with
`brew install semgrep trivy gitleaks hadolint shellcheck` and `uv tool install pip-audit complexipy`.

## Conventions

- The generic noun in code and docs is **PDP**. Write "Axiomatics ADS" only for ADS-specific
  features or when naming the example domains.
- Every enforcement point sends the request shapes documented in
  [docs/enforcement-points.md](docs/enforcement-points.md); keep them stable, since policies
  depend on them.
- Add a unit test with every behaviour change. Adapters for new frameworks should be thin
  mappings onto the helpers in `axiomatics_ai.core`, like the existing ones.
- Model strings in examples are fully qualified LiteLLM ids.
- Docstrings follow the Google style, with the type in parentheses:
  `Args:` / `Returns:` / `Raises:` (and `Attributes:` on classes, `Yields:` on generators),
  e.g. `subject (dict): Subject attributes.` Every public module, class and function gets
  one; a public function or method documents each parameter and its return value, and a
  class documents its constructor arguments. Private helpers keep a one-line summary,
  optionally followed by a short paragraph. ruff enforces the format (`convention = "google"`).

## Reporting a security issue

Please do not open a public issue. See [SECURITY.md](SECURITY.md).
