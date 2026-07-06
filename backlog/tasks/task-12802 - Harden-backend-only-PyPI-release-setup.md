---
id: TASK-12802
title: Harden backend-only PyPI release setup
status: Done
labels:
- packaging
- pypi
- release
priority: Medium
modified_files:
- .github/workflows/publish-pypi.yml
- .github/workflows/pypi-package.yml
- Docs/Development/Packaging_and_Distribution_Strategy.md
- Docs/Development/PyPI_Publishing.md
- Docs/superpowers/plans/2026-06-24-backend-api-pypi-release.md
- Helper_Scripts/Packaging/__init__.py
- Helper_Scripts/Packaging/check_pypi_artifacts.py
- Makefile
- tldw_Server_API/tests/Helper_Scripts/test_check_pypi_artifacts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Lock the root PyPI release scope to the backend/API package, keep the Next.js WebUI out of PyPI, and harden/document the release path for first PyPI/TestPyPI publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PyPI packaging documentation clearly states backend/API-only scope and frontend distribution via GHCR/release artifacts.
- [x] #2 Root publish workflow is reviewed and hardened where needed without changing package runtime behavior.
- [x] #3 Local package build/check and isolated wheel smoke validation are documented or run.
- [x] #4 Frontend artifacts remain excluded from the Python wheel/sdist.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-24-backend-api-pypi-release.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented backend/API-only PyPI hardening in the isolated worktree. Added a distribution content guard, wired it into `make pypi-check`, pinned PyPI workflow actions to the SHA refs already used by the MCP publish workflow, expanded package-check triggers, and clarified that the WebUI is distributed separately via container/release artifacts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification completed:
- make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python pypi-check: passed; built wheel and sdist, twine check passed, backend/API-only artifact guard passed.
- /tmp/tldw-pypi-smoke-backend-api-2363 wheel smoke: installed dist/tldw_server-0.1.31-py3-none-any.whl with --no-deps and imported tldw_Server_API from site-packages.
- PR review fixes: artifact path normalization now treats Windows separators and absolute names safely, blocked path checks use exact path components/sequences, required package roots use exact components, the guard uses Loguru, and focused regression tests cover these cases.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Helper_Scripts/test_check_pypi_artifacts.py -q: passed, 4 tests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py -q: passed, 3 tests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check Helper_Scripts/Packaging/check_pypi_artifacts.py tldw_Server_API/tests/Helper_Scripts/test_check_pypi_artifacts.py: passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile Helper_Scripts/Packaging/check_pypi_artifacts.py tldw_Server_API/tests/Helper_Scripts/test_check_pypi_artifacts.py: passed.
- Workflow YAML parse for .github/workflows/pypi-package.yml and .github/workflows/publish-pypi.yml: passed.
- python -m bandit -r Helper_Scripts/Packaging -f json -o /tmp/bandit_backend_api_pypi.json: passed with 0 findings.
- git diff --check: passed.
Known note: smoke install used --no-deps to avoid downloading the full media/ML dependency stack; dependency resolution is still covered by wheel metadata and should be exercised in TestPyPI/PyPI publish smoke.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
