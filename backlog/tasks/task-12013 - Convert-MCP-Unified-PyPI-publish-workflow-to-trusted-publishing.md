---
id: TASK-12013
title: Convert MCP Unified PyPI publish workflow to trusted publishing
status: Done
labels:
- mcp
- packaging
- pypi
- ci
priority: high
modified_files:
- .github/workflows/mcp-unified-publish.yml
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- backlog/tasks/task-12013 - Convert-MCP-Unified-PyPI-publish-workflow-to-trusted-publishing.md
references:
- https://github.com/rmusser01/tldw_server/pull/2514
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the standalone MCP Unified production PyPI publish workflow to use the pending PyPI trusted publisher configured for repository rmusser01/tldw_server, workflow mcp-unified-publish.yml, environment pypi, instead of a long-lived PyPI API token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GitHub Actions environment pypi exists for the publish workflow.
- [x] #2 Production publish-pypi job requests id-token: write and keeps environment name pypi.
- [x] #3 Production publish-pypi job no longer references MCP_UNIFIED_PYPI_API_TOKEN or TWINE_PASSWORD.
- [x] #4 Production publish-pypi job publishes .artifacts/mcp-unified-rc/dist with pypa/gh-action-pypi-publish.
- [x] #5 TestPyPI publish path remains token-based and unchanged.
- [x] #6 Workflow boundary tests cover the trusted publishing shape.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR: https://github.com/rmusser01/tldw_server/pull/2514

Validation:
- GitHub Actions environment `pypi` verified via `gh api`: id `17193755583`.
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q -k publish_workflow` -> 3 passed.
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q` -> 47 passed.
- `make PYTHON=python mcp-unified-publish-dry-run` -> passed; helper build and TestPyPI dry-run plan still work.
- `python -m ruff check tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py` -> passed.
- `git diff --check` -> passed.
- Full Bandit on touched test file exits nonzero for existing low-severity pytest/subprocess/test-secret baseline; filtered touched-scope Bandit with `-s B101,B404,B603,B105` -> no findings.
- Review follow-up validation: rebased on latest `origin/dev`; production PyPI publish now downloads the prebuilt publish-plan artifact, uses SHA-pinned artifact download and PyPI publish actions, and keeps the OIDC job free of checkout/build/run steps.
- Skipped Qodo marker item as already satisfied: `test_runtime_package_boundary.py` has module-level `pytestmark = pytest.mark.unit`, so the touched tests already have exactly one effective approved classification marker.
- Local `actionlint` check was attempted but unavailable in this environment (`command not found`); the PR's GitHub actionlint check remains the workflow validation source.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused workflow tests pass.
- [x] #2 MCP Unified publish workflow is YAML-valid under existing test helpers.
- [x] #3 Bandit run for touched code when applicable or documented non-code skip.
- [x] #4 PR opened against dev with validation summary.
<!-- DOD:END -->
