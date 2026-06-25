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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the standalone MCP Unified production PyPI publish workflow to use the pending PyPI trusted publisher configured for repository rmusser01/tldw_server, workflow mcp-unified-publish.yml, environment pypi, instead of a long-lived PyPI API token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GitHub Actions environment pypi exists for the publish workflow.
- [ ] #2 Production publish-pypi job requests id-token: write and keeps environment name pypi.
- [ ] #3 Production publish-pypi job no longer references MCP_UNIFIED_PYPI_API_TOKEN or TWINE_PASSWORD.
- [ ] #4 Production publish-pypi job publishes .artifacts/mcp-unified-rc/dist with pypa/gh-action-pypi-publish.
- [ ] #5 TestPyPI publish path remains token-based and unchanged.
- [ ] #6 Workflow boundary tests cover the trusted publishing shape.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validation:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q -k publish_workflow` -> 3 passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q` -> 47 passed.
- `make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python mcp-unified-publish-dry-run` -> passed; helper build and TestPyPI dry-run plan still work.
- `git diff --check` -> passed.
- Full Bandit on touched test file exits nonzero for existing low-severity pytest/subprocess/test-secret baseline; filtered touched-scope Bandit with `-s B101,B404,B603,B105` -> no findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Focused workflow tests pass.
- [ ] #2 MCP Unified publish workflow is YAML-valid under existing test helpers.
- [ ] #3 Bandit run for touched code when applicable or documented non-code skip.
- [ ] #4 PR opened against dev with validation summary.
<!-- DOD:END -->
