---
id: TASK-12143
title: Update PyPI workflow contract for gated auto-publish
status: Done
assignee: []
created_date: '2026-07-04 07:04'
updated_date: '2026-07-04 07:06'
labels:
  - tests
  - ci
  - pypi
dependencies: []
references:
  - TASK-12123
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The broad pytest slice stops in tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py because the test still requires publish-pypi.yml to be workflow_dispatch-only. TASK-12123 intentionally added push-to-main auto-publish for missing PyPI versions, so update the contract test to assert the new gated behavior without losing manual TestPyPI/PyPI dispatch coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused PyPI workflow contract tests pass.
- [x] #2 Contract test verifies push publishing is limited to main pyproject.toml changes and gated by detect-version should_publish.
- [x] #3 Manual workflow_dispatch TestPyPI/PyPI behavior remains asserted.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: TASK-12123 intentionally added push-to-main PyPI auto-publishing for new pyproject.toml versions, but test_pypi_workflow_contracts.py still asserted that publish-pypi.yml was workflow_dispatch-only. Reverting the workflow would lose the intended auto-publish behavior, so the contract test now asserts both preserved manual dispatch and the new safety gates: push limited to main/pyproject.toml, build gated by detect-version should_publish, TestPyPI manual-only, and PyPI publishing allowed only for manual target=pypi or push with should_publish=true.

Verification: focused PyPI workflow contract file passed: 3 passed, 12 warnings. Bandit on test_pypi_workflow_contracts.py with B101 skipped exited 0 with zero findings. git diff --check exited 0. Full CI directory retry moved past the PyPI contract tests and stopped at an unrelated shard coverage failure in test_required_workflow_contracts.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the PyPI workflow contract test to match the intended TASK-12123 behavior: manual TestPyPI/PyPI dispatch remains covered, and push-to-main PyPI publishing is asserted to be scoped to pyproject.toml changes and gated by detect-version should_publish. Focused tests and Bandit passed; the broader CI retry exposed the next unrelated workflow-shard coverage failure.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Root cause documented.
- [x] #8 Focused verification recorded.
- [x] #9 Bandit or non-code skip recorded.
- [x] #10 Final summary added.
<!-- DOD:END -->
