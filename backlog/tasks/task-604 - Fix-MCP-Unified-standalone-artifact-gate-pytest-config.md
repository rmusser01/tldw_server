---
id: TASK-604
title: Fix MCP Unified standalone artifact gate pytest config
status: Done
labels:
- ci
- mcp-unified
- standalone-package
priority: High
modified_files:
- .github/workflows/pypi-package.yml
- .github/tests/test_mcp_unified_artifact_gate.py
- Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md
- mcp_unified/pytest-artifact-gate.ini
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
documentation:
- 'Verification recorded: clean temporary Python 3.11 venv with only mcp_unified[dev]
  plus packaging tools had no httpx installed and passed `python -m pytest -c mcp_unified/pytest-artifact-gate.ini
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_distribution_metadata_matches_extras
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_sdist_contains_only_package_boundary
  -q`. Project venv also passed the same gate and the root workflow-contract/package-boundary
  selection. Ruff passed for touched Python files. Bandit returned 0 findings for
  the new gate test and 0 medium/high findings across touched Python tests. `git diff
  --check` passed.'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-merge CI follow-up for PR #2235: the standalone artifact validation job invokes pytest with the repository-wide config, which loads root test plugins and requires root dependencies such as httpx. Keep the gate independent of root dependencies by running the selected package-boundary tests with a minimal pytest config and add regression coverage for the workflow command.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the failing workflow gate and current package-boundary tests.\n2. Add a minimal pytest config for the standalone artifact gate and update the workflow invocation.\n3. Add regression coverage that prevents the workflow from invoking pytest without the gate config.\n4. Validate focused tests, diff hygiene, and Bandit on touched Python test code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the post-merge standalone artifact gate failure by moving the CI gate tests out of the host package path, adding a minimal package-local pytest config, updating the PyPI workflow and docs to use the non-host gate, and extending workflow contract coverage. Verified with a clean temporary Python 3.11 venv containing only mcp_unified[dev] plus packaging tools, with httpx absent.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
