---
id: TASK-2227
title: Add MCP Unified standalone typed package marker
status: Done
labels:
- mcp-unified
- packaging
- typing
priority: medium
modified_files:
- .github/tests/test_mcp_unified_artifact_gate.py
- .github/workflows/pypi-package.yml
- Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md
- mcp_unified/py.typed
- mcp_unified/pyproject.toml
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- backlog/tasks/task-2227 - Add-MCP-Unified-standalone-typed-package-marker.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add and verify a PEP 561 py.typed marker for the standalone mcp_unified package so typed downstream users and package artifacts can recognize the package as typed. Keep the slice scoped to package metadata/tests/docs and the marker file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a PEP 561 `mcp_unified/py.typed` marker for the standalone MCP Unified package and configured setuptools package data so the marker is included in built wheel and sdist artifacts. Extended package-boundary tests, the isolated `.github` artifact-gate shim, and the PyPI package workflow command so CI checks the typed marker along with metadata/extras/sdist boundaries. Updated standalone gateway admin docs to describe the typed-package marker. TDD red checks failed first for the missing marker/config/artifact/docs surfaces; verification passed with focused marker tests, the full runtime package-boundary plus gateway CLI suites, isolated artifact-gate shim, Bandit medium+ on touched Python with zero results, and git diff --check.
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
