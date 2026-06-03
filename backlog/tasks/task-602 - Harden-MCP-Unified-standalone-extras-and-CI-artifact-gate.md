---
id: TASK-602
title: Harden MCP Unified standalone extras and CI artifact gate
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-03 03:14
labels:
- mcp-unified
- packaging
- ci
dependencies: []
priority: medium
modified_files:
- .github/workflows/pypi-package.yml
- Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md
- Docs/superpowers/plans/2026-06-03-mcp-unified-standalone-extras-ci-gate-plan.md
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a focused release gate that builds the package-local mcp_unified distribution artifact and proves the standalone dependency extras are independently represented/tested without pulling in the root tldw-server dependency surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Built standalone mcp_unified wheel metadata is checked for package name, version, console script, extras, and dependency boundary.
- [x] #2 Built standalone mcp_unified sdist is checked for package boundary contents and absence of host application trees.
- [x] #3 PyPI Package Check workflow runs the standalone artifact gate for mcp_unified/** changes while preserving the root package build/check/upload flow.
- [x] #4 Standalone admin documentation describes the artifact gate and publishing status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-03-mcp-unified-standalone-extras-ci-gate-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a package-local MCP Unified artifact gate that builds wheel and sdist artifacts, validates wheel metadata/extras/entry point/dependency boundary, validates sdist contents, wires the PyPI Package Check workflow to run the focused gate for mcp_unified/** changes, and documents the pre-publish gate. Verification: focused MCP package-boundary pytest passed; Bandit medium+ threshold passed on the touched Python test file; default Bandit output contains only existing low-severity test-scope findings.
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
