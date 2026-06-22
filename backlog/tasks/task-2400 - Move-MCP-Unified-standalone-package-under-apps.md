---
id: TASK-2400
title: Move MCP Unified standalone package under apps
status: Done
labels:
- mcp
- refactor
modified_files:
- apps/mcp-unified
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete Task 1 relocation of the mcp_unified standalone package into apps/mcp-unified/src/mcp_unified, move package project files to apps/mcp-unified, add package-resource documentation copies, and update focused boundary tests. Scope excludes Task 2 artifact boundary helper changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Focused relocation tests passed: 3 passed. Bandit was run on the touched Python scope and returned 82 low-severity baseline findings, with no findings on newly added docs assertion lines.
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
