---
id: TASK-2400
title: Fix relocated MCP Unified boundary path tests
status: Done
labels:
- mcp
- tests
modified_files:
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Targeted follow-up for Task 1 spec review: fix subprocess PYTHONPATH, standalone artifact build source path, and artifact gate config path expectations after moving MCP Unified under apps/mcp-unified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Full test_runtime_package_boundary.py passed: 35 passed, 5 warnings. Bandit on touched Python files reported 82 low-severity baseline findings and no findings in the new helper or package_metadata.py.
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
