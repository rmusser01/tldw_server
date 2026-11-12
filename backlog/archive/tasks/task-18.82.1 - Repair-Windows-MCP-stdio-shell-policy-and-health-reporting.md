---
id: TASK-18.82.1
title: Repair Windows MCP stdio shell policy and health reporting
status: In Progress
assignee: []
created_date: '2026-09-29 07:31'
updated_date: '2026-09-29 07:32'
labels: []
dependencies: []
references:
  - Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md
parent_task_id: TASK-18.82
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Default stdio process policy denies Windows shell executables, including bash.exe, unless explicitly allowlisted.
- [ ] #2 Focused and full stdio transport tests pass; security scan reports no new findings.
- [ ] #3 The exited-process health test waits for the child process to exit before asserting disconnected status on Windows.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root-cause correction: docs.exit writes its reply before sys.exit; on Windows the subprocess returncode may remain None during the immediate health check. This is a fixture timing race, not a proven production health defect.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
