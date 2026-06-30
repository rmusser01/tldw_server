---
id: TASK-2249
title: Finalize Workspace file inventory worker verification
status: Done
labels:
- workspaces
- file-inventory
- verification
priority: high
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 8 from the Workspace file inventory Jobs implementation plan. Run focused Workspace verification, compile checks, Bandit over the touched Workspace/file-inventory backend scope, diff hygiene, and record final closeout for the file inventory worker sequence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused Workspace test suite passes or any scoped failures are investigated and fixed.
- [x] #2 Workspace/file-inventory compile check exits successfully.
- [x] #3 Bandit touched backend scope reports no new findings or documents known baseline findings.
- [x] #4 Diff hygiene passes with no whitespace errors.
- [x] #5 Backlog records final verification evidence and known skips/blockers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 8 full Workspace verification initially exposed one outdated sandbox-volume fixture expectation in `test_workspace_capabilities_include_attached_primary_root`: the test expected `write_files` to be allowed while omitting `sandbox_mount_state`. The current Workspace Core contract intentionally fail-closes sandbox roots until mounted, so the fixture now sets `sandbox_mount_state: mounted` and asserts the mounted state before checking allowed actions.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed final Workspace file inventory worker verification. Evidence: targeted regression rerun passed `1 passed, 6 warnings`; full Workspace suite passed `238 passed, 8 warnings`; compileall over Workspace core, services, workspaces endpoint, and workspace schemas exited 0; Bandit JSON at `/tmp/bandit_workspace_file_inventory.json` reported zero results, zero errors, and zero skipped files; `git diff --check` produced no output.
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
