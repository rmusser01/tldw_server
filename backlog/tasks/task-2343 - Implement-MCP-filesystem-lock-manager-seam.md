---
id: TASK-2343
title: Implement MCP filesystem lock manager seam
status: Done
labels:
- mcp
- filesystem
- locks
modified_files:
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
- mcp_unified/USER_GUIDE.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-lock-manager-seam-implementation-plan.md
references:
- https://github.com/rmusser01/tldw_server/pull/2335
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an injectable filesystem lock-manager interface/config seam while keeping the in-memory process-local manager as the default backend. This prepares shared/persistent lock backends without changing existing tool behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FilesystemModule depends on a narrow lock-manager contract instead of directly hard-coding the in-memory manager at call sites.
- [x] #2 Default behavior remains process-local and in-memory unless an injected/configured manager is supplied.
- [x] #3 Tests prove two FilesystemModule instances can coordinate locks when they share an injected manager.
- [x] #4 Existing lock acquire/release and mutation lock validation behavior remains compatible.
- [x] #5 Docs/plan note persistence is a future backend slice, not part of this seam.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-mcp-filesystem-lock-manager-seam-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the MCP filesystem lock-manager seam: added a protocol and memory-backend factory, wired FilesystemModule to accept an injected manager while defaulting to process-local memory, added regression tests for shared manager coordination and unsupported backend config, and clarified the user guide. PR review follow-up: fixed fail-closed backend validation so explicit falsy lock_manager_backend values are rejected, synchronized the Backlog acceptance/DoD checklist with Done status, and verified Gemini's async protocol suggestion against the current offloaded sync helper design; no async refactor was applied in this seam. Verification: pytest test_filesystem_module.py -q (99 passed, 4 warnings); py_compile on touched implementation files; Bandit on touched implementation files (0 findings); git diff --check.
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
