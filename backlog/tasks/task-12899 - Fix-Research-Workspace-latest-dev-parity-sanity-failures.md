---
id: TASK-12899
title: Fix Research Workspace latest-dev parity sanity failures
status: Done
labels:
- research-workspace
- notebooklm
- uat
- frontend
- bug
priority: high
modified_files:
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
references:
- https://github.com/rmusser01/tldw_server/pull/2674
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Latest-dev Research Workspace / NotebookLM parity sanity pass found two isolated WorkspaceHeader test failures: the Workspaces dropdown can remain visible when opening Workspace settings, and the agent-task modal close assertion does not account for Ant modal teardown timing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspaceHeader focused tests pass on latest dev.
- [x] #2 Focused Research Workspace/WebClipper frontend sanity suite passes.
- [x] #3 Changes stay limited to the WorkspaceHeader sanity failures unless another directly related regression is found.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm root cause from WorkspaceHeader and WorkspaceAgentTaskHandoffModal behavior.
2. Apply minimal frontend fix/test hardening for the two latest-dev sanity failures.
3. Re-run targeted failing tests, full WorkspaceHeader test file, focused frontend sanity suite, and lightweight diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review follow-up: rebased on origin/dev at 5605c36949, renumbered this task from TASK-12898 to TASK-12899 after dev introduced a different TASK-12898, checked AC/DOD items to match Done status, simplified the redundant null assertion, and made the manual agent-task modal lookup select the non-leaving dialog when Ant leaves the prior portal mounted.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Latest-dev Research Workspace sanity failures are fixed in the focused test coverage. PR: https://github.com/rmusser01/tldw_server/pull/2674. Verification: targeted failing WorkspaceHeader cases passed; full WorkspaceHeader suite passed (69 tests); focused Research Workspace/WebClipper sanity suite passed (12 files, 296 tests); git diff --check and git diff --cached --check passed. Frontend package typecheck was rerun with NODE_OPTIONS=--max-old-space-size=8192 and failed on unrelated existing test type errors outside the touched file: ChatGreetingPicker, MCPHub first-run, background-session-store, useSetupOnboarding, TldwChat abort, and character-export SSRF tests. Bandit skipped because the change only touches a TypeScript test and Backlog task metadata.
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
