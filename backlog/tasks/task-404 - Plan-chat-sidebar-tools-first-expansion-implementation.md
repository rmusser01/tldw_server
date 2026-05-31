---
id: TASK-404
title: Plan chat sidebar tools-first expansion implementation
status: Done
labels:
- webui
- extension
- planning
references:
- TASK-401
documentation:
- Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md
- Docs/superpowers/plans/2026-05-17-chat-sidebar-tools-first-expansion-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-chat-sidebar-tools-first-expansion-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an execution-ready implementation plan for the shared ChatSidebar tools-first expansion behavior approved in TASK-401.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is saved under Docs/superpowers/plans.
- [x] #2 Plan decomposes ChatSidebar, layout reset signal, lazy-history gating, and tests into reviewable TDD tasks.
- [x] #3 Plan records exact files and verification commands.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed after implementation verification was recorded in TASK-567 and merged through PR #2168.

- Implementation plan: `Docs/superpowers/plans/2026-05-17-chat-sidebar-tools-first-expansion-implementation-plan.md`.
- The plan decomposes ChatSidebar reset behavior, recent disclosure/history gating, coordinator visibility wiring, layout `openResetKey` behavior, and focused regression tests.
- Verification evidence now lives in TASK-567, including focused ChatSidebar unit tests, the WebLayout chat scroll contract test, and a browser smoke pass on `/chat`.
- This closeout changes only Backlog Markdown task records. Bandit is not applicable to this non-code closeout.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the sidebar tools-first planning task. The execution plan is saved and has been validated by the merged implementation and verification recorded in TASK-567 and PR #2168.
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
