---
id: TASK-401
title: Design chat sidebar tools-first expansion behavior
status: Done
labels:
- webui
- extension
- design
documentation:
- Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md
modified_files:
- Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the shared WebUI/extension chat sidebar behavior so every sidebar open/expand presents tools/shortcuts expanded and recent conversations collapsed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec captures the approved tools-first sidebar open behavior.
- [x] #2 Spec identifies shared ChatSidebar state ownership and boundaries.
- [x] #3 Spec covers lazy recent-history loading and regression tests.
- [x] #4 Backlog task links the resulting design artifact.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed after implementation verification was recorded in TASK-567 and merged through PR #2168.

- Design artifact: `Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md`.
- The spec captures the shared ChatSidebar ownership boundary, the tools-first reset behavior, recent-history lazy loading, and regression coverage expectations.
- Verification evidence now lives in TASK-567, including focused ChatSidebar unit tests, the WebLayout chat scroll contract test, and a browser smoke pass on `/chat`.
- This closeout changes only Backlog Markdown task records. Bandit is not applicable to this non-code closeout.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the sidebar tools-first design task. The approved spec defines the shared ChatSidebar ownership and boundaries, tools-first reset behavior, recent-history lazy loading, and regression coverage. Implementation and verification are now recorded in TASK-567 and PR #2168.
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
