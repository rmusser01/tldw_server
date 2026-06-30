---
id: TASK-400
title: Design WebUI and extension exposure for folder-to-notes Sources sync
status: Done
labels:
- design
- webui
- extension
- sources
- notes-sync
documentation:
- Docs/superpowers/specs/2026-05-17-folder-notes-sources-ui-exposure-design.md
modified_files:
- Docs/superpowers/specs/2026-05-17-folder-notes-sources-ui-exposure-design.md
- backlog/tasks/task-400 - Design-WebUI-and-extension-exposure-for-folder-to-notes-Sources-sync.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the approved design/spec work for exposing existing local-directory-to-notes Sources sync in the shared WebUI/extension surface, including a Notes entry point and Sources shortcut modal coverage before implementation planning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec phase only. Critique pass applied after user review. The direct Notes folder-sync entry point now has a required server-owned local-directory source capability/enforcement precondition; generic Sources-page routing can proceed on the existing ingestion-sources capability. Implementation planning must also cover shortcut-config default merging, header launcher legacy-default migration, and rendering the existing schedule_enabled control before claiming scheduled rescans.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed the design tracker after PR #2099 merged. The spec is approved for implementation planning, and the downstream implementation plan already exists in Docs/superpowers/plans/2026-05-17-folder-notes-sources-ui-exposure-implementation-plan.md under completed TASK-403. Verification for this closeout: inspected the spec, downstream plan, active backlog inventory, and current origin/dev merge state; no product code changed. Bandit is not applicable for this docs/backlog-only closeout.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design tracker complete. The folder-to-notes Sources UI exposure spec now reflects approved-for-implementation-planning status, with the implementation plan already created and completed under TASK-403. This closeout changes only spec/backlog metadata and leaves implementation work to the existing plan stages.
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
