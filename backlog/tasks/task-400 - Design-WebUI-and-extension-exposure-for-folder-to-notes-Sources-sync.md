---
id: TASK-400
title: Design WebUI and extension exposure for folder-to-notes Sources sync
status: In Progress
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

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
