---
id: TASK-158
title: Expose persona visual pack workflow from Buddy shell
status: Done
assignee: []
created_date: '2026-05-09 05:41'
updated_date: '2026-05-09 05:47'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1410'
  - 'https://github.com/rmusser01/tldw_server/pull/1412'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-persona-buddy-visual-workflow-entry-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the floating Persona Buddy surface a direct entry point into the existing persona visual pack workflow. The merged foundation already supports active visual packs in BuddyShellHost, VisualPackEditor creation/upload/import/export, and persona_visuals MCP runtime state overrides; this slice closes the usability gap by linking the live Buddy popover to the selected persona's Visuals tab instead of sending users through unrelated VN surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buddy popover for an active persona exposes a direct action to open that persona's Visuals pack workflow.
- [x] #2 The action preserves the selected persona id and lands on the existing Visuals tab used for create, upload, import, export, and review flows.
- [x] #3 Buddy still renders active visual packs and falls back to summary text when no pack or a broken pack is available.
- [x] #4 Focused Buddy shell tests cover the new direct Visuals action.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused implementation plan at Docs/superpowers/plans/2026-05-09-persona-buddy-visual-workflow-entry-plan.md.

RED: bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx initially failed because the Buddy popover had no Open Visuals link.
GREEN: bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx passed with 16 tests after adding the direct Visuals action.
RELATED VERIFICATION: bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx src/utils/__tests__/persona-garden-route.test.ts passed with 37 tests.
HYGIENE: git diff --check passed.
BANDIT: not applicable; touched code is frontend TypeScript plus Backlog/plan metadata only.

Opened draft PR #1412 for the direct Buddy/Persona visual-pack workflow entry point.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a direct Open Visuals action to the floating Persona Buddy popover. The action uses the existing Persona Garden route helper to preserve the active persona id and open the Visuals tab that already hosts visual-pack create/upload/import/export/review workflows. Existing active-pack rendering and fallback behavior remain unchanged and covered by focused tests.
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
