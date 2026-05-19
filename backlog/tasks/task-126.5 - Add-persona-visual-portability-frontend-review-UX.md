---
id: TASK-126.5
title: Add persona visual portability frontend review UX
status: Done
assignee: []
created_date: '2026-05-09 02:54'
updated_date: '2026-05-09 03:00'
labels:
  - persona
  - visual-packs
  - portability
  - frontend
dependencies:
  - TASK-126.4
  - TASK-137
references:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
  - 'https://github.com/rmusser01/tldw_server/pull/1135'
parent_task_id: TASK-126
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first frontend review surface for PR1135-aligned persona visual pack portability. Users should be able to queue an export for the selected pack, see/poll the export job status, download the completed archive through the authenticated client, upload a .tldw-persona-vpack archive for import preview, and inspect the preview summary without committing any import. This depends on the completed persona visual portability API endpoints and should stay inside the existing VisualPackEditor/client surface for a focused reviewable slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona visual frontend types and API helpers cover export start/status/download and import-preview upload/status responses.
- [x] #2 VisualPackEditor exposes export controls for the selected pack and shows queued/processing/completed/failed job state.
- [x] #3 Completed exports can be downloaded through the authenticated frontend client rather than relying only on an unauthenticated href.
- [x] #4 VisualPackEditor exposes import-preview upload controls for .tldw-persona-vpack archives and shows preview status plus summary/warnings/conflicts/proposed plan when available.
- [x] #5 Import-preview UX clearly remains review-only and does not mutate or activate persona visual packs.
- [x] #6 Focused frontend tests cover export queue/status/download and import-preview upload/status rendering.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added persona visual frontend portability types and service helpers for export start/status, authenticated archive download, import-preview upload, and import-preview status.

Extended VisualPackEditor with a Portability panel that queues export jobs, refreshes export status, downloads completed archives through fetchWithAuth as arrayBuffer data, uploads .tldw-persona-vpack archives for import preview, and displays review-only summary/warnings/conflicts/proposed plan output.

RED verification: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx initially failed on missing persona-visual-export-button and persona-visual-import-preview-input, confirming the new tests covered missing UI.

Focused verification passed: cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx => 6 passed.

Regression verification passed: cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/store/__tests__/persona-visual-runtime.test.ts => 37 passed.

git diff --check passed with no output.

Bandit is not applicable for this frontend-only slice because no Python production code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the frontend review UX for PR1135-aligned persona visual pack portability. The existing VisualPackEditor now has a focused Portability panel for export queue/status/download and import-preview upload/status review. New frontend contracts and service helpers wrap the backend API, and completed export archives are downloaded through the authenticated client as binary data instead of relying on a plain href. The import-preview flow remains review-only and renders summary, warning, conflict, and proposed-plan metadata without mutating packs. Focused and related persona visual frontend tests pass; Bandit is not applicable to this frontend-only change.
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
