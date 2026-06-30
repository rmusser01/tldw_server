---
id: TASK-137
title: Add persona visual pack editor
status: Done
assignee: []
created_date: '2026-05-09 00:47'
updated_date: '2026-05-09 00:55'
labels:
  - frontend
  - persona
  - visuals
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the Persona Garden editor surface for V1 persona visual packs so users can create draft packs, upload/select assets, edit sprite-frame manifests, validate required states, activate a pack, and deactivate back to derived Buddy rendering. This implements Task 8 from Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md and builds on the existing persona visual API/client/renderer work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loads visual pack list for the selected persona and handles empty/error states
- [x] #2 Allows creating a draft visual pack and uploading assets with selectable roles
- [x] #3 Edits required and optional state mappings plus ordered frame definitions without relying on filename order
- [x] #4 Supports sprite-sheet region fields preview frame frame rate loop alignment fallbacks and authored triggers
- [x] #5 Displays validation errors and disables activation when required states are missing
- [x] #6 Activates a valid pack and supports deactivating the active pack back to derived Buddy rendering
- [x] #7 Adds a Persona Garden visuals tab only in persona mode
- [x] #8 Includes focused Vitest coverage for editor workflows
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Persona Garden visual pack editor for draft creation, pack loading, asset upload, state mapping, ordered frames, sprite-sheet region fields, preview frame, fallbacks, authored triggers, validation, activation, and deactivate. Added the persona `visuals` route tab and sidepanel tab wiring.

Verification red: `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/utils/__tests__/persona-garden-route.test.ts` failed before implementation because VisualPackEditor was missing and the visuals tab was rejected.

Verification green: `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/utils/__tests__/persona-garden-route.test.ts` passed 10 tests; `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx` passed 73 tests; touched-file TypeScript filter produced no diagnostics; `git diff --check` passed.

Bandit not run for this slice because it touched frontend TypeScript/React files and Backlog/plan docs only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Persona Garden visual pack editor and visuals tab route wiring. The editor uses the existing persona visual API client to load packs, create drafts, upload assets, edit sprite-frame manifests, validate required states, activate valid packs, and deactivate the active pack back to derived Buddy rendering. Focused editor, route utility, and sidepanel tests pass; no backend Python files were touched in this slice.
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
