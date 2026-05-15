---
id: TASK-390
title: Implement first-run Persona Visual Buddy setup choices
status: Done
assignee: []
created_date: '2026-05-15 21:34'
labels:
  - persona
  - webui
  - persona-visuals
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1695'
documentation:
  - apps/packages/ui/src/routes/sidepanel-persona.tsx
  - apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx
  - apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1695 under epic #1510. The user-facing goal is a clear first-run Persona Garden setup path for personas without an active visual: use a bundled default pack, import a portable pack, or start blank, while preserving the existing draft-first and explicit-activation Persona Visual contract. Keep scope on Persona Garden / Buddy visual setup and avoid VN/CYOA, Live2D runtime adapter work, or external provider execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Garden exposes discoverable first-run setup choices for a persona with no active visual.
- [x] #2 Choosing a default pack creates and selects a draft without activating it.
- [x] #3 Choosing import enters the existing import preview/commit flow and lands on the committed draft without activating it.
- [x] #4 Choosing blank leaves a coherent editable no-visual or empty-draft state consistent with current backend support.
- [x] #5 Focused UI tests cover the setup choice rendering and primary state transitions.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a first-run setup panel in `VisualPackEditor` that appears when the selected persona has no active visual pack.
- The setup panel routes to existing draft-first primitives: bundled starter copy, import archive picker, and blank draft creation.
- Added localized copy for the setup panel and default blank draft title/status message.
- Added focused VisualPackEditor tests covering default/import choice rendering, no implicit activation, and blank draft creation.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFY:BEGIN -->
- `bun run test:run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` passed: 33 tests.
- `git diff --check` passed.
- `bun run lint` passed with existing warnings only: 0 errors, 155 warnings.
- Bandit skipped: frontend-only TypeScript/locale/task change, no Python touched.
<!-- SECTION:VERIFY:END -->

## Final Summary

<!-- SECTION:SUMMARY:BEGIN -->
Implemented the first-run Persona Visual Buddy setup choice layer for issue #1695. Users without an active visual now see a clear setup panel with Use default, Import pack, and Start blank paths, all preserving the existing draft-first and explicit-activation contract.
<!-- SECTION:SUMMARY:END -->
