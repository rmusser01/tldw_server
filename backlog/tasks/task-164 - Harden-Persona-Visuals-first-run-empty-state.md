---
id: TASK-164
title: Harden Persona Visuals first-run empty state
status: Done
assignee: []
created_date: '2026-05-09 15:39'
updated_date: '2026-05-09 15:47'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1415'
  - 'https://github.com/rmusser01/tldw_server/pull/1416'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/superpowers/plans/2026-05-09-persona-visual-empty-state-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the Persona Garden Visuals no-pack empty state from a passive message into a concise, action-oriented first-run guide for the existing Persona Buddy visual-pack workflow. This should remain centered on Persona Buddy/Persona Live and explain that users start by creating a draft pack, then upload assets, map visual states, import/export packs, queue generation where configured, review results, and explicitly activate a valid pack.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VisualPackEditor no-pack state explains the first action is creating a draft visual pack for the selected persona.
- [x] #2 No-pack state references the existing follow-on workflows without exposing controls that require a selected pack.
- [x] #3 No-pack state keeps Persona Buddy/Persona Live framing and does not mention VN/CYOA surfaces.
- [x] #4 Focused VisualPackEditor tests cover the no-pack first-run copy and existing draft creation flow still works.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added implementation plan at Docs/superpowers/plans/2026-05-09-persona-visual-empty-state-plan.md.

RED: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx failed because the empty state still rendered only "No visual packs yet."
GREEN: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx passed with 7 tests after the empty-state copy update.
RELATED VERIFICATION: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/utils/__tests__/persona-garden-route.test.ts passed with 31 tests.
HYGIENE: git diff --check passed.
BANDIT: not applicable; touched code is frontend TypeScript plus Backlog/plan metadata only.

Opened draft PR #1416 for the Persona Visuals first-run empty-state slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Improved the Persona Garden Visuals no-pack empty state so it frames the workflow around the selected persona's Persona Buddy, names draft creation as the first action, and explains that upload, state mapping, import, generation, review, and activation follow after a draft exists. Existing draft creation and asset upload flow remains covered by the focused VisualPackEditor test.
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
