---
id: TASK-409
title: Implement Persona Visual management header
status: Done
labels:
- persona
- visual-packs
- frontend
- webui
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/issues/1769
documentation:
- Docs/superpowers/specs/2026-05-16-persona-visual-post-setup-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 2 of the Persona Visual post-setup management design: render a compact management header at the top of Persona Garden Visuals using the existing PersonaVisualManagementSummary helper. The header should show active-pack status, lifecycle counts, and the highest-priority attention item while preserving existing VisualPackEditor behavior and backend/API contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VisualPackEditor renders a compact management header derived from `PersonaVisualManagementSummary` before the detailed editor controls.
- [x] #2 Header shows selected persona context, active-pack status, lifecycle counts, and the highest-priority attention item with accessible text labels.
- [x] #3 Header uses existing loaded pack/candidate/library/job/readiness state and does not introduce backend/API changes or duplicate server rules.
- [x] #4 Existing first-run, reuse, import/export, generation, candidate review, validation, and activation controls remain available below the header.
- [x] #5 Focused shared UI tests cover empty/no-active state, active-pack summary, counts, attention priority, and preservation of existing editor behavior.
- [x] #6 Verification records focused Vitest, lint or documented baseline skip, diff check, and Bandit non-applicability for frontend-only changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a compact management band in `VisualPackEditor` that uses `buildPersonaVisualManagementSummary()` from the existing summary helper. The band is rendered only after setup choices are no longer active, so first-run paths remain unchanged, and it uses already-loaded pack, candidate, library, job, validation, and readiness state without adding backend calls.

Verification:
- `bun run test src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx -t "management header"` passed.
- `bun run test src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` passed with 63 tests.
- `git diff --check` passed.
- `bunx tsc --noEmit -p tsconfig.json` was attempted from `apps/packages/ui` and failed on existing package-wide baseline type errors outside the touched PersonaGarden files; no errors were reported for the touched files in the visible output.
- Bandit is not applicable to this frontend-only slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Persona Visual post-setup management header so users see the selected persona, active visual pack, lifecycle counts, and the top actionable attention item before detailed editor controls. The implementation reuses the existing management summary model and preserves current setup, import/export, generation, candidate review, validation, and activation workflows below the header.
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
