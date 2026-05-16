---
id: TASK-408
title: Implement Persona Visual management summary model
status: Done
references:
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/issues/1769
documentation:
- Docs/superpowers/specs/2026-05-16-persona-visual-post-setup-management-design.md
modified_files:
- apps/packages/ui/src/components/PersonaGarden/personaVisualManagementSummary.ts
- apps/packages/ui/src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts
- backlog/tasks/task-408 - Implement-Persona-Visual-management-summary-model.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 1 from the Persona Visual post-setup management design: add a pure shared UI helper that derives PersonaVisualManagementSummary and attention rows from existing Persona Garden visual pack, candidate, library, import/export job, and generation-readiness state. Keep this slice frontend-only with deterministic tests and no backend/API behavior changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Slice 1 of the Persona Visual post-setup management design. Added a pure shared UI helper that derives PersonaVisualManagementSummary plus attention rows from existing Persona Garden visual state: packs, active pack, selected validation errors, generated candidates, library items, import/export jobs, and generation readiness. Added deterministic Vitest coverage for empty state, active-pack dedupe, validation/candidate attention, import/export completion attention, library stale/unavailable state, generation unavailability, pending jobs, and failed jobs.

Verification:
- RED: bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts failed because ../personaVisualManagementSummary did not exist.
- GREEN: bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts passed 5 tests.
- Focused regression: bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx passed 59 tests.
- git diff --check passed.
- Full package TypeScript check was attempted with bunx tsc --noEmit -p tsconfig.json and failed on existing unrelated repo-wide type debt outside this slice; no errors referenced the new Persona Visual helper files.
- Bandit skipped because this slice only changes TypeScript frontend files and Backlog metadata, with no Python touched.
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
