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
- [x] Pure Persona Visual management summary model derives active-pack identity, pack lifecycle counts, and attention rows from existing Persona Garden editor state without backend or API changes.
- [x] Attention model covers invalid selected manifests, generated candidate review/failure, stale or unavailable library sources, generation unavailability, import/export completion, pending jobs, and failed terminal jobs.
- [x] Tests cover empty state, active-pack dedupe including duplicate active-pack inputs, validation/candidate attention, import/export completion, deleted import preview failure, conflicting job-state precedence, library source state, generation unavailability, pending jobs, and failed jobs.
- [x] Focused Persona Garden visual regression tests pass and verification details are recorded in the Final Summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Slice 1 of the Persona Visual post-setup management design. Added a pure shared UI helper that derives PersonaVisualManagementSummary plus attention rows from existing Persona Garden visual state: packs, active pack, selected validation errors, generated candidates, library items, import/export jobs, and generation readiness. Added deterministic Vitest coverage for empty state, active-pack dedupe, validation/candidate attention, import/export completion attention, library stale/unavailable state, generation unavailability, pending jobs, failed jobs, deleted import preview failure, and conflicting job-state failure precedence.

Review follow-up:
- Verified and fixed the deleted import-preview terminal state so it contributes to failed job attention rather than disappearing from the management model.
- Replaced OR-based job classification with canonical state precedence where failure terminal states win over pending/completed conflicts.
- Added the missing actual dedupe regression where the same active pack is present in both active_pack and packs.
- Simplified pack count initialization and completed the Backlog AC/DoD metadata.

Verification:
- RED: bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts failed before the helper existed.
- Review RED: bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts failed on deleted preview and conflicting job status regressions before the review fixes.
- GREEN: bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts passed 8 tests.
- Focused regression: bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx passed 62 tests.
- git diff --check passed.
- Full package TypeScript check was attempted with bunx tsc --noEmit -p tsconfig.json and failed on existing unrelated repo-wide type debt outside this slice; no errors referenced the new Persona Visual helper files.
- Bandit skipped because this slice only changes TypeScript frontend files and Backlog metadata, with no Python touched.
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
