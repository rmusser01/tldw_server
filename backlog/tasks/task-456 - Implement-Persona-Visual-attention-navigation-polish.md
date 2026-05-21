---
id: TASK-456
title: Implement Persona Visual attention navigation polish
status: Done
labels:
- persona-visual
- webui
- frontend
references:
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/issues/1899
- https://github.com/rmusser01/tldw_server/issues/1769
- https://github.com/rmusser01/tldw_server/pull/1771
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Management header attention rows expose an action that focuses or scrolls to relevant existing controls for candidate review, import preview/commit, export download, validation/activation, personal library, and generation readiness.
- [x] #2 Navigation reuses existing VisualPackEditor refs/sections and does not add backend calls or change server behavior.
- [x] #3 Review-before-activation and explicit activation semantics remain unchanged.
- [x] #4 Focused shared UI tests cover supported attention target mappings and a no-op/no-target case.
- [x] #5 No Buddy animation, VN/CYOA, backend schema, generation orchestration, marketplace/shared-library, renderer, MCP provider execution, or auto-activation work is included.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Slice 4 from Docs/superpowers/specs/2026-05-16-persona-visual-post-setup-management-design.md: add focus/scroll navigation from Persona Visual management attention rows to existing VisualPackEditor controls and sections while preserving backend/API behavior and explicit review/activation semantics.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Slice 4 from the Persona Visual post-setup management design. Added a management-header attention action that routes the top attention row to existing VisualPackEditor sections or controls through refs. Candidate and job attention focuses the jobs/review section, library attention focuses the personal-library section, validation attention uses the existing activation controls focus path, import/export attention targets the portable actions section, generation attention targets the readiness panel, and pack failures target the pack basics section. The change is frontend-only and does not add backend calls or alter activation semantics.

Verification:
- `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx -t "management header|library attention|validation attention|generation attention|workspace sections"` passed with 5 focused tests.
- `bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` passed with 70 tests.
- `git diff --check` passed.
- `bunx tsc --noEmit -p tsconfig.json` was attempted from `apps/packages/ui` and failed on existing package-wide TypeScript debt outside the touched PersonaGarden files; no errors referenced the touched files in the visible output.
- Bandit is not applicable to this frontend-only TypeScript/Backlog slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Persona Visual attention navigation polish. The management header now exposes an attention action that focuses the relevant existing VisualPackEditor section/control for review, library, generation, validation, import/export, jobs, or pack-state attention without changing backend behavior.
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
