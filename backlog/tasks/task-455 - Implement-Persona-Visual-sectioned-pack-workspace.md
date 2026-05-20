---
id: TASK-455
title: Implement Persona Visual sectioned pack workspace
status: In Progress
labels:
- persona
- persona-visual
- webui
- frontend
references:
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/issues/1894
- https://github.com/rmusser01/tldw_server/issues/1769
- https://github.com/rmusser01/tldw_server/pull/1771
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VisualPackEditor groups existing controls into clear Persona Visual workspace sections for pack basics/status, assets, animations, state mappings/fallbacks, authored triggers, validation/activation, jobs/review, and reuse/portability without backend behavior changes.
- [x] #2 Existing first-run setup, management header, import/export, generation, candidate review, validation, activation, library, and duplicate-to-persona controls remain available and keep current test coverage.
- [x] #3 Focused shared UI tests cover the section headings/landmarks and preservation of existing behavior.
- [x] #4 No Buddy animation, VN/CYOA, backend schema, generation orchestration, marketplace, shared-library, renderer, MCP provider execution, or auto-activation work is included.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Slice 3 from Docs/superpowers/specs/2026-05-16-persona-visual-post-setup-management-design.md: refactor the Persona Garden VisualPackEditor surface into clearer post-setup workspace sections while preserving existing behavior and backend/API contracts. Keep this Persona Visual only and leave attention navigation polish for the next slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Slice 3 from the Persona Visual post-setup management design. Added semantic section landmarks and compact headings around the existing VisualPackEditor control groups: pack basics/status, reuse and portability entry points, personal library, assets, portable archive/duplicate actions, state mappings/fallbacks, animations, authored triggers, validation/activation, and jobs/review. Existing controls remain in place and continue to use the current frontend state and backend/API contracts.

Verification:
- `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx -t "workspace sections|management header"` passed with 2 focused tests.
- `bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` passed with 67 tests.
- `git diff --check` passed.
- `bunx tsc --noEmit -p tsconfig.json` was attempted from `apps/packages/ui` and failed on existing package-wide TypeScript debt outside the touched PersonaGarden files; no errors referenced the touched files in the visible output.
- Bandit is not applicable to this frontend-only TypeScript/Backlog slice.

Review follow-up:
- Added the missing reuse/portability section heading.
- Converted the personal library container to a semantic section with a matching section test id and aria label.
- Removed duplicate personal-library heading description text.
- Hardened the new workspace-section test so unmocked API paths throw instead of rendering a silent error banner.
- Verification after review fixes: `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx -t "workspace sections|management header"` passed with 2 focused tests; `bunx vitest run src/components/PersonaGarden/__tests__/personaVisualManagementSummary.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` passed with 67 tests; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Persona Visual sectioned pack workspace for post-setup management. VisualPackEditor now exposes testable, accessible sections for the existing lifecycle areas without changing backend behavior, generation orchestration, activation semantics, or Buddy animation functionality.
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
