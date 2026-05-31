---
id: TASK-459
title: Implement Persona Buddy live-control frontend service
status: Done
labels:
- persona
- buddy
- frontend
- implementation
references:
- TASK-457
- Docs/superpowers/plans/2026-05-20-persona-buddy-interaction-text-slice.md
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
- 'issue #1510'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the Persona Buddy interaction text-slice plan: shared UI service for Persona live-control list/create/focus/stop endpoints, OpenAPI guard paths, server capability detection, and focused Vitest coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend live-control service normalizes backend summaries with safe defaults.
- [x] #2 Create/focus/stop service calls use authenticated tldw client paths and encode session IDs safely.
- [x] #3 OpenAPI guard includes the new live-control paths.
- [x] #4 Server capability normalization exposes Persona live-control availability.
- [x] #5 Focused service tests pass and verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `persona-live-control.ts` with safe response normalization, typed list/create/focus/stop helpers, default `resume_compatible` create policy, and encoded session IDs.
- Added focused Vitest coverage for service normalization and authenticated endpoint calls.
- Added live-control paths to the OpenAPI client guard and surfaced `hasPersonaLiveControl` in server capability normalization.
- Verification: `bunx vitest run src/services/__tests__/persona-live-control.test.ts src/services/__tests__/server-capabilities.test.ts` passed with 43 tests.
- Verification: `bun run verify:openapi` passed for the new live-control paths, with only the existing reviewed exception paths reported.
- Verification note: `bunx tsc --noEmit` still fails on pre-existing UI type debt outside this slice; no failures referenced the new live-control service or `hasPersonaLiveControl` field.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the shared Persona Buddy live-control frontend service and capability detection needed for the next hook/UI slice.
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
