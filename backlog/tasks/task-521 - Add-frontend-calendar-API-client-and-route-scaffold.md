---
id: TASK-521
title: Add frontend calendar API client and route scaffold
status: Done
labels:
- implementation
- calendar
- frontend
- api-client
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- apps/packages/ui/src/services/calendar.ts
- apps/packages/ui/src/services/__tests__/calendar.test.ts
- apps/packages/ui/src/components/Option/Calendar/CalendarPage.tsx
- apps/tldw-frontend/extension/routes/option-calendar.tsx
- apps/tldw-frontend/pages/calendar.tsx
- apps/tldw-frontend/extension/routes/route-registry.tsx
- apps/packages/ui/src/public/_locales/en/option.json
- vitest.config.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from the Calendar module implementation plan: typed frontend calendar service client, service tests, option/Next route wrappers, route registry entry, and locale navigation key.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation complete for Task 5 frontend scope. Added typed calendar API client and service tests, minimal CalendarPage placeholder because Task 6 UI component did not exist, option/Next route wrappers, `/calendar` route registry entry, and `option:calendar.nav` locale key. Added root `vitest.config.ts` shim because the required root command `bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts` otherwise did not load the UI package alias config; the existing scheduled-tasks test showed the same root alias failure. Verification: initial red run failed as expected with missing `@/services/calendar`; final run passed with 1 file / 5 tests. Bandit not applicable: frontend-only TypeScript/TSX changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Task 5 frontend calendar scaffold: typed calendar service client, contract tests, minimal placeholder CalendarPage, option and Next route wrappers, route registry navigation, and the `calendar.nav` locale key. Added a root Vitest config shim so the required root-level service test command resolves UI aliases and excludes copied worktree tests. Verification recorded: initial red run failed on missing `@/services/calendar`; final `bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts` passed with 1 file and 5 tests. Bandit not applicable because no backend Python was touched.
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
