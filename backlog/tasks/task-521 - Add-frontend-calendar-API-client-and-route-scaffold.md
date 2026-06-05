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
Implementation complete for Task 5 frontend scope. Added typed calendar API client and service tests, minimal CalendarPage placeholder because Task 6 UI component did not exist, option/Next route wrappers, `/calendar` route registry entry, and `option:calendar.nav` locale key. Follow-up review remediation added recursive secret redaction for non-create/verify payloads, required delete targets with provider/read-only mutation context, and removed the broad root Vitest config by switching the calendar service/test imports to relative paths. Verification: initial red run failed as expected with missing `@/services/calendar`; final root and UI-package runs passed with 1 file / 8 tests. Bandit not applicable: frontend-only TypeScript/TSX changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Task 5 frontend calendar scaffold: typed calendar service client, contract tests, minimal placeholder CalendarPage, option and Next route wrappers, route registry navigation, and the `calendar.nav` locale key. Review remediation hardened non-create/verify secret stripping, required delete calls to carry item mutation context before local/provider read-only checks, and removed the temporary root Vitest shim so broad root test behavior is not hijacked. Verification recorded: initial red run failed on missing `@/services/calendar`; final `bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts` and UI-package `bunx vitest run src/services/__tests__/calendar.test.ts` both passed with 1 file and 8 tests. Bandit not applicable because no backend Python was touched.
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
