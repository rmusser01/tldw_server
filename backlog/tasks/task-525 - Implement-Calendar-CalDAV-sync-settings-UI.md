---
id: TASK-525
title: Implement Calendar CalDAV sync settings UI
status: Done
labels:
- implementation
- calendar
- frontend
- caldav
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Calendar/CalendarSyncSettings.tsx
- apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarPage.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarFilterRail.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarItemDrawer.tsx
- apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx
- apps/packages/ui/src/services/calendar.ts
- apps/packages/ui/src/services/__tests__/calendar.test.ts
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 9 from the Calendar module plan: frontend CalDAV sync settings, account/discovery/binding/sync UI, provider-owned drawer UX tightening, service contract updates, and focused Vitest coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Calendar CalDAV sync settings UI and service contract updates. Added a Sync view in the Calendar page, account add/verify flow, discovery results, local calendar binding with lookback/lookahead windows, manual sync queueing, account revoke/delete actions, and focused tests for secret hygiene and sync trigger payloads. The existing provider-owned drawer behavior was covered by CalendarItemDrawer tests; this slice also fixed local Calendar TypeScript issues in the filter rail, drawer source-owner narrowing, and calendar service view-query cast found during verification.

Verification:
- RED: bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx apps/packages/ui/src/services/__tests__/calendar.test.ts failed for missing CalendarSyncSettings and dropped sync trigger body.
- PASS: bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx apps/packages/ui/src/services/__tests__/calendar.test.ts -> 2 files, 12 tests passed.
- PASS: bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx apps/packages/ui/src/services/__tests__/calendar.test.ts -> 4 files, 25 tests passed.
- PASS: git diff --check on touched frontend/plan files.
- INFO: bunx tsc --noEmit --project apps/packages/ui/tsconfig.json --pretty false initially stops on the repo tsconfig baseUrl deprecation. With --ignoreDeprecations 6.0, it still fails on existing repo-wide UI type issues outside this slice; the Calendar-specific errors reported during the first compile pass were fixed and no Calendar paths appear in the final typecheck error list.

Bandit: not applicable for this frontend-only slice; no Python code was touched.
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
