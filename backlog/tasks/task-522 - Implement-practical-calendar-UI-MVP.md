---
id: TASK-522
title: Implement practical calendar UI MVP
status: In Progress
labels:
- implementation
- calendar
- frontend
- ui
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Calendar/CalendarPage.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarAgenda.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarWeekView.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarFilterRail.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarItemDrawer.tsx
- apps/packages/ui/src/components/Option/Calendar/CalendarOwnershipBadge.tsx
- apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx
- apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx
- apps/packages/ui/src/components/Option/Calendar/__tests__/root-dom-setup.ts
- apps/packages/ui/src/services/calendar.ts
- apps/packages/ui/src/services/__tests__/calendar.test.ts
- tldw_Server_API/app/api/v1/endpoints/calendar.py
- tldw_Server_API/app/api/v1/schemas/calendar_schemas.py
- tldw_Server_API/app/core/Calendar/view_service.py
- tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
- vitest.config.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the Calendar module implementation plan: practical Calendar UI MVP with page shell, agenda, week view, filter rail, item drawer, ownership badge, and focused UI tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation complete for Task 6 frontend UI scope. Added the CalendarPage workspace shell, agenda list, week grid, filter rail, item drawer, ownership badge, and focused UI tests. Added a scoped root Vitest config and Calendar-only DOM setup because the required root-level `bunx vitest` UI command uses a temporary Vitest package that cannot resolve the app workspace `jsdom` environment directly.

Verification:
- Initial Task 6 root UI run failed before the root DOM setup existed because `jsdom` could not be resolved by the temporary root `bunx` Vitest package.
- Review-fix root UI/service run `bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx` passed with 3 files / 22 tests.
- Review-fix UI-package run from `apps/packages/ui`, `bunx vitest run src/services/__tests__/calendar.test.ts src/components/Option/Calendar/__tests__/CalendarPage.test.tsx src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx`, passed with 3 files / 22 tests.
- Review-fix backend run `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_service.py tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v` passed with 39 tests.
- Review-fix Bandit run `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/calendar.py tldw_Server_API/app/api/v1/schemas/calendar_schemas.py tldw_Server_API/app/core/Calendar/view_service.py -f json -o /tmp/bandit_calendar_task6_review_fix.json` reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a practical Calendar workspace MVP with agenda and week views, calendar/source/kind filters, ownership badges for local/org/provider/linked items, and a drawer for local item create/edit plus provider copy and linked-source handling. Added focused UI coverage for data loading, unsupported backend recovery, ownership labels, provider/linked read-only behavior, and local create/edit flows. Scoped root Vitest support was added for Calendar UI tests so the plan's root-level command works without changing app runtime behavior.
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
