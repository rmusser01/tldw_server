---
id: TASK-520
title: Expose local calendar API and reminder handoff
status: Done
labels:
- implementation
- calendar
- backend
- api
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- tldw_Server_API/app/api/v1/schemas/calendar_schemas.py
- tldw_Server_API/app/api/v1/endpoints/calendar.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/app/core/Calendar/recurrence.py
- tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from the Calendar module implementation plan: Pydantic calendar schemas, thin FastAPI calendar router, router group registration, reminder handoff endpoint, and integration tests for local calendars/items/views/annotations/links/copy/privacy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 4 execution notes: Red run verified missing endpoint module. Implemented calendar schemas/router, registered content/minimal route specs, and hardened raw RRULE parsing for unsupported keys and bad numeric/UNTIL values. Integration API test command now passes: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v (14 passed).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 4: added Calendar API schemas and thin FastAPI router for local calendars, memberships, items, agenda views, annotations, links, provider-item copy, reminder handoff, and external account/binding/sync placeholders. Registered the calendar route in content and minimal router groups. Hardened raw RRULE parsing so unsupported keys and invalid numeric/UNTIL values are surfaced as CalendarValidationError/client errors instead of raw ValueError/500. Added permission-aware owner checks for external account/binding/sync placeholder routes. Verification: red API test failed on missing endpoint first; additional external placeholder ownership tests failed before the guard and passed after it. Final runs passed: calendar API integration 16 passed, OpenAPI config/jobs 4 passed, Calendar unit/property suite 56 passed, Bandit exited 0 with results [].
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
