---
id: TASK-519
title: Implement calendar recurrence views and scheduled-task projections
status: Done
labels:
- implementation
- calendar
- backend
- recurrence
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the Calendar module implementation plan: bounded recurrence expansion, agenda/week view service, scheduled-task linked projections, and recurrence/unit/property tests without adding API/frontend/CalDAV sync scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Add bounded local recurrence expansion for daily, weekly-by-weekday, monthly-by-date, interval, count, until, and all-day DST-stable cases.
- [x] Add backend agenda/week view service that expands Calendar rows through existing Calendar permissions.
- [x] Add read-only scheduled-task linked projections from `ScheduledTasksControlPlaneService.list_tasks()`.
- [x] Keep Task 3 scoped to backend recurrence/view projections; no API/router/frontend/reminder handoff/CalDAV sync work added.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Calendar.recurrence` as an explicit local recurrence subset over `python-dateutil`, with required query-window validation and expansion caps.
- Added `Calendar.view_service` for `agenda`, `week`, `expand_items_window`, and `load_scheduled_task_projections`.
- Added minimal `CalendarDatabase` helpers for the already-existing `calendar_recurrences` table: `upsert_recurrence`, `list_recurrences_for_items`, and `list_items_for_expansion`. These were necessary so recurring master rows that start before the query window can be loaded through the DB abstraction and expanded without raw SQL in the service layer.
- TDD red run before implementation: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_recurrence.py tldw_Server_API/tests/Calendar/property/test_calendar_recurrence_properties.py tldw_Server_API/tests/Calendar/unit/test_calendar_service.py -v` -> 11 failed, 14 passed; failures were missing `Calendar.recurrence` and `Calendar.view_service`.
- Follow-up review fix: raw ISO timestamp comparisons in `CalendarDatabase.list_items_for_expansion()` could drop valid offset-aware events before Python-aware overlap checks. Added failing regressions for offset-aware agenda/week inclusion, broadened candidate loading to a bounded widened date-prefix range per readable calendar, kept authoritative overlap in `CalendarViewService`, and added `idx_calendar_recurrences_item`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented Task 3 recurrence and backend view expansion. Local recurrence supports daily, weekly weekday lists, monthly-by-date skipping, interval/count/until bounds, all-day date-stable expansion across DST, max query-window rejection, and capped occurrence expansion.
- Implemented agenda/week view dataclasses and service methods, permission-aware Calendar item expansion through `CalendarService`, provider tombstones hidden by default, and read-only scheduled-task linked projections with `source_owner/read_only_reason == "linked_projection"`.
- Added minimal `CalendarDatabase` helpers for recurrence persistence and expansion candidate loading, including `upsert_recurrence`, `list_recurrences_for_items`, `list_items_for_expansion`, and `idx_calendar_recurrences_item`.
- TDD red run before implementation: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_recurrence.py tldw_Server_API/tests/Calendar/property/test_calendar_recurrence_properties.py tldw_Server_API/tests/Calendar/unit/test_calendar_service.py -v` -> 11 failed, 14 passed; failures were missing recurrence/view services.
- Initial verification: recurrence/property/service tests -> 25 passed, 7 warnings; Calendar DB + permissions + service + recurrence/property tests -> 53 passed, 7 warnings; Bandit `/tmp/bandit_calendar_task519.json` -> exit 0, 0 findings; compileall over Calendar + Calendar_DB -> exit 0.
- Follow-up review fix: raw ISO timestamp comparisons in `CalendarDatabase.list_items_for_expansion()` could drop valid offset-aware events before Python-aware overlap checks. Added regressions for offset-aware agenda/week inclusion and all-day date-only agenda behavior, broadened bounded candidate loading, and kept authoritative overlap in `CalendarViewService`.
- Follow-up verification: offset-aware red regression run -> 2 failed, 1 passed; green regression run -> 3 passed, 7 warnings; recurrence/property/service tests -> 28 passed, 7 warnings; Calendar DB + permissions + service + recurrence/property tests -> 56 passed, 7 warnings; Bandit `/tmp/bandit_calendar_task519.json` -> exit 0, 0 findings.
- Final local review evidence: original offset-aware repro now returns `['Offset event']`; fresh Calendar DB + permissions + service + recurrence/property tests -> 56 passed, 7 warnings; Bandit `/tmp/bandit_calendar_task519_final_review.json` JSON totals show 0 findings.
- Spec review: passed.
- Code-quality re-review: no Critical or Important findings. Minor follow-ups noted for later API-facing work: raw RRULE parsing should reject unsupported keys and wrap invalid numeric fields in `CalendarValidationError` before API accepts raw recurrence strings; `include_provider_tombstones=True` remains intentionally non-functional until a separate permission-aware tombstone read path exists.
- Known skips/blockers: none for this Task 3 slice. API/router/frontend/reminder-handoff/CalDAV sync remain intentionally out of scope for later tasks.
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
