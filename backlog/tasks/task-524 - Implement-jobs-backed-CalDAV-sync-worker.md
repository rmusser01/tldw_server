---
id: TASK-524
title: Implement jobs-backed CalDAV sync worker
status: Done
labels:
- implementation
- calendar
- backend
- jobs
- caldav
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Calendar/calendar_sync_worker.py
- tldw_Server_API/app/services/calendar_sync_scheduler.py
- tldw_Server_API/app/services/shutdown_calendar_sync_worker.py
- tldw_Server_API/app/core/Calendar/calendar_service.py
- tldw_Server_API/app/api/v1/endpoints/calendar.py
- tldw_Server_API/app/api/v1/schemas/calendar_schemas.py
- tldw_Server_API/app/core/Calendar/providers/caldav.py
- tldw_Server_API/app/core/DB_Management/Calendar_DB.py
- tldw_Server_API/app/services/startup_content_jobs_pollers.py
- tldw_Server_API/app/services/startup_worker_groups.py
- tldw_Server_API/tests/Calendar/unit/test_calendar_sync_worker.py
- tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 8 from the Calendar module plan: Jobs-backed Calendar sync queue helper, worker, optional scheduler bridge, startup/shutdown wiring, and focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Task 8. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py tldw_Server_API/tests/Calendar/unit/test_calendar_db.py tldw_Server_API/tests/Calendar/unit/test_calendar_sync_worker.py tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v` -> 65 passed. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/calendar.py tldw_Server_API/app/api/v1/schemas/calendar_schemas.py tldw_Server_API/app/core/Calendar/calendar_service.py tldw_Server_API/app/core/Calendar/calendar_sync_worker.py tldw_Server_API/app/core/Calendar/providers/caldav.py tldw_Server_API/app/core/DB_Management/Calendar_DB.py tldw_Server_API/app/services/calendar_sync_scheduler.py tldw_Server_API/app/services/shutdown_calendar_sync_worker.py tldw_Server_API/app/services/startup_content_jobs_pollers.py tldw_Server_API/app/services/startup_worker_groups.py -f json -o /tmp/bandit_calendar_task8.json` -> 0 findings.
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
