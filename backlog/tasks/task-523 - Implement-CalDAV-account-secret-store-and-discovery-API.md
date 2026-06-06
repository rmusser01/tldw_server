---
id: TASK-523
title: Implement CalDAV account secret store and discovery API
status: Done
labels:
- implementation
- calendar
- backend
- security
- caldav
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Calendar/secret_store.py
- tldw_Server_API/app/core/Calendar/providers/__init__.py
- tldw_Server_API/app/core/Calendar/providers/caldav.py
- tldw_Server_API/app/core/DB_Management/Calendar_DB.py
- tldw_Server_API/app/api/v1/schemas/calendar_schemas.py
- tldw_Server_API/app/api/v1/endpoints/calendar.py
- tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py
- tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py
- tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 7 from the Calendar module plan: encrypted Calendar secret store, read-only CalDAV provider adapter/discovery, external account/binding API endpoints, and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 7 implemented: encrypted CalDAV account credentials, read-only CalDAV verify/discovery provider behavior, redacted account responses, account revoke/delete cleanup, binding management/status/events endpoints, and focused unit/integration/security verification.

Verification:
- source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py tldw_Server_API/tests/Calendar/integration/test_calendar_api.py tldw_Server_API/tests/Calendar/unit/test_calendar_db.py -v -> 58 passed.
- source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/calendar.py tldw_Server_API/app/api/v1/schemas/calendar_schemas.py tldw_Server_API/app/core/Calendar/secret_store.py tldw_Server_API/app/core/Calendar/providers/caldav.py tldw_Server_API/app/core/DB_Management/Calendar_DB.py -f json -o /tmp/bandit_calendar_task7.json -> 0 findings.
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
