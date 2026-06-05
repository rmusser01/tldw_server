---
id: TASK-517
title: Implement calendar database foundation
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-05 19:58'
labels:
  - implementation
  - calendar
  - backend
  - database
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
  - Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the Calendar module implementation plan: create Calendar_DB.py, Calendar package constants/errors, and DB unit tests for schema, local calendar creation, owner membership, provider-owned guards, tombstones, account secret references, and binding sync metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Calendar database foundation for Task 1, then hardened account cleanup semantics after spec compliance review.

Files changed:
- tldw_Server_API/app/core/DB_Management/Calendar_DB.py
- tldw_Server_API/app/core/Calendar/__init__.py
- tldw_Server_API/app/core/Calendar/constants.py
- tldw_Server_API/app/core/Calendar/errors.py
- tldw_Server_API/tests/Calendar/unit/test_calendar_db.py

Initial verification:
- Red phase confirmed: `python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_db.py -v` failed before implementation because `tldw_Server_API.app.core.DB_Management.Calendar_DB` did not exist.
- Green phase: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_db.py -v` -> 6 passed, 7 warnings.
- Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/Calendar_DB.py tldw_Server_API/app/core/Calendar -f json -o /tmp/bandit_calendar_task517.json` -> 0 findings.

Follow-up fix:
- `delete_secret_ref_in_connection()` now wipes `encrypted_payload` while tombstoning secret rows, so revoke/delete removes credential material rather than only hiding the row.
- Imported provider-row destructive cleanup and remote tombstone cleanup now detach copied local items before deleting provider-owned rows, and the new schema declares `copied_from_item_id ... ON DELETE SET NULL` for fresh databases.
- Added regression coverage for secret payload wiping after both revoke and delete, destructive account cleanup preserving copied tldw-owned items, and tombstone cleanup preserving copied tldw-owned items.

Follow-up verification:
- Red phase confirmed: the new regression tests failed before the fix with the secret payload still present and copied-item FK failures on destructive cleanup/tombstone cleanup.
- Green phase: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_db.py -v` -> 10 passed, 7 warnings.
- Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/Calendar_DB.py tldw_Server_API/app/core/Calendar -f json -o /tmp/bandit_calendar_task517_fix.json` -> 0 findings.

Known skips or concerns:
- Focused this slice on repository/schema methods only; no API router, Pydantic schemas, recurrence service, frontend, provider adapter, or sync worker work was included by design.
- Existing pytest run emits unrelated project warning/log output; the focused Calendar tests pass.
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
