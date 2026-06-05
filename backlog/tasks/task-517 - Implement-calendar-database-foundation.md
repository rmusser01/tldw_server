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
Implemented the Calendar database foundation for Task 1, then hardened account cleanup and external import ownership semantics after follow-up reviews.

Files changed:
- tldw_Server_API/app/core/DB_Management/Calendar_DB.py
- tldw_Server_API/app/core/Calendar/__init__.py
- tldw_Server_API/app/core/Calendar/constants.py
- tldw_Server_API/app/core/Calendar/errors.py
- tldw_Server_API/tests/Calendar/unit/test_calendar_db.py

Initial implementation:
- Added the calendar schema, repository methods, Calendar constants/errors package, secret-reference storage, external accounts/bindings, provider-owned item import paths, tombstone handling, and sync-event records.
- Added focused unit tests for local calendar creation, automatic owner membership, provider-owned local write guards, remote tombstone visibility, account secret references, and binding sync metadata.
- Red phase confirmed missing module before implementation; green phase passed the focused Calendar DB tests; Bandit reported 0 findings.

Account cleanup follow-up:
- `delete_secret_ref_in_connection()` now wipes `encrypted_payload` while tombstoning secret rows, so revoke/delete removes credential material rather than only hiding the row.
- Imported provider-row destructive cleanup and remote tombstone cleanup now detach copied local items before deleting provider-owned rows, and the schema declares `copied_from_item_id ... ON DELETE SET NULL` for fresh databases.
- Added regression coverage for secret payload wiping after both revoke and delete, destructive account cleanup preserving copied tldw-owned items, and tombstone cleanup preserving copied tldw-owned items.

External import ownership follow-up:
- Personal external accounts can bind only to active private calendars owned by the account user; org calendars and other users' calendars are rejected.
- Provider upsert validates the binding, active account, and bound calendar, and rejects caller-supplied calendar IDs that do not match the binding.
- Secret refs are validated against tenant/user/provider on account creation; scoped resolve/delete helpers were added and account cleanup paths use scoped deletion.
- Soft-deleted bindings can be resurrected when rebinding the same account/remote calendar instead of failing the full unique constraint.
- Due-scan now joins account rows and excludes inactive, revoked, or deleted accounts.
- Added regressions for org/other-user binding rejection, provider upsert calendar mismatch, secret ref tenant/user/provider mismatch, scoped secret access mismatch, rebind after soft delete, non-active account binding rejection, and due-scan account-state filtering.

Local code-quality gate after reviewer quota limit:
- Re-read the prior reviewer findings against the current code and verified the critical/important issues are addressed in the Task 1 repository layer.
- Fresh verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_db.py -v` -> 22 passed, 7 warnings.
- Fresh security check: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/Calendar_DB.py tldw_Server_API/app/core/Calendar -f json -o /tmp/bandit_calendar_task517_local_review.json`; JSON totals show 0 findings.

Known skips or concerns:
- Focused this slice on repository/schema methods only; no API router, Pydantic schemas, recurrence service, frontend, provider adapter, or sync worker work was included by design.
- Existing pytest run emits unrelated project warning/log output; the focused Calendar tests pass.
- The unscoped secret-store convenience methods remain for direct repository use/tests, while account creation and cleanup paths use scoped validation/deletion.
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
