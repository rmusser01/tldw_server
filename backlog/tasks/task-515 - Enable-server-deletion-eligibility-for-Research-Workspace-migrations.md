---
id: TASK-515
title: Enable server deletion eligibility for Research Workspace migrations
status: Done
labels:
- research-workspace
- migration
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend follow-up that can safely emit client_delete_eligible=true for Research Workspace migration sessions after server-side read-back/integrity verification proves all declared payloads were accepted and recoverable. Until this exists, WebUI migration must retain local legacy data and show recovery copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migration sessions expose client_delete_eligible=true only after explicit server-side verification of all declared chunks and manifest hash.
- [x] #2 Client delete acknowledgement succeeds only for eligible finalized sessions with matching manifest hash.
- [x] #3 Conflicts and failed verification remain recoverable and do not mark sessions delete-eligible.
- [x] #4 Focused backend/API tests cover eligible, ineligible, mismatch, and idempotent retry paths.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added finalize-time read-back verification in `CharactersRAGDB` that compares persisted declared chunks against accepted chunk receipts by id, hash, byte count, and kind.
- `finalize_workspace_migration()` now sets `client_delete_eligible=true` only for non-empty, fully verified migrations and records verification details in the recovery manifest.
- Zero-declared-chunk sessions and receipt drift/corruption remain finalized but deletion-ineligible and recoverable.
- `record_workspace_migration_client_delete_ack()` now also requires finalized status, matching manifest hash, and deletion eligibility; repeated eligible acknowledgements remain accepted.
- Updated `Docs/Design/Research_Workspace_Migration_Protocol_API.md` to describe verified eligibility and ineligible recovery states.
- Verification: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py -q` passed with 10 tests and 5 warnings.
- Verification: `git diff --check` passed.
- Security: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py -f json -o /tmp/bandit_task515.json` exited 0; `/tmp/bandit_task515.json` has an empty `results` array.
- Known skip: no PostgreSQL-backed workspace migration test was run in this slice; the code uses the existing backend abstraction and stores a native bool for PostgreSQL, matching the existing schema.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented server-side deletion eligibility for Research Workspace migrations. Finalized sessions now authorize client legacy-storage deletion only after read-back verification proves all declared chunks have matching accepted receipts; ineligible and failed-verification sessions remain recoverable and cannot be acknowledged for local deletion.
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
