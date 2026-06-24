---
id: TASK-12017
title: Implement Jobs backend parity refactor first slice
status: In Progress
created_date: 2026-06-24 21:44
labels:
- jobs
- implementation
- refactor
priority: medium
references:
- TASK-12015
- TASK-12016
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- Docs/superpowers/plans/2026-06-24-jobs-backend-parity-refactor-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- Docs/superpowers/plans/2026-06-24-jobs-backend-parity-refactor-implementation-plan.md
modified_files:
- Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md
- tldw_Server_API/tests/Jobs/parity/__init__.py
- tldw_Server_API/tests/Jobs/parity/scenarios.py
- tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py
- tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py
- tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py
- tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_postgres.py
- tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py
- tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py
- tldw_Server_API/app/core/Jobs/settings.py
- tldw_Server_API/tests/Jobs/test_jobs_settings.py
- backlog/tasks/task-12017 - Implement-Jobs-backend-parity-refactor-first-slice.md
updated_date: 2026-06-24 23:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the first safety-net PR from the Jobs backend parity implementation plan. Scope includes inventory, shared SQLite/Postgres parity scenarios, public admin and Chatbooks mapping contract tests, JobsSettings semantics, operation result contracts, and verification gates. Production SQL extraction is explicitly out of scope for this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 inventory created at Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md. It classifies admin direct SQL, read-model SQL, service/worker operational SQL, and first-slice domain mapping coverage.
Follow-up inventory update added the public admin stale-processing read-model boundary and clarified the Prompt Studio status dashboard first-slice action.
Follow-up inventory update made the stale-processing boundary explicit as GET /jobs/stale at jobs_admin.py:1513.
Task 2 started: adding shared Jobs backend parity scenario helpers and refactoring the owned SQLite/Postgres idempotency tests to call them. Applying the plan correction to update TASK-12017, not TASK-12016.
Task 2 completed: added shared Jobs backend parity scenario helpers under tldw_Server_API/tests/Jobs/parity and refactored the owned SQLite/Postgres idempotency tests to call shared scenarios. Verification: py_compile for scenarios.py exited 0; SQLite selected tests passed (3 passed, 19 warnings); Postgres selected tests exited 0 with 3 skips because Postgres was not reachable ("Postgres not reachable; skipping Postgres-backed tests"); git diff --check exited 0; Bandit full touched-scope report contained only low-severity B101 pytest assert findings, and the follow-up run with B101 excluded exited 0 with 0 findings.
Task 3 started: adding first SQLite and Postgres parity wrapper tests around the existing shared Jobs parity scenarios. Applying the plan correction to update TASK-12017 only.
Task 3 completed: added SQLite and Postgres parity wrapper tests around the shared Jobs parity scenarios. Verification: `RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py -q` exited 0 with 6 passed and 26 warnings; `RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py -q` exited 0 with 6 skipped and 14 warnings because Postgres was unavailable; follow-up `-q -rs` confirmed the skip reason: "Postgres not reachable; skipping Postgres-backed tests". `git diff --check` exited 0. Bandit on the touched wrapper test files exited 0, and the B101-excluded follow-up reported no issues. No production code was changed.
Task 4 completed: added public Jobs admin and Chatbooks adapter contract tests, then corrected the new admin list contract assertion after validation showed `/api/v1/jobs/list` intentionally returns the existing `JobItem` public field set without storage-only owner fields. Code-quality review findings were addressed by forcing the SQLite admin contract away from ambient `JOBS_DB_URL`, faking the Chatbooks Jobs manager for mapping-only tests, asserting the adapter query shape, and covering terminal export/import status preservation. Verification: `RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py -q` exited 0 with 20 passed and 52 warnings; `git diff --check` with the new file included exited 0; Bandit on the touched test files with B101 excluded exited 0.
Task 5 started: adding the JobsSettings snapshot/refresh contract with TDD. Applying the required correction that refresh() preserves construction-time db_url/db_path from the existing snapshot while refreshing JSON, lease, outbox, counters, and allowed queue values from the supplied environment. Updating TASK-12017 only.
Task 5 completed: added tldw_Server_API.app.core.Jobs.settings with immutable JobsSettings snapshots, setting-mode classification, domain-aware allowed queue merging without duplicates, and refresh semantics that preserve construction-time db_url/db_path while refreshing max_json_bytes, lease_max_seconds, events_outbox_enabled, counters_enabled, and allowed queue values. Added TDD coverage in tldw_Server_API/tests/Jobs/test_jobs_settings.py, including a regression assertion that a naive type(self).from_env(env) refresh would fail. Red verification: initial worktree run failed with ModuleNotFoundError for tldw_Server_API.app.core.Jobs.settings as expected. Verification: `RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_settings.py -q` exited 0 with 7 passed and 26 warnings; `git diff --check` exited 0 after intent-to-add for the new files; `python -m bandit -q -s B101 tldw_Server_API/app/core/Jobs/settings.py tldw_Server_API/tests/Jobs/test_jobs_settings.py` exited 0. Bandit initially flagged /tmp example path literals in the tests as B108; the examples were changed to neutral data/ paths and verification was rerun cleanly.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
