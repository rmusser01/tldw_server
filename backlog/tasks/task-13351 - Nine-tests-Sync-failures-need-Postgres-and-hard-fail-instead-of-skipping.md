---
id: TASK-13351
title: Nine tests/Sync failures need Postgres and hard-fail instead of skipping
status: To Do
assignee: []
created_date: '2026-09-23 00:52'
labels:
  - tests
  - sync
  - postgres
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enumerated by junit-xml under TASK-13344. All reproduce deterministically; none are parallelism artifacts.

CLAUDE.md states: 'Skip Postgres-dependent tests only when the fixture reports Postgres unavailable; never roll your own database setup.' These do not skip -- they fail, which is why tests/Sync looks worse than it is on a machine without Postgres.

  test_sync_v2_personal_context_certification.py
    ::test_postgres_two_connections_choose_exactly_one_existing_authority
       psycopg_pool.PoolTimeout: couldn't get a connection after 30.00 sec
       (this one also costs 30s of the suite's runtime by waiting for a pool that will never arrive)
    ::test_bootstrap_reuses_existing_nondefault_authority_without_creating_default
       PersonalContextBootstrapError: personal_context_snapshot_unavailable
       NOT postgres-named and uses the production_factories fixture -- needs a separate look to
       confirm whether it is the same environmental cause or a genuine defect.

  test_sync_v2_personal_context_conflicts.py
    ::test_postgres_candidate_attachment_replay_and_retention[skip]
    ::test_postgres_candidate_attachment_replay_and_retention[overwrite]
    ::test_postgres_candidate_attachment_replay_and_retention[duplicate_rename]
    ::test_stale_purge_after_failed_ingress_requires_refresh_and_reconfirmation[postgres-insertion]
    ::test_stale_purge_after_failed_ingress_requires_refresh_and_reconfirmation[postgres-preflight]
       all PersonalContextBootstrapError: personal_context_snapshot_unavailable

  test_sync_v2_store.py::test_postgres_personal_context_receipt_locks_binding_before_upsert
       SyncStoreError: personal_context_link_binding_stale
       TASK-13344 recorded this as link_state fixture drift, red since 2026-09-03 (8c97f181e5).
       Worth confirming whether it is fixture drift or simply the absent backend.

  test_sync_v2_notes_organization_postgres_contract.py
    ::test_postgres_predecessor_selector_uses_dataset_cursor_and_nonapplied_status
       Asserts the emitted SQL, gets 'SQLite query execution failed' -- it ran against SQLite
       because Postgres was unavailable, then compared Postgres dialect text.

Source: TASK-13344 AC2/AC3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each test skips with a reason when the fixture reports Postgres unavailable, rather than failing
- [ ] #2 test_bootstrap_reuses_existing_nondefault_authority and the store link-binding test are confirmed environmental or filed as defects
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
