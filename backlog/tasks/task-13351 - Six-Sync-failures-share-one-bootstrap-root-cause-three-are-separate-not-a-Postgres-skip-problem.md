---
id: TASK-13351
title: >-
  Six Sync failures share one bootstrap root cause; three are separate (not a
  Postgres skip problem)
status: Done
assignee: []
created_date: '2026-09-23 00:52'
updated_date: '2026-09-23 02:23'
labels:
  - tests
  - sync
  - postgres
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CORRECTED. The original premise of this task -- that nine tests need Postgres and hard-fail instead of skipping -- is WRONG, and the correction matters more than the original framing.

Postgres IS running on this machine (port 5432 is open), and the fixture chain already skips correctly when it is not: pg_database_config -> pg_temp_db -> pg_server, and pg_server calls pytest.skip("Postgres not reachable") at tests/_plugins/postgres.py:188. Nothing needs a skip guard added. These tests fail for real reasons.

They split into two families.

FAMILY A -- ONE shared root cause, six tests, and one of them is not a Postgres test at all:

    SyncStoreError: Notes suggestion authority requires the owned default dataset
    raised at core/Sync/v2/service.py:2519 in prepare_notes_suggestion_authority,
    reached from profile.py:707 _bind_personal_context_dataset
    <- profile.py:531 bootstrap_personal_context

  which then surfaces to the caller as
    PersonalContextBootstrapError: personal_context_snapshot_unavailable

  Affected:
    conflicts::test_postgres_candidate_attachment_replay_and_retention[skip]
    conflicts::test_postgres_candidate_attachment_replay_and_retention[overwrite]
    conflicts::test_postgres_candidate_attachment_replay_and_retention[duplicate_rename]
    conflicts::test_stale_purge_after_failed_ingress_requires_refresh_and_reconfirmation[postgres-insertion]
    conflicts::test_stale_purge_after_failed_ingress_requires_refresh_and_reconfirmation[postgres-preflight]
    certification::test_bootstrap_reuses_existing_nondefault_authority_without_creating_default

  That last one uses the production_factories fixture, which sets USER_DB_BASE_DIR and runs
  entirely on SQLite under tmp_path -- no Postgres anywhere. It was grouped with the Postgres
  tests only because it reports the same PersonalContextBootstrapError. Confirming that before
  adding a skip is what stopped six tests being silently hidden behind a guard that would have
  been wrong.

  prepare_notes_suggestion_authority requires the dataset to satisfy ALL of: it exists, is
  owned by the user, scope_type == "personal", metadata["default_personal"] is True, and
  metadata["client_family"] == "chatbook". One of those is not being met during bootstrap.
  Whether the precondition or the bootstrap is at fault is the open question.

FAMILY B -- three genuinely separate failures:

  store::test_postgres_personal_context_receipt_locks_binding_before_upsert
    SyncStoreError: personal_context_link_binding_stale
    (TASK-13344 recorded this as link_state fixture drift, red since 2026-09-03, 8c97f181e5)

  certification::test_postgres_two_connections_choose_exactly_one_existing_authority
    psycopg_pool.PoolTimeout: couldn't get a connection after 30.00 sec
    The test sets pool_size = 2 and max_overflow = 0 and is ABOUT two connections racing for
    one authority, so a timeout means either a connection is never released or the test's
    expectation of the pool is wrong. Possible connection leak; worth its own look. It also
    costs 30 seconds of suite runtime every run.

  notes_organization_postgres_contract::test_postgres_predecessor_selector_uses_dataset_cursor_and_nonapplied_status
    Asserts the emitted SQL text and gets "SQLite query execution failed", so it ran against
    SQLite and then compared Postgres dialect SQL. This one IS a backend-selection problem,
    but the backend is available -- the test is not asking for it.

Source: TASK-13344 triage, corrected while working the fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Family A's shared cause is resolved: prepare_notes_suggestion_authority and bootstrap_personal_context agree on the dataset contract
- [x] #2 The three Family B failures are each classified and fixed or filed
- [x] #3 The PoolTimeout test is checked for a connection leak, since it is about two connections racing
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC1 DONE in c09e47c98e -- Family A's shared cause resolved; it was TWO bugs, not one. tests/Sync 14 failures to 8. AC2 and AC3 remain.

Both bugs sat on the path bootstrap_personal_context -> _bind_personal_context_dataset -> prepare_notes_suggestion_authority, and both surfaced as "personal_context_snapshot_unavailable" because profile.py:552 maps any other SyncStoreError to it. That mapping is why this looked like a snapshot problem for so long.

BUG 1 -- the fence demanded the chatbook default of a dataset not required to be one.
  Sync_DB.personal_context_bootstrap_transaction (Sync_DB.py:4433) deliberately selects
  bound_rows[0] as the authority whatever its markers, and line 4432 sets
  require_chatbook_default = not bound_rows and bool(default_rows) -- an explicit
  relaxation for precisely that case. That landed 2026-09-04 (0507921d63), the same day
  as the test asserting it. The fence call was added to _bind_personal_context_dataset
  13 days later (f0536ee5cf, 2026-09-17) and applied the default requirement
  unconditionally, refusing a legitimately bound non-default authority.
  Fix: prepare_notes_suggestion_authority takes require_default, still True by default,
  and the caller passes `existing_state is None` -- the binding it already computed one
  line earlier at profile.py:700.

BUG 2 -- a Postgres-only transaction-visibility defect.
  The fence read through self.store, while the dataset existed only inside the bootstrap
  guard's transaction. On PostgreSQL that uncommitted row is invisible to any other
  connection, so get_dataset returned None and the fence refused a dataset that plainly
  existed. On SQLite it resolved, which is why only the postgres-parametrised cases
  failed after Bug 1 was fixed.
  Fix: the fence reads through a store passed by the caller; _bind_personal_context_dataset
  hands it selected_store, which carries the transaction. That required moving the
  selected_store assignment above the fence call, where it had been one line below.

METHOD. Both were found by instrumenting the five conditions rather than reasoning about
them. First pass: metadata carried only ['personal_context'], default markers absent ->
Bug 1. Second pass, after fixing Bug 1: actual_is_none True with store_has_connection
False -> Bug 2. Worth recording because the first instrumentation attempt printed nothing
and I nearly concluded the code path was not reached; writing the diagnostic to a file
instead of stdout is what made it visible under pytest.

SECURITY NOTE. Bug 1's fix relaxes a fence, so its boundary is pinned separately in
tests/Sync/test_sync_v2_notes_suggestion_authority_fence.py: ownership and personal scope
are unconditional and asserted under BOTH values of require_default, as is the
dataset-not-visible case. Those tests were verified to be capable of failing -- deleting
the owner check from the fence trips two of them, and it restores clean. The strict
behaviour remains the default, so any new caller gets the full fence.

STILL OPEN (Family B):
  AC2 -- store::test_postgres_personal_context_receipt_locks_binding_before_upsert
         (personal_context_link_binding_stale) and
         notes_organization_postgres_contract::test_postgres_predecessor_selector_uses_
         dataset_cursor_and_nonapplied_status (ran on SQLite, compared Postgres SQL).
  AC3 -- certification::test_postgres_two_connections_choose_exactly_one_existing_authority
         still times out. It sets pool_size = 2, max_overflow = 0 and is ABOUT two
         connections racing for one authority, so a 30s PoolTimeout means either a
         connection is never released or the test's expectation of the pool is wrong.
         It also costs 30 seconds of every suite run. Given Bug 2 above was a connection
         -scoping defect on this same path, a leak is plausible and worth checking first.

AC2 and AC3 DONE in d389329118. tests/Sync 8 failures to 5.

AC3 -- the PoolTimeout was NOT a connection leak. The test deadlocks against itself: the pool is deliberately sized at 2 with no overflow so the two binds genuinely contend, the test checks out BOTH connections to drive the race, and then verified with list_datasets_for_user while still holding them. That method takes no connection argument and reaches for the pool, so it waited 30 seconds for a connection the test itself held. bind_personal_context_dataset threads its connection correctly throughout -- checked before concluding, because Bug 2 above made a leak plausible. The connections are now returned after the race and before verification; 30.11s to 0.98s, and every suite run gets 30 seconds back.

AC2 -- both classified as test drift against deliberate product changes:
  predecessor selector asserted one exact SQL string. The selector gained a clause
  exempting personal-context envelopes under unresolved conflict review, so such an
  envelope does not block projection. Each predicate the test is named for is now
  asserted separately, plus the exemption, because a blob comparison only reports
  'index 0 differs' and does not name the clause that moved.
  link receipt: the fake backend's dataset row carried no link_state, which
  complete_personal_context_link_receipt has required to be bootstrap_pending or
  complete since 2026-09-03 (Sync_DB.py:3972). Added bootstrap_pending, the
  pre-receipt state.

Whole directory now 5 failed / 2995 passed: the three exchange-gate fixtures (TASK-13349), the registry inconsistency that is correctly failing (TASK-13350), and one known regex drift.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All nine resolved or correctly reattributed. Six shared two real bugs in Personal Context bootstrap, one of them a Postgres-only transaction-visibility defect. Of the remaining three, the PoolTimeout was the test holding the pool it needed rather than a leak, and two were test drift against deliberate product changes. tests/Sync went 14 failures to 5 across this task.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
