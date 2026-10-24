---
id: TASK-13901
title: 'Tests: five PostgreSQL tests have failed on dev since #3084'
status: Done
labels:
- chat
- postgres
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13861's PR D (#3300), whose sweep of the pg_restricted_backend users hit them. Five tests fail on dev: (1) tests/DB_Management/test_chacha_postgres_sequence_ownership.py::test_chacha_sync_continues_after_table_error_without_settling_caller: DatabaseError: PostgreSQL query execution failed, raised in the call phase at :251 (db._sync_postgres_sequences(conn) -> postgres_sequence_maintenance.sync_owned_postgres_sequences -> backend.execute of the test's patched MAX(id) query), not at teardown (the #3300 reviewer confirmed this), so the test expects the sync to continue past a failing table and the code does not. (2) tests/DB_Management/test_history_selection_transactions.py::test_native_admission_capture_refuses_changed_input_or_capture[sqlite-metadata] and (3) [postgres-metadata]: assert 2 == 1 at :1022 (the input message's version after a metadata-only change; 'Metadata must be fenced without a physical version bump'). (4) test_native_admission_capture_refuses_metadata_change_and_restore[sqlite] and (5) [postgres]: assert 3 == 1 at :1050 ('Metadata ABA leaves the physical version unchanged'). Cases 2-5 fail on SQLite too. Measured on 2026-10-22: all five fail identically on origin/dev ba84588d12, on d47c9ccf58 (#3084's merge, before PR C #3297), and on eb0c8c9e52, the #3084 commit that added the sequence-maintenance change and these tests, which contains none of TASK-13861's code (git merge-base --is-ancestor ad946dd687 eb0c8c9e52 is false). On 5c2e2b4c80 (before #3084) the sequence test passes and the four history cases do not exist yet. Not caused by TASK-13861.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the five tests passes on dev, or is fixed or removed with a recorded reason
- [x] #2 The cause is identified and stated in the task
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Case 1 cause (measured on 617094d014; the touched files are unchanged since the plan's e06eeec68c): #3253 (1f7f780a4d) added advance_postgres_serial and this test and kept ChaCha's per-table try/except that logs and continues ('Preserve each wrapper's existing ownership and error policy'); #3084 (eb0c8c9e52) replaced ChaCha's loop with one call to its new sync_owned_postgres_sequences, an inlined copy with no error handling, and its plan names no error-policy change (it ran the 3 ownership tests that predate #3253). Fix: postgres_sequence_maintenance.py is now byte-identical to open PR #3269's copy (81b0bb5836, blob f169543591): the loop calls advance_postgres_serial and takes continue_on_error=False; ChaCha passes True. RED: 1 failed, DatabaseError at :251. GREEN: the five sequence files 130 passed (129 + test_sync_postgres_sequences_raises_a_failing_table, which pins Media's raise; flipping the default fails it: DID NOT RAISE). uvx bandit -q -ll on postgres_sequence_maintenance.py, postgres_sequences.py and ChaChaNotes_DB.py: 0 results (0 on dev).
Cases 2-5 cause: the product is right; the tests' write path and one expectation were not. #3229 (69e594c495, TASK-13569, merged 2026-10-15) made a metadata rewrite bump messages.version ('Undo must not restore an opaque revision within one timestamp tick', message_store.py:3210-3212); a first write stays version-neutral. Both pinned by its tests. #3084 later added these cases, admitting with extra_metadata so that its drift was a rewrite, and asserting the version stays 1; its design record wanted 'metadata and asset drift without physical-version changes' and 'does not redefine either revision domain'. With the bump disabled these 4 pass and #3229's 4 undo cases fail (4 failed, 22 passed). Fix: the parametrized test admits without extra_metadata (:1014), so its metadata drift is a version-neutral first write and :1022's == 1 holds unchanged; that case now pins match='stale_parent' (:1040), the input-state-digest refusal (disabling that check turns it into stale_source and fails the pin). The A->B->A test (:1050) now expects 3: under #3229 a version-neutral ABA is impossible (no metadata delete path). The design record's claim still holds for the parametrized metadata and asset cases; it no longer covers ABA, which is refused through the version bump. Cases 2-5 failed on SQLite too. RED: 4 failed, 14 passed. GREEN: the two functions 18 passed (9 sqlite, 9 postgres); test_history_selection_transactions.py + test_history_branch_admission.py 200 passed, 1 skipped (the SQLite-only lifetime test on postgres; 924 s at load ~30).
No docs change: the dated #3084 design record (Docs/Design/uat_a01_native_retry_validation_20261011.md) is left as written; the note above records where its claim no longer applies. PR #3269: the add/add conflict on postgres_sequence_maintenance.py disappears; test_history_selection_transactions.py now conflicts with it on three one-line hunks (:1014, :1040, :1050; keep dev's side). Plan: Docs/superpowers/plans/2026-10-22-pg-dev-test-failures.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The five tests pass. Case 1 was a #3084 regression: its new shared sequence helper had no error handling and replaced ChaCha's loop, which #3253 had kept tolerant of a failing table. The helper now takes continue_on_error (the same file as open PR #3269), ChaCha opts in, and Media still raises, pinned by a new unit guard. Cases 2-5 were tests #3084 wrote against TASK-13569's contract (#3229), which bumps the message version on a metadata rewrite. The metadata-drift case now writes metadata the version-neutral way #3084 intended and pins the stale_parent refusal; only the A->B->A case changes its expectation, to the version 3 that #3229's bump produces.
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
