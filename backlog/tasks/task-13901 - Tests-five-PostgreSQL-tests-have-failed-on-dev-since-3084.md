---
id: TASK-13901
title: 'Tests: five PostgreSQL tests have failed on dev since #3084'
status: To Do
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
- [ ] #1 Each of the five tests passes on dev, or is fixed or removed with a recorded reason
- [ ] #2 The cause is identified and stated in the task
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
