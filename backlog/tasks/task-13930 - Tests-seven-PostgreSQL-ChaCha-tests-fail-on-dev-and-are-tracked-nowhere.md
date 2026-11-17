---
id: TASK-13930
title: 'Tests: seven PostgreSQL ChaCha tests fail on dev and are tracked nowhere'
status: To Do
labels:
- chacha
- postgres
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13896/TASK-13899's review sweep, and re-run on origin/dev e06eeec68c with the same result (12 failed, 200 passed, 1 skipped across their 4 files; 5 of the 12 are TASK-13901's). Untracked: tldw_Server_API/tests/Character_Chat_NEW/unit/test_postgres_backends.py x4 (StubDB has no _verify_dictionary_schema_postgres) and tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py x3 ('CharactersRAGDB' object has no attribute '_owner_user_id'). Each sweep that touches ChaCha on PostgreSQL has to re-baseline them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 7 tests pass on PostgreSQL, or each is fixed or removed with its reason
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
