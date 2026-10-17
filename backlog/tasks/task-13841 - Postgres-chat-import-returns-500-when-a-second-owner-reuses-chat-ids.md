---
id: TASK-13841
title: Postgres chat import returns 500 when a second owner reuses chat ids
status: To Do
labels:
- chat
- postgres
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat[postgres] fails on dev 7b5ba7aa6d with `assert 500 == 201` at line 798; the sqlite variant passes. The 500 comes from POST /api/v1/chats/import, whose trace records error.type psycopg_pool.PoolTimeout. The test only runs when a Postgres is reachable, so CI without Postgres does not see it. Found while sweeping the TASK-13560 change (PR #3247); it fails on unchanged dev too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A second owner importing chats that reuse the first owner's ids gets 201 on Postgres, and the first owner's chat is untouched
- [ ] #2 The cause of the pool timeout is fixed, not hidden by a longer timeout
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
