---
id: TASK-13900
title: RAG restricted Media DB test leaves a session idle in transaction and hangs
  its own teardown
status: Done
labels:
- rag
- postgres
- tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13861 (measured on origin/dev 2b925562e5 and with TASK-13861's changes alike; re-measured on 2026-10-22 on origin/dev ba84588d12, which has PRs A-C, and on PR D's branch, identically). In tests/RAG/test_restricted_postgres_media_retrieval.py, the 19th test in collection order (test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True]) fails when the file runs in order (alone it passes, 18 s on ba84588d12). The restricted_media_store fixture's own teardown then runs DROP OWNED BY on its rag_retrieval_* role (:74 on ba84588d12; :75 after PR D adds an import), which waits on a Media DB session of that role left idle in transaction (SELECT * FROM MediaFiles WHERE media_id = $1 AND file_type = $2 ... in pg_stat_activity, wait_event_type Lock on the DROP). pytest-timeout (-o timeout=150) did not end the run: the DROP was still waiting after 5 min on both commits (13 min in the original measurement), and the run had to be killed; the role is left behind in the cluster. Not ChaCha: the session is the Media DB's, and the fixture is not pg_restricted_backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The failing test is fixed or its cause recorded, and no Media DB session of the test role is left in a transaction at teardown
- [x] #2 The file's teardown fails fast on a leaked session instead of hanging (as pg_restricted_backend does after TASK-13861)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Correction (PR #3300 review, M3): the description names the wrong test. test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True] is the sqlite param, whose fixture runs no DROP, and it is not the 19th test: that count took only the <Function> lines of --collect-only and skipped two <Coroutine> items. Re-run with pytest -v on 2026-10-22 (PR D's branch, -o timeout=150): the 19th test executed, test_permission_first_auth_trash_restore_without_preseeded_scope[postgresql-X-API-KEY], is the one that FAILED, and its own restricted_media_store teardown (postgresql param) is the DROP OWNED BY that then waits on its rag_retrieval_* role's Media DB session, idle in transaction on SELECT * FROM MediaFiles WHERE media_id = $1 AND file_type = $2 .... The review's guess, [postgresql-Rowan-True] (21st by the <Function> count), is not it either; nothing after the 19th test ran. Start from [postgresql-X-API-KEY].
Cause (measured on 617094d014; the touched files are unchanged since the plan's e06eeec68c): the failing node, [postgresql-X-API-KEY], fails alone too; it is the first PostgreSQL case of authenticated_media_client. Its first request, GET /media/{id}, reaches MediaFilesRepository.get_for_media through has_original_file. get_for_media used db.get_connection(), which on PostgreSQL pins a pooled connection in the request's context var; its SELECT left the session idle in transaction holding AccessShareLock on MediaFiles. The request's next statements (_load_keywords, _load_versions) needed a second slot on the fixture's 1-slot pool, waited pool_timeout (30 s) and the GET answered 500 (assert 500 == 200). The test's media_db override never calls release_context_connection, so the session stayed in its transaction, and the fixture's bare DROP OWNED BY waited on its lock past pytest-timeout (killed at 4:40 while planning, role left behind and dropped by name). Production's get_media_db_for_user releases at request end, but each detail GET held two of the pool's 10 slots, and any non-releasing path keeps the session.
Fix: get_for_media reads through db.execute_query, as list_for_media has since TASK-13233 (76d60dbbd6); restricted_media_store's teardown uses _drop_restricted_role (TASK-13861). RED: test_lookup_leaves_no_postgres_session_in_a_transaction assert 1 == 0 on dev (the in-transaction read-your-writes guard passes before and after); with the teardown only, the RAG node gave 1 failed, 1 error in 68 s at load ~30 (GET 500; teardown named the idle-in-transaction MediaFiles session; no role left). GREEN: the 4 trash_restore cases 4 passed; the whole RAG file 68 passed (nothing after the 19th item had run on dev). test_database_query_failure_is_not_reported_as_missing_file patched _fetchall_with_connection, which get_for_media no longer calls (2 failed, DID NOT RAISE); its patch moved to execute_query, expectation unchanged; the contract file 13 passed. The 24 test files that reach get_for_media (the six-name grep plus test_media_versions.py): 563 passed. The RAG file never ran without the fail-fast teardown.
Verification: ruff All checks passed; uvx bandit -q -ll on media_files_repository.py: 0 results (0 on dev). No docs change. Follow-ups: TASK-13910 (six more Media DB calls pin the request connection on PostgreSQL), TASK-13911 (34 test files hand-roll a bare DROP OWNED BY teardown), TASK-13912 (pytest-timeout does not end a PostgreSQL lock wait). Known: one run of the contract file aborted at interpreter exit after '13 passed' (libc++abi recursive_mutex lock failed, rc 134) and passed clean on re-run; other sessions' unrelated runs show the same exit abort. Plan: Docs/superpowers/plans/2026-10-22-pg-dev-test-failures.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed. The hang came from a product leak: MediaFilesRepository.get_for_media read on the request's pinned PostgreSQL connection and left it idle in a transaction, so GET /media/{id} stalled for a second pool slot (500 on the fixture's one-slot pool) and the session's lock blocked the fixture's DROP OWNED BY for good. get_for_media now reads through execute_query, as list_for_media does, pinned by a contract test; and the fixture's teardown uses TASK-13861's fail-fast drop, so a future leak fails the test with the session named and leaves no role behind. The whole file passes on PostgreSQL. TASK-13910, TASK-13911 and TASK-13912 track the same connection pattern elsewhere, the other hand-rolled teardowns, and pytest-timeout not firing.
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
