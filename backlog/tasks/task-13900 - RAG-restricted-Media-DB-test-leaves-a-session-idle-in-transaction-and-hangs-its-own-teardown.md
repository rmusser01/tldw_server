---
id: TASK-13900
title: RAG restricted Media DB test leaves a session idle in transaction and hangs
  its own teardown
status: To Do
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
- [ ] #1 The failing test is fixed or its cause recorded, and no Media DB session of the test role is left in a transaction at teardown
- [ ] #2 The file's teardown fails fast on a leaked session instead of hanging (as pg_restricted_backend does after TASK-13861)
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
