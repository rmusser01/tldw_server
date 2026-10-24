---
id: TASK-13910
title: Media DB calls that use the request connection outside a transaction pin a
  PostgreSQL pool slot idle in a transaction
status: To Do
labels:
- media
- postgres
- backend
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13900. On PostgreSQL, MediaDatabase.get_connection() checks a connection out and pins it in the request's context var (media_db/runtime/connection_lifecycle.py:67-85); a statement on it opens a transaction nothing ends, so the session sits idle in transaction until release_context_connection() runs, and every later execute_query in the same request takes a second pool slot. TASK-13900 measured it on MediaFilesRepository.get_for_media: with a 1-slot pool the next statement waited pool_timeout (30 s) and GET /media/{id} answered 500; with a dependency that never releases, the session held AccessShareLock on MediaFiles for good and hung DROP OWNED BY. get_for_media and list_for_media now read through execute_query, pinned by test_lookup_leaves_no_postgres_session_in_a_transaction. The same pattern remains in keyword_access_ops.py:86 (fetch_media_for_keywords), sync_log_ops.py:35 (get_sync_log_entries), chunk_batch_ops.py:181 (process_chunks' parent check), visual_document_ops.py:114 (list_visual_documents_for_media), and the writes visual_document_ops.py:59 (insert_visual_document) and :142 (soft_delete_visual_documents_for_media) (measured on 617094d014). backup_ops.py:37 and api.py describe_media_schema are SQLite-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed read goes through execute_query or an explicit transaction, so it leaves no PostgreSQL session idle in a transaction, with a PostgreSQL test like test_lookup_leaves_no_postgres_session_in_a_transaction
- [ ] #2 The two visual-document writes run in db.transaction() (or are shown to commit), with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Correction (PR #3305 clean review, M2): the description overstates the coverage. test_lookup_leaves_no_postgres_session_in_a_transaction exercises only MediaFilesRepository.get_for_media. list_for_media has read through execute_query since TASK-13233, but no test pins it against leaving a PostgreSQL session idle in a transaction; add one here alongside the listed calls.
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
