---
id: TASK-13354
title: >-
  DatabaseError wrapping discards the original exception, hiding constraint and
  driver detail
status: To Do
assignee: []
created_date: '2026-09-23 02:35'
labels:
  - db
  - diagnostics
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The DB_Management layer wraps sqlite3 errors as DatabaseError('SQLite query execution failed') WITHOUT chaining the original: __cause__ is None, so the underlying message -- for example 'CHECK constraint failed: source_path_hash = source_key_hash' -- is unrecoverable by any caller, test or operator.

Confirmed empirically: a deliberate CHECK violation against sync_notes_attachment_cleanup_candidates raises DatabaseError with str() == 'SQLite query execution failed' and e.__cause__ is None.

Cost observed today, twice:
- test_cleanup_candidate_schema_rejects_path_hash_identity_drift matched on 'CHECK constraint failed' and had been red since the wrapping changed. It could not be repaired by matching the new text, because 'SQLite query execution failed' is satisfied by ANY SQLite failure, so the test was rewritten to assert the schema and the accept/reject pair instead (d-commit above).
- Every diagnosis of a constraint failure in this layer starts by re-running the query by hand to find out which constraint fired.

Fix is small: raise the wrapper with  so the original is chained, and ideally include the driver message in the wrapper text. Callers that deliberately hide SQL from logs can still do so -- chaining does not force anything into a log line, it just stops the information being destroyed.

Compare core/Sync/v2/profile.py:552, which maps any unrecognised SyncStoreError to 'personal_context_snapshot_unavailable' and hid a real bug for weeks (TASK-13351). Same failure mode, different layer.

Source: found while fixing TASK-13352/13344 follow-ups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DatabaseError chains the original driver exception so __cause__ is populated
- [ ] #2 A constraint violation is identifiable from the raised exception without re-running the query
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
