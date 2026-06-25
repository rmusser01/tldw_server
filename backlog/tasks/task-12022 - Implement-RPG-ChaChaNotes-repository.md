---
id: TASK-12022
title: Implement RPG ChaChaNotes repository
status: Done
created_date: 2026-06-25 03:24
labels:
- rpg
- ttrpg
- backend
- implementation
- persistence
priority: high
references:
- TASK-12018
- TASK-12019
- TASK-12020
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/DB_Management/RPG_DB.py
- tldw_Server_API/tests/RPG/test_rpg_db.py
updated_date: 2026-06-25 03:37
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the ChaChaNotes-backed RPG repository slice from the reviewed plan: schema creation, campaign/session creation with initial snapshot, row mapping, operation-scoped idempotency, and atomic event/snapshot/session cursor commits. Scope excludes service, REST, MCP, and proposal repository methods beyond nullable proposal id support in event commits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RPGRepository initializes required tables in a CharactersRAGDB transaction
- [x] #2 Campaign/session creation stores JSON fields and creates an initial snapshot
- [x] #3 commit_events_and_snapshot assigns contiguous sequences, stores events/snapshot, updates session cursors, and records operation idempotency in one transaction
- [x] #4 Idempotent replay returns stored event rows without advancing snapshot version
- [x] #5 Conflicting idempotency hash and stale expected sequence raise RPGConflictError
- [x] #6 Focused repository tests are written test-first and pass
- [x] #7 Bandit/diff checks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing repository tests for campaign/session/initial snapshot, atomic commit cursor updates, idempotent replay, idempotency conflict, and stale expected sequence.
2. Implement `tldw_Server_API/app/core/DB_Management/RPG_DB.py` using `CharactersRAGDB.transaction()` and parameterized SQL.
3. Run focused repository tests, compileall, Bandit, and diff checks.
4. Record modified files and final notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ChaChaNotes-backed RPG repository schema and idempotent campaign/session/event/snapshot writes. Review found and fixed a session-create replay bug where the idempotency lookup required an unknown session_id; added a regression test. Verification: repository RED was confirmed before implementation; combined RPG focused tests passed (30 passed); compileall passed; Bandit on core RPG/DB touched scope reported 0 results; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the ChaChaNotes-backed RPG repository with schema initialization, campaign/session creation, initial snapshots, row mappers, operation-scoped idempotency, and atomic event/snapshot/session cursor commits. Review added a regression for idempotent session creation replay and fixed the lookup to use operation scope when no session id is known yet.
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
