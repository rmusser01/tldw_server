---
id: TASK-13899
title: 'ChaCha operations: serialize managed transactions per connection state, or
  fail loudly on overlap'
status: To Do
labels:
- chacha
- postgres
- bug
dependencies:
- TASK-13861
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13861's review. Inside one chacha_operation every thread shares one ConnectionState and one physical connection, and BackendManagedTransaction keeps tx_depth on that state without a lock: __enter__ reads it (ChaChaNotes_DB.py:835 on origin/dev d47c9ccf58; :731 on 2cae53929f, before TASK-13861 PR A) and __exit__ writes back the value it read (:889; :785 on 2cae53929f). Two overlapping db.transaction() blocks on two threads leave the depth at 1, so every later transaction in the operation is unmanaged, never commits, and is rolled back when the operation returns its checkout. Measured: a Chat_Macros run wrapped in one operation lost every branch row and its final status ('running', []); a replay of its branch fan-out lost writes in 7 of 50 runs. TASK-13861 avoids it by rule (ADR-080; the Chat_Macros executor gives each call an independent operation). No HTTP route was found that overlaps managed transactions on one handle (plan Findings F13, static audit). Request-path fan-outs that read one ChaCha handle concurrently, to re-check if they ever write: core/RAG/rag_service/unified_pipeline.py:1699, :1736, :4819, :5306, :9456, :9488 and database_retrievers.py:2613, :5014 (notes and chat-history retrievers through execute_query, no managed transaction). Fix: a per-state lock taken in __enter__ and released in __exit__, reentrant per thread, or a loud error on overlap; it changes a pinned transaction contract, so it needs its own tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two overlapping db.transaction() blocks on one ConnectionState from two threads either serialize or raise; neither loses a write (PostgreSQL test)
- [ ] #2 test_chacha_operation_scope.py and the HTTP operation lifecycle tests still pass
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
