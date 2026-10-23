---
id: TASK-13899
title: 'ChaCha operations: serialize managed transactions per connection state, or
  fail loudly on overlap'
status: Done
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
- [x] #1 Two overlapping db.transaction() blocks on one ConnectionState from two threads either serialize or raise; neither loses a write (PostgreSQL test)
- [x] #2 test_chacha_operation_scope.py and the HTTP operation lifecycle tests still pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md. PR: the PR from branch fix/chacha-off-loop-and-transaction-guard (its URL is appended in a later note).
Chose fail loudly, not serialize. ConnectionState gains tx_owner, the (thread ident, asyncio task) key use() already uses; BackendManagedTransaction claims it before reading tx_depth (enter_transaction), releases it in __exit__ and on a failed __enter__ (leave_transaction); the same key nests and joins as before; another key raises ConcurrentChaChaTransactionError (a RuntimeError, logged at ERROR). Legacy thread-local and SQLite paths unchanged. Why not a lock: a lock would deadlock a transaction that waits on another thread's, block the event loop, and, per thread, miss two tasks on one thread (mutation: a thread-only key loses writes in the two-task case exactly as dev). Raise versus log-only was the owner's Decision 1: raise (AC1 as written; refusal also covers overlaps that happened to commit on dev, so a missed caller becomes a 500 or a failed job).
Overlap audit: the first scans (1228 sync transaction blocks in app/ on e06eeec68c, none awaits or hands work to a thread inside; the 2 that hold a ChaCha transaction across a yield have no awaiting caller) missed fan-outs. Widened (plan Design choice 1): 486 ChaCha methods that open a managed transaction, 135 fan-out sites (gather, create_task, TaskGroup, as_completed, wait, start_soon, run_in_executor, submit, run_coroutine_threadsafe, Thread, copy_context) in the 268 app files that name ChaCha, matched by name within 3 hops of their module; 17 flagged, each read. One overlaps: SemanticProjector.validate_conversion gathered two to_thread(list_visible_vector_ids), each opening db.transaction(), inside the request's operation, so POST /notes/{id}/links with semantic_conversion would answer 500 (the review measured 40 of 40 refused). It now awaits the two lookups in turn. The audit is name-based within one module and skips modules with no ChaCha marker; the reviews spot-read the gather/wait sites, a gap scan of direct-transaction fan-outs (13 sites), and the MCP and Workflows fan-outs, and found no second overlap. Cross-module chains are not exhaustively covered.
RED on origin/dev e06eeec68c: tests/DB_Management/test_chacha_postgres_concurrent_transactions.py 2 failed: ([], ['r1']) == (['ConcurrentChaChaTransactionError'], ['r1', 'r3']) for two threads and for two tasks on the loop thread (r2 and r3 rolled back with the operation). GREEN: 2 passed; with test_chacha_operation_scope.py 21 passed. Semantic conversion (tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py): dev (None, 2), both lookups at once; guard without the fix ('ConcurrentChaChaTransactionError', 2) 3 of 3; with the fix (None, 1), 38 passed with the existing semantic projector and manual-conversion tests. AC2: the 13 operation-focused PostgreSQL files (operation scope, HTTP operation lifecycle, chat retry, enrichment, study pack, per-call handles, legacy checkouts, close_all, locking reads, notes-graph, character_chat, Jobs workers incl. the Chat_Macros concurrent-branch run) 162 passed with RUN_JOBS=1, 163 passed with the semantic test; the 263-file sweep (RAG/test_restricted_postgres_media_retrieval.py excluded: TASK-13900 hangs it on dev too) 7318 passed, 12 failed, 20 skipped, 1 error. The 12 are the known dev failures: TASK-13901's 5 and TASK-13930's 7. The error is the pre-existing Writing/test_writing_owner_postgres.py teardown flake: alternating dev with dev plus this diff in one worktree, each errored in 2 of 10 runs. 244 files ran in 6 shards, then the 19 files with @pytest.mark.timeout(30) markers in one process (timeout_method=signal); a first attempt at load ~48 lost whole shards when two semantic-migration tests ran past their 30 s markers during schema initialization.
Docs: ADR-080's 'Concurrent transactions' Consequences bullet now states the guard and its cost (Decision unchanged); ChaChaNotes_DB.md names the error; Docs/Published refreshed. Bandit: nothing in operation_scope.py, ChaChaNotes_DB.py or semantic_projector.py; the two tests have B101 LOW x1 each.
Executed on origin/dev 617094d014 (2026-10-23): the RED ([], ['r1']) x2, GREEN 21 passed, the thread-only-key mutation, the semantic RED 3 of 3 and 38 passed, and the 14-file operation set (163 passed) all matched the plan's Expected lines.
PR: https://github.com/rmusser01/tldw_server/pull/3312.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Inside one chacha_operation, a db.transaction() that enters while another thread's or task's transaction is open on the same ConnectionState now raises ConcurrentChaChaTransactionError instead of joining it; nesting on one thread and task joins as before. Before, the overlap left tx_depth wrong and every later write in the operation was silently rolled back (measured: r2 and r3 lost for both a two-thread and a two-task overlap). Raising rather than serializing avoids deadlocks and event-loop stalls, and covers two tasks on one thread; it also refuses overlaps that happened to work, so the one current caller, semantic conversion's two lookups, now runs them in turn. ADR-080's rule is now enforced.
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
