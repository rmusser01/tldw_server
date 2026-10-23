---
id: TASK-13897
title: 'Workflows engine: detached run tasks inherit the caller''s ChaCha operation'
status: Done
labels:
- workflows
- postgres
- bug
dependencies:
- TASK-13861
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13861 PR B's clean review (#3295). The workflow engine's detached tasks copy the caller's contextvars, including an open ChaCha operation, so a step that reaches ChaCha after the caller's operation has closed gets ClosedChaChaOperationError. The tasks (tldw_Server_API/app/core/Workflows/engine.py, origin/dev ad946dd687 line numbers): the continue_run task at :1807, the human-step timeout task at :1829, and the orphan reaper's continue_run task at :2051, which start_run creates through _reap_orphans at :688 (so it can continue unrelated runs). Callers that run inside an operation: HTTP requests (api/v1/endpoints/workflows.py :3497, :4228, :4350, :4566, each asyncio.create_task inside the request's operation; this predates TASK-13861, see its plan's Review Focus 3) and, after TASK-13861 PR B, a voice command's SYNC workflow (process_command runs inside chacha_operation(independent=True)). ASYNC runs start on their own thread (WorkflowScheduler._spawn) and inherit nothing. Affected adapters on PostgreSQL: chatbooks (run_chatbooks_adapter), covered by TASK-13861 PR C with independent=True; character_chat (run_character_chat_adapter), covered by PR B's review round 1 (each ChaCha call in its own independent operation); notes (Workflows/adapters/knowledge/crud.py :171-265, NotesInteropService called synchronously on the loop), not covered, it returns notes_error; and possibly RAG search through unified_rag_pipeline (adapters/rag/search.py), unaudited. Proposed fix: the engine runs these detached tasks with no inherited ChaCha operation, by resetting the operation contextvar at each task's start. It must not wrap the run in one operation instead: concurrent branches would then share one connection and its transaction depth (TASK-13861 plan, F13).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A workflow run continued by a detached engine task (continue_run, human-step timeout, orphan reaper) does not inherit the creating caller's ChaCha operation
- [x] #2 On PostgreSQL, a notes step in a run continued after its creating request or voice command returned succeeds, pinned by a test
- [x] #3 The RAG search adapter's ChaCha access from such a run is audited and covered or documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC 1 RED on 617094d014 (the plan measured e06eeec68c; no file this plan touches changed between them): test_engine_detached_tasks_chacha_operation.py gave 4 failed: ('failed', 'The ChaCha operation has finished') for a route's continue_run task, a human step's timeout, and the orphan reaper (a step joining with chacha_operation()); and the human-step timeout task itself held its creator's closed ChaChaOperation in its context for its whole sleep. Detached callers found: engine.py:1807 and :2054, endpoints/workflows.py:3503, :4234, :4358 and :4575, and research_wait_bridge.py:58 (not in the task text); all go through continue_run. Fix: operation_scope.outside_chacha_operation() clears the inherited operation; continue_run runs inside it, and the timeout task is created inside it; the run is not wrapped in an operation (ADR-080, F13). GREEN: 4 passed; with test_chacha_operation_scope.py and test_orphan_requeue_unit.py, 24 passed.
AC 2 RED on e06eeec68c while planning (PostgreSQL, pg_database_config): ('failed', 'Unexpected error initializing DB for user 1: The ChaCha operation has finished'). The per-call CharactersRAGDB raised a CharactersRAGDBError, which is outside the adapter's noncritical tuple, so the run failed (the task text expected a notes_error output). With AC 1 fixed, a second, pre-existing failure, re-run RED on 617094d014 (the plan measured e06eeec68c; no file this plan touches changed between them) with AC 1 in: TypeError: Object of type datetime is not JSON serializable, from WorkflowsDatabase.update_run_status (:2120), and in the SQLite unit test from complete_step_run (:2513). A control probe on dev (a SYNC run inside an open operation) failed the same way, with the notes step run left running, because complete_step_run's json.dumps failed inside a suppress. Fix: update_run_status and complete_step_run store a date or datetime as ISO text (_json_default); the 7 new lines shifted three Workflows_DB.py entries in Helper_Scripts/ci/scope_predicate_baseline.txt (the ratchet gave 1 failed, 12 passed), regenerated with --write-baseline ('Recorded 28 conditional scope predicates', exactly :1808/:1811/:1960 -> :1815/:1818/:1967). GREEN on PostgreSQL: 35 passed with the AC 1 file, test_workflows_db.py and lint/test_scope_predicate_ratchet.py. Mutant (the notes adapter skips its close): assert 1 == 0 outstanding checkouts.
AC 3: the rag_search adapter passes only media_db_path; MultiDatabaseRetriever builds a ChaCha retriever only for notes, character, world-book or dictionary paths, or a passed chacha_db. Probe (real adapter, sources media_db, notes, chats, characters, world_books, dictionaries): CharactersRAGDB constructed 0 times; positive control, the workflow notes adapter through the same recorder: 1. (The operation-lookup count is not evidence: on SQLite with no open operation even the control records 0.) RAGSearchConfig has no sources field. Documented in a comment at the pipeline call and here; no ChaCha operation needed.
Docs: ADR-080's Consequences bullet 'Detached tasks' updated to the new state (status facts only; it names the notes-graph pattern the notes adapter follows; precedent b3c1a1fa56), with Docs/Published refreshed (only the ADR-080 copy changed); tests/Docs 212 passed. Bandit -ll on operation_scope.py, engine.py, Workflows_DB.py and search.py: 0 results. Follow-up: TASK-13921 (workflow notes on SQLite use a nested ChaChaNotes.db). Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A workflow run continued by a detached task (a route's retry, approve or reject, a human step's timeout, the orphan reaper, a research checkpoint) now runs outside its creator's ChaCha operation, through outside_chacha_operation() in continue_run; it is never wrapped in one operation of its own. On PostgreSQL a notes step in such a run now succeeds and returns its checkout, which also needed WorkflowsDatabase to store PostgreSQL's datetimes as ISO text (a pre-existing failure for any notes step on PostgreSQL). The RAG search adapter reaches no ChaCha data, documented at the call. ADR-080's detached-task bullet describes the new state.
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
