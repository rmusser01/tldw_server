---
id: TASK-13897
title: 'Workflows engine: detached run tasks inherit the caller''s ChaCha operation'
status: To Do
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
- [ ] #1 A workflow run continued by a detached engine task (continue_run, human-step timeout, orphan reaper) does not inherit the creating caller's ChaCha operation
- [ ] #2 On PostgreSQL, a notes step in a run continued after its creating request or voice command returned succeeds, pinned by a test
- [ ] #3 The RAG search adapter's ChaCha access from such a run is audited and covered or documented
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
