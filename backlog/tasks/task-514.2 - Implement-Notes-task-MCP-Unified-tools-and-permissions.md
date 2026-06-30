---
id: TASK-514.2
title: Implement Notes task MCP Unified tools and permissions
status: Done
parent_task_id: TASK-514
references:
- TASK-512
- TASK-513
- TASK-514
documentation:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
- tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py
- tldw_Server_API/app/core/DB_Management/chacha/task_store.py
- tldw_Server_API/app/core/Notes_Tasks/models.py
- tldw_Server_API/app/core/Notes_Tasks/reconciler.py
- tldw_Server_API/app/core/Notes_Tasks/service.py
- tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py
- tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose Notes task-backed to-do list operations through MCP Unified with strict validation, permissions, approval-required handling, denied autonomous writes, idempotency, and scope enforcement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Task 5. Keep autonomous writes denied or approval-required until the activity slice enables them after durable notice tests pass. Run `tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py` and Bandit for touched backend scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started MCP Unified task-tool slice after TASK-514.1 backend foundation completion. Scope: `notes_module.py` task tool definitions/dispatch/validation/permission policy behavior and `test_notes_task_tools.py` coverage. Autonomous task writes remain denied or approval-required until the later durable activity/notice slice enables them.

Implemented Task 5 with subagent-driven TDD and review checkpoints. Added MCP Unified task tool definitions and handlers for list/get/create/update/set_status/delete/reconcile_note, strict validator coverage, runtime-metadata write confirmation, denied autonomous-write behavior, idempotency handling, optimistic task/note version checks, scoped listing/filtering, reconciliation summaries, and audit metadata. Follow-up review fixes hardened caller-controlled confirmation, schema completeness, idempotency concurrency, read/write classification, server-side filtering/pagination, skipped status version checks, internal reconciliation audit metadata, current note-row version checks, and high-offset scoped listing.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 is complete. MCP Unified now exposes notes task-backed to-do operations with strict schemas, persona/scope checks, runtime-governed write approval, denied autonomous writes without mutation, idempotent retries, optimistic task and note version checks, batch status results, and reconciliation-aware listing. Supporting task-store/service/reconciler paths were hardened for filtered pagination, audit metadata, projection safety, and current-note conflict detection.

Review: spec review approved after contract hardening; final quality re-review of `bf97c1d4b3` found no findings. Residual noted gap is that the high-offset scoped-list regression uses the module fake DB rather than a full SQLite integration fixture.

Verification recorded 2026-06-05:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py -v` -> 37 passed, 7 warnings.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_notes_crud_tags.py tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py tldw_Server_API/app/core/MCP_unified/tests/test_persona_scope_stage3.py -v` -> 11 passed, 5 warnings.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py -v` -> 71 passed, 7 warnings.
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py tldw_Server_API/app/core/DB_Management/chacha/task_store.py tldw_Server_API/app/core/Notes_Tasks/models.py tldw_Server_API/app/core/Notes_Tasks/reconciler.py tldw_Server_API/app/core/Notes_Tasks/service.py -f json -o /tmp/bandit_notes_task_mcp_task5_final.json` -> results=0, errors=0.
- `git diff --check 31ea82c592..HEAD -- <Task 5 touched files>` -> clean.
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
