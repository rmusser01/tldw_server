---
id: TASK-514.1
title: Implement Notes task parser, storage, reconciliation, and REST API foundation
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-05 09:32
labels: []
dependencies: []
references:
- TASK-512
- TASK-513
- TASK-514
documentation:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
parent_task_id: TASK-514
modified_files:
- tldw_Server_API/app/api/v1/endpoints/notes.py
- tldw_Server_API/app/api/v1/endpoints/notes_tasks.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/chacha/task_store.py
- tldw_Server_API/app/core/Notes_Tasks/__init__.py
- tldw_Server_API/app/core/Notes_Tasks/markdown_parser.py
- tldw_Server_API/app/core/Notes_Tasks/models.py
- tldw_Server_API/app/core/Notes_Tasks/reconciler.py
- tldw_Server_API/app/core/Notes_Tasks/service.py
- tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
- tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py
- tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py
- tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py
- tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py
- tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the core backend foundation for Notes task-backed to-do lists: parser/projection utilities, task tables and store, reconciliation after note saves, and REST API routes/schemas.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Tasks 1-4 from `Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md`. Use TDD. Record verification from parser, migration/store, reconciler, notes integration, task API, router contract, and Bandit on touched backend scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Parser/projection utilities checkpoint complete. Commits: b303ea276f24a9be9fb89feb478486cbfd813888, f0612cce7b991f0795f3b4cb5aec970a95096cd9, 49d0b9aed993412051fd6639d406d9e770badbb3, ef61f60de1b881572ad819132fd400812abc912e, 57d22bdcb2952d4f072408c6a1d020a5436cefaa, 60093c6ca058a626a4b1a823f207f259cf6cc1ab, 91521b6a098433264e56feaa95cc22231c953610, 6ce58385cec4658b3f109c2ec8bc22ceaf610e20, 7b2a188aa4049fa4ed062b549fe873ef97eac46e, e0ae1b043d8c8df6f69d4a204b49aee4afd6853b. Spec compliance review approved. Final code-quality review approved with no Critical or Important issues. Local verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py -v` -> 31 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Notes_Tasks -f json -o /tmp/bandit_notes_tasks_parser_final.json` -> 0 findings, 0 errors. Python 3.10 runtime was not available locally; code was statically reviewed as Python 3.10-safe and tests include enum behavior guards.

Storage/migration checkpoint complete. Commits: 7d5ac09b7855c2bf69fe2cb579369c92f269f4bb, 4281e84cad4a31da4e7da448957f7330dc77bb85, 324635e5ebec6dd937f564883776f85c5ffa9729, 5142d203717b753fcef4f96a36f1f6cba86ae299, 174241232693670f31e0478c6199fd865b2d94df, d9d71e971038b69ee0a2a0918277290e81e11eb2, 455370c458177960f280b87e43fb54be0fd93646, 57391be4469855c1f7e87faef0d9ada4c2242578, 36f653a59fd8176160ea3db30c2f01fa027cacc3, 1000fef8aa6f78c5e73dac2aa4b2e3f8d173cf45, a22dca9337c936f892e6137440349b42c784ac9f. Spec compliance review approved. Final code-quality review approved with no Critical, Important, or Minor issues. Local verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py -v` -> 56 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha/task_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_notes_tasks_storage_final.json` -> 0 findings, 0 errors. Internal task table is `note_tasks` to avoid collision with Scheduler `tasks`; storage tests cover active-note write guards, projection drift/race handling, event/read-state/reconciliation error mapping, and migration table/check constraints.

Reconciler and Notes save integration checkpoint complete. Commits: c63c27aafa43bd53d0ef382748026a7cc371dcd9, 8219abf78451ee06c36b4e7914efd3fc6e6424ec, 0a13e7f64bd8814c5f57b42e1cef636ea17079c7, fa56ba896d7f2860ece14be4756692d3705bc4cf, 51d5faabdf9a86cdd34587fce6709b3ed0e8f7d1, eb95f49314d6e2443da0b8269f071f4180edba41. Spec compliance review approved after fixes for duplicate stable-line matching, block-context fallback, child/detail-only exact-line identity, and title-only update reconciliation state. Code-quality review approved after non-blocking post-save reconciliation, public live projection helper/fail-closed projection reads, empty placeholder warnings, bulk/import API coverage, and public note reconciliation snapshot boundary. Local verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py -v` -> 21 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Notes_Tasks tldw_Server_API/app/api/v1/endpoints/notes.py tldw_Server_API/app/core/DB_Management/chacha/task_store.py -f json -o /tmp/bandit_notes_tasks_reconciler_final.json` -> 0 findings, 0 errors. `git diff --check 0d008a2288..HEAD -- <Task 3 touched files>` -> clean. Residual risk: import coverage exercises Markdown create-copy import; JSON/overwrite import branches share the same reconciliation helper and are left for broader API regression coverage.

Task 4 REST API checkpoint complete. Commits: 5c4100cbffe8, a260326f8135, 5e5fd43f56e168cd1893a7ae08748951dc89c2c8, aeab203a5572da0466a770e04cd6635a837de22f, 139249d601cd3d5b56fc228a0f9dfb13d38589d1. Spec compliance review approved after activity recency/window fixes and status-only unknown-token preservation. Code-quality review approved after fixes for stale sibling projections, newline task-text injection, malformed metadata token preservation, atomic batch status rollback, strict metadata schema/service validation, and strict due-date shape. Local verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py -v` -> 27 passed, 7 warnings; `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py -v` -> 52 passed, 7 warnings; `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py -v` -> 57 passed, 7 warnings. Bandit: `/tmp/bandit_notes_tasks_api_task4_strict_due_date.json` -> 0 findings, 0 errors. `git diff --check af349586e6..HEAD -- <Task 4 files>` -> clean. Router contract suite remains blocked by unrelated pre-existing dirty edits in `tldw_Server_API/tests/Services/test_router_groups_contract.py`, not by this task slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backend foundation for Notes task-backed to-do lists is complete across parser/projection utilities, task storage/migration, reconciliation after note saves, and REST task APIs. The accepted implementation adds durable note task records, projection reconciliation, task activity/read-state helpers, note-save reconciliation hooks, `/api/v1/notes/tasks*` routes, note-scoped task create/reconcile/list routes, and regression coverage for ambiguous/unlinked/projected mutation safety, atomic status updates, strict metadata/text validation, and stale projection refresh. Spec and code-quality reviews are approved for Tasks 1-4. Final verification for the accepted Task 4 REST slice: Task API tests 27 passed, parser/reconciler/reconciliation API overlap tests 52 passed, task-store tests 57 passed, Bandit JSON results empty, and diff whitespace clean. Known external caveat: the router groups contract suite currently has unrelated pre-existing dirty-test failures outside this task's files.
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
