---
id: TASK-514.1
title: Implement Notes task parser, storage, reconciliation, and REST API foundation
status: In Progress
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
- tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py
- tldw_Server_API/app/api/v1/endpoints/notes_tasks.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/core/Notes_Tasks/service.py
- tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py
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

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Parser/projection utilities checkpoint complete. Commits: b303ea276f24a9be9fb89feb478486cbfd813888, f0612cce7b991f0795f3b4cb5aec970a95096cd9, 49d0b9aed993412051fd6639d406d9e770badbb3, ef61f60de1b881572ad819132fd400812abc912e, 57d22bdcb2952d4f072408c6a1d020a5436cefaa, 60093c6ca058a626a4b1a823f207f259cf6cc1ab, 91521b6a098433264e56feaa95cc22231c953610, 6ce58385cec4658b3f109c2ec8bc22ceaf610e20, 7b2a188aa4049fa4ed062b549fe873ef97eac46e, e0ae1b043d8c8df6f69d4a204b49aee4afd6853b. Spec compliance review approved. Final code-quality review approved with no Critical or Important issues. Local verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py -v` -> 31 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Notes_Tasks -f json -o /tmp/bandit_notes_tasks_parser_final.json` -> 0 findings, 0 errors. Python 3.10 runtime was not available locally; code was statically reviewed as Python 3.10-safe and tests include enum behavior guards.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 REST API slice added `notes_tasks` schemas/endpoints, registered the router before generic notes routes, and extended `NotesTaskService` with note-scoped reconciliation, projected task create/update/status/delete, unlinked record-only metadata/delete handling, and unread agent task activity read/dismiss state. RED: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py -v` initially failed 14 tests on missing 404 route behavior. GREEN: same command now passes 14 tests. Router contract verification currently has 6 unrelated failures from pre-existing dirty edits in `tldw_Server_API/tests/Services/test_router_groups_contract.py` expecting missing `router_groups.selection` and minimal route_key changes; this file was dirty before Task 4 edits and was not modified by this slice. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/notes_tasks.py tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py tldw_Server_API/app/core/Notes_Tasks tldw_Server_API/app/core/DB_Management/chacha/task_store.py -f json -o /tmp/bandit_notes_tasks_api_task4.json` -> 0 findings, 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Storage/migration checkpoint complete. Commits: 7d5ac09b7855c2bf69fe2cb579369c92f269f4bb, 4281e84cad4a31da4e7da448957f7330dc77bb85, 324635e5ebec6dd937f564883776f85c5ffa9729, 5142d203717b753fcef4f96a36f1f6cba86ae299, 174241232693670f31e0478c6199fd865b2d94df, d9d71e971038b69ee0a2a0918277290e81e11eb2, 455370c458177960f280b87e43fb54be0fd93646, 57391be4469855c1f7e87faef0d9ada4c2242578, 36f653a59fd8176160ea3db30c2f01fa027cacc3, 1000fef8aa6f78c5e73dac2aa4b2e3f8d173cf45, a22dca9337c936f892e6137440349b42c784ac9f. Spec compliance review approved. Final code-quality review approved with no Critical, Important, or Minor issues. Local verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py -v` -> 56 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha/task_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_notes_tasks_storage_final.json` -> 0 findings, 0 errors. Internal task table is `note_tasks` to avoid collision with Scheduler `tasks`; storage tests cover active-note write guards, projection drift/race handling, event/read-state/reconciliation error mapping, and migration table/check constraints.

Reconciler and Notes save integration checkpoint complete. Commits: c63c27aafa43bd53d0ef382748026a7cc371dcd9, 8219abf78451ee06c36b4e7914efd3fc6e6424ec, 0a13e7f64bd8814c5f57b42e1cef636ea17079c7, fa56ba896d7f2860ece14be4756692d3705bc4cf, 51d5faabdf9a86cdd34587fce6709b3ed0e8f7d1, eb95f49314d6e2443da0b8269f071f4180edba41. Spec compliance review approved after fixes for duplicate stable-line matching, block-context fallback, child/detail-only exact-line identity, and title-only update reconciliation state. Code-quality review approved after non-blocking post-save reconciliation, public live projection helper/fail-closed projection reads, empty placeholder warnings, bulk/import API coverage, and public note reconciliation snapshot boundary. Local verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py -v` -> 21 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Notes_Tasks tldw_Server_API/app/api/v1/endpoints/notes.py tldw_Server_API/app/core/DB_Management/chacha/task_store.py -f json -o /tmp/bandit_notes_tasks_reconciler_final.json` -> 0 findings, 0 errors. `git diff --check 0d008a2288..HEAD -- <Task 3 touched files>` -> clean. Residual risk: import coverage exercises Markdown create-copy import; JSON/overwrite import branches share the same reconciliation helper and are left for broader API regression coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
