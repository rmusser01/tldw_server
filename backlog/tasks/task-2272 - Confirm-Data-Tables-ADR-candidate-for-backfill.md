---
id: TASK-2272
title: Confirm Data Tables ADR candidate for backfill
status: Done
assignee: []
created_date: '2026-06-07 02:48'
updated_date: '2026-06-07 02:59'
labels:
  - docs
  - process
  - adr
  - data-tables
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confirm whether INV-025 from Docs/ADR/inventory/2026-06-03-decision-inventory.md is current and bounded enough for ADR backfill. Verify Docs/Design/Data_Tables_Backend.md against current code/tests for per-user Media DB ownership, async generation/Jobs ownership, RAG source snapshot behavior, server-side exports/File_Artifacts ownership, external UUID behavior, caveats, and any scope that should remain inventory-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create a Data Tables confirmation audit under Docs/ADR/inventory/ using current origin/dev evidence.
- [x] #2 Classify INV-025 as ready for bounded ADR backfill, needing code/doc alignment, or inventory-only, with explicit caveats.
- [x] #3 Update the decision inventory only if the confirmation result changes the tracked next action.
- [x] #4 Create a follow-up Backlog task only if the candidate is ready for ADR backfill.
- [x] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started in isolated worktree .worktrees/confirm-data-tables-adr-candidate from origin/dev. Initial plan: inspect INV-025 source and implementation evidence; write bounded confirmation audit; update inventory if disposition changes; create a follow-up ADR backfill task only if ready; verify docs/references and focused tests where applicable.

Confirmation result: INV-025 is current governing and ready for a bounded Data Tables backend ADR backfill. Added Docs/ADR/inventory/2026-06-07-data-tables-confirmation-audit.md, updated the inventory disposition, and created follow-up TASK-2273. Caveats recorded for numeric job IDs, wait-for-completion/direct-export paths, source ownership scope, bounded snapshots, and File Artifacts internals.

Verification recorded before finalization: git diff --check passed; reference scan for TASK-2272/TASK-2273/INV-025/audit path found expected docs/task references and no developer-machine absolute paths in touched files; focused tests passed with source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DataTables/test_data_tables_api.py tldw_Server_API/tests/DataTables/test_data_tables_export.py tldw_Server_API/tests/DataTables/test_data_tables_jobs_integration.py tldw_Server_API/tests/DataTables/test_data_tables_worker.py tldw_Server_API/tests/DB_Management/test_data_tables_crud.py tldw_Server_API/tests/DB_Management/test_media_db_data_table_child_ops.py tldw_Server_API/tests/DB_Management/test_media_db_data_table_generation_ops.py (77 passed).

Bandit applicability: skipped because touched files are Markdown docs and Backlog task records only; no Python/code paths changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed INV-025 as current governing Data Tables backend behavior. Added the confirmation audit, updated the ADR inventory/default next action, and created TASK-2273 for the bounded accepted ADR backfill. Verification passed; Bandit was not applicable to docs-only changes.
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
