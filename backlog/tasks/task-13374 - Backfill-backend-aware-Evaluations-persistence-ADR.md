---
id: TASK-13374
title: Backfill backend-aware Evaluations persistence ADR
status: Done
assignee: []
created_date: '2026-09-26 04:31'
updated_date: '2026-09-26 04:47'
labels:
  - docs
  - adr
  - evaluations
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill the approved current EvaluationsDatabase persistence boundary for INV-009 and INV-012: SQLite/PostgreSQL support with JSON TEXT/JSONB and normalized reads. No runtime changes or broad isolation/migration guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add one accepted implementation-backed ADR with alternatives, evidence, path overrides, and fallback caveats.
- [x] #2 Link covering ADR from the index, inventories, historical Evaluations design, and module README while keeping INV-014 separate.
- [x] #3 Verify source/published consistency and focused tests; record applicable skips.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm manager, backend selection, schema, JSON conversion, and caller evidence. 2. Record bounded ADR-048 and references. 3. Sync mirrors, run focused verification, and finalize task/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR check: yes; ADR-048 covers EvaluationsDatabase backend-aware persistence and representation, complementing ADR-020 without extending its guarantees. Human approved the bounded decision and documentation scope in this task conversation.

Implementation: ADR-048 records current EvaluationsDatabase SQLite/PostgreSQL storage and JSON TEXT/JSONB normalization, with path overrides, known resolver fallbacks, and non-lossless scalar caveats. Index, both inventories, historical confirmation audit/design, and module README now link the covering decision; INV-014 remains separate. Focused existing SQLite smoke, unified fallback, and migration/CRUD tests passed (3 tests); no runtime code changed.

Verification: 31 docs refresh tests passed; 2 full content-equality tests excluded because dev baseline contains unrelated tokenizer mirror drift in Docs/Published/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md and Docs/Published/Env_Vars.md. The first broader run reproduced one failure only on those two files; reported hashes exactly match untouched baseline contents. All 48 ADRs are indexed/mirrored; all 30 changed ADR docs and the historical Evaluations plan have byte-identical published copies. Three focused SQLite persistence tests passed (8 pre-existing warnings). Live PostgreSQL tests were not run; PostgreSQL claims are bounded source/schema inspection, not live parity certification. Bandit not applicable: Markdown-only changes, no Python code touched. Unresolved historical mappings/duplicate IDs, INV-014, SecretManager adoption, and proposed ADR-029 remain separate; no new blockers.

Reproduce from an ordinary repository root with its .venv: source .venv/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tldw_Server_API/tests/Docs/test_docs_published_refresh.py -k 'not refresh_replaces_clean_and_stale_destinations_deterministically and not refresh_preserves_committed_destination_when_backup_cleanup_fails'. SQLite checks: source .venv/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tldw_Server_API/tests/Evaluations/test_evaluations_backend_dual.py::test_sqlite_evaluations_basic tldw_Server_API/tests/Evaluations/test_evaluations_postgres_crud.py::test_sqlite_evaluations_unified_fallback tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py::test_evaluations_unified_sqlite_migration_and_crud. This isolated worktree used the shared repository venv via source ../../.venv/bin/activate; all recorded commands avoid developer-machine absolute paths.

Final combined verification after whitespace cleanup: 34 passed, 2 deselected, 8 warnings; git diff --cached --check clean. Fresh origin/dev is 59bd584503. ADR/Evaluations source files are unchanged between pinned evidence baseline 3f909e133b and this PR base.

Independent review found one normalization overclaim: get_unified_evaluation returns raw rows on unified and legacy paths. Verified the code and narrowed ADR-048, index, and inventory to converted primary CRUD normalization with an explicit raw unified-read caveat. No runtime fix or parity claim introduced.

Post-review verification: reviewer confirmed P2 resolved with no further findings; fresh combined run passed 34 tests, 2 deselected, 8 warnings. All 48 ADR source/published mirrors and all inventory mirrors match; staged whitespace check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added ADR-048 for backend-aware Evaluations persistence and DB-owned JSON normalization. Updated index, inventories, historical design/audit, module README, and published mirrors. INV-009/INV-012 now have a covering accepted record; no runtime behavior changed.
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
