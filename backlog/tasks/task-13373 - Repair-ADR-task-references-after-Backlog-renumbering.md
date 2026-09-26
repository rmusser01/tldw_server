---
id: TASK-13373
title: Repair ADR task references after Backlog renumbering
status: Done
assignee: []
created_date: '2026-09-26 04:30'
updated_date: '2026-09-26 04:47'
labels:
  - docs
  - adr
  - process
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair verified historical-to-current task references in the ADR work stream, using Git rename evidence rather than title guesses. Preserve accepted rationale and explicitly record unresolved historical identities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record provenance-backed mappings and unresolved or ambiguous historical identities.
- [x] #2 Repair active inventory pointers and add dated reference metadata to affected accepted ADRs without rewriting rationale.
- [x] #3 Verify current task targets, Markdown links, and published mirrors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit ADR references and rename evidence. 2. Add identity reconciliation and update current pointers/metadata. 3. Verify docs and finalize task/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR check: no new ADR required; this corrects references under ADR-001/ADR-002 without changing a durable decision. Human approved bounded design in this task conversation.

Implementation: recorded ten R099 Git renames, corrected current inventory pointers, and added dated metadata to 24 ADRs. A read-only verifier confirmed original ADR contents are unchanged after removing the new metadata note, current task frontmatter/title and rename provenance match, and 90 local Markdown links resolve. Unmapped early task identities and five duplicate current IDs remain explicit rather than guessed.

Verification: 31 docs refresh tests passed; 2 full content-equality tests excluded because dev baseline contains unrelated tokenizer mirror drift in Docs/Published/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md and Docs/Published/Env_Vars.md. The first broader run reproduced one failure only on those two files; reported hashes exactly match untouched baseline contents. All 48 ADRs are indexed/mirrored; all 30 changed ADR docs and the historical Evaluations plan have byte-identical published copies. Three focused SQLite persistence tests passed (8 pre-existing warnings). Live PostgreSQL tests were not run; PostgreSQL claims are bounded source/schema inspection, not live parity certification. Bandit not applicable: Markdown-only changes, no Python code touched. Unresolved historical mappings/duplicate IDs, INV-014, SecretManager adoption, and proposed ADR-029 remain separate; no new blockers.

Reproduce from an ordinary repository root with its .venv: source .venv/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tldw_Server_API/tests/Docs/test_docs_published_refresh.py -k 'not refresh_replaces_clean_and_stale_destinations_deterministically and not refresh_preserves_committed_destination_when_backup_cleanup_fails'. SQLite checks: source .venv/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tldw_Server_API/tests/Evaluations/test_evaluations_backend_dual.py::test_sqlite_evaluations_basic tldw_Server_API/tests/Evaluations/test_evaluations_postgres_crud.py::test_sqlite_evaluations_unified_fallback tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py::test_evaluations_unified_sqlite_migration_and_crud. This isolated worktree used the shared repository venv via source ../../.venv/bin/activate; all recorded commands avoid developer-machine absolute paths.

Final combined verification after whitespace cleanup: 34 passed, 2 deselected, 8 warnings; git diff --cached --check clean. Fresh origin/dev is 59bd584503. ADR/Evaluations source files are unchanged between pinned evidence baseline 3f909e133b and this PR base.

Independent review found one normalization overclaim: get_unified_evaluation returns raw rows on unified and legacy paths. Verified the code and narrowed ADR-048, index, and inventory to converted primary CRUD normalization with an explicit raw unified-read caveat. No runtime fix or parity claim introduced.

Post-review verification: reviewer confirmed P2 resolved with no further findings; fresh combined run passed 34 tests, 2 deselected, 8 warnings. All 48 ADR source/published mirrors and all inventory mirrors match; staged whitespace check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled ten verified task renames and corrected active inventory references. Added dated task identity metadata to 24 ADRs without changing their original content, documented unresolved/colliding identities, and synchronized published copies.
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
