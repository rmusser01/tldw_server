---
id: TASK-501
title: Fix ChaChaNotes v44 to v45 migration path registration
status: Done
labels:
- bug
- database
- migration
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair the user-reported ChaChaNotes `rag_char_chat_schema` migration failure: `Migration path undefined ... from version 44 to 45`. Add focused regression coverage for the exact v44 -> v45 upgrade path and verify the migration no longer fails for users updating through that version boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Exact v44 -> v45 ChaChaNotes SQLite migration path is covered by a focused regression test.
- [x] Migration dispatcher exposes an explicit registered path for v44 -> v45 so future releases cannot silently omit it from path resolution.
- [x] Focused tests, compile check, diff check, and Bandit verification are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Worktree: `<local_worktree_path_redacted>/codex-fix-rag-char-chat-v44-v45-path`
- Branch: `codex/fix-rag-char-chat-v44-v45-path`
- Initial inspection: current `dev` already has a fallback branch for v44 -> v45 and a manual exact-target smoke test succeeds, so the durable fix is explicit migration-path registration plus regression coverage for the exact reported boundary.
- Red/green: `test_sqlite_linear_migration_registry_maps_v44_to_v45` failed before implementation with `AttributeError: 'CharactersRAGDB' object has no attribute '_sqlite_linear_migration_steps'`; passed after adding the registry and dispatcher wiring.
- PR review follow-up: removed the absolute local worktree path from this task file, added index assertions to the exact v44 -> v45 test, and refactored fresh database initialization to use the same SQLite linear migration registry as fallback migrations.
- PR review red/green: `test_new_database_initialization_uses_linear_migration_registry` failed before the refactor with `Failed: DID NOT RAISE <class '...SchemaError'>`; passed after routing fresh DB migrations through `_migrate_sqlite_linearly_to_target`.
- Verification:
  - `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py::test_migration_v44_to_v45_creates_persona_visual_tables tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py::test_sqlite_linear_migration_registry_maps_v44_to_v45 -q` -> 2 passed.
  - `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -q` -> 12 passed.
  - `python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v10.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v39.py tldw_Server_API/tests/DB_Management/test_chacha_conversations_fts_healing.py -q` -> 4 passed.
  - `python -m py_compile tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py` -> passed.
  - `git diff --check` -> passed.
  - `python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -s B101 -f json -o /tmp/bandit_chacha_v44_v45_path.json` -> 0 findings.
- PR review verification:
  - `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py::test_migration_v44_to_v45_creates_persona_visual_tables tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py::test_sqlite_linear_migration_registry_maps_v44_to_v45 tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py::test_new_database_initialization_uses_linear_migration_registry -q` -> 3 passed.
  - `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -q` -> 13 passed.
  - `python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v10.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v39.py tldw_Server_API/tests/DB_Management/test_chacha_conversations_fts_healing.py -q` -> 4 passed.
  - `python -m py_compile tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py` -> passed.
  - `git diff --check` -> passed.
  - `python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -s B101 -f json -o /tmp/bandit_chacha_v44_v45_path_review_final.json` -> 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a SQLite migration-step registry to `CharactersRAGDB` and routed fallback migrations through it, including an explicit v44 -> v45 handler. Added regression coverage for the exact reported v44 -> v45 update boundary, dispatcher registration, exact-path index creation, and fresh database initialization through the same registry so future schema bumps cannot omit this path silently. Redacted the local worktree path from this task record after PR review.
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
