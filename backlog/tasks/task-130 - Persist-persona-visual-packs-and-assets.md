---
id: TASK-130
title: Persist persona visual packs and assets
status: Done
assignee: []
created_date: '2026-05-09 00:10'
updated_date: '2026-05-09 00:18'
labels:
  - persona
  - webui
  - implementation
  - database
dependencies:
  - TASK-129
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the persistence slice for Persona Visual Packs: per-user persona visuals storage directory helper, ChaChaNotes schema migration for visual packs/assets/generated candidates, and PersonaStateStore/CharactersRAGDB methods for pack, asset, activation, deactivation, and candidate lifecycle. Keep this slice backend-only and do not add API endpoints yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DatabasePaths exposes a per-user persona_visuals directory helper that creates the directory safely
- [x] #2 ChaChaNotes migrations create persona_visual_packs, persona_visual_assets, and persona_visual_candidates tables plus indexes for fresh and migrated databases
- [x] #3 PersonaStateStore and CharactersRAGDB expose scoped CRUD methods for packs, assets, active pack lookup, activation/deactivation, and candidates
- [x] #4 Activation archives any prior active pack so at most one non-deleted active pack exists for a user/persona
- [x] #5 Focused tests cover migration, scoping by user/persona/pack, active-pack transitions, deactivation, and candidate accept/reject lifecycle
- [x] #6 Verification commands and Bandit result for touched backend scope are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Work in /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/persona-visual-packs-plan on branch codex/persona-visual-packs-plan.

1. Inspect existing ChaChaNotes migrations and PersonaStateStore patterns for persona buddy/exemplar persistence.
2. Write failing tests in tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py for storage helper, table creation, pack CRUD, active-pack transitions, deactivation, asset scoping, and candidate lifecycle.
3. Run focused tests to confirm expected missing-method failures.
4. Implement DatabasePaths persona_visuals helper, ChaChaNotes migration tables/indexes, PersonaStateStore row mappers and methods, and CharactersRAGDB delegation.
5. Re-run focused DB tests plus existing nearby persona state tests if needed.
6. Run Bandit on touched backend persistence files and record results.
7. Commit this slice with TASK-130 metadata.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -q --tb=short` failed with expected missing `DatabasePaths.get_user_persona_visuals_dir`, missing visual tables, and missing PersonaStateStore/CharactersRAGDB visual pack methods.

Green run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -q --tb=short` passed 7 tests.

Adjacent regression run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q --tb=short` passed 15 tests.

Whitespace/security checks: `git diff --check` passed; `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/db_path_utils.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py -f json -o /tmp/bandit_persona_visuals_persistence.json` exited 0 with no results/errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the backend persistence slice for persona visual packs: added the per-user persona_visuals storage path helper, schema v45 tables/indexes for packs/assets/candidates, PersonaStateStore row mappers and scoped lifecycle methods, CharactersRAGDB delegation, and focused tests for migration, scoping, activation/deactivation, and candidate accept/reject lifecycle.
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
