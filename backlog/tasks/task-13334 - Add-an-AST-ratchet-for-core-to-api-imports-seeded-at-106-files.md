---
id: TASK-13334
title: Add an AST ratchet for core to api imports seeded at 106 files
status: In Progress
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-22 20:42'
labels:
  - architecture
  - tests
  - ci
dependencies: []
references:
  - 'tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py:14'
  - 'tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py:2699'
  - 'tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py:20'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured today: 157 import lines across 106 files. Correctly scoped: 23 lines are inside core/MCP_unified/tests/ (an in-app test tree, not production), 98 are schema-only (the schemas are in the wrong package), and 33 are PRODUCTION TRUE INVERSIONS across 20 distinct files - core importing api/v1/endpoints/* or api/v1/API_Deps/*.

Sharpest instance: Ingestion_Media_Processing/persistence.py, where FOUR of five imports exist ONLY so tests can monkeypatch endpoints.media.*. Production resolves its file validator, temp-dir manager and template classifier via getattr on an API module at request time, so a stray attribute on endpoints.media silently reconfigures ingestion. The docstrings say so verbatim (:1884-1886, :2693-2695).

Cheap sub-case: three core/Chat files import the CONSTANT DEFAULT_CHARACTER_NAME from api/v1/API_Deps; moving one constant clears 3 of the 20 files.

AND THE EXISTING GUARD PROTECTS ONLY ONE DIRECTION OF A REAL CYCLE: tests/lint/test_endpoint_auth_deps_import_boundary.py:14-15 bans endpoints from importing core.AuthNZ.User_DB_Handling, while core/AuthNZ/User_DB_Handling.py:20 imports oauth2_scheme FROM api/v1/API_Deps/v1_endpoint_deps.

Proportionate fix: a sibling AST ratchet seeded at 106 so the number can only go down, with a HARD BAN on the 20 production true-inversion files. NOT a mass refactor. Helper_Scripts/ci/rls_coverage_ratchet.py is the freshest in-repo precedent for the shape.

Source: synthesis F34
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ratchet test exists with a recorded baseline of 106 files
- [ ] #2 The 20 production true-inversion files are on a hard ban list
- [ ] #3 persistence.py monkeypatch seam replaced with explicit DI
- [ ] #4 DEFAULT_CHARACTER_NAME moved into core
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RATCHET LANDED: tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py.

Three tests, each PROVEN to fail rather than assumed to work:
- test_no_new_core_to_api_imports - baseline 105 files. Adding a schema-only import to a new core file failed it (verified with a temporary probe file, then reverted).
- test_no_new_true_inversions - baseline 20 files. Adding an api/v1/API_Deps import failed BOTH tests, as designed.
- test_baselines_only_shrink - a stale baseline entry failed it, so fixing a violation forces the baseline down rather than letting it drift.

BASELINE IS AST-DERIVED, AND THAT CORRECTED THE AUDIT: the synthesis said 106 files based on grep; the true number is 105. grep counted core/DB_Management/db_errors.py, whose only match is a usage example inside a DOCSTRING. The ratchet cannot make that mistake.

Deliberately excludes core/MCP_unified/tests/ from the true-inversion set - it is an in-app test tree, not production core (23 of the 157 import lines).

SYNTHESIS CORRECTION FILED IN THE SAME CHANGE: the "cheap sub-case - move DEFAULT_CHARACTER_NAME and clear 3 of the 20 files" recommendation is WRONG and acting on it would introduce a defect. There are two constants of that name with DIFFERENT VALUES - core/Character_Chat/modules/character_utils.py:16 is "Character", api/v1/API_Deps/ChaCha_Notes_DB_Deps.py:350 is "Helpful AI Assistant". chat_history.py:112 passes it to get_character_card_by_name(), a DB lookup on that exact string, so repointing would fetch the wrong character card. Consolidating them is owner-only and is a behaviour decision.

Still open: the 20 true inversions themselves, particularly persistence.py where four of five exist only as monkeypatch seams.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
