---
id: TASK-16
title: Address PR 1240 character exemplar deleted-filter review
status: In Progress
assignee: []
created_date: '2026-05-03 21:07'
labels:
  - pr-review
  - chacha
  - phase-2
  - postgresql
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1240'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the extracted CharacterStore character exemplar methods to use backend-aware deleted predicates instead of hardcoded SQLite literals, matching existing cross-backend helpers and keeping the Phase 2.3 tranche narrow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 get_character_exemplar_by_id uses backend-safe deleted filtering when include_deleted is false.
- [ ] #2 list_character_exemplars uses backend-safe deleted filtering for non-deleted rows.
- [ ] #3 Focused tests, Bandit touched-source scope, and git diff --check are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regression coverage for PostgreSQL deleted filtering in CharacterStore exemplar read/list methods. 2. Patch only the extracted character exemplar methods to use _deleted_literal/_deleted_value consistently. 3. Run focused ChaChaNotes tests, Bandit on touched source, and git diff --check. 4. Push the review-fix commit and resolve the PR thread.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added focused PostgreSQL regression coverage for the extracted exemplar read/list helpers in `test_chacha_character_store.py`.
- Fixed `get_character_exemplar_by_id()` to use `_deleted_literal(False)` when `include_deleted` is false.
- Fixed `list_character_exemplars()` to bind `_deleted_value(False)` instead of hardcoding SQLite `0`.
- Verification:
  - `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py -k "backend_safe_deleted" -q` -> `2 passed`
  - `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py -q` -> `21 passed`
  - `python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha/character_store.py -f json -o /tmp/bandit_pr1240_character_store.json` -> `0 results, 0 errors`
  - `git diff --check` passed
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the remaining PR #1240 review item by making the extracted CharacterStore exemplar read/list queries use the existing backend-aware deleted helpers. The fix stays narrow, adds direct regression coverage for PostgreSQL semantics, and leaves the broader Phase 2.3 tranche unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->
