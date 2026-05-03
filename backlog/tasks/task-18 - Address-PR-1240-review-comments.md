---
id: TASK-18
title: 'Address PR #1240 review comments'
status: Done
assignee:
  - codex
created_date: '2026-05-03 21:40'
updated_date: '2026-05-03 21:48'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1240'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve remaining review feedback on PR #1240 for ChaCha exemplar delegation. Scope is limited to the existing PR branch and the Qodo/Gemini comments around CharacterStore delegation, exemplar normalization duplication, SQL construction, and test helper typing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CharacterStore no longer exposes an unbounded parent DB __getattr__ proxy while preserving required delegated behavior.
- [x] #2 Character and persona exemplar normalization share one implementation with unchanged public facade behavior.
- [x] #3 Character exemplar SQL paths avoid the review-flagged string formatting and keep PostgreSQL-safe soft-delete values.
- [x] #4 New or modified test helpers have explicit parameter and return type annotations.
- [x] #5 Focused ChaChaNotesDB tests and Bandit on touched code pass or have documented baseline-only findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing tests for the unbounded CharacterStore proxy and shared exemplar normalization behavior. 2. Move common exemplar normalization helpers into a shared chacha module and update both stores to call it. 3. Replace CharacterStore.__getattr__ with explicit _db calls/properties needed by exemplar methods, and remove review-flagged format_map usage in exemplar SQL paths. 4. Add missing test helper type annotations. 5. Run focused pytest, Bandit on touched app code, and git diff checks; update this task with results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py -k 'does_not_proxy or format_map or shared_by_stores' -v failed as expected. Failures covered unbounded CharacterStore proxy, remaining exemplar .format_map usage, and missing shared exemplar_normalization module.

Green verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -v passed 31 tests. git diff --check passed. Bandit on touched app code wrote /tmp/bandit_pr1240_review.json with zero findings; warnings were existing nosec notices only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1240 review feedback by removing CharacterStore's unbounded parent DB __getattr__ proxy, moving shared exemplar normalization into chacha/exemplar_normalization.py, converting character exemplar methods to explicit parent DB calls, replacing exemplar .format_map() query assembly with parameterized/static predicates, and annotating the flagged test helpers. Verified with focused ChaChaNotesDB pytest coverage, git diff --check, and Bandit on touched app code.
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
