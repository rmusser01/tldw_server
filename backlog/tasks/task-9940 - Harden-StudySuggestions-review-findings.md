---
id: TASK-9940
title: Harden StudySuggestions review findings
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-24 04:51
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix validated current-code review findings in `tldw_Server_API/app/core/StudySuggestions` and its API boundary: collision-resistant fingerprints, recency-aware anchor status, real source availability evidence, and cleanup for failed action finalization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collision-safe bounded StudySuggestions selection fingerprints implemented
- [x] #2 Anchor status ignores stale failed/pending jobs when a newer active snapshot exists
- [x] #3 Snapshot live evidence checks backing quiz/flashcard source availability
- [x] #4 Generated follow-up quiz/deck targets are cleaned up if link finalization fails
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan file: IMPLEMENTATION_PLAN_study_suggestions_review_fixes_9940.md
Initial scope: fix collision-safe fingerprints, recency-aware status, live evidence availability, and finalization cleanup with failing-first tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented hashed v2 fingerprints with legacy lookup candidates, recency-aware anchor status, DB-backed source availability for live evidence, and generated-target cleanup on finalization failure.
Verification: StudySuggestions worker/API test files pass; Bandit touched-scope report has zero findings at /tmp/bandit_study_suggestions_9940.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed all original StudySuggestions review findings plus PR #2490 review follow-up comments. Added core-owned generated-target cleanup, observable best-effort cleanup/release failure logging, helper docstrings, and live-evidence DB failure degradation with regression coverage. Final verification passed for 43 affected StudySuggestions tests, Ruff touched-file checks, Bandit touched backend scope with zero findings, and whitespace checks.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2490 review follow-up after rebasing onto latest `origin/dev`: address Qodo comments about moving generated-target cleanup into core StudySuggestions actions, documenting helper functions, logging best-effort cleanup/release failures, and degrading live evidence source lookup DB failures to unavailable instead of returning 500.
Review follow-up completed: moved generated target cleanup into `app/core/StudySuggestions/actions.py`, replaced silent endpoint cleanup suppression with warning logs carrying operation identifiers, added docstrings to the reviewed snapshot helpers, and made live-evidence DB lookup failures return unavailable evidence. Verification: focused red tests failed before the fix; final `python -m pytest tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py -q --tb=short` passed 43 tests. Ruff touched-file check passed. Bandit touched backend scope reported zero findings in `/tmp/bandit_study_suggestions_9940_reviewfix_worktree.json`. `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
