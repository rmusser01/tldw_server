---
id: TASK-9940
title: Harden StudySuggestions review findings
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-23 22:05'
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
Fixed all four validated StudySuggestions review findings. Added regression coverage for fingerprint delimiter collisions, stale job status masking newer snapshots, deleted evidence sources, and failed follow-up finalization cleanup. Verification passed for the StudySuggestions worker/API test files and Bandit returned zero findings on the touched scope.
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
