---
id: TASK-484
title: Address PR review for Watchlists durable audio projection
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-22 21:49'
labels:
  - watchlists
  - audio
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable PR review findings on durable Watchlists audio artifact projection: undefined run variable typos, mirror timing after scheduler fallback status normalization, path-safe artifact metadata scrubbing, and idempotency lookup before paginated Workflow run scan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Undefined run row naming in get_run_audio is clarified without behavior change.
- [x] #2 Mirrored audio projection persists finalized Scheduler fallback status after normalization.
- [x] #3 Mirrored artifact metadata is path-safe and Workflow run lookup uses idempotency before paginated scans.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR review findings for Watchlists durable audio projection. Renamed the local run row variable in get_run_audio for clarity, moved audio projection mirroring until after Scheduler fallback status normalization, scrubbed uri keys recursively from mirrored artifact metadata, and added O(1) Workflow run lookup by Watchlists audio idempotency key before paginated scans. Verification: focused regressions first failed on the old behavior and then passed; expanded backend suite passed with 335 passed, 1 skipped; Bandit on touched Python scope reported 0 results and 0 errors; git diff --check passed.
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
