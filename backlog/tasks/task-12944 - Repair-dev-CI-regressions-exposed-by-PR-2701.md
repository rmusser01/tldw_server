---
id: TASK-12944
title: Repair dev CI regressions exposed by PR 2701
status: Done
assignee: []
created_date: '2026-07-10 03:25'
updated_date: '2026-07-10 03:44'
labels:
  - ci
  - frontend
  - research
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the latest dev baseline regressions surfaced by PR 2701: smoke auth no longer reaches the app-shell runtime auth path, and research tests still assert pre-hardening clear-text artifact paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared smoke auth seeds credentials through the runtime-config contract without persisting API keys.
- [x] #2 Research tests validate hashed artifact storage through ResearchArtifactStore instead of clear-text paths.
- [x] #3 The focused UX smoke and critical deep-research tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both CI failures. 2. Update the shared smoke fixture and stale research assertions. 3. Run focused frontend/backend verification and Bandit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root causes: fecb7e4b22 moved smoke credentials out of persistent storage without routing them through runtime-config, and hardened research artifact storage while leaving clear-text path assertions behind. Verification: Stage 6 Playwright 6/6; critical deep-research E2E 1/1; research worker tests 21/21; frontend ESLint and TypeScript noEmit; Python Ruff; git diff --check. Bandit: touched Python test files scanned with B101 excluded; 0 findings. The unfiltered scan reported only expected B101 pytest assertions (278 low, 0 medium/high). Docs: no user-facing documentation change required. Blockers: none.

Independent final-diff review found that manifest-backed assertions should also prove storage-path confinement. Updated both research test helpers to require each resolved artifact path to remain under the configured research output root; focused E2E and worker suites were rerun successfully.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired latest-dev CI regressions by seeding smoke auth through the runtime-config endpoint and validating research artifacts through their manifest-backed hashed storage paths.
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
