---
id: TASK-80
title: Stabilize Watchlists Vitest gates
status: Done
assignee: []
created_date: '2026-05-05 17:53'
updated_date: '2026-05-05 18:13'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the next tldw-frontend local test slice after PR #1318 by stabilizing Watchlists Vitest failures found on fresh origin/dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused Watchlists Vitest slice passes
- [x] #2 TypeScript frontend gate remains clean
- [x] #3 Diff check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the fresh origin/dev Watchlists Vitest failures. 2. Update stale Watchlists test mocks and assertions. 3. Keep heavyweight interaction tests stable under the full Watchlists directory run. 4. Verify Watchlists Vitest, TypeScript, and diff check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline: tsc was clean. Watchlists Vitest initially failed 10 tests across stale Modal mock, stale output-link test IDs, obsolete snapshot assertions, a renamed cron help aria label, and timeout-only heavy interaction tests under the full 96-file run.

Fix: replaced brittle Watchlists external snapshots with explicit copy/ARIA contracts, updated output relationship jump expectations to current test IDs and status-filter reset, added Modal.confirm to the JobsTab delete mock, and gave known heavy Watchlists interaction tests targeted larger timeouts.

Verification: bunx vitest run ../packages/ui/src/components/Option/Watchlists --reporter=verbose passed 96 files / 423 tests; node node_modules/typescript/bin/tsc --noEmit --pretty false -p tsconfig.json passed; git diff --check passed. Bandit skipped because only frontend TS/TSX test files and Backlog task metadata changed.

PR: https://github.com/rmusser01/tldw_server/pull/1321
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stabilized the Watchlists frontend Vitest gate after PR #1318 by refreshing stale mocks and assertions, replacing obsolete snapshot checks with explicit copy/ARIA contracts, and adding targeted timeouts for heavy Watchlists interaction tests under the full local directory run. Verification passed: Watchlists Vitest 96 files / 423 tests, frontend TypeScript, and git diff --check. Bandit skipped because the change is frontend test-only plus Backlog metadata.
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
