---
id: TASK-12050
title: Fix PR 1982 Watchlists extension E2E CI browser launch
status: In Progress
labels:
- ci
- pr-1982
- extension
- watchlists
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/1982
- https://github.com/rmusser01/tldw_server/actions/runs/28281338011/job/83797553514
- /tmp/ci1982_logs/job_83797553514_watchlists_extension_e2e_206.log
modified_files:
- .github/workflows/ui-watchlists-extension-e2e.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the PR #1982 Watchlists Extension E2E gate after the current-head run built the extension and passed backend health but skipped all 14 tests because no extension service worker/background target appeared. Local workflow patch restores the passing Playwright Chromium pattern from UI Research Workspace Parity. Verification so far: apps/extension `bun run compile` passed; workflow YAML parses with Ruby YAML; `git diff --check` passed. Focused Vitest launcher test could not be run through the current repo config because bare Vitest resolves to the shared UI include filter and finds no extension test files. Bandit is not applicable for this workflow-only YAML change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Root cause evidence:
- Watchlists Extension E2E builds the extension and backend health passes.
- All 14 watchlists tests skip after repeated 90s waits because no extension service worker appears.
- The failing workflow forces TLDW_E2E_PLAYWRIGHT_CHANNEL=chrome and only verifies system google-chrome 149.0.7827.155.
- The passing UI Research Workspace Parity extension job installs Playwright Chromium and lets the launcher use its CI default channel=chromium.
Plan:
1. Restore the Watchlists Extension E2E workflow to the passing Playwright Chromium pattern.
2. Verify workflow diff and local launcher/unit coverage where possible.
3. Push and monitor the current PR head checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
