---
id: TASK-540
title: Address PR 2091 review comments after rebase
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 16:49'
labels:
  - research-workspace
  - review-fix
  - workspaces
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2091'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase codex/workspaces-next onto latest dev and address still-valid review comments on PR #2091 with focused frontend/backend/test fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2091 rebased onto latest origin/dev.
- [x] #2 Still-valid review comments addressed with focused backend, frontend, and test changes.
- [x] #3 Focused frontend/backend regression tests, TypeScript, Bandit, and diff hygiene recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased codex/workspaces-next onto origin/dev and reviewed unresolved PR #2091 comments from Gemini, Qodo, and CodeRabbit.

Backend fixes: Redis backpressure now honors explicit REDIS_ENABLED=false before URL auto-enable; Redis close failures are debug-logged instead of suppressed; sandbox diagnostics validate source_label, run blocking store calls off the event loop, propagate cancellations, document response models, and coerce pagination totals once; ACP session/new fallback also removes mcpServers when runners reject that field; Redis worker failure handling avoids masking original errors with missing locals.

Frontend fixes: sandbox diagnostics strings use i18n fallbacks and runtime_config links are restricted to same-origin absolute app paths; migration recovery details labels use translated strings; workspace bundle import handles null/malformed payloads; migration cleanup blocks final tombstones when local deletion preflight cleanup is unavailable; Agent Tasks project fetches ignore stale responses.

Test/support fixes: added focused regressions for each review path, changed sandbox admin list endpoint tests to use a minimal router fixture with dependency cleanup in finally, and made TASK-478 verification command portable.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2091 review comments after rebasing on origin/dev. Verification: focused UI Vitest slice passed (4 files, 79 tests); TypeScript noEmit passed; focused backend pytest slice passed (60 tests, 2 warnings); Bandit over touched backend files/tests produced 0 results and 0 errors; git diff --check passed. Known unrelated files left untouched: two untracked watchlist template files under tldw_Server_API/Config_Files/templates/watchlists/.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
