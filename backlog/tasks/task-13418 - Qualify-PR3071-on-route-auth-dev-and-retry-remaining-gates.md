---
id: TASK-13418
title: Qualify PR3071 on route-auth dev and retry remaining gates
status: In Progress
created_date: 2026-10-02 05:59
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/3071
documentation:
- IMPLEMENTATION_PLAN_pr3071_route_auth_dev_retry_2026_10_01.md
updated_date: 2026-10-02 06:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved PR3071 retry on frozen dev3caebcfc. Preserve the prior TASK-13417 qualification history and incoming route-map TASK-13417 without active ID collision. Retain source-equivalent Chat Workspace and current-head CI successes; run incoming route-auth/benchmark gates and actual read-only Chrome acceptance without mocks. Retry PostgreSQL only through official fixtures; do not restart Docker, remove shared containers, fabricate a database or bypass gates. Publish to the existing draft PR with requester Change summary unchanged and no merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preserve the prior tracker byte-identically and integrate frozen dev3caebcfc without losing either branch history.
- [x] #2 Incoming authentication/ratchet checks, scoped Bandit and actual Chat Workspace acceptance are qualified, with unchanged-source evidence reused only where byte-identical.
- [ ] #3 Retry official PostgreSQL and head-bound CI, record real remaining limits, and normally publish the existing draft PR without merging or changing requester summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Archive the collided qualification record intact and merge frozen dev. 2. Run incoming gate tests and review, scoped Bandit, and source-equivalent native Chrome checks. 3. Retry official PG/current-head CI and publish evidence without claiming incomplete gates passed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Independent review confirmed two incoming route-auth gaps: JSON enable lists lose configured route keys during CSV-only copying, and dotenv-only config selection is pinned away before canonical dotenv loading. Add real subprocess inventory regressions first, then reuse canonical config loading/cache refresh and preserve supported list syntax. Hosted backend-required at head7274 failed OpenAPI fingerprint drift (2105 paths/3259 schemas vs3245 checked in); mypy is explicitly non-blocking, so do not expand into inherited type debt. Direct auth-postgres job log verifies56 actual tests passed but excludes durable-turn/image-recovery PG arms; local7 official-fixture skips remain unqualified.
Frozen dev3cae still latest at final fetch; clean merge staged and prior tracker archive SHA546a0220d918a561522667767d85ea78fc1bb25d2d798af999ce1b0ff9dde93a retained intact. Four strengthened subprocess regressions first fail (JSON configured jobs lost; FILE/PATH/DIR dotenv selections incorrectly include disabled chat). Minimal shared-loader fix reuses canonical route-policy parser, early dotenv loading and existing cache resets; owning17 pass, final lint+benchmark delta87 pass/1 inherited skip/4 warnings. Independent follow-up review no actionable findings, static only; JSON/all3 dotenv selectors covered, newline/dotenv test-mode flags not newly covered. Final4-module Bandit0findings/0errors; changed Ruff clean, connector I001/SIM114 unchanged against published7274. OpenAPI gate reproduced CI hashf4609bf with isolated FastAPI0.142.2/Pydantic2.13.5/Starlette1.7 (root venv/live services untouched), frozen-dev controle384a65. Delta14 added durable/recovery schemas,3 updated models,9 intended path parameter/capability changes, no removals. Regenerated fingerprint/types with existing exporter/openapi-typescript; fresh drift check passes. All7217 qualified frontend entries still identical except the metadata-only fingerprint. Fresh native Chrome/CDP reload/mobile passes actual protectedGET200, new loader, retained nonempty draft,92 Tabs to composer,zero overflow/zero completions; inspected desktop/mobile PNGs. Actual original10/eight-row hashes, served contract,6 baseline tabs and68 stashes preserved; historical missing targets remain absent, not claimed preserved. Hosted auth-integration-b-z JUnit at7274:149 actual passes/0skips, including all6 durable-user-turn PostgreSQL tests (concurrency, migration, owner/order/history/image fences). This closes those6 only on source-equivalent scope; the separate image-recovery PostgreSQL case and next-head CI remain unqualified. Local official7 PostgreSQL cases still skip because service unavailable; no Docker restart/bypass.
Final CI-version route-auth plus route-map inventory rerun:23 passed/6 warnings. Full frontend TypeScript passes after generated types refresh with existing8192MB allowance. Incoming unrelated TASK13417 blobc89000751b matches frozen origin/dev exactly. Preparing normal merge commit/hooks and normal push to existing draft PR3071; next-head hosted results remain separate.
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
