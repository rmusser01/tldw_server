---
id: TASK-378.5
title: Stage 3E Watchlist alerts overview QA and closeout
status: Done
assignee: []
created_date: '2026-05-15 14:53'
updated_date: '2026-05-15 17:26'
labels:
  - watchlists
  - stage3
  - qa
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-378
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate content alert and health summaries into Overview, run focused verification including Bandit and CDP, and close Stage 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overview shows unread content alerts separately from health issues.
- [x] #2 Focused backend/frontend tests, Bandit, git diff check, and CDP desktop/mobile smoke are recorded.
- [x] #3 Stage 3 Backlog tasks and docs are closed out with known skips or blockers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 3E after Stage 3D commit 1a8378570. Scope: add Overview separation for unread content alerts versus health issues, run focused backend/frontend verification, Bandit, git diff check, and CDP desktop/mobile smoke.

Implemented Overview alert/health separation: unread content alerts now load from the content-alerts API and render separately from health issues with actions to Alerts and Activity.

Added alerts route deep-link support so /watchlists?tab=alerts resolves to the Alerts tab; discovered during real-server CDP verification.

Real server only: discarded the earlier mocked CDP attempt. FastAPI ran on 127.0.0.1:8000 with AUTH_MODE=single_user and the WebUI dev server ran on 127.0.0.1:18001.

Real server seed: created Watchlist/source/job/rule through the live API and used the Watchlists DB helper only for run/item/content-alert seed data because there is no public create-alert endpoint.

Real API probes: /api/v1/watchlists/3/alerts?status=unread returned total=1; /api/v1/watchlists/runs?q=failed&watchlist_id=3 returned total=1.

CDP smoke: node /private/tmp/watchlists-real-cdp-smoke.cjs passed against live WebUI and live FastAPI. Screenshots: /tmp/watchlists-stage3-alerts-desktop-cdp.png and /tmp/watchlists-stage3-alerts-mobile-cdp.png.

Verification: focused Stage 3E Vitest passed 4 files / 13 tests; option-watchlists route-state test passed 4 tests; git diff --check passed; filtered tsc for Watchlists/option-watchlists paths had no output. Earlier Stage 3 backend tests passed 43 tests and Bandit exited 0 with only existing nosec B608 warnings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3E closes the Watchlists content-alerts UI loop by surfacing unread content alerts separately from operational health on Overview, linking users to Alerts or Activity with clearer copy, and making Alerts first-class in route query handoff. Verification used the real FastAPI server and real WebUI via CDP, with desktop/mobile screenshots captured.
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
