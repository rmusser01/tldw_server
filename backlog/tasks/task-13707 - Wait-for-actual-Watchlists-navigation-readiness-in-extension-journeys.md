---
id: TASK-13707
title: Wait for actual Watchlists navigation readiness in extension journeys
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current PR3229 hosted extension E2E exposes a race: the Watchlists heading renders before selected-watchlist navigation. The driver snapshots desktop alternatives and then waits permanently for an absent mobile trigger. Reproduce delayed real loading and await the existing shared content shell before branching; preserve original assertions, deadlines and supported desktop/mobile navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deterministic delayed-loading probe reproduces the hosted wrong-mobile-branch failure before repair
- [x] #2 Shared navigation helper waits for the actual content shell without changing production behavior or deadlines
- [x] #3 Original strict Watchlists and prompt-improvement extension suites, relevant checks and independent reviews pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR https://github.com/rmusser01/tldw_server/pull/3229; new failure on source-identical0f14 job116502852782/run38758382084. Screenshot shows desktop Updates while helper awaits mobiletrigger. ReviewerB independently confirms heading is unconditional and both navigation branches appear with watchlists-tab-content-shell only when watchlistViewsAvailable. ADR assessment: no new durable rule; test-driver readiness implements existing responsive navigation contract, ADR-059 governs tracking. Local historical planning (untracked; retired after completion): Docs/Design/IMPLEMENTATION_PLAN_watchlists_navigation_readiness_13707.md. Committed evidence: Docs/Reviews/CHAT_FINAL_RECOVERY_SETUP_HANDOFF_13673_13680_2026_10_14.md, Extension navigation admission section.
Causal private actual-packaged-extension probe delays only keyboard fixture GET/watchlists by1000ms. Original helper RED unexpected1, exact hosted absent mobile-trigger timeout10000ms; all original keyboard assertions/deadlines retained. One existing content-shell visibility assertion before layout probes yields GREEN1/1,0skip/retry/flaky/errors (case30.206s). Fresh native production build94s; isolated profile/backends and headless contexts stop in finally. First private adapter used wrong monorepo CLI path and ran no tests; excluded, corrected to actual installed extension dependency. ReviewerB confirms render coupling and minimal fix. Complete original14Watchlists+22prompt inventories now running against isolated profile. No production/workflow change; Bandit N/A TS-only supplement.
Complete original strict inventories GREEN14Watchlists/77.12s and22prompt-improvement/138.07s. Original no-skip validators pass;0skip/retry/flaky/errors, isolated backend healthy after suites, owned processes stopped. One readiness assertion only; no product/workflow change, all original source/assertions/deadlines retained. Lint0errors103parent-identicalwarnings. Independent working reviews A/B CLEAR. Implementation qualified; normalhooks/exactcommitreviews/fresh hosted CI and merge remain landing gates. Evidence Docs/Reviews/CHAT_FINAL_RECOVERY_SETUP_HANDOFF_13673_13680_2026_10_14.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the shared extension navigation driver by awaiting its existing content shell before one-shot responsive-layout probes. Deterministic actual packaged-extension delayed-fixture RED matches hosted absent-mobile-trigger failure; same case GREEN after one assertion. Full original14+22 cases and strict no-skip validators pass, preserving every deadline/assertion; lint0newissues and both bounded working reviews clear. No production change, new dependency or Bandit-relevant source. Final exact-head CI/reviews/authorized merge remain required.
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
