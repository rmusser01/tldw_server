---
id: TASK-12946
title: Unify Watchlists briefing pipeline contract and harden latest-briefing UX
status: In Progress
documentation:
- Docs/superpowers/specs/2026-07-09-watchlists-briefing-contract-ux-hardening-design.md
- Docs/superpowers/plans/IMPLEMENTATION_PLAN_watchlists_briefing_contract_ux_2026_07_09.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the confirmed UAT-driven Watchlists remediation: route every setup path through one versioned briefing pipeline contract, guarantee truthful scheduled text/audio fulfillment semantics, consolidate setup into Sources → Cadence → Briefing → Delivery → Test, add a natural-language activation receipt and Latest briefing surface, fix accessibility/live-region defects, and retest WebUI plus browser extension against matched frontend/backend revisions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All Watchlists setup entry points serialize through one versioned canonical pipeline contract while preserving unknown advanced fields and legacy watchlists.
- [x] #2 Scheduled runs create explicit text and selected audio fulfillment outcomes, including no-material-update briefings, without false success or silent stage failure.
- [x] #3 Retries are stage-scoped and idempotent so reports, audio, and delivery are not duplicated.
- [x] #4 The shared WebUI/extension setup flow is Sources → Cadence → Briefing → Delivery → Test and ends with an exact timezone-aware natural-language receipt.
- [x] #5 A Latest briefing surface exposes playback, text/audio readiness, delivery state, next run, provenance, and recovery actions.
- [x] #6 Repeated or unlabeled accessible names are fixed, background transitions are announced reliably, and responsive/localized states meet WCAG AA expectations.
- [x] #7 Focused frontend/backend tests, Bandit on touched Python, matched-revision WebUI CDP UAT, and browser-extension verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Final independent-review remediation completed on the rebuilt branch. Default-tenant normalization now covers all Workflow endpoints, generated `audio/*` artifacts are permitted by the default download policy, setup renders a localized natural-language receipt, selected audio requires a successful current-draft sample/full test before activation, and missing current audio changes aggregate readiness to Partial. The branch was rebased onto exact `origin/dev` merge base `42631082bb32611e6a52abe1cf468822477e7c44` with a dated safety ref retained. Rebuilt verification: 476 backend tests passed; Watchlists accessibility 100/100; focused receipt/status UI 53/53; static type guard 3/3; Chrome MV3 production build passed; Bandit scanned 41,401 lines with 0 findings and 0 errors; diff and locale-key checks were clean. Real FastAPI plus unpacked-extension Playwright/CDP UAT (no CUA, no mocked backend) loaded occurrence 17, observed two authenticated artifact responses at HTTP 200, advanced the real 24.55-second audio to `currentTime=0.11059` with `readyState=4`, and completed a 196,845-byte `Real-Backend-CreatorCast.mp3` browser download. PR #2710 must remain draft until the requester supplies the repository-required human-written Change summary.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Watchlists briefing hardening and independent validity remediation on a clean branch based directly on current dev. The canonical pipeline now produces truthful text/audio/delivery outcomes, the outcome-first setup includes a localized natural-language receipt and successful-audio activation gate, Latest briefing exposes coherent partial/recovery semantics, and Workflow artifacts enforce the correct single-user tenant boundary while allowing generated audio downloads. Rebuilt automated verification passed 476 backend tests, 100 accessibility tests, 53 focused UI tests, 3 static type guards, the Chrome MV3 production build, clean diff/locale checks, and Bandit with zero findings. Final real-backend extension CDP UAT successfully played and downloaded the actual generated briefing audio. The only remaining merge gate is the repository policy requiring the human requester to write the PR Change summary.
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
