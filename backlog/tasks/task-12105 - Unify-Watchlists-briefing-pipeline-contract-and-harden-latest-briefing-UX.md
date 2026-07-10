---
id: TASK-12105
title: Unify Watchlists briefing pipeline contract and harden latest-briefing UX
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the confirmed UAT-driven Watchlists remediation: route every setup path through one versioned briefing pipeline contract, guarantee truthful scheduled text/audio fulfillment semantics, consolidate setup into Sources → Cadence → Briefing → Delivery → Test, add a natural-language activation receipt and Latest briefing surface, fix accessibility/live-region defects, and retest WebUI plus browser extension against matched frontend/backend revisions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All Watchlists setup entry points serialize through one versioned canonical pipeline contract while preserving unknown advanced fields and legacy watchlists.
- [ ] #2 Scheduled runs create explicit text and selected audio fulfillment outcomes, including no-material-update briefings, without false success or silent stage failure.
- [ ] #3 Retries are stage-scoped and idempotent so reports, audio, and delivery are not duplicated.
- [ ] #4 The shared WebUI/extension setup flow is Sources → Cadence → Briefing → Delivery → Test and ends with an exact timezone-aware natural-language receipt.
- [ ] #5 A Latest briefing surface exposes playback, text/audio readiness, delivery state, next run, provenance, and recovery actions.
- [ ] #6 Repeated or unlabeled accessible names are fixed, background transitions are announced reliably, and responsive/localized states meet WCAG AA expectations.
- [ ] #7 Focused frontend/backend tests, Bandit on touched Python, matched-revision WebUI CDP UAT, and browser-extension verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Confirmed Impeccable shape brief and hardening amendments: fulfillment-stage truth, no-material-update artifacts, delivery ordering, idempotent retries, one bounded selection policy, safe Test behavior, versioned legacy normalization, capability validation, and timezone/DST clarity.
- Confirmed design specification: `Docs/superpowers/specs/2026-07-09-watchlists-briefing-contract-ux-hardening-design.md`.
- Clean baseline before implementation planning: frontend contract/setup suite 35 passed; backend Watchlists briefing/audio suite 56 passed and 4 external-feed tests skipped.
- Spec review broadened the audience and editorial model beyond news/OSINT. The same contract now supports concise briefings, solo updates, multi-host discussions, sportscasts, culture roundtables, and custom source-grounded programs. Added durable script stages, show identity/notes, 60-second sample tests, honest target-duration copy, prompt-injection/copyright/disclosure/impersonation safeguards, and explicit podcast publishing non-goals.
- Approved implementation plan: `Docs/superpowers/plans/IMPLEMENTATION_PLAN_watchlists_briefing_contract_ux_2026_07_09.md`. Five stages cover canonical contract, durable fulfillment/delivery, editorial programs, outcome-first UI/accessibility, and matched-revision UAT/polish.
- Plan review corrected external-delivery semantics: providers without durable idempotency can return an unknown outcome after timeout, so acknowledged or uncertain sends are never automatically replayed and reviewed retry is required.
Stage 1 Task 1 complete. Added the versioned backend `briefing_pipeline` contract and compatibility projection; canonical create/update writes preserve unknown/ingest fields, fail closed on malformed sections and boolean values, and route pipeline/audio/retry/diagnostics through the contract. Reports-only destinations are not advertised as retryable external delivery. Commits: f9f224ed32, acbded7ad2, 1699747bfc. Verification: implementer combined suite 80/80 and Bandit 0 findings; independent final review SPEC PASS / CODE QUALITY APPROVED / CLEAN with 82 tests.
- Stage 1 Task 2 complete. Added the shared frontend v1 contract, one canonical job serializer, legacy/unknown preference preservation, actual four-route parity including JobForm, generalized briefing/episode formats, and an exact timezone/DST-aware receipt with reviewed delivery destinations. Commits: 51d2eef8ed, db97bde17e. Verification: 91/91 independent tests, touched ESLint/diff clean, zero touched TypeScript errors; independent review SPEC PASS / CODE QUALITY APPROVED / CLEAN.
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
