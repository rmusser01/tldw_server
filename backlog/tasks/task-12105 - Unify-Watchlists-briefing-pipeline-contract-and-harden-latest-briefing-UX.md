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
- Stage 1 Task 1 complete. Added the versioned backend `briefing_pipeline` contract and compatibility projection; canonical create/update writes preserve unknown/ingest fields, fail closed on malformed sections and boolean values, and route pipeline/audio/retry/diagnostics through the contract. Reports-only destinations are not advertised as retryable external delivery. Commits: f9f224ed32, acbded7ad2, 1699747bfc. Verification: implementer combined suite 80/80 and Bandit 0 findings; independent final review SPEC PASS / CODE QUALITY APPROVED / CLEAN with 82 tests.
- Stage 1 Task 2 complete. Added the shared frontend v1 contract, one canonical job serializer, legacy/unknown preference preservation, actual four-route parity including JobForm, generalized briefing/episode formats, and an exact timezone/DST-aware receipt with reviewed delivery destinations. Commits: 51d2eef8ed, db97bde17e. Verification: 91/91 independent tests, touched ESLint/diff clean, zero touched TypeScript errors; independent review SPEC PASS / CODE QUALITY APPROVED / CLEAN.
- Stage 2 Task 3 complete. Added owned, atomic briefing occurrence persistence across SQLite/PostgreSQL schemas with race-safe create-or-get, leak-safe user/run/job reads, named lifecycle updates, explicit nullable-ID clearing, closed status validation, and non-negative counts. Commits: 22db76f8ff, f5cccc140b. Verification: 36/36 independent compatibility tests, guarded official PostgreSQL behavior coverage, Bandit/compile/diff clean; independent review SPEC PASS / CODE QUALITY APPROVED / CLEAN.
- Stage 2 Task 4 complete and independently CLEAN. Implemented occurrence-backed, idempotent scheduled briefing fulfillment in commits 95be22be8b, 252a97a68f, and fdf1f61625: shared ordered selection for text/audio, deterministic zero-item text/audio, crash-safe output/audio recovery, stable idempotency keys, durable stage/status transitions, and accurate audio projection. Independent review reported SPEC COMPLIANCE PASS and CODE QUALITY APPROVED with 193 tests passing. PostgreSQL runtime and external RSS cases remain environment-gated; Bandit/diff checks were clean. Two non-blocking future optimizations were noted: deterministic ordering for unused include-deleted tombstone lookup and bounding cached rendered content in stages JSON.
- Stage 2 Task 5 complete and independently CLEAN after three rigorous hardening cycles. Commits: 338299476d, 5d133f6065, ac31973f3a, 20b7730bce, b8b4195edb. Delivered owned latest/exact briefing projections, stage-scoped recovery, backward-compatible retry wrappers, terminal-audio-verified delivery ordering, durable SQLite/PostgreSQL attempt ledger, attempt-specific Scheduler keys, concurrency-safe atomic external-delivery/audio finalization, explicit confirmation for ambiguous sends, ledger-authoritative duplicate prevention, tenant-correct retries, pure read projections, validated destinations, redacted Chatbook errors, frontend overview services/types, and canonical restore fidelity. Final independent review: SPEC COMPLIANCE PASS / CODE QUALITY APPROVED / CLEAN, with 92 occurrence/delivery/projection tests, 12 delivery truth-table cases, deterministic sibling-adapter interleaving, import-order checks, compileall, Bandit, and full-range diff checks clean. Broad implementation gates included 239 backend and 10 UI/type tests. Live PostgreSQL behavior remained environment-skipped after attempted Docker startup; transformed PostgreSQL DDL and locking/transaction structure were verified.
- Stage 3 Task 6 complete and independently CLEAN. Commits: 6c1015027a, c3d66fdf9a, b7437db3da, 350d510df1, ae979de52e. One mapping-driven composer/workflow now supports concise briefings, solo updates, host discussions, sportscasts, culture roundtables, and custom programs with immutable grounding prompts, subordinate bounded editorial data, exact ordered selection representation, deterministic no-update LLM bypass, safe 1-4 speaker markers/voices, URL-free speech, public show-note provenance, final-artifact disclosure/metadata, and cross-platform secret/path scrubbing. Hardening covered pathological 1,000-item XML, four-host long copy, colliding/Unicode cast IDs, nested/deep encoded credentials, alternate/bare URI forms, and version-vs-IP speech. Final independent review: SPEC COMPLIANCE PASS / CODE QUALITY APPROVED / CLEAN; 323 focused tests, Bandit 0 findings, diff clean. Task 4-5 lifecycle/idempotency compatibility remained green with documented external RSS skips.
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
