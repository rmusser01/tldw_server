---
id: TASK-12947
title: Fix browser extension E2E launch and validate Quick Ingest
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-11 17:58'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-11-extension-e2e-launch-cancel-race-design.md
  - Docs/superpowers/plans/2026-07-11-extension-quick-ingest-cancellation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate why the local Playwright extension harness fails to expose or complete an MV3 extension context, root-cause the headful timeout, validate PDF/link/YouTube Quick Ingest through the actual browser extension against a live backend, add regression coverage, and open a separate PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Extension launch behavior is proven with a clean headed/headless and minimal/full locale matrix; any missing-target failure must be reproduced with retained diagnostics before launcher code changes.
- [ ] #2 The supported extension launcher reliably returns an extension id, seeded storage, and a loaded options page without relying on CI-only behavior.
- [ ] #3 Host-side browser-extension UAT validates PDF, reachable link, repeated link, exact YouTube Short, and repeated YouTube in one extension context with no page or console errors.
- [ ] #4 Focused automated regression coverage fails before the fix and passes afterward, with adjacent extension tests and compile green.
- [ ] #5 Changes are rebased on current dev and published in a separate reviewed PR against dev.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-11-extension-quick-ingest-cancellation.md using host-side TDD and UAT. Order: terminal runtime fence; async setup/start guards; persisted reattach guard; installed-extension PDF/link/YouTube UAT; packaged regression/static verification; review/rebase/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-11 investigation evidence:
- Packaged-extension launch health passes headed/minimal in 4.7s, headless/minimal in 3.1s, and headless/full in 2.7s after installing apps workspace dependencies and building the production extension. No launch-helper or manifest-key change is justified.
- The exact Quick Ingest extension regression launches Chromium in about 2s, reaches Cancel All, and then fails waiting for cancelled/error results while the UI reports Succeeded (1).
- Root cause: cancellation is fenced only in a later passive effect, so immediate completion can win.

2026-07-11 design review corrections:
- Add synchronous run-level cancellation intent and one shared Cancel All handler.
- Cover cancellation during payload preparation and before extension/direct acknowledgement.
- Cancel a late extension session and suppress direct backend batch submission.
- Ignore every post-cancel runtime message, including progress.
- Stop persisted direct-job reattachment from overwriting cancelled results or scheduling another poll.
- Keep current launcher behavior unless a missing-target failure is reproduced with retained diagnostics.

2026-07-11 plan review: cancellation intent must be initialized once per keyed wizard-session mount and never reset inside startRun, otherwise a delayed start effect could erase an already-recorded cancellation. The implementation plan now includes idempotent cancellation and distinct late-processing/late-completed reattachment cases.

2026-07-11 Stage 1 TDD evidence (host): focused Vitest RED had 2/2 failures. Immediate completion produced complete:1 instead of cancelled:1; immediate progress preserved outcome processed under cancelled status. After adding the modal-owned synchronous idempotent fence and rejecting all fenced runtime messages, the same 2 tests passed (25 unrelated tests skipped by the name filter). No sleeps were used.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
