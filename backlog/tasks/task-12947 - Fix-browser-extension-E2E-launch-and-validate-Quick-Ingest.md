---
id: TASK-12947
title: Fix browser extension E2E launch and validate Quick Ingest
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-11 18:02
labels: []
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-11-extension-e2e-launch-cancel-race-design.md
- Docs/superpowers/plans/2026-07-11-extension-quick-ingest-cancellation.md
modified_files:
- apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx
- apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
- apps/extension/tests/e2e/quick-ingest-cancel.spec.ts
- apps/extension/tests/unit/wxt-config-public-dir.test.ts
- apps/packages/ui/src/public/fonts/
- Docs/superpowers/specs/2026-07-11-extension-e2e-launch-cancel-race-design.md
- Docs/superpowers/plans/2026-07-11-extension-quick-ingest-cancellation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate why the local Playwright extension harness fails to expose or complete an MV3 extension context, root-cause the headful timeout, validate PDF/link/YouTube Quick Ingest through the actual browser extension against a live backend, add regression coverage, and open a separate PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extension launch behavior is proven with a clean headed/headless and minimal/full locale matrix; any missing-target failure must be reproduced with retained diagnostics before launcher code changes.
- [x] #2 The supported extension launcher reliably returns an extension id, seeded storage, and a loaded options page without relying on CI-only behavior.
- [x] #3 Host-side browser-extension UAT validates PDF, reachable link, repeated link, exact YouTube Short, and repeated YouTube in one extension context with no page or console errors.
- [x] #4 Focused automated regression coverage fails before the fix and passes afterward, with adjacent extension tests and compile green.
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

2026-07-11 Stage 2 TDD evidence (host): four deferred tests failed before the async guards: late extension ack was not cancelled, late direct ack invoked submitQuickIngestBatch, resumed setup started a session, and late start rejection changed cancelled status to error. After adding cancellation checks after awaited setup/start boundaries and in catch, the focused selection passed 5/5. Two persisted-reattach characterization tests passed before any poll code change, proving the existing effect cleanup already blocks late processing/completed snapshots; no redundant poll guard was added. Full session file: 33/33 passed.
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
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-11 implementation and validation evidence:
- Launch investigation: clean packaged MV3 launch health passes headed/minimal, headless/minimal, and headless/full; no launcher or manifest change is justified.
- Cancellation TDD: synchronous run-level intent and session fencing make Cancel All terminal. Deferred extension/direct acknowledgements, setup continuations, errors, progress, completion, and persisted reattachment cannot revive or continue a cancelled run. Full session suite passed 33/33.
- Installed-extension UAT: one headed context against isolated Auth, Jobs, and Media stores completed PDF, RFC 9110 URL, duplicate URL skip, exact Short https://www.youtube.com/shorts/6-rf_YXDpPg, and duplicate YouTube skip in 44.9 seconds. Browser diagnostics had zero page errors, console errors, or failed media requests; jobs reached completed/100%; Media DB contained exactly three unique active rows.
- Font packaging: live UAT exposed missing Inter requests. A static publicDir contract failed first, then passed 2/2 after packaging all nine referenced shared fonts.
- Maximum-depth root cause: the installed MV3 cancellation regression reproduced React invariant 185. React replayed terminal reducer updates after synchronous wizard-state persistence rerendered the parent; fresh completion timestamps made equivalent cancelled snapshots appear distinct and caused repeated Zustand upserts.
- Persistence fix: derive a semantic signature from the existing persisted patch, normalize only volatile timestamp presence, and suppress an equivalent replay before the synchronous write. Real timestamps and all meaningful status, queue, progress, tracking, and result values remain persisted.
- Browser regression: the test now uses the production MV3 direct HTTP path against a strict local server, defers only process-web-scraping, cancels before response, releases late success, and asserts cancelled remains terminal with zero page, console, or unexpected-request errors. It passed in 22.6 seconds.
- Adjacent verification: Quick Ingest tests passed 56/56, font tests 2/2, TypeScript compile passed. ESLint has no applicable workspace configuration; Bandit is not applicable to TypeScript/assets. No credentials, auth headers, request bodies, or backend payloads are logged.
2026-07-11 post-rebase verification: origin/dev was already current. Production Chrome MV3 build passed in 37.0s and packaged all nine fonts. The strict headed cancellation regression passed in 23.3s, Quick Ingest unit tests passed 56/56, font contract tests passed 2/2, TypeScript compile passed, and launch health passed headed/minimal 4.2s, headless/minimal 3.4s, and headless/full 2.9s. git diff --check was clean. Existing build warnings about duplicate imports, circular chunks, and bundle size remain outside this diff.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
