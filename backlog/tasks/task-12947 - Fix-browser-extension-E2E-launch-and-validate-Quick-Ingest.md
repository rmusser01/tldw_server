---
id: TASK-12947
title: Fix browser extension E2E launch and validate Quick Ingest
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-11 16:54'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-11-extension-e2e-launch-cancel-race-design.md
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
