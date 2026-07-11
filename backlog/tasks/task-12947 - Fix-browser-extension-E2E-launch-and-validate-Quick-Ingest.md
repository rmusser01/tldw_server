---
id: TASK-12947
title: Fix browser extension E2E launch and validate Quick Ingest
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-11 16:44'
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
- [ ] #1 Root cause of missing MV3 targets and the headful timeout is proven with step-level evidence.
- [ ] #2 The supported extension launcher reliably returns an extension id, seeded storage, and a loaded options page without relying on CI-only behavior.
- [ ] #3 Host-side browser-extension UAT validates PDF, reachable link, repeated link, exact YouTube Short, and repeated YouTube in one extension context with no page or console errors.
- [ ] #4 Focused automated regression coverage fails before the fix and passes afterward, with adjacent extension tests and compile green.
- [ ] #5 Changes are rebased on current dev and published in a separate reviewed PR against dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-07-11 root cause evidence: headed packaged-extension launch health passed in 4.7 seconds. The exact Quick Ingest extension regression launched Chromium in 2 seconds and reached Cancel All. Playwright API tracing failed at quick-ingest-cancel.spec.ts:163 because the cancelled/error region never appeared; the final screenshot showed Succeeded (1). ProcessingStep dispatches cancellation before QuickIngestWizardModal fences the session in a passive effect, allowing an immediate stale completion to win. Approved design records an explicit headed local launch default, deterministic manifest staging, and a synchronous cancellation fence.
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
