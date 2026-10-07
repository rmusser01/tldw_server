---
id: TASK-13520
title: 'WebUI perf batch W0: baseline harness'
status: To Do
created_date: 2026-10-07 04:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program stage 0 (index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md). Plan: Docs/Plans/2026-10-06-webui-perf-batch-0-baseline-harness-implementation-plan.md. Streaming-render bench, persistence micro-bench, bundle baseline, PERF_BASELINE_WEBUI_2026_10.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

### Stage 1 / Task 1B — Playwright streaming perf e2e spec (2026-10-07)

- Added "Canned 100-chunk stream (no live LLM)" suite to
  `apps/extension/tests/e2e/performance-streaming.spec.ts` (path already held
  the live-backend suites; appended a new describe so no existing test was
  disabled). Loopback mock server serves a paced SSE stream of exactly 100
  chunks into the sidepanel `/chat`; an in-page bench on the real
  `useStoreMessageOption` store records `performance.mark`/`performance.measure`
  deltas plus store-notification, DOM-mutation and long-task counts; prints
  `[performance-streaming] {json}` (mirrors the 1A vitest bench convention).
  Sanity-level assertions only. Runs under `test:e2e:perf`.
- Verified: `bunx playwright test tests/e2e/performance-streaming.spec.ts` →
  1 passed / 5 skipped (pre-existing env-gated suites), stable across 3 runs
  (100 chunks served, 15 applied store-growth events, ~1.15s stream phase).
- Bandit: N/A — TypeScript test file only, no Python touched.
- Full report: `.superpowers/sdd/2026-10-06-webui-perf-batch-0-baseline-harness-implementation-plan/task-1b-report.md` (untracked by design).
- Prior stage 1 work: Task 1A vitest bench landed in commit 66c8a21c80.

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
