---
id: TASK-13520
title: 'WebUI perf batch W0: baseline harness'
status: Done
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

### Stage 2 / Task 2 — persistence micro-benches (2026-10-07, commit 736b1a32a6)

- Landed `apps/packages/ui/src/store/__tests__/workspace-persist.bench.test.ts`
  (one trivial `set()` on a 10 workspaces x 50 artifacts fixture re-serializes
  ~2.45 MB across 31 JSON.stringify calls; ~19 of those are dev-only
  diagnostics under NODE_ENV=test — absolute number overstates prod, W2
  deltas still valid) and `apps/tldw-frontend/lib/__tests__/history.bench.test.ts`
  (50 x 100 KiB `addRequestHistory` → ~130.8 MB attempted setItem bytes for a
  ~4.93 MB final history, ~50x average-history-size amplification). Count/byte
  metrics deterministic across runs; no product-code changes.
- Full report: `.superpowers/sdd/2026-10-06-webui-perf-batch-0-baseline-harness-implementation-plan/task-2-report.md` (untracked by design).

### Stage 3 / Task 3 — bundle-budget baseline + PERF_BASELINE_WEBUI_2026_10.md (2026-10-07)

- Created `Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md`: environment,
  re-run commands, and baseline tables for all four measurement families
  (unit streaming bench, e2e canned 100-chunk spec incl. the verbatim
  msPerChunk pacing caveat labelled ms-per-APPLIED-chunk, persistence
  benches, build/bundle numbers), plus PENDING-MANUAL rows for the three
  deferred manual metrics (10-item ingest request count + page-load request
  counts → W4; copilot long tasks → W5) and the W1-W5 append protocol.
- Build numbers (production profile): `bun run build:prod` green —
  `[bundle-budget] shared _app: 593.4 KB gzip (budget 600.0 KB, 34 files)`,
  heaviest route `/__debug__/mermaid-chat-cards` 897.2 KB gzip (budget
  900.0 KB), check ok; reproduced identically by standalone
  `bun run check:bundle-budget`. Default `bun run build` fails pre-build
  off-main without NEXT_PUBLIC_API_URL (advanced-profile validation) —
  recorded in the doc as the default invocation's status, not fixed.
- Bandit: N/A — docs + backlog markdown only, no Python touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All three stages landed on `codex/webui-perf-w0-baseline`: commit `66c8a21c80`
(streaming-render vitest bench), `e645ed7a23` (canned 100-chunk Playwright e2e
perf spec), `736b1a32a6` (persistence micro-benches), and `2f2e3b6f3a`
(freeze of `Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md`, the program's
authoritative baseline). Headline baseline numbers now frozen in that doc:
500 streamed chunks → 502 store notifications / 501 message-array updates
(Family A); one trivial workspace `set()` = 31 JSON.stringify calls /
~2.45 MB re-serialized (Family C1); request-history sidecar write
~2.6 MB per request against a ~4.93 MB final history (Family C2); bundle
budgets effectively saturated at 98.9% (shared `_app`, 6.6 KB headroom) and
99.7% (heaviest route, 2.8 KB headroom) (Family D). Verification: three
in-task reviews (one per stage) plus the final whole-branch review all
approved with zero fix rounds inside the tasks; this final-review fix wave is
doc/process edits only. Known follow-ups: the pre-existing
`workspace.split-storage.test.ts` failure is recorded (not introduced by this
batch); the shared mock-server helper should be extracted from the e2e spec
before W1 reuses it; PENDING-MANUAL rows M1–M3 remain open by design, owned by
W4 (M1/M2) and W5 (M3).
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
