---
id: TASK-13522
title: 'WebUI perf batch W2: persistence write-path'
status: In Progress
created_date: 2026-10-07 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program. Index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md. Plan: Docs/Plans/2026-10-06-webui-perf-batch-2-persistence-write-path-implementation-plan.md. Scope: Request-history sidecar, workspace persist chain debounce+diff, ModelDb namespacing, legacy chrome.storage quarantine, session-state serialization, settings registry. WP-15..22.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 (WP-15 request-history sidecar) started: in-memory array + idle/trailing-debounce(500ms) flush + visibilitychange/beforeunload flush + redact/truncate at insert per Docs/Plans/2026-10-06-webui-perf-batch-2-persistence-write-path-implementation-plan.md Stage 1. ADR assessment: no new ADR needed - mechanical perf fix inside one lib module following the approved batch plan (no new durable architecture rule); governing plan referenced above.
Stage 1 (WP-15 request-history sidecar) COMPLETE (partial prior attempt audited, corrected, and finished): lib/history.ts buffers entries in a module array; redact + 2KB responseBody truncation applied exactly once at insert (historyInternals seam for the spy); flush = 500ms trailing debounce + requestIdleCallback (sync fallback under jsdom) + immediate flush on visibilitychange->hidden and beforeunload; MAX=200 newest-first unchanged; legacy un-truncated entries hydrate readable (migration-on-read kept, never re-truncated); new public flushRequestHistory() lands pending writes for readers/tests. W0 bench kept green and retargeted (100KB blob requestBody->responseBody, drives now-async flush, resets module buffer between runs, exactly-1-setItem assertion): post-fix shape 1 setItem/run, 115,651 bytes attempted=landed, 0 quota errors, 1.0x amplification (pre-fix: 50 calls, ~130.8MB attempted, 2 quota errors). Existing api-client.fetch.test.ts (8 red under async flush) fixed via helper flush only - assertions untouched. Verified: lib/__tests__ 84/84 green, scoped eslint clean, tsc --noEmit clean, Bandit n/a for TS-only scope (0 findings). Baseline-doc delta append belongs to the batch final task. Stages 2-6 remain.
Stage 2 sub-step 3 (artifact offload dedup, WP-16) committed as 55b070bb90, completing over a partial left by a prior session: offload stamps (WeakMap per adapter, record key -> content hash + updatedAt) skip re-puts of unchanged payloads and reuse the stored pointer timestamp for snapshot byte-stability; remaining puts parallelized via Promise.all; stamps invalidated on record deletion paths and removeItem. New tests workspace.persist-artifact-dedup.test.ts (3/3). Store __tests__ 368 pass / 1 documented pre-existing split-storage failure; tsc --noEmit 400 pre-existing errors, none new; W0 bench green untouched, numbers unchanged at this sub-step (jsdom lacks IndexedDB).
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
