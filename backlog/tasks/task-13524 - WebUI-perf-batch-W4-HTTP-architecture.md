---
id: TASK-13524
title: 'WebUI perf batch W4: HTTP architecture'
status: In Progress
created_date: 2026-10-07 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program. Index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md. Plan: Docs/Plans/2026-10-06-webui-perf-batch-4-http-architecture-implementation-plan.md. Scope: Ingest batching + parallel polling, polling to SSE gating, N+1 fan-outs, auth/config read caching, model-catalog fetch discipline. WP-36..55. BE-marked stages file against backend TASK-13515.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 (WP-36/37/38, extension quick-ingest batching) implemented in worktree codex/webui-perf-w4_http-arch: same-config URL/file submission grouping into single /media/ingest/jobs requests, bounded pool (4) for heterogeneous submissions and conference planning POSTs, parallel poll rounds (per-job session poll + multi-batch tracker rounds), concurrent terminal PATCH / batch-cancel / per-job DELETE fan-out. Per-item progress events preserved (order + shape pinned by test). Overlap check found batched per-batch status polling already landed (PR #3210 follow-up); only real gaps closed. Full flow map + verdicts in .superpowers/sdd/2026-10-06-webui-perf-batch-4-http-architecture-implementation-plan/task-1-report.md
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
