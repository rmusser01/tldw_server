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
Stage 2 (WP-39/40/41/42, polling to SSE/visibility gating) implemented in worktree codex/webui-perf-w4_http-arch: buddy host poll pins to 60s while its panel is closed and adds focus refresh (BuddyInteraction's 4s loop joins the shared XP-16 schedule); research run list refetchInterval becomes activity-gated with SSE terminal-event invalidation; notification unread poll runs only while the stream/bootstrap is degraded and the stored-auth read moves out of render; the workspace's two 5s pollers consolidate into one scheduler gated on processing sources + page visibility. Overlap verdicts and full verification in .superpowers/sdd/2026-10-06-webui-perf-batch-4-http-architecture-implementation-plan/task-2-report.md
Stage 3 (WP-43, media search correctness + cost) implemented in worktree codex/webui-perf-w4_http-arch: media and notes searches now dispatch together via Promise.all inside runSearch (per-side result arrays keep media-before-notes row order and per-side error handling); the mount-time type-discovery effect is deleted (it fetched /api/v1/media/ pages 1-3 at per_page=50 and re-called refetch, so mount ran 2 searches + up to 3 discovery pages); media-type facets derive from the first search response via the existing per-row derivation, with an empty first response leaving the facet list empty (same as HEAD's empty discovery result); the W3 Stage 5 Set snapshot (availableMediaTypesSet) is applied here because that sweep (841b440710) is not in this branch's lineage. Overlap verdicts: Promise.all GAP, duplicate refetch GAP, discovery pages GAP, Set conversion GAP-in-this-branch (landed only on another branch). Existing tests that pinned the old mount fan-out were re-anchored (outage pending>=2 waits, search-experience settle proxy, stage12 sampling tests); one false-positive permalink test (its page=2 wait was satisfied by the discovery fan-out, not the mocked-search pagination) was replaced by a real hook-level page-2 pin. Full verification in .superpowers/sdd/2026-10-06-webui-perf-batch-4-http-architecture-implementation-plan/task-3-report.md
Stage 4 (WP-44/45/46/47/48/51, N+1 fan-outs) implemented in worktree codex/webui-perf-w4-http-arch: ACP run-history modal fetches task details lazily on row expand (first page of tasks only, merged detail-over-list state, per-row retry) instead of the eager 2+N+M fan-out; ChatPane ask-with-sources and StudioPane artifact generation share a new source-content-text resolver backed by an in-memory (never persisted) workspace-store cache keyed by mediaId+version, fetching only misses (cache is an optional store capability so partial store mocks degrade to fetch instead of breaking resolution); flashcard due-count queries cache with staleTime 60s; KnowledgeQA retrySync defers per-message rag-context writes off the parent-chained message sequence and settles them via Promise.all with failure semantics preserved; character-list filtering verified as already routed server-side via listCharactersPage (default-on rollout predating the plan; listAllCharacters fetch-all survives only as explicit opt-out and route-conflict fallback) with a request-shape pin added. WorldBooks stats gating verified already landed at base (openGlobalStats + stats tab) and pinned by a new test. Overlap verdicts and verification in .superpowers/sdd/2026-10-06-webui-perf-batch-4-http-architecture-implementation-plan/task-4-report.md
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
