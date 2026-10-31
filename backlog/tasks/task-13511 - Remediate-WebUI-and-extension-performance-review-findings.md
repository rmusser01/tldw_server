id: TASK-13511
title: Remediate WebUI and extension performance review findings
status: Done
labels:
- frontend
- performance
- architecture
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements the tractable fixes from the 2026-10-06 performance/architecture review of `apps/` (WebUI + WXT extension + `packages/ui`). Branch `codex/frontend-perf-arch-remediation-20261006` off dev; PR against dev. Plan: `Docs/Plans/2026-10-06-frontend-perf-arch-remediation-implementation-plan.md`.

In scope: extension content-script weight (all_frames, chunk split, parser lazy-import), MV3 worker cold-start fan-out, sidepanel streaming throttle, ingest N+1 polling, web load waterfall (auth/health gate), provider/model refetch caching, chat runtime (store selectors, PlaygroundMessage memo, web chat virtualization, KnowledgeQA throttle, debug-clone gating, Markdown hoists, debounce), dead-file cleanup, doc fixes. The large architecture refactors (API client unification, codegen sharing, package split, vitest consolidation, dependency major alignment, auth/primitives dedupe, store splits, font/locale asset work) are filed as follow-up tasks, not silently dropped.

NOTE: this record was created by manually writing the task file because `backlog task create` crashes with "Maximum call stack size exceeded" on all inputs (CLI v1.44.0, retried 3x incl. enlarged stack; list/search/show still work). Per AGENTS.md exception path; user explicitly instructed the work to proceed on 2026-10-06.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extension: copilot content script no longer injected into all frames and no longer ships TldwApiClient eagerly in the every-page bundle
- [x] #2 cheerio/Readability/Turndown absent from every-page content scripts and the eager sidepanel chunk
- [x] #3 Background cold start uses cached capabilities; no forced refresh_openrouter; model-warm alarm relaxed; ingest polling uses batch endpoint
- [x] #4 Sidepanel streaming flushes throttled at 80ms like the web pipelines
- [x] #5 Web: no serial auth->health render gate on reload; no forceRefresh model fetch per chat mount; provider status cached
- [x] #6 Chat streaming re-renders bounded to visible messages (selectors + memoized rows + virtualized web chat)
- [x] #7 KnowledgeQA deltas throttled; chat debug payload clone gated
- [x] #8 Dead files removed; .gitignore/doc drift fixed
- [x] #9 Touched test suites pass; extension builds; bundle budget check passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Staged execution per Docs/Plans/2026-10-06-frontend-perf-arch-remediation-implementation-plan.md (all stages Complete). Branch codex/frontend-perf-arch-remediation-20261006, five commits, PR against dev.

Measured results (chrome build, vs 2026-10-06 baseline):
- content-scripts/copilot-popup.js: 560,769B -> 4,498B (stub + web-accessible copilot-popup-main.js loaded on first popup-open)
- content-scripts/web-clipper.js: 337,615B -> 8,247B (parser-main.js WAR chunk on first capture)
- manifest: no all_frames on http(s) content scripts; WAR resources registered
- sidepanel/options eager closures: cheerio/Readability/Turndown removed (2.86MB remaining eager closure is the deferred A1-class refactor, tracked in TASK-13526)
- Model warm alarm 60min -> 6h; OpenAPI drift check 15min -> 24h; capabilities cold start cached; ingest polling N+1 -> one batched request/cycle; getProvidersStatus 60s TTL cache

Verification: ~500 packages/ui tests green across scheduler/useMessage/useMessageOption/Message/PlaygroundChat/KnowledgeQA/Markdown/Review/chat-debug suites + playground composer/device/a11y groups; 86 web tests green (readiness gate); extension bun test unit = zero new failures vs dev (9 failures pre-existing, verified against clean dev checkout); chrome build green; tldw-frontend tsc --noEmit clean. Pre-existing on dev and NOT introduced here: Message.dynamic-ui-surface guard test (1 assertion, stale after VirtualChatTimeline refactor) and the 9 extension unit failures above. Bandit: N/A - no Python files touched.

Findings that turned out stale against current dev (documented, no change needed): web chat virtualization already exists (VirtualChatTimeline); getPreviousUserMessage map + KnowledgeQA throttle were partially in flight on the reviewed branch.

Note: implemented across parallel subagents + direct edits after several subagents were killed by harness inactivity limits; all diffs reviewed manually before commit. TASK-13526 tracks the deferred architecture refactors (API client unification, codegen sharing, dependency alignment, vitest consolidation, package split, auth/primitives dedupe, store splits, font/locale assets, ChatPane virtualization, composer leaf extraction, voice-assistant-sdk decision).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the tractable performance fixes from the 2026-10-06 WebUI/extension review: extension per-page JS cost cut ~99% (4.5KB+8.2KB top-frame-only stubs vs 560KB+337KB in every frame), MV3 worker cold-start fan-out removed, ingest N+1 polling batched, web auth/health gate parallelized with non-blocking reconnect banner, chat streaming re-render cost bounded to visible rows (selectors + memoized rows + shared 80ms scheduler), KnowledgeQA throttled, per-request debug deep-clones made lazy, keyword searches debounced, dead files removed. Large architectural refactors deferred to TASK-13526. PR: https://github.com/rmusser01/tldw_server/pull/3210 (branch codex/frontend-perf-arch-remediation-20261006, base dev).
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
