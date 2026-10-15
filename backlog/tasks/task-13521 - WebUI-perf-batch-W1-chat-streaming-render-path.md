---
id: TASK-13521
title: 'WebUI perf batch W1: chat streaming render path'
status: In Progress
created_date: 2026-10-07 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program. Index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md. Plan: Docs/Plans/2026-10-06-webui-perf-batch-1-chat-streaming-render-implementation-plan.md. Scope: Throttle sidepanel streaming updates, memoize PlaygroundMessage, virtualize PlaygroundChat and ChatPane transcripts, per-row O(n) scans and selector/context hygiene. WP-01..14.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 0 (pre-Stage-1 prep): extracted shared sidepanel mock-server helper apps/extension/tests/e2e/utils/sidepanel-mock-server.ts (startSidepanelMockServer/stopSidepanelMockServer); refactored performance-streaming.spec.ts canned suite and sidepanel-chat-smoke.spec.ts to consume it, behavior-preserving (same routes/headers/lifecycle; chat-completions payload differences kept per-spec via handleChatCompletions opt). Verified: scoped perf spec 1 passed (canned 100-chunk) + 5 live tests skip env-gated unchanged; scoped smoke spec 2 passed + 1 pre-existing failure (handoff title assertion fails identically on pristine tree, stash-verified). No product code.
Stage 1 (WP-01) audit+completion of a partial prior attempt: extracted the 80ms trailing-flush streaming throttle into shared helper apps/packages/ui/src/hooks/chat/streamingThrottle.ts (schedule/flushNow/cancel/disarm + replaceStreamingMessage last-index slice update) and adopted it in hooks/useMessage.tsx (4 loops), hooks/chat/useChatActions.ts and hooks/chat/useCharacterChatMode.ts, replacing the per-flush prev.map full-transcript clone. New tests: streamingThrottle.test.ts (9), useMessage.streaming-throttle.test.ts (4), useCharacterChatMode.streaming-throttle.test.ts (3) - 16 green. W0 bench untouched and green: transcriptArrayMapClones 2->1 (remaining 1 = once-per-stream final commit; per-token/per-flush clones 0), store notifications 3 (frozen W0 baseline was 502; HEAD already carried the TASK-13511 80ms throttle, so this stage's delta is the narrowed flush + helper unification). Scoped suites: useChatActions 328 passed, useCharacterChatMode+useMessage-related 150 passed. Scoped tsc --noEmit: error set byte-identical to HEAD (400 pre-existing, none in touched files; needs NODE_OPTIONS=--max-old-space-size=8192). Bandit: n/a, TypeScript-only change. Report: .superpowers/sdd/2026-10-06-webui-perf-batch-1-chat-streaming-render-implementation-plan/task-1-report.md (untracked program artifact).
Stage 2 (WP-02, WP-03 partial, WP-09) audit+completion of the TASK-13511 partial attempt: React.memo+comparator, per-row stable ref-dispatched callbacks, EMPTY_IMAGES and previousUserMessageByIndex were already in HEAD; this stage landed the missing pieces. Narrowed isStreaming to last-row-only in both parents (PlaygroundChat.tsx block.index formulation, Sidepanel/Chat/body.tsx index formulation - ChatPane verbatim copy); memoized highlightText inside the row via useMemo([message, searchQuery]); fixed a memo-defeating on-mount clear of the per-row handler caches in both parents (rowHandlerCache/researchActionsCacheRef now clear only on an actual conversationInstanceId change, skip initial mount). Tests written first (red run: 5 failing / 3 green locks): Message.memoization.test.tsx (50-row token-flush render count with last-row-still-renders guard via MessageActionsBar render spy; highlightText memo spy incl. recompute-on-query-change control; source guards for memo export, EMPTY_IMAGES and narrowed isStreaming in both parents) and Sidepanel body.message-memo.test.tsx (memoized row spy through real SidePanelBody: stream-start + token flush re-render only the last row of 50; only last row receives isStreaming=true; flag clears on stream end). Scoped suites green: 32 files / 433 tests (all PlaygroundMessage-referencing suites incl. cockpit-controls + coordinator, Sidepanel body, ChatPane stages). Scoped tsc --noEmit: 400 pre-existing errors, 0 in touched files (NODE_OPTIONS=--max-old-space-size=8192). Lint: tldw-frontend eslint 0 errors (packages/ui outside its base path; no packages/ui lint script). Bandit: n/a, TypeScript-only. Report: .superpowers/sdd/2026-10-06-webui-perf-batch-1-chat-streaming-render-implementation-plan/task-2-report.md.
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
