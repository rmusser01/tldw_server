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
