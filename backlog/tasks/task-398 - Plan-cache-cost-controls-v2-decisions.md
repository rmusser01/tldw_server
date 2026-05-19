---
id: TASK-398
title: Plan cache-cost controls v2 decisions
status: Done
assignee: []
created_date: '2026-05-16 01:25'
updated_date: '2026-05-16 01:28'
labels:
  - chat
  - cost-control
  - llm-cache
  - world-books
  - planning
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md
  - Docs/superpowers/plans/2026-05-16-chat-cache-cost-v2-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Capture the approved v2 direction for chat/world-book cache cost controls and turn the remaining open design questions into a reviewable implementation plan. The v2 scope follows TASK-377 and TASK-378: debug-gated raw prompt-envelope persistence, user-pinned world-book stable-prefix entries, cache-aware reporting staying in existing usage analytics, and local prefill-latency reporting with explicit source/confidence labels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approved v2 decisions are recorded in the cache-cost design spec without reopening completed v1 scope.
- [x] #2 A new implementation plan decomposes v2 into reviewable slices with tests and safety constraints.
- [x] #3 The plan distinguishes observed local prefill timing from app-observed TTFT and token-count estimates.
- [x] #4 Verification is recorded for docs/backlog changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Resolve the remaining open questions in the approved cache-cost design spec using the user-approved v2 decisions.
2. Add a dedicated v2 implementation plan for debug-gated prompt envelope persistence, stable-prefix world-book pinning, existing usage analytics reporting, and source-labeled local prefill latency.
3. Verify docs/backlog changes with git diff checks and targeted content searches.
4. Complete TASK-398 with verification notes and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created v2 implementation plan at Docs/superpowers/plans/2026-05-16-chat-cache-cost-v2-implementation-plan.md and updated the approved design spec with resolved v2 direction.

Verification: git diff --check passed; targeted rg checks confirmed Resolved V2 Direction, debug-gated/stable-prefix decisions, UsageAnalyticsPage/WorldBookEntryManager implementation paths, and local latency source labels; ASCII scan with rg -nP "[^\x00-\x7F]" returned no matches. Bandit is not applicable because this task changed only docs and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the v2 cache-cost design decisions in the approved spec and added Docs/superpowers/plans/2026-05-16-chat-cache-cost-v2-implementation-plan.md. The plan decomposes debug-gated prompt envelope persistence, stable-prefix world-book pinning, existing usage analytics reporting, and source-labeled local prefill latency into reviewable tasks with tests, safety constraints, and verification commands.
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
