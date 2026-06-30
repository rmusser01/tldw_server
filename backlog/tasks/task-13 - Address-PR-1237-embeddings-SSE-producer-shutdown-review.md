---
id: TASK-13
title: Address PR 1237 embeddings SSE producer shutdown review
status: Done
assignee: []
created_date: '2026-05-03 20:16'
updated_date: '2026-05-03 20:19'
labels:
  - pr-review
  - openapi
  - embeddings
  - phase4
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1237'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the non-inline CodeRabbit PR #1237 review-body finding that the unified embeddings orchestrator SSE normal shutdown path can await an infinite producer without cancelling it. Keep the change narrow to producer cancellation/lifecycle semantics and focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unified embeddings orchestrator SSE normal generator close cancels the producer before awaiting it.
- [x] #2 A focused regression fails before the fix and passes after it.
- [x] #3 Focused tests, Bandit touched-source scope, and git diff --check are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression around STREAMS_UNIFIED orchestrator_events body iterator close that proves generator shutdown returns promptly and cancels the producer. 2. Patch only the normal shutdown path to cancel the producer before gather, matching the existing cancellation path. 3. Run focused embeddings SSE tests, Bandit on touched source/tests, and git diff --check. 4. Record verification and push the PR review-fix commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: test_embeddings_orchestrator_events_unified_normal_close_cancels_producer timed out before the fix while awaiting the still-running producer task. GREEN: focused normal_close regression passed after the production change. Adjacent verification: test_orchestrator_sse_unified_flag.py plus test_orchestrator_sse.py passed with existing Redis-fixture skips (1 passed, 5 skipped); OpenAPI embeddings orchestrator contract selection passed (1 passed). Bandit source scope reported 0 findings in /tmp/bandit_pr1237_embeddings_sse_producer.json. git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the PR #1237 non-inline embeddings SSE review finding by cancelling the long-running unified orchestrator producer before awaiting it on normal stream completion. Added a focused regression proving the body iterator completes promptly and the producer task is cancelled.
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
