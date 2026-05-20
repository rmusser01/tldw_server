---
id: TASK-442
title: Implement Watchlists demo rescue slice
status: In Progress
labels:
- watchlists
- demo-readiness
- implementation
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR A from the Watchlists demo remediation plan: template contract hotfix, audio Scheduler submit hotfix, minimal audio/status truthfulness, source/run health truthfulness, demo runbook, and live verification gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Template payloads from quick setup and pipeline output/job creation send backend template names such as briefing_markdown while preserving UI recipe ids where appropriate.
- [x] #2 Watchlist audio briefing requests use Scheduler submit(...) and focused backend tests catch the enqueue API mismatch.
- [ ] #3 Watchlists UI exposes pending/skipped/enqueue_failed audio states without implying a finished playable artifact.
- [ ] #4 Active source fetch failures and source-error zero-item runs block clean System healthy state and surface warning evidence.
- [ ] #5 Demo runbook and focused WebUI/extension smoke coverage document verified claims and hard stops.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 committed in 7de556bad. Task 2 replaces the audio briefing bridge with Scheduler submit(...), includes Scheduler-required user_id metadata, scopes idempotency by user/job/run, and updates the implementation plan. Verification: test_audio_briefing_workflow.py 12 passed; watchlists API generate_audio metadata subset 3 passed; test_audio_output_delivery.py 8 passed; git diff --check passed; Bandit on audio_briefing_workflow.py reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed.
- [ ] #2 Focused frontend and backend tests recorded.
- [ ] #3 Bandit run on touched Python paths or documented skip.
- [ ] #4 Backlog final summary updated.
- [ ] #5 Known skips or blockers documented.
<!-- DOD:END -->
