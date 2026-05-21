---
id: TASK-442
title: Implement Watchlists demo rescue slice
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-05-20 22:51
labels:
- watchlists
- demo-readiness
- implementation
dependencies: []
priority: high
modified_files:
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
- apps/tldw-frontend/e2e/workflows/watchlists-demo-readiness.spec.ts
- apps/extension/tests/e2e/watchlists.spec.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR A from the Watchlists demo remediation plan: template contract hotfix, audio Scheduler submit hotfix, minimal audio/status truthfulness, source/run health truthfulness, demo runbook, and live verification gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Template payloads from quick setup and pipeline output/job creation send backend template names such as briefing_markdown while preserving UI recipe ids where appropriate.
- [x] #2 Watchlist audio briefing requests use Scheduler submit(...) and focused backend tests catch the enqueue API mismatch.
- [x] #3 Watchlists UI exposes pending/skipped/enqueue_failed audio states without implying a finished playable artifact.
- [x] #4 Active source fetch failures and source-error zero-item runs block clean System healthy state and surface warning evidence.
- [x] #5 Demo runbook and focused WebUI/extension smoke coverage document verified claims and hard stops.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 committed in 7de556bad. Task 2 replaces the audio briefing bridge with Scheduler submit(...), includes Scheduler-required user_id metadata, scopes idempotency by user/job/run, and updates the implementation plan. Verification: test_audio_briefing_workflow.py 12 passed; watchlists API generate_audio metadata subset 3 passed; test_audio_output_delivery.py 8 passed; git diff --check passed; Bandit on audio_briefing_workflow.py reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 added the Watchlists demo-readiness runbook, a focused WebUI same-origin Playwright smoke for pipeline creation and output/audio failure truthfulness, and an extension route smoke for the same Reports/Activity recovery path. During WebUI verification, caught output regeneration failures were still surfacing through the Next.js runtime overlay because OutputsTab logged the raw error object; OutputsTab now logs a safe string for that caught path while preserving the in-app error message and live region. Verification recorded: WebUI Playwright smoke 2 passed; focused OutputsTab/OutputPreviewDrawer Vitest 3 files/11 tests passed; backend watchlists demo/pipeline/audio pytest set 29 passed, 4 skipped, 1 xpassed; git diff --check passed; Bandit on touched Watchlists Python paths reported 0 results/errors. Extension Playwright could not execute because global setup hangs in WXT build; isolated bun run build:chrome:prod also timed out after 120 seconds after the same WXT duplicate-import warnings, before .output/chrome-mv3 was produced.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed.
- [x] #2 Focused frontend and backend tests recorded.
- [x] #3 Bandit run on touched Python paths or documented skip.
- [x] #4 Backlog final summary updated.
- [x] #5 Known skips or blockers documented.
<!-- DOD:END -->
