---
id: TASK-486
title: Implement Watchlists status UX and setup hardening
status: Done
labels:
- watchlists
- ux
- status
- frontend
- backend
priority: High
documentation:
- Docs/superpowers/specs/2026-05-22-watchlists-status-ux-setup-hardening-design.md
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts
- apps/packages/ui/src/components/Option/Watchlists/shared/runStatus.ts
- apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/runStatus.test.ts
- apps/packages/ui/src/components/Option/Watchlists/shared/StatusTag.tsx
- apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/StatusTag.accessibility.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/polling-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/polling-utils.test.ts
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/run-notifications.ts
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/run-notifications.test.ts
- apps/packages/ui/src/services/watchlists-overview.ts
- apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts
- apps/packages/ui/src/types/watchlists.ts
- apps/packages/ui/src/assets/locale/en/watchlists.json
- apps/packages/ui/src/public/_locales/en/watchlists.json
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Watchlists follow-up focused on /watchlists: beginner variable cadence controls, frontend run status normalization, audio health attention states, and no-task audio status visibility for requested audio briefings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quick setup supports manual, interval, daily, weekdays, weekly, and advanced cron cadence without leaving /watchlists.
- [x] #2 Frontend treats succeeded and completed run statuses consistently in previews, badges, terminal polling, and notifications.
- [x] #3 Overview health flags actionable audio states including queue_unavailable and configuration_required without treating disabled audio as a failure.
- [x] #4 Run audio endpoint and Run Detail show requested audio statuses without task ids while preserving 404 for runs with no audio request.
- [x] #5 Focused frontend/backend tests and Bandit for touched Python scope are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-watchlists-status-ux-setup-hardening-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Quick Setup now stores a `scheduleCadence` draft and renders beginner controls for manual, interval, daily, weekdays, weekly, and advanced cron cadence.
- Added shared frontend run-status normalization so `succeeded` behaves as completed in badges, preview lookup, terminal stream handling, polling helpers, and notifications.
- Overview health now treats `queue_unavailable` and `configuration_required` as actionable audio output attention states while leaving `disabled` and `skipped_no_items` informational.
- `GET /api/v1/watchlists/runs/{run_id}/audio` returns metadata-only requested audio status when no task was created, and still returns `404 no_audio_briefing_for_run` when no audio request/status evidence exists.
- Verification recorded: focused Vitest Watchlists status/setup group passed (8 files, 72 tests); RunsTab accessibility live-region test passed (2 tests); full `test_audio_output_delivery.py` passed (26 tests); locale JSON parse passed; Bandit on `tldw_Server_API/app/api/v1/endpoints/watchlists.py` passed with 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Watchlists status UX and setup hardening completed for the approved scope. The /watchlists beginner setup path supports variable cadence, successful run status handling is consistent for succeeded/completed, actionable audio setup/queue states surface in health, and requested audio states remain visible even without an enqueued task.
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
