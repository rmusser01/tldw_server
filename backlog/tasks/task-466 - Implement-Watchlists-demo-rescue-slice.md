---
id: TASK-466
title: Implement Watchlists demo rescue slice
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-20 23:22
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
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx
- apps/packages/ui/src/services/watchlists-overview.ts
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py
- tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py
- tldw_Server_API/app/core/Watchlists/pipeline.py
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
Clean PR branch replayed the Watchlists demo-rescue commits onto current dev, renumbered stale Backlog task IDs to TASK-464/TASK-465/TASK-466 to avoid collisions, and retained current dev's newer Watchlists route/audio state. Final review fixes added root Watchlist container and scoped alert stubs to browser smoke coverage, kept pipeline wizard errors in context when digest output creation fails, and surfaced report-level audio briefing status on markdown report previews.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the scoped Watchlists demo-readiness rescue slice and review fixes on a clean dev-based branch. Verification recorded: focused Watchlists Vitest suite 10 files / 79 tests passed; focused backend Watchlists pytest set 35 passed with 5 warnings; WebUI Playwright demo-readiness smoke 3 passed; extension strict demo-readiness route smoke 1 passed after WXT Chrome build; git diff --check passed; Bandit on touched Watchlists Python paths reported 0 results/errors.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed.
- [x] #2 Focused frontend and backend tests recorded.
- [x] #3 Bandit run on touched Python paths or documented skip.
- [x] #4 Backlog final summary updated.
- [x] #5 Known skips or blockers documented.
<!-- DOD:END -->
