---
id: TASK-442
title: Implement Watchlists demo rescue slice
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-21 04:06
labels:
- watchlists
- demo-readiness
- implementation
dependencies: []
priority: high
modified_files:
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
- Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
- apps/tldw-frontend/e2e/workflows/watchlists-demo-readiness.spec.ts
- apps/extension/tests/e2e/watchlists.spec.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx
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

Task 3 frontend audio-status slice implemented. Added run-audio service contract, explicit output audio trigger fields, defensive output metadata audio status parsing, non-playable audio briefing status rendering in OutputPreviewDrawer, and RunDetailDrawer status lookup/404 handling. Review fix: RunDetailDrawer now scans linked output metadata before claiming no audio was requested, preserves skipped/failed linked-output fallback states, and normalizes succeeded/success audio runs as complete. Canonical audio metadata tests cover pending/skipped/enqueue_failed/failed/completed. Verification from apps/packages/ui: focused and adjacent vitest suite passed 6 files and 51 tests. git diff --check passed. Bandit skipped because this task touched no Python files.

Task 4 implemented. Backend now persists source_statuses/source_errors in run stats with safe source error text, redacts common secret formats, and preserves structured run-detail stats through the API schema. Frontend overview health counts active error:* source states, source-error zero-item runs, and enqueue_failed/skipped audio outputs as attention. Verification: pytest test_watchlists_operator_recovery.py + test_watchlists_pipeline.py + test_run_detail_filters_totals.py passed 16 passed, 1 skipped; focused Vitest passed 2 files/6 tests; git diff --check passed; Bandit on pipeline.py and endpoints/watchlists.py reported 0 results/errors.
<!-- SECTION:NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 review hardening completed after the first demo-readiness commit and post-rebase repair. The WebUI smoke now covers the first-time guided source preflight and source/monitor creation path, fails closed on unmatched Watchlists API mocks, uses normal Playwright actionability checks, scopes regenerate modal actions, and keeps diagnostics strict with only narrow expected mock-environment allowances. OutputsTab regenerate logging sanitizes server errors and redacts key/token-looking values instead of stringifying arbitrary error objects. Post-rebase fixes restored persistent in-modal pipeline output-create errors, aligned the WebUI/extension smokes with the selected-Watchlist collection model and current five-step pipeline wizard, and made markdown report previews show truthful flat backend audio briefing metadata (`audio_briefing_status`, task id, and error) for queued/failed/skipped states. PR #1906 review fix added string-error handling for pipeline setup error extraction. Runbook safe claims now reflect that the extension build and route smoke passed. Verification: WebUI Playwright smoke 3 passed; extension Watchlists Playwright smoke 14 passed; focused Watchlists UI Vitest 10 files/69 tests passed; focused backend Watchlists pytest 54 passed, 5 skipped, 1 xpassed; PR review focused `OverviewTab.quick-setup` Vitest 22 tests passed; git diff --check passed; Bandit on touched Watchlists Python paths reported 0 results/errors.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed.
- [x] #2 Focused frontend and backend tests recorded.
- [x] #3 Bandit run on touched Python paths or documented skip.
- [x] #4 Backlog final summary updated.
- [x] #5 Known skips, warnings, or blockers documented.
<!-- DOD:END -->
