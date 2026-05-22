---
id: TASK-471
title: Implement Watchlists first-time cadence and review cleanup
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-21 20:18
labels:
- watchlists
- webui
- ux
- pr-b
dependencies: []
priority: high
modified_files:
- Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-wizard-state.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts
- backlog/tasks/task-471 - Implement-Watchlists-first-time-cadence-and-review-cleanup.md
references:
- https://github.com/rmusser01/tldw_server/pull/1921
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the Watchlists demo remediation plan: support variable first-time cadence choices in the guided setup flow and make the review summary accurately reflect source count, cadence, output, delivery, audio speaker count, and first-run behavior while reusing the existing schedule/backend contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Guided Watchlists setup supports manual, interval-minute, interval-hour, daily, weekdays, weekly, and advanced cron cadence drafts without introducing a new backend schedule contract.
- [x] #2 Pipeline and quick-setup payloads serialize supported cadence drafts into the existing schedule fields and preserve raw cron when selected.
- [x] #3 Pipeline wizard review summary reflects one-source setup correctly and uses the selected audio cast size instead of a fixed podcast assumption.
- [x] #4 Focused Watchlists cadence/review tests pass, with verification recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md#task-6-first-time-cadence-and-review-cleanup
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added red tests for variable cadence drafts, weekdays/advanced wizard cadence, one-source summary, audio-off summary behavior, and pure pipeline review-summary cadence labels. Initial focused runs failed as expected because current dev lacked cadence-draft serialization, the wizard did not expose weekdays/advanced cron modes, and buildPipelineReviewSummary still labeled scheduleCadence drafts from schedulePreset. Implemented cadence-draft conversion through existing cron schedule fields, added weekdays and advanced cron wizard modes, and kept wizard/contract review summaries aligned with selected source, cadence, delivery, and optional 1-4 speaker audio state. Verification from apps/packages/ui: focused Task 6 suite passed with 6 files and 35 tests; Watchlists static guard passed with 1 file and 3 tests; git diff --check passed. Bandit skipped because this slice touched TypeScript, Markdown, and Backlog task metadata only.

Reopened for PR #1921 review follow-up after Gemini, CodeRabbit, Qodo, and cubic flagged still-valid cadence issues: advanced cron validation was weaker than the existing SchedulePicker guardrails, advanced cron serialization accepted malformed values, submission/review schedule precedence could diverge, cadence time parsing was duplicated, and shared cron validation accepted out-of-range field values.

Review follow-up centralized cron format, range, and frequency validation plus cadence time parsing in JobsTab schedule utilities, reused those guards from SchedulePicker, PipelineWizard, and quick setup serialization, aligned pipeline payload and review-summary schedule precedence, and added copy-injection for pure review-summary cadence labels so UI callers can localize generated labels. Verification after review fixes: focused Watchlists suite passed with 6 files and 42 tests; Watchlists static guard passed with 1 file and 3 tests; git diff --check passed. Bandit skipped because this follow-up touched TypeScript and Backlog task metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the remaining Watchlists Task 6 cadence gaps and PR #1921 review fixes on current dev. Quick setup and pipeline drafts now serialize variable cadence through the existing cron schedule fields, advanced cron values share the SchedulePicker format/range/min-frequency validation, and malformed, out-of-range, or too-frequent cron is blocked before payload serialization. PipelineWizard exposes weekdays and advanced cron choices, distinguishes format, range, and frequency errors, and keeps review summaries aligned with selected cadence and optional 1-4 speaker audio. Pipeline payload and review summary precedence now match when both scheduleExpr and scheduleCadence are present, cadence time parsing is shared through schedule-utils, and the pure pipeline review summary accepts localization copy hooks. Verification: focused Watchlists Vitest suite passed with 6 files and 42 tests; Watchlists static guard passed with 1 file and 3 tests; git diff --check passed. Bandit skipped because this slice touched TypeScript and Backlog task metadata only.
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
