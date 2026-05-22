---
id: TASK-473
title: Address PR 1925 Watchlists digest output review comments
status: Done
labels:
- watchlists
- webui
- review-fix
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/1925
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-wizard-state.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/job-summaries.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/job-summaries.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review comments on PR #1925: add explicit scheduled-output intent to PipelineWizard, prevent stale auto_output.enabled from surviving monitor edits, and require enabled === true when summarizing delivery channels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pipeline Wizard exposes and persists an explicit Scheduled reports choice instead of deriving auto-output from schedule presence.
- [x] #2 Job form output prefs clear auto_output.enabled when scheduled reports are toggled off.
- [x] #3 Job summaries only report email/chatbook delivery when the delivery record explicitly has enabled === true.
- [x] #4 Focused Watchlists tests are updated and pass after the review fixes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify current PipelineWizard state model and render controls. 2. Add explicit createScheduledOutput wizard state and UI switch. 3. Fix stale auto_output.enabled and strict delivery summary checks. 4. Update focused tests for red/green review cases. 5. Run focused Watchlists verification, update task, commit, push, and reply/resolve PR comments.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1925 review comments: Pipeline Wizard now has an explicit Scheduled reports switch/default false state, scheduled wizard monitors remain manual/test-only until opted in, JobForm edit payloads send auto_output.enabled=false when disabling previously scheduled reports, and job summaries require delivery.enabled === true before reporting email/chatbook delivery. Also stabilized the existing missing-template pipeline error test by giving it the same 20s timeout as adjacent long-running pipeline tests after the rebased a11y suite exposed a default-timeout flake. Verification on the rebased branch: focused Watchlists suite 64 passed, static guard 3 passed, bun run test:watchlists:a11y 91 passed, backend Watchlists output/newsletter pytest 47 passed, git diff --check passed. Bandit is not applicable because no Python source changed.
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
