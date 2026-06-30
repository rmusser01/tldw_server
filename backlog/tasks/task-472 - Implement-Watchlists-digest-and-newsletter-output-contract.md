---
id: TASK-472
title: Implement Watchlists digest and newsletter output contract
status: Done
labels:
- watchlists
- webui
- ux
- pr-b
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/1921
modified_files:
- Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-wizard-state.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/job-summaries.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.scope-filter-summary.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.regenerate-modal.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 7 from the Watchlists demo remediation implementation plan: make scheduled digest/newsletter output intent explicit, keep manual/test output separate from scheduled auto-output, and ensure Jobs/Reports UI describes delivery/audio status truthfully inside /watchlists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pipeline job payloads only enable scheduled auto-output when explicitly requested and preserve backend template naming.
- [x] #2 Job form summaries distinguish scheduled reports, manual/test reports, delivery targets, and audio briefing request state.
- [x] #3 Reports/Outputs surface digest/newsletter artifacts from actual output records and metadata without claiming delivery or audio success prematurely.
- [x] #4 Focused frontend/backend Watchlists output-contract tests pass, with verification recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 7 implementation completed locally: explicit scheduled output contract, monitor form scheduled reports switch, expanded job output linkage summary, Reports metadata regression coverage, and plan status update. Next: final static checks, Bandit skip note because no Python source changed, commit, and PR preparation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification so far: focused vitest suite 49 passed; watchlists static guard 3 passed; backend pytest for test_job_output_prefs_roundtrip.py and test_newsletter_briefing_gaps.py 47 passed; bun run test:watchlists:a11y 90 passed; git diff --check passed. No Python source was changed, so Bandit touched-scope scan is not applicable for this slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Watchlists digest/newsletter output contract slice: scheduled auto-output is now an explicit user choice, pipeline wizard drafts preserve that explicit intent, monitor summaries distinguish scheduled reports from manual/test reports, output linkage includes delivery/audio state, and Reports has regression coverage for metadata-driven delivery visibility. Verified after rebasing onto origin/dev with focused Watchlists frontend tests (49 passed), Watchlists static guard (3 passed), backend Watchlists output/newsletter tests (47 passed), the broader Watchlists a11y suite (90 passed), and git diff --check. Bandit is not applicable because this slice changed frontend/planning files and tests only.
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
