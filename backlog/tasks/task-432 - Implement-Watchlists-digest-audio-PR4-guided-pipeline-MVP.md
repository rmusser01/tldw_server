---
id: TASK-432
title: Implement Watchlists digest audio PR4 guided pipeline MVP
status: Done
labels:
- watchlists
- frontend
- implementation
- guided-pipeline
priority: High
references:
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
- Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md
modified_files:
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-wizard-state.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 8 / PR4 from the Watchlists digest/audio plan: guided pipeline MVP inside /watchlists. Scope includes pure wizard state helpers, an additive PipelineWizard component, source/monitor/digest/optional-audio/review steps, and focused frontend tests while preserving existing full controls and OSINT/CTI workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pure wizard state helpers validate source, monitor, digest, delivery, and optional 0-4 speaker audio configuration.
- [x] #2 The /watchlists overview exposes a guided pipeline entry point that can create a new source or use existing sources, then create monitor/output settings without leaving /watchlists.
- [x] #3 Optional audio supports no audio and 1-4 speaker cast configuration while preserving existing voice_map/audio_cast contracts.
- [x] #4 Existing full controls, quick setup, and advanced tabs remain reachable and existing pipeline tests continue to pass.
- [x] #5 Focused Vitest tests and diff checks pass; Bandit is documented as not applicable unless backend Python is touched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline existing pipeline tests passed: OverviewTab.quick-setup.test.tsx + pipeline-contract.test.ts, 24 tests. Red PR4 tests added for missing PipelineWizard and pipeline-wizard-state modules; focused run failed as expected on unresolved imports while existing pipeline-contract tests still passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the PR4 guided briefing pipeline MVP inside /watchlists. Added pure wizard state helpers, an additive five-step PipelineWizard (source, monitor, digest, optional audio, review), overview integration for existing or newly-created sources, variable cadence, optional 0-4 speaker audio contracts, preview/test-generation actions, and focused tests. Verification: focused Vitest suite passed (36 tests), watchlists static typecheck passed, and git diff --check passed. Bandit not run because this PR touches frontend/docs/backlog files only.
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
