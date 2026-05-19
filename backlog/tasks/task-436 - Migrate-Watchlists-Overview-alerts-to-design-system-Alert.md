---
id: TASK-436
title: Migrate Watchlists Overview alerts to design-system Alert
status: Done
labels:
- design-system
- watchlists
- ui
priority: medium
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.alerts-health.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx
- apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx
- apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx
- apps/packages/ui/src/utils/watchlists-onboarding-telemetry.ts
- apps/packages/ui/src/utils/__tests__/watchlists-onboarding-telemetry.test.ts
- apps/packages/ui/scripts/design-system-product-state-baseline.json
references:
- https://github.com/rmusser01/tldw_server/pull/1863
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Guard-backed design-system slice: replace remaining AntD product-state Alert surfaces in Watchlists OverviewTab and PipelineWizard with canonical design-system Alert primitives, add focused coverage, remove stale baseline entries, and initialize the source_create telemetry failure bucket exposed by the touched Overview pipeline path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation completed in branch codex/design-system-next-slice-10 and opened as draft PR #1863. After rebasing onto current origin/dev, the design-system guard exposed new unbaselined Setup Readiness Alert/Empty findings from upstream; migrated that small ReadinessSetupScreen surface in the same PR instead of adding new baseline debt. Verification: focused Vitest passed for OverviewTab alerts, PipelineWizard, ReadinessSetupScreen, and watchlists onboarding telemetry (23 tests); design-system product-state guard passed with 346 allowed legacy exceptions and no blocked/stale findings for this slice; git diff --check passed; full UI tsc remains red from inherited package-wide TypeScript debt, and a filtered tsc pass confirmed no diagnostics for touched files. Bandit skipped because this is TypeScript/JSON/Backlog-only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Watchlists OverviewTab, PipelineWizard, and rebased Setup Readiness product-state callouts from AntD Alert/Empty to canonical design-system Alert/EmptyState primitives. Added regression coverage for health/setup/load failure/validation/preview/setup-readiness alert wrappers and profile-empty state wrappers, removed stale Overview Alert baseline entries, and initialized the source_create pipeline failure telemetry bucket so source-creation failures are counted instead of producing NaN.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
