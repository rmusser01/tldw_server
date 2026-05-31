---
id: TASK-45.44.3.2
title: Migrate RunDetailDrawer alerts and Watchlists output labels to design-system
  primitives
status: Done
labels:
- design-system
- webui
- watchlists
priority: medium
parent_task_id: TASK-45.44.3
references:
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/issues/1660
- https://github.com/rmusser01/tldw_server/pull/1855
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.source-column.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportEvidencePanel.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
- apps/packages/ui/src/assets/locale/en/watchlists.json
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Watchlists design-system product-state migration by replacing RunDetailDrawer's remaining product-state AntD Alert usage with design-system primitives and routing Watchlists output Ready/Blocked/Loading labels through the shared design-system state registry. This also resolves the current verifier drift where RunDetailDrawer alert IDs became unbaselined and stale baseline entries remain.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RunDetailDrawer linkage, recovery, common-causes, filtered-sample, stream-error, truncated-log, and load-error product-state surfaces render through design-system primitives.
- [x] #2 Watchlists output Ready/Blocked and audio-ready labels are routed through the design-system state registry.
- [x] #3 Migrated RunDetailDrawer and outputMetadata baseline exceptions are removed, with no new blocked product-state findings.
- [x] #4 Focused tests, guard tests, verifier, diff check, and TypeScript debt classification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing design-system marker coverage for RunDetailDrawer linkage, remediation, filtered-sample, stream-error, and truncated-log product-state surfaces. 2. Add a registry-label regression for Watchlists output readiness/audio labels with a mocked design-system registry. 3. Replace RunDetailDrawer product-state AntD Alert usages with design-system Alert or RecoveryCallout and route Loading through LOADING_STATE_LABEL. 4. Route outputMetadata Ready/Blocked labels through the design-system state registry and remove migrated baseline exceptions. 5. Run focused tests, guard tests, verifier, diff check, and TypeScript debt classification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Used TDD: first added failing marker assertions for RunDetailDrawer design-system Alert/RecoveryCallout usage and a mocked-registry test for outputMetadata canonical labels. Then migrated the component/helper code and removed the corresponding stale baseline entries.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated RunDetailDrawer's flagged product-state alerts to the shared design-system Alert and RecoveryCallout primitives, including linkage, recovery, common-causes, filtered-sample, stream-error, truncated-log, and load-error surfaces. Routed Watchlists output Ready/Blocked and audio-ready labels through the design-system state registry, added translation-key label maps for remaining readiness/audio labels, and made mocked registry cleanup exception-safe. Preserved warning-level recovery as the degraded state and migrated ReportEvidencePanel's evidence error, legacy, and warning callouts to design-system Alert instead of rebaselining shifted AntD debt. Removed stale/migrated baseline exceptions for RunDetailDrawer, ReportEvidencePanel, and outputMetadata. PR: https://github.com/rmusser01/tldw_server/pull/1855. Verification: focused Watchlists Vitest suite passed 38 tests; product-state guard tests passed 52 tests; bun run verify:design-system-state exited 0 with baseline exceptions reduced to 350; git diff --check passed; bunx tsc --noEmit --pretty false still exits 2 on existing unrelated UI TypeScript debt with no touched-file matches. Bandit skipped because this slice only changes frontend TypeScript/JSON and Backlog task metadata.
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
