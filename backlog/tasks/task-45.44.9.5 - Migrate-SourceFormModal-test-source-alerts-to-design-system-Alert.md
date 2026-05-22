---
id: TASK-45.44.9.5
title: Migrate SourceFormModal test-source alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
- watchlists
priority: medium
parent_task_id: TASK-45.44.9
references:
- https://github.com/rmusser01/tldw_server/issues/1666
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1936
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Watchlists SourceFormModal test-source diagnostics and error alert UI off AntD Alert and onto the canonical design-system Alert, clearing the current guard blocked/stale mismatch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SourceFormModal test-source diagnostics and error alerts render the design-system Alert primitive instead of AntD Alert.
- [x] #2 Retry behavior and remediation copy remain covered by focused tests.
- [x] #3 Design-system product-state verifier passes with stale SourceFormModal Alert baseline entries removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing SourceFormModal test-source assertions requiring the diagnostics and error alerts to render with the design-system Alert marker.
2. Replace the relevant SourceFormModal AntD Alert usages with the canonical design-system Alert primitive while preserving text, variant, and retry behavior.
3. Remove stale SourceFormModal Alert baseline entries and run focused test plus product-state verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. Added failing assertions for SourceFormModal summary, diagnostics, and error callouts requiring the design-system Alert marker. Replaced the AntD Alert import/usages in the test-source block with the canonical design-system Alert primitive while preserving summary, diagnostics, remediation copy, and retry behavior. Removed the two stale SourceFormModal Alert baseline entries.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated SourceFormModal test-source summary, diagnostics, and error callouts from AntD Alert to the design-system Alert primitive. Added focused test-source coverage for the design-system Alert marker while preserving retry behavior and remediation guidance. Removed stale SourceFormModal Alert exceptions from the product-state baseline. Verification: bunx vitest run src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx; bun run verify:design-system-state; git diff --check. Bandit skipped: frontend-only TSX/test/JSON change.
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
