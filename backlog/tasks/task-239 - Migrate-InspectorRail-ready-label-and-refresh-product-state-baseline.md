---
id: TASK-239
title: Migrate InspectorRail ready label and refresh product-state baseline
status: Done
assignee: []
created_date: '2026-05-10 19:06'
updated_date: '2026-05-10 19:14'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1536'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the frontend design-system product-state cleanup by replacing InspectorRail's hardcoded idle Ready runtime label with the canonical ready state registry value while preserving current streaming and unavailable behavior. The latest dev baseline also had bounded product-state guard drift, so this task refreshes those baseline records to keep the verifier green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 InspectorRail uses the design-system ready state registry label for its idle non-streaming runtime state.
- [x] #2 Focused component coverage proves the ready label is supplied through the registry rather than a hardcoded literal.
- [x] #3 The matching canonical-state-label baseline entry is removed and the design-system verifier passes.
- [x] #4 Current dev product-state baseline drift is reconciled without reintroducing InspectorRail's Ready exception.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red: InspectorRail test mocked the design-system ready label as 'Ready via registry' and failed while the component still rendered literal Ready.

Green verification: InspectorRail focused Vitest passed (4 tests); product-state guard unit test passed (52 tests); verify:design-system-state exited 0 with blocked=0 stale=0 baselineErrors=0; git diff --check exited 0. Broad bunx tsc still exits 2 with 239 baseline errors, and touched-scope filtering found no InspectorRail/baseline/touched-path matches.

Bandit skipped: touched implementation is TypeScript/TSX/JSON/Backlog only, with no Python runtime code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
InspectorRail now uses getDesignSystemState('ready').label for the idle non-streaming runtime label, with a mocked-registry regression test proving the label is not hardcoded. Removed the InspectorRail Ready baseline exception and refreshed bounded current-dev product-state baseline drift so the design-system verifier has blocked=0, stale=0, and baselineErrors=0.

PR: https://github.com/rmusser01/tldw_server/pull/1536
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
