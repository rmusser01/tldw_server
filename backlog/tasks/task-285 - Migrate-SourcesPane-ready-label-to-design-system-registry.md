---
id: TASK-285
title: Migrate SourcesPane ready label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 03:37'
labels:
  - design-system
  - frontend
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the hardcoded ready source-status label in WorkspacePlayground SourcesPane with the design-system state registry fallback while preserving status guardrail behavior and source row rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ready source status badge in SourcesPane displays the design-system ready state label.
- [x] #2 Focused tests prove the ready badge label comes from the registry without source-string assertions.
- [x] #3 The matching canonical-state-label baseline exception is removed and the design-system state guard passes.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused SourcesPane test that mocks the design-system registry and verifies ready source status badges use the registry-provided label.
2. Replace the hardcoded ready source status label with `getDesignSystemState("ready").label` while leaving processing/error labels and guardrail behavior unchanged.
3. Remove the matching `canonical-state-label` baseline exception and verify the product-state guard still passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Red test first: `bunx vitest run src/components/Option/WorkspacePlayground/__tests__/SourcesPane.design-system.test.tsx --reporter=dot` failed because the source row still rendered `Ready` instead of the mocked registry label `Registry Ready`.
- The ready status badge now reads the label from the design-system state registry; processing and error status labels remain on their existing translated text paths.
- Removed the `canonical-state-label:src/components/Option/WorkspacePlayground/SourcesPane/index.tsx:Ready` baseline entry.
- Review fix: updated the focused design-system test mock to preserve the real `@/design-system` module exports and override only `getDesignSystemState`, matching the existing registry-label test pattern.
- Review disposition: did not apply optional chaining to `readyState.label` because `getDesignSystemState("ready")` is a typed lookup into the static design-system state registry; a missing `ready` key would violate the registry contract and should fail loudly instead of rendering an undefined badge.
- Broad `SourcesPane` TypeScript filtering surfaced an existing unrelated error in `SourcesPane.stage5.transfer.test.tsx`; an exact touched-path filter for `SourcesPane/index.tsx`, `SourcesPane.design-system.test.tsx`, and the baseline file returned no output.
- Bandit skipped: touched implementation is frontend TypeScript/test JSON only, with no Python code path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the WorkspacePlayground SourcesPane ready source-status badge to the design-system state registry by resolving `getDesignSystemState("ready")` and rendering its label for ready rows. Added focused coverage that mocks the registry and scopes the assertion to a source row, proving the badge uses the registry-provided label without source-string assertions.

Removed the now-obsolete canonical state label baseline exception. Review feedback was addressed by preserving real design-system module exports in the test mock. Verification passed with the focused SourcesPane design-system test, existing SourcesPane filters/sort coverage, the product-state guard unit suite, the design-system guard CLI, `git diff --check`, and an exact touched-path TypeScript error filter. Repo-wide TypeScript still has an unrelated pre-existing `SourcesPane.stage5.transfer.test.tsx` error.
<!-- SECTION:FINAL_SUMMARY:END -->
