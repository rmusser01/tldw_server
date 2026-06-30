---
id: TASK-282
title: Migrate SourceStatusPanels degraded label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 02:05'
updated_date: '2026-05-12 03:04'
labels:
  - design-system
  - frontend
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the hardcoded degraded status label in SourceStatusPanels with the design-system state registry fallback while preserving the numeric summary text and existing tag rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused tests prove the rendered degraded summary uses the registry fallback label without source-string assertions.
- [x] #2 The matching canonical-state-label and stale AntD Tag baseline exceptions are removed and the design-system state guard passes.
- [x] #3 The degraded status label in SourceStatusPanels comes from getDesignSystemState("degraded").label.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red test first: SourceStatusPanels.design-system.test.tsx initially failed because the component rendered Degraded 2 instead of the mocked registry label Registry Degraded 2.

Verification passed: SourceStatusPanels design-system test; SourceDetailPage plus SourceStatusPanels tests; product-state guard tests; bun run verify:design-system-state; git diff --check; touched-path tsc filter produced no SourceStatusPanels or baseline errors. Bandit skipped because this is frontend TypeScript/test-only work.

PR review follow-up: Gemini flagged the module-level degraded label constant and remaining AntD Tag usage for the degraded state; Qodo flagged that variable-based registry labels make the AntD product-state guard lose visibility. Plan is to remove the AntD degraded Tag by using a design-system badge primitive and to keep label lookup inside the component render path.

PR review fix implemented: removed the module-level DEGRADED_STATE_LABEL constant, resolves the degraded state inside SourceStatusPanels render, and renders the degraded summary with the design-system Badge primitive using getBadgeVariantForDesignSystemSeverity. Added test assertions that the registry lookup happens during render and the degraded summary renders as data-ds-component=Badge with warning variant, not AntD Tag.

Review-fix verification passed: SourceDetailPage plus SourceStatusPanels tests; product-state guard tests; bun run verify:design-system-state; git diff --check; touched-path tsc filter produced no SourceStatusPanels or baseline output.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated SourceStatusPanels degraded summary labels to the design-system state registry and badge primitive, added focused render coverage for registry lookup and Badge rendering, and removed the resolved baseline exceptions for the component.
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
