---
id: TASK-281
title: Migrate ReviewPage ready status label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 01:27'
updated_date: '2026-05-12 01:50'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the ReviewPage hardcoded Ready product-state label from the guard baseline by sourcing the ready status fallback from the design-system state registry without changing review status behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ReviewPage ready status fallback label is sourced from the design-system state registry instead of a hardcoded canonical label.
- [x] #2 The product-state guard baseline no longer includes the ReviewPage Ready-label exception.
- [x] #3 Focused regression coverage proves the registry fallback is used for the ReviewPage ready status label.
- [x] #4 Design-system guard verification passes for this slice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_server/pull/1585

Updated ReviewPage ready status fallback to use getDesignSystemState("ready")?.label. Added focused behavior coverage in ReviewPage.connection.test.tsx by mocking the design-system registry to return a distinctive ready label and asserting ReviewPage renders that fallback for a ready media result. Removed the matching canonical-state-label baseline exception.

Verification: red ReviewPage guard failed before implementation, then ReviewPage guard passed, product-state guard tests passed, verify:design-system-state passed, git diff --check passed, touched-path tsc filter returned no matches. Bandit skipped because this is a frontend TypeScript/test-only slice with no Python runtime changes.

Addressed PR review feedback by making the ready state lookup optional and replacing the source-string guard with behavior coverage. This removed the __dirname source-path helper called out in review. Re-ran ReviewPage guard, product-state guard tests, verify:design-system-state, git diff --check, and touched-path tsc filter after the review fixes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the ReviewPage ready status fallback label to the design-system state registry and removed its product-state guard baseline exception. Focused coverage now prevents the ready status fallback from regressing to a hardcoded canonical label.
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
