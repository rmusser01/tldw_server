---
id: TASK-45.28
title: Adapt PresentationStudioStatusBadge to design-system state registry
status: Done
assignee: []
created_date: '2026-05-09 18:40'
updated_date: '2026-05-09 18:48'
labels:
  - design-system
  - ui
  - product-state
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioStatusBadge.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Presentation Studio asset status badge from direct variant mapping to the canonical design-system state registry while preserving the existing compact badge UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Presentation Studio asset statuses resolve through getDesignSystemState before choosing Badge severity styling.
- [x] #2 The badge continues to render the shared Badge primitive with dot, sm sizing, caller className support, and stable labels for missing, ready, stale, generating, failed, and nullish statuses.
- [x] #3 Focused tests cover status-to-label/variant behavior and nullish fallback behavior.
- [x] #4 The design-system product-state baseline no longer contains the PresentationStudioStatusBadge local-status-badge exception.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented with TDD: added a failing adapter test proving PresentationStudioStatusBadge calls getDesignSystemState for missing/ready/stale/generating/failed/nullish statuses, then routed the component through getBadgeVariantForDesignSystemSeverity while preserving Badge dot, sm size, className, and visible labels. Verification: focused PresentationStudioStatusBadge test passed; product-state guard unit test passed; verify:design-system-state passed; git diff --check passed. Repo-wide tsc remains red on unrelated existing UI type debt and produced no diagnostics for PresentationStudioStatusBadge or its new test. Bandit skipped because touched implementation is TS/TSX/JSON/Backlog only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PresentationStudioStatusBadge now resolves asset statuses through the canonical design-system state registry before selecting the shared Badge severity variant, with regression coverage for all known statuses and nullish fallback. Removed the obsolete PresentationStudioStatusBadge local-status-badge baseline exception.
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
