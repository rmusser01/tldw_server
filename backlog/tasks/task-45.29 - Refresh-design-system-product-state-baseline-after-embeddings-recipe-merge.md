---
id: TASK-45.29
title: Refresh design-system product-state baseline after embeddings recipe merge
status: Done
assignee: []
created_date: '2026-05-09 18:44'
updated_date: '2026-05-09 18:48'
labels:
  - design-system
  - ui
  - product-state
dependencies: []
references:
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx
  - >-
    apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile the product-state guard baseline with the current dev branch after the embeddings recipe flow merge introduced new AntD Alert findings and left stale RecipesTab baseline IDs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current live embeddings recipe product-state findings are represented in the baseline as legacy design-system debt rather than blocked findings.
- [x] #2 Stale RecipesTab baseline IDs that no longer match live findings are removed.
- [x] #3 The design-system product-state verifier exits successfully after the baseline refresh and the Presentation Studio badge migration.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled baseline drift from the current dev branch: removed stale RecipesTab baseline IDs plus the migrated PresentationStudioStatusBadge entry, and added current embeddings recipe Alert findings as allowed legacy design-system debt. Verification: verify:design-system-state passed with 513 baseline exceptions and local-status-badge reduced to 2. Bandit skipped because touched scope is TS/TSX/JSON/Backlog only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the product-state guard baseline so current embeddings recipe findings are represented as legacy debt instead of blocked findings, and stale RecipesTab IDs are removed.
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
