---
id: TASK-45.12
title: Adapt WatchlistsEmptyState to shared EmptyState
status: Done
assignee: []
created_date: '2026-05-06 19:09'
updated_date: '2026-05-06 19:26'
labels:
  - design-system
  - frontend
  - watchlists
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_inventory.md
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the shared product-state design-system migration by replacing the Watchlists shared empty-state wrapper's direct AntD Empty/Button rendering with the canonical components/ui/feedback/EmptyState primitive. Keep this as a narrow compatibility slice for Watchlists shared empty-state behavior; do not migrate unrelated Watchlists alerts, status tags, loading states, page shells, or modal footers in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WatchlistsEmptyState renders the canonical EmptyState design-system marker while preserving entity-specific descriptions, context hints, icons, primary and secondary actions, override labels, test IDs, and i18n fallbacks.
- [x] #2 Focused tests cover at least one entity with primary and secondary actions and one entity without secondary action, including absent secondary-action behavior where relevant.
- [x] #3 The product-state guard passes without WatchlistsEmptyState local-empty-state or AntD Empty baseline debt, and any migrated stale baseline entries are removed.
- [x] #4 Scope remains limited to the Watchlists shared empty-state wrapper and direct tests, without migrating unrelated Watchlists AntD Alert/Tag/loading/status surfaces.
- [x] #5 Focused Vitest coverage, design-system state verification, git diff checks, and Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented WatchlistsEmptyState as a compatibility adapter over components/ui/feedback/EmptyState. Preserved entity descriptions, contextual hint copy, entity icons, primary/secondary action behavior, override labels, and legacy test IDs via EmptyState action passthroughs.

Verification: red WatchlistsEmptyState test failed on missing data-ds-component marker before implementation. After implementation, bunx vitest run src/components/Option/Watchlists/shared/__tests__/WatchlistsEmptyState.test.tsx src/components/Common/__tests__/FeatureEmptyState.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed 43/43; bun run verify:design-system-state exited 0 with 523 allowed legacy exceptions and no stale WatchlistsEmptyState entries; git diff --check exited 0. Package-wide bunx tsc --noEmit --pretty false -p tsconfig.json still exits 2 on unrelated existing frontend type errors outside touched Watchlists/EmptyState files. Bandit is not applicable to this frontend-only TypeScript/JSON slice.

PR review pass started for PR #1343. Actionable findings: remove unused AntD mock and consolidate lucide-react imports. Gemini EmptyStateAction.icon type comment is response-only because action icons intentionally follow Button.icon React.ReactNode.

PR review pass complete. Removed the unused AntD mock from WatchlistsEmptyState.test.tsx and consolidated the lucide-react type/value import in WatchlistsEmptyState.tsx. Rechecked EmptyStateAction.icon against Button.icon and kept it as React.ReactNode intentionally; this preserves JSX button adornment compatibility while EmptyStateProps.icon remains the hero LucideIcon type.

Verification after review fixes: bunx vitest run src/components/Option/Watchlists/shared/__tests__/WatchlistsEmptyState.test.tsx src/components/Common/__tests__/FeatureEmptyState.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed 43/43; bun run verify:design-system-state exited 0 with 523 allowed legacy exceptions; git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted the shared WatchlistsEmptyState wrapper to the canonical EmptyState primitive, added focused coverage for feed and monitor empty states, extended EmptyState actions with icon and data-testid passthroughs needed for compatibility wrappers, and removed the migrated WatchlistsEmptyState AntD Empty/local-empty-state baseline debt.
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
