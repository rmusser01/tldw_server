# Migrate design-system product state: Document and Workspace surfaces

## Scope

Owned paths and product surfaces from the ordered path ownership map:

- `src/components/DocumentWorkspace`
- `src/components/Option/Workspace`

## Current Baseline Debt

Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
Snapshot date: 2026-05-14

- Total: 13
- `antd-product-state-import`: 13
- `canonical-state-label`: 0

Top current path groups:

- `src/components/DocumentWorkspace/DocumentViewer`: 4
- `src/components/DocumentWorkspace/DocumentPickerModal.tsx`: 3
- `src/components/DocumentWorkspace/LeftSidebar`: 3
- `src/components/DocumentWorkspace/DocumentWorkspacePage.tsx`: 2
- `src/components/DocumentWorkspace/DocumentWorkspaceErrorBoundary.tsx`: 1

## Done Criteria

- This area has zero current product-state baseline exceptions.
- Focused tests cover migrated behavior.
- `bun run verify:design-system-state` passes from `apps/packages/ui`.
- `git diff --check` passes.
- Touched-file TypeScript filtering reports no diagnostics, or unrelated baseline diagnostics are documented.
- Bandit is run for Python touches or explicitly skipped for UI-only work.

## Tracking

- Parent epic: TBD
- Backlog task: TBD
- PRs: TBD

## Notes

- Keep AntD where it is only mechanics.
- Migrate product state language to shared primitives or the state registry.
- Split implementation into reviewable PRs when the area is too broad.
