# Migrate design-system product state: Chat and Playground

## Scope

Owned paths and product surfaces from the ordered path ownership map:

- `src/components/Option/Playground`
- `src/components/Common/Playground`
- `src/components/Sidepanel/Chat`
- `src/routes/sidepanel-chat.tsx`

## Current Baseline Debt

Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
Snapshot date: 2026-05-14

- Total: 2
- `antd-product-state-import`: 0
- `canonical-state-label`: 2

Top current path groups:

- `src/components/Option/Playground`: 1
- `src/routes/sidepanel-chat.tsx`: 1

## Done Criteria

- This area has zero current product-state baseline exceptions.
- Focused tests cover migrated behavior.
- `bun run verify:design-system-state` passes from `apps/packages/ui`.
- `git diff --check` passes.
- Touched-file TypeScript filtering reports no diagnostics, or unrelated baseline diagnostics are documented.
- Bandit is run for Python touches or explicitly skipped for UI-only work.

## Tracking

- Parent epic: https://github.com/rmusser01/tldw_server/issues/1655
- Backlog task: TASK-45.44.1 (`backlog/tasks/task-45.44.1 - Migrate-design-system-product-state-Chat-and-Playground.md`)
- PRs:

| PR | Notes |
| --- | --- |
| TBD | |

## Notes

- Keep AntD where it is only mechanics.
- Migrate product state language to shared primitives or the state registry.
- Split implementation into reviewable PRs when the area is too broad.
