# Migrate design-system product state: Jobs, Scheduler, and Watchlists

## Scope

Owned paths and product surfaces from the ordered path ownership map:

- `src/components/Option/Watchlists`
- `src/components/Option/AgentTasks`
- `src/components/Common/Workflow`

## Current Baseline Debt

Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
Snapshot date: 2026-05-14

- Total: 52
- `antd-product-state-import`: 52
- `canonical-state-label`: 0

Top current path groups:

- `src/components/Option/Watchlists`: 38
- `src/components/Common/Workflow`: 8
- `src/components/Option/AgentTasks`: 6

## Done Criteria

- This area has zero current product-state baseline exceptions.
- Focused tests cover migrated behavior.
- `bun run verify:design-system-state` passes from `apps/packages/ui`.
- `git diff --check` passes.
- Touched-file TypeScript filtering reports no diagnostics, or unrelated baseline diagnostics are documented.
- Bandit is run for Python touches or explicitly skipped for UI-only work.

## Tracking

- Parent epic: https://github.com/rmusser01/tldw_server/issues/1655
- Backlog task: TASK-45.44.3 (`backlog/tasks/task-45.44.3 - Migrate-design-system-product-state-Jobs-Scheduler-and-Watchlists.md`)
- PRs:

| PR | Notes |
| --- | --- |
| TBD | |

## Notes

- Keep AntD where it is only mechanics.
- Migrate product state language to shared primitives or the state registry.
- Split implementation into reviewable PRs when the area is too broad.
