# Migrate design-system product state: Writing and Review surfaces

## Scope

Owned paths and product surfaces from the ordered path ownership map:

- `src/components/Option/WritingPlayground`
- `src/components/Review`

## Current Baseline Debt

Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
Snapshot date: 2026-05-14

- Total: 22
- `antd-product-state-import`: 21
- `canonical-state-label`: 1

Top current path groups:

- `src/components/Option/WritingPlayground`: 21
- `src/components/Review/MediaReviewReadingPane.tsx`: 1

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
