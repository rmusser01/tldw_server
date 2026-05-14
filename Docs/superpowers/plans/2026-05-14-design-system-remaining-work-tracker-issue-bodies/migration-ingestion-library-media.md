# Migrate design-system product state: Ingestion, Library, and media

Draft only: human review and approval are required before creating or updating any public GitHub issue from this body.

## Scope

Owned paths and product surfaces from the ordered path ownership map:

- `src/components/Option/Ingestion`
- `src/components/Option/Library`
- `src/components/Option/Media`
- `src/components/Option/Sources`
- `src/components/Option/DataTables`
- `src/components/Option/AudiobookStudio`
- `src/components/Option/ChunkingPlayground`
- `src/components/Common/QuickIngest`
- `src/components/Timeline`

## Current Baseline Debt

Baseline source: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
Snapshot date: 2026-05-14

- Total: 39
- `antd-product-state-import`: 39
- `canonical-state-label`: 0

Top current path groups:

- `src/components/Option/AudiobookStudio`: 11
- `src/components/Option/ChunkingPlayground`: 11
- `src/components/Option/Sources`: 7
- `src/components/Option/DataTables`: 5
- `src/components/Common/QuickIngest`: 3
- `src/components/Timeline/TimelineModal.tsx`: 2

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
