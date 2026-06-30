---
id: TASK-45.44.2.6
title: Migrate SourceDetailPage alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.2
references:
- apps/packages/ui/src/components/Option/Sources/SourceDetailPage.tsx
- apps/packages/ui/src/components/Option/Sources/__tests__/SourceDetailPage.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/issues/1659
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Ingestion/Library/media product-state migration by replacing SourceDetailPage last-error and source-identity AntD Alerts with the shared design-system Alert primitive, then remove the matching baseline exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SourceDetailPage last-error alert renders through the shared design-system Alert primitive while preserving the error text.
- [x] #2 SourceDetailPage immutable source identity hint renders through the shared design-system Alert primitive as a polite status notice.
- [x] #3 The two SourceDetailPage AntD Alert baseline entries are removed without introducing new guard findings.
- [x] #4 Focused Vitest, design-system guard verification, TypeScript touched-scope check, and diff whitespace checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Migrated SourceDetailPage `detail.last_error` and source-identity notices from AntD `Alert` to `@/components/ui/primitives` `Alert` using `variant="error"` and `variant="info"`.
- Extended `SourceDetailPage.test.tsx` to assert both notices render inside `[data-ds-component="Alert"]`; the error notice keeps `role="alert"`, and the source-identity hint renders as a polite `role="status"` notice.
- Removed the two matching `src/components/Option/Sources/SourceDetailPage.tsx` baseline entries from `design-system-product-state-baseline.json`; SourceDetailPage has zero baseline entries.
- Refreshed inherited Watchlists baseline drift by replacing stale current-line IDs and adding current missing IDs, so the design-system product-state verifier passes for the PR branch.
- Verification:
  - RED: `bunx vitest run src/components/Option/Sources/__tests__/SourceDetailPage.test.tsx --maxWorkers=1 --no-file-parallelism` failed before the migration because the notices were not inside `[data-ds-component="Alert"]`.
  - GREEN: `bunx vitest run src/components/Option/Sources/__tests__/SourceDetailPage.test.tsx --maxWorkers=1 --no-file-parallelism` passed, 3 tests.
  - GREEN: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --no-file-parallelism` passed, 52 tests.
  - GREEN: `git diff --check`.
  - TYPECHECK: `bunx tsc --noEmit --pretty false` exits 2 on the existing broad WebUI TypeScript backlog; filtering `/tmp/ds-sourcedetail-tsc-after-rebase.log` for `SourceDetailPage`, `design-system-product-state-baseline`, and `TASK-45.44.2.6` returned no matches.
  - GREEN: `bun run verify:design-system-state` passed after refreshing inherited Watchlists baseline drift; report shows 404 allowed baseline exceptions, including 401 `antd-product-state-import` and 3 `canonical-state-label` entries, with no blocked or stale sections.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
SourceDetailPage now uses the shared design-system Alert primitive for its last-error and immutable source-identity product-state notices, with focused coverage proving the canonical primitive and expected ARIA behavior. The two SourceDetailPage legacy baseline exceptions were removed, inherited Watchlists baseline drift was refreshed, and the repo-wide design-system product-state verifier passes on the PR branch.
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
