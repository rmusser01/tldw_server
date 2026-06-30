---
id: TASK-349.3.2
title: Stage 6B Sources and Monitors constrained management
status: Done
dependencies:
- TASK-349.3.1
labels:
- watchlists
- stage6
- frontend
- sources
- monitors
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace wide table-only Feeds and Monitors management with constrained list/detail patterns while preserving desktop tables, source bulk actions, source CRUD, OPML import, monitor CRUD, run now, preview, delete/undo, and active toggles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Feeds/Sources constrained viewport renders a list/detail management view instead of the wide table while preserving desktop table behavior.
- [x] #2 Source add, edit, delete/undo, active toggle, check now, seen details, OPML import, filters, group/tag context, and bulk actions remain reachable at 420x760.
- [x] #3 Monitors constrained viewport renders a list/detail management view instead of the wide table while preserving desktop table behavior.
- [x] #4 Monitor add, edit, delete/undo, active toggle, run now, preview, schedule, scope/filter summary, output linkage, and pagination remain reachable at 420x760.
- [x] #5 Focused Vitest coverage proves constrained source/monitor management and existing delete/bulk/advanced-details regressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 6B implemented constrained-width Feeds and Monitors management using the Stage 6A viewport helper. Feeds now render card/list management at constrained width with search, filters, group context, OPML import, advanced toggle, selection, bulk actions, active toggle, check-now, seen details, edit, and delete reachable without the wide table. Monitors now render card/list management at constrained width with schedule, scope, filters, output linkage, last/next run, active toggle, run now, preview, edit, and delete reachable without the wide table. Desktop table paths and existing service/store contracts remain intact.

Verification:
- cd apps/packages/ui && bunx vitest run src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.extension-management.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.bulk-move.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.delete-confirm.test.tsx src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.extension-management.test.tsx src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.advanced-details.test.tsx src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.undo-delete.test.tsx --maxWorkers=1 --no-file-parallelism: 6 files, 21 tests passed.
- cd apps/packages/ui && bun run test:watchlists:typecheck: 1 file, 3 tests passed.
- cd apps/packages/ui && node -e JSON locale parse for src/assets/locale/en/watchlists.json and src/public/_locales/en/watchlists.json: both ok.
- git diff --check: passed.
- Bandit: not applicable; touched files are frontend TypeScript/tests, locale JSON, docs, and Backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6B replaced the table-only constrained Feeds and Monitors surfaces with card/list management views while preserving desktop table behavior. Focused tests cover constrained source and monitor workflows plus existing bulk/delete/advanced-details regressions. Static Watchlists guard, locale JSON parsing, and whitespace checks passed; Bandit was not applicable for this frontend-only slice.
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
