---
id: TASK-349.1.3
title: Stage 4C Watchlist item triage client contract
status: Done
assignee: []
created_date: '2026-05-15 18:18'
updated_date: '2026-05-15 18:49'
labels:
  - watchlists
  - stage4
  - frontend
dependencies:
  - TASK-349.1.1
  - TASK-349.1.2
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
documentation:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-349.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the frontend TypeScript and service contract for Stage 4 item triage after the backend sort/filter, batch, and saved-view APIs exist. Scope is limited to types, service methods, query serialization, and saved-view migration helpers; do not redesign ItemsTab in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend Watchlists types expose item alert summaries, item sort/filter literals, batch update payloads/results, and saved view contracts.
- [x] #2 watchlists.ts service methods serialize Stage 4 item filters/sort, batch triage requests, and saved view CRUD routes correctly.
- [x] #3 ItemsTab utility helpers normalize saved views and preserve a recoverable localStorage-to-server migration path.
- [x] #4 Focused Vitest service and utility tests cover query serialization, invalid localStorage data, Watchlist-scoped saved view payloads, and batch endpoint calls.
- [x] #5 No visible ItemsTab behavior is changed outside type/service/helper wiring in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after Stage 4B commit 550f5ab6b. Scope: frontend TypeScript types, watchlists service methods/query serialization, and ItemsTab saved-view migration helpers only; no visible ItemsTab behavior changes.

Implemented Stage 4C frontend client contract: item alert summary/sort/filter/batch/saved-view types, watchlists service serialization and routes, and ItemsTab local saved-view migration helpers. No visible ItemsTab behavior changes were made.

Verification: ./node_modules/.bin/vitest run src/services/__tests__/watchlists-items-triage.test.ts src/services/__tests__/watchlists-first-class.test.ts src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts --maxWorkers=1 --no-file-parallelism passed 3 files / 45 tests. git diff --check passed.

Known skip: frontend TypeScript-only task, so Bandit is not applicable. Earlier tsc --noEmit still fails on unrelated repo-wide baseline TypeScript errors outside this touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 4C frontend client contract for watchlist item triage: shared types for alert summaries, alert filters, batch update responses, and saved views; service methods for item filters/sort, batch triage, and saved-view CRUD; and ItemsTab utility helpers to map existing local saved presets into server saved-view payloads. Verified with focused Vitest coverage and diff whitespace checks; no visible ItemsTab behavior changed in this task.
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
