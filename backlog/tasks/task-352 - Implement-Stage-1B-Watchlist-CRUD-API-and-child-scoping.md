---
id: TASK-352
title: Implement Stage 1B Watchlist CRUD API and child scoping
status: In Progress
assignee: []
created_date: '2026-05-15 01:43'
updated_date: '2026-05-15 01:43'
labels:
  - watchlists
  - backend
  - api
  - stage1b
dependencies:
  - TASK-351
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the API slice from the Stage 1 first-class Watchlists plan. Scope is limited to Pydantic schemas, watchlists router endpoints, and backend calls needed for CRUD plus watchlist_id scoping on sources/jobs/runs/items while preserving existing unscoped behavior and legacy /{watchlist_id}/clusters route semantics. Do not start frontend work in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 API tests cover Watchlist CRUD, patch lifecycle fields, delete/restore, default scope for omitted watchlist_id, explicit watchlist_id filtering, and legacy /{watchlist_id}/clusters stability.
- [ ] #2 watchlists_schemas.py exposes Watchlist create/update/container/list/delete schemas and backward-compatible watchlist_id fields on source/job contracts.
- [ ] #3 watchlists.py adds root CRUD endpoints without shadowing static child routes and wires watchlist_id to source/job/run/item create/list paths.
- [ ] #4 Focused Watchlists API and DB tests pass after implementation.
- [ ] #5 Bandit is run on touched backend API/schema/router code or any skip is explicitly documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 1B in isolated worktree .worktrees/watchlists-stage1a on branch codex/watchlists-stage1a. Scope is limited to Watchlist CRUD API schemas/router tests and child endpoint watchlist_id scoping; frontend work remains out of scope.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
