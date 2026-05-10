---
id: TASK-249
title: Implement VN Play generation persistence
status: Done
assignee:
  - codex
created_date: '2026-05-10 21:44'
updated_date: '2026-05-10 21:57'
labels:
  - vn
  - scripted-generation
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: durable VN Play generation point, request, action, and revision persistence with idempotent command support. Scope is backend repository/storage behavior only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Schema initializes generation point request action and revision tables on fresh and upgraded DBs
- [x] #2 Generation point uniqueness is enforced per owner session and generation_point_key
- [x] #3 Generation action idempotency replays stored responses and rejects conflicting payloads
- [x] #4 Active revision updates require same generation and succeeded revision status
- [x] #5 Revision listing returns stable offset pagination order
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused DB tests first for generation schema initialization, point uniqueness, generation action idempotency conflict/replay, request/revision JSON/status fields, active revision validation, and stable revision pagination.
2. Extend VN_PLAY_SCHEMA_SQL with generation point/request/action/revision tables, indexes, and upgrade helpers for existing SQLite DBs.
3. Add VNPlayRepository helpers matching existing session-action style: owner-scoped create/get/update methods, JSON encode/decode, idempotent action creation, and conflict recovery.
4. Add revision lifecycle helpers, stable list pagination, and active revision pointer updates guarded by same-generation plus succeeded status.
5. Run the focused VN_Play tests changed, Bandit on touched backend code, and record results/known skips in TASK-249.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 2 in the vn-scripted-generation-runtime-plan worktree. MCP task_view could not resolve TASK-249 because the task file is untracked in this worktree; using backlog CLI fallback for task updates.

Implemented durable VN Play generation persistence in VNPlayRepository. Added guards so generation actions cannot point at revisions/requests from different generations, relation updates cannot create cross-session links, parent generation status stays synchronized with request status, and non-empty legacy generation tables migrate without SQLite non-constant default failures. Verification run: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py -q -> 33 passed, 5 warnings. compileall and git diff --check passed. Bandit command: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/VNPlay_DB.py -f json -o /tmp/bandit_vn_play_generation_persistence.json -> exit 0; only nosec B608 informational warnings for fixed internal schema statements.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 2 repository primitives for VN Play scripted generation persistence. Added generation point, request, action, and revision schema plus upgrade helpers, owner-scoped repository methods, JSON encode/decode handling, idempotent generation actions, request/revision status metadata storage, same-generation revision/action guards, migration-safe legacy table upgrades, succeeded-only active revision activation, and stable revision history pagination. Focused VN Play tests and touched-scope Bandit pass. Known skip: full repository test suite was not run; scope was narrowed to Task 2 repository primitives and the listed VN Play tests.
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
